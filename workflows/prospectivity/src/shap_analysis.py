"""Outer-OOF TreeSHAP attribution, reliability gates, and locked bridge outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score

from export_utils import (
    IntegrityLedger,
    load_config,
    read_lines,
    save_json,
    save_table,
    sha256_file,
    sha256_payload,
    stage_complete,
)
from model_core import (
    MODEL_INDEX,
    FoldModelMatrix,
    build_model,
    fit_model,
    predict_score,
)
from nested_tuning import load_primary_input


def _load_selection(paths: dict[str, Path]) -> dict[str, Any]:
    path = paths["results"] / "model_selection_decision.json"
    if not path.exists():
        raise FileNotFoundError("Run repeated nested model evaluation before TreeSHAP.")
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_fold_parameters(paths: dict[str, Path], selected_model: str) -> pd.DataFrame:
    table = pd.read_csv(paths["optuna"] / "best_trial_by_outer_fold.csv", low_memory=False)
    result = table[table["model"].eq(selected_model)].copy()
    if result.empty:
        raise ValueError(f"No selected outer-fold parameters were saved for {selected_model}.")
    result["parameters"] = result["parameters_json"].map(json.loads)
    return result


def _tree_shap_values(
    model_name: str, model: Any, matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import shap
    except ImportError as exc:
        raise ImportError("SHAP is required. Create the environment from environment.yml and select that Jupyter kernel.") from exc
    explainer = shap.TreeExplainer(model)
    explanation = explainer(matrix, check_additivity=False)
    values = np.asarray(explanation.values)
    base = np.asarray(explanation.base_values)
    if values.ndim == 3:
        if values.shape[2] != 2:
            raise ValueError(f"Unexpected multiclass TreeSHAP dimensions: {values.shape}")
        values = values[:, :, 1]
        base = base[:, 1] if base.ndim == 2 else np.repeat(np.asarray(explainer.expected_value)[1], len(matrix))
    elif values.ndim != 2:
        raise ValueError(f"Unexpected TreeSHAP dimensions: {values.shape}")
    if base.ndim == 0:
        base = np.repeat(float(base), len(matrix))
    if model_name == "RF":
        raw_output = model.predict_proba(matrix)[:, 1]
    elif model_name == "XGBoost":
        try:
            import xgboost as xgb
            raw_output = model.get_booster().predict(xgb.DMatrix(matrix), output_margin=True)
        except Exception as exc:
            raise RuntimeError("Unable to calculate XGBoost raw margins for SHAP additivity audit.") from exc
    else:
        raise ValueError("OOF TreeSHAP is restricted to the objectively selected tree model.")
    return values, np.asarray(base, dtype=float).reshape(-1), np.asarray(raw_output, dtype=float)


def run_selected_model_oof_shap(config_path: str | Path) -> dict[str, Any]:
    config, root, paths, frame, features = load_primary_input(config_path)
    protocol_contract = json.loads(
        (paths["audit"] / "analysis_protocol_contract.json").read_text(encoding="utf-8")
    )
    protocol_hash = str(protocol_contract["analysis_protocol_hash"])
    run_id = f"{config['analysis_revision']}::{protocol_hash[:12]}"
    ledger = IntegrityLedger(paths["audit"] / "preflight_and_integrity_checks.json", run_id=run_id)
    decision = _load_selection(paths)
    selected_model = str(decision["selected_model"])
    status_path = paths["shap"] / "selected_model_treeshap_status.json"
    if selected_model not in {"RF", "XGBoost"}:
        status = {
            "status": "not_applicable_selected_model_is_not_tree_based",
            "selected_model": selected_model,
            "substitution_performed": False,
            "note": "No alternative model was substituted for the objectively selected non-tree model.",
        }
        save_json(status_path, status)
        save_json(paths["bridge"] / "taskA_bridge_contract.json", {
            "analysis_revision": config["analysis_revision"],
            "analysis_protocol_hash": protocol_hash,
            "selected_model": selected_model,
            "bridge_status": "not_generated_selected_model_is_not_tree_based",
            "coupling_use": "paused",
            "causal_interpretation_prohibited": True,
        })
        return status
    gate_config_at_start = dict(config["shap"])
    gate_hash = sha256_payload(gate_config_at_start)
    sample_oof = pd.read_csv(paths["oof"] / "sample_level_repeated_oof.csv", low_memory=False)
    sample_oof = sample_oof[sample_oof["model"].eq(selected_model)].copy()
    split_registry = pd.read_csv(paths["oof"] / "outer_split_registry.csv", low_memory=False)
    fold_parameters = _selected_fold_parameters(paths, selected_model)
    output_parts = []
    reproduction_rows = []
    columns = config["columns"]
    for parameter_row in fold_parameters.itertuples(index=False):
        repeat = int(parameter_row.repeat)
        outer_fold = int(parameter_row.outer_fold)
        params = parameter_row.parameters
        parameter_hash = str(parameter_row.parameter_hash)
        split = split_registry[
            split_registry["repeat"].eq(repeat) & split_registry["outer_fold"].eq(outer_fold)
        ]
        train_groups = set(split.loc[split["partition"].eq("train"), "Geological Group ID"])
        valid_groups = set(split.loc[split["partition"].eq("validation"), "Geological Group ID"])
        train = frame[frame[columns["group_id"]].isin(train_groups)].reset_index(drop=True)
        valid = frame[frame[columns["group_id"]].isin(valid_groups)].reset_index(drop=True)
        train_blocks = set(train[columns["cv_block_id"]])
        valid_blocks = set(valid[columns["cv_block_id"]])
        ledger.check(
            f"shap_refit_block_disjoint_r{repeat}_f{outer_fold}", train_blocks.isdisjoint(valid_blocks),
            {"overlap": sorted(train_blocks.intersection(valid_blocks))},
        )
        matrix = FoldModelMatrix(features, config)
        x_train = matrix.fit_transform(train)
        x_valid = matrix.transform(valid)
        completed = matrix.completed_raw(valid)
        model_seed = int(config["validation"]["base_seed"]) + repeat * 10000 + outer_fold * 1000 + MODEL_INDEX[selected_model] * 100
        model = build_model(selected_model, params, model_seed)
        fit_model(selected_model, model, x_train, train, config, model_seed)
        model_score = predict_score(selected_model, model, x_valid)
        saved = sample_oof[
            sample_oof["repeat"].eq(repeat) & sample_oof["outer_fold"].eq(outer_fold)
        ].set_index("Record ID")
        valid_ids = valid[columns["record_id"]].astype(str)
        if set(valid_ids) != set(saved.index.astype(str)):
            raise RuntimeError(f"SHAP refit validation records do not match saved OOF records for r{repeat} f{outer_fold}.")
        saved_score = saved.loc[valid_ids, "model_score"].to_numpy(float)
        score_error = float(np.max(np.abs(model_score - saved_score)))
        ledger.check(
            f"shap_refit_matches_saved_oof_r{repeat}_f{outer_fold}",
            score_error <= float(config["shap"]["maximum_oof_reproduction_absolute_error"]),
            {"maximum_absolute_model_score_error": score_error},
        )
        values, base, raw_output = _tree_shap_values(selected_model, model, x_valid)
        additivity_error = np.abs(base + values.sum(axis=1) - raw_output)
        part = valid[[
            columns["record_id"], columns["group_id"], columns["cv_block_id"],
            columns["reference_id"], columns["label"]
        ]].copy()
        part.columns = ["Record ID", "Geological Group ID", "CV Block ID", "Reference ID", "target"]
        part["repeat"] = repeat
        part["outer_fold"] = outer_fold
        part["model"] = selected_model
        part["parameter_hash"] = parameter_hash
        part["model_score"] = model_score
        part["score_type"] = "predict_proba"
        part["decision_threshold"] = float(parameter_row.threshold)
        part["analysis_protocol_hash"] = protocol_hash
        part["base_value"] = base
        part["raw_model_output"] = raw_output
        part["additivity_absolute_error"] = additivity_error
        shap_columns = {}
        for index, feature in enumerate(features):
            shap_columns[f"SHAP::{feature}"] = values[:, index]
            shap_columns[f"COMPLETED::{feature}"] = completed[feature].to_numpy()
            shap_columns[f"OBSERVED::{feature}"] = pd.to_numeric(valid[feature], errors="coerce").to_numpy()
            shap_columns[f"IMPUTED::{feature}"] = valid[feature].isna().astype(int).to_numpy()
        part = pd.concat([part, pd.DataFrame(shap_columns, index=part.index)], axis=1)
        output_parts.append(part)
        reproduction_rows.append({
            "repeat": repeat, "outer_fold": outer_fold, "model": selected_model,
            "parameter_hash": parameter_hash,
            "maximum_model_score_reproduction_error": score_error,
            "maximum_additivity_absolute_error": float(additivity_error.max()),
            "mean_additivity_absolute_error": float(additivity_error.mean()),
            "validation_records": len(valid),
        })
    paired = pd.concat(output_parts, ignore_index=True)
    reproduction = pd.DataFrame(reproduction_rows)
    expected = len(frame) * int(config["validation"]["outer_repeats"])
    ledger.check("selected_model_oof_shap_has_expected_record_repeats", len(paired) == expected, {
        "observed_rows": len(paired), "expected_rows": expected
    })
    parameter_lookup = fold_parameters.set_index(["repeat", "outer_fold"])["parameter_hash"]
    parameter_match = paired.apply(
        lambda row: row["parameter_hash"] == parameter_lookup.loc[(row["repeat"], row["outer_fold"])], axis=1
    )
    ledger.check("shap_parameter_hash_matches_outer_fold_selection", bool(parameter_match.all()), {
        "mismatches": int((~parameter_match).sum())
    })
    save_table(paired, paths["shap"] / "selected_model_paired_oof_shap.csv")
    save_table(reproduction, paths["shap"] / "selected_model_oof_reproduction_and_additivity.csv")
    importance, repeat_ranks = summarize_global_shap(paired, features, config, selected_model)
    missingness = pd.read_csv(paths["audit"] / "missingness_by_label_audit.csv", low_memory=False)
    importance = importance.merge(
        missingness[["feature", "absolute_label_missingness_gap"]],
        on="feature", how="left", validate="one_to_one",
    )
    save_table(importance, paths["shap"] / "global_shap_importance_and_stability.csv")
    save_table(repeat_ranks, paths["shap"] / "global_shap_repeat_ranks.csv")
    quality_sensitivity = run_quality_panel_sensitivity(
        frame, features, paired, importance, fold_parameters, selected_model, config, paths
    )
    gate = build_global_attribution_gate(
        paired, importance, repeat_ranks, reproduction, quality_sensitivity,
        config, gate_hash, protocol_hash
    )
    config_reloaded = json.loads(Path(config_path).read_text(encoding="utf-8"))
    gate_hash_at_end = sha256_payload(config_reloaded["shap"])
    ledger.check("shap_gate_configuration_immutable_during_run", gate_hash == gate_hash_at_end, {
        "start_hash": gate_hash, "end_hash": gate_hash_at_end
    })
    save_json(paths["shap"] / "global_attribution_reliability_gate.json", gate)
    locked = read_lines(root / config["shap"]["locked_bridge_feature_file"])
    for feature in locked:
        if feature not in features:
            paired[f"SHAP::{feature}"] = np.nan
            paired[f"OBSERVED::{feature}"] = np.nan
            paired[f"COMPLETED::{feature}"] = np.nan
            paired[f"IMPUTED::{feature}"] = np.nan
    locked_table = build_locked_feature_reproducibility(
        importance, locked, config, quality_sensitivity
    )
    ledger.check("bridge_uses_exact_locked_nine_features", locked_table["feature"].tolist() == locked, {
        "features": locked_table["feature"].tolist()
    })
    save_table(locked_table, paths["shap"] / "locked_nine_feature_reproducibility.csv")
    candidates = build_candidate_stability_table(importance, config, quality_sensitivity)
    save_table(candidates, paths["shap"] / "optuna_derived_candidate_features_v2.csv")
    save_table(candidates, paths["shap"] / "candidate_feature_stability_table.csv")
    missingness_sensitivity = run_missingness_indicator_sensitivity(
        frame, features, fold_parameters, selected_model, config, paths
    )
    bridge, contract = build_record_bridge(
        paired, features, locked, locked_table, selected_model, gate,
        protocol_contract, candidates, config
    )
    save_table(bridge, paths["bridge"] / "taskA_oof_bridge_one_row_per_Record_ID.csv")
    save_table(locked_table, paths["bridge"] / "taskA_locked_bridge_features_v1.csv")
    save_json(paths["bridge"] / "taskA_bridge_contract.json", contract)
    status = {
        "status": "completed",
        "selected_model": selected_model,
        "global_gate_passed": gate["global_gate_passed"],
        "bridge_status": contract["bridge_status"],
        "paired_oof_rows": len(paired),
        "record_bridge_rows": len(bridge),
        "missingness_indicator_sensitivity": missingness_sensitivity,
    }
    save_json(status_path, status)
    stage_complete(
        paths["logs"] / "stage_03_oof_shap.complete.json",
        "oof_tree_shap", {"model_selection": sha256_file(paths["results"] / "model_selection_decision.json")},
        [
            paths["shap"] / "selected_model_paired_oof_shap.csv",
            paths["shap"] / "global_attribution_reliability_gate.json",
            paths["bridge"] / "taskA_oof_bridge_one_row_per_Record_ID.csv",
            paths["bridge"] / "taskA_bridge_contract.json",
        ],
        analysis_revision=config["analysis_revision"],
        analysis_protocol_hash=protocol_hash,
    )
    return status


def run_missingness_indicator_sensitivity(
    frame: pd.DataFrame,
    features: list[str],
    fold_parameters: pd.DataFrame,
    selected_model: str,
    config: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    """Audit whether a target feature's missingness pattern is exploited by the model.

    The selected model is refit once without and once with a raw binary missingness
    indicator for the target feature, on the same frozen folds. If the consensus
    group-level ROC-AUC and F1 do not change beyond the prespecified tolerances, the
    missingness pattern is not treated as a separate predictive signal.
    """
    target_feature = str(config["shap"].get("missingness_indicator_target_feature", "Fe2O3 (wt.%)"))
    output_path = paths["shap"] / "missingness_indicator_sensitivity.csv"
    if target_feature not in features:
        payload = [{
            "status": "skipped_target_feature_not_in_predictors",
            "target_feature": target_feature,
        }]
        save_table(pd.DataFrame(payload), output_path)
        return {"status": "skipped_target_feature_not_in_predictors", "target_feature": target_feature}

    columns = config["columns"]
    group_id = columns["group_id"]
    label = columns["label"]
    split_registry = pd.read_csv(paths["oof"] / "outer_split_registry.csv", low_memory=False)
    group_rows: list[pd.DataFrame] = []
    fold_aucs: list[dict[str, Any]] = []
    fold_thresholds: list[float] = []

    for parameter_row in fold_parameters.itertuples(index=False):
        repeat = int(parameter_row.repeat)
        outer_fold = int(parameter_row.outer_fold)
        params = parameter_row.parameters
        fold_thresholds.append(float(parameter_row.threshold))
        split = split_registry[
            split_registry["repeat"].eq(repeat) & split_registry["outer_fold"].eq(outer_fold)
        ]
        train_groups = set(split.loc[split["partition"].eq("train"), "Geological Group ID"])
        valid_groups = set(split.loc[split["partition"].eq("validation"), "Geological Group ID"])
        train = frame[frame[group_id].isin(train_groups)].reset_index(drop=True)
        valid = frame[frame[group_id].isin(valid_groups)].reset_index(drop=True)

        matrix = FoldModelMatrix(features, config)
        x_train = np.asarray(matrix.fit_transform(train), dtype=float)
        x_valid = np.asarray(matrix.transform(valid), dtype=float)
        model_seed = (
            int(config["validation"]["base_seed"])
            + repeat * 10000
            + outer_fold * 1000
            + MODEL_INDEX[selected_model] * 100
        )
        indicator_train = train[target_feature].isna().astype(float).to_numpy().reshape(-1, 1)
        indicator_valid = valid[target_feature].isna().astype(float).to_numpy().reshape(-1, 1)

        variants = {
            "baseline": (x_train, x_valid),
            "with_missingness_indicator": (
                np.hstack([x_train, indicator_train]),
                np.hstack([x_valid, indicator_valid]),
            ),
        }
        for variant, (x_tr, x_va) in variants.items():
            model = build_model(selected_model, params, model_seed)
            fit_model(selected_model, model, x_tr, train, config, model_seed)
            scores = np.asarray(predict_score(selected_model, model, x_va), dtype=float)
            grouped = pd.DataFrame({
                "repeat": repeat,
                "outer_fold": outer_fold,
                "variant": variant,
                group_id: valid[group_id].to_numpy(),
                label: valid[label].to_numpy(float),
                "model_score": scores,
            }).groupby(group_id, as_index=False).agg(
                **{label: (label, "first"), "model_score": ("model_score", "median")}
            )
            grouped["repeat"] = repeat
            grouped["outer_fold"] = outer_fold
            grouped["variant"] = variant
            group_rows.append(grouped)
            fold_aucs.append({
                "repeat": repeat,
                "outer_fold": outer_fold,
                "variant": variant,
                "group_level_roc_auc": float(roc_auc_score(grouped[label], grouped["model_score"])),
            })

    pooled = pd.concat(group_rows, ignore_index=True)
    threshold_used = float(np.median(fold_thresholds))
    summary_rows: list[dict[str, Any]] = []
    for variant in ["baseline", "with_missingness_indicator"]:
        subset = pooled[pooled["variant"].eq(variant)]
        auc = float(roc_auc_score(subset[label], subset["model_score"]))
        prediction = (subset["model_score"] >= threshold_used).astype(int)
        f1 = float(f1_score(subset[label], prediction, pos_label=1, zero_division=0))
        summary_rows.append({
            "variant": variant,
            "consensus_group_roc_auc": auc,
            "consensus_group_f1": f1,
            "threshold_used": threshold_used,
            "n_groups": int(len(subset)),
        })
    summary = pd.DataFrame(summary_rows)
    baseline = summary.loc[summary["variant"].eq("baseline")].iloc[0]
    augmented = summary.loc[summary["variant"].eq("with_missingness_indicator")].iloc[0]
    auc_delta = abs(float(augmented["consensus_group_roc_auc"] - baseline["consensus_group_roc_auc"]))
    f1_delta = abs(float(augmented["consensus_group_f1"] - baseline["consensus_group_f1"]))
    max_auc_delta = float(config["shap"].get("missingness_indicator_max_auc_delta", 0.02))
    max_f1_delta = float(config["shap"].get("missingness_indicator_max_f1_delta", 0.05))
    not_exploited = bool(auc_delta <= max_auc_delta and f1_delta <= max_f1_delta)
    summary["auc_absolute_delta_vs_baseline"] = [0.0, auc_delta]
    summary["f1_absolute_delta_vs_baseline"] = [0.0, f1_delta]
    summary["missingness_pattern_not_significantly_exploited"] = [bool(not_exploited), bool(not_exploited)]
    save_table(summary, output_path)
    save_table(pd.DataFrame(fold_aucs), paths["shap"] / "missingness_indicator_sensitivity_per_fold.csv")
    return {
        "status": "completed",
        "target_feature": target_feature,
        "consensus_group_roc_auc_baseline": float(baseline["consensus_group_roc_auc"]),
        "consensus_group_roc_auc_with_indicator": float(augmented["consensus_group_roc_auc"]),
        "consensus_group_f1_baseline": float(baseline["consensus_group_f1"]),
        "consensus_group_f1_with_indicator": float(augmented["consensus_group_f1"]),
        "auc_absolute_delta": auc_delta,
        "f1_absolute_delta": f1_delta,
        "missingness_pattern_not_significantly_exploited": not_exploited,
    }


def _bootstrap_mean_abs(
    group_values: pd.DataFrame, feature: str, iterations: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = group_values[feature].to_numpy(float)
    estimates = []
    for _ in range(iterations):
        sample = values[rng.integers(0, len(values), len(values))]
        estimates.append(np.mean(np.abs(sample)))
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _bootstrap_spearman_direction(
    table: pd.DataFrame,
    value_column: str,
    shap_column: str,
    iterations: int,
    seed: int,
    cluster_column: str | None = None,
) -> dict[str, float]:
    subset = table[[value_column, shap_column] + ([cluster_column] if cluster_column else [])].dropna()
    if len(subset) < 8 or subset[value_column].nunique() < 4:
        return {"estimate": np.nan, "bootstrap_median": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    estimate = float(spearmanr(subset[value_column], subset[shap_column]).statistic)
    rng = np.random.default_rng(seed)
    estimates = []
    if cluster_column:
        cluster_array = subset[cluster_column].to_numpy()
        clusters = subset[cluster_column].drop_duplicates().to_numpy()
        rows_by_cluster = {cluster: np.flatnonzero(cluster_array == cluster) for cluster in clusters}
        value_array = subset[value_column].to_numpy(float)
        shap_array = subset[shap_column].to_numpy(float)
        for _ in range(iterations):
            sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
            row_index = np.concatenate([rows_by_cluster[cluster] for cluster in sampled_clusters])
            value = spearmanr(value_array[row_index], shap_array[row_index]).statistic
            if np.isfinite(value):
                estimates.append(float(value))
    else:
        for _ in range(iterations):
            sampled = subset.iloc[rng.integers(0, len(subset), len(subset))]
            value = spearmanr(sampled[value_column], sampled[shap_column]).statistic
            if np.isfinite(value):
                estimates.append(float(value))
    if not estimates:
        return {"estimate": estimate, "bootstrap_median": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    values = np.asarray(estimates)
    return {
        "estimate": estimate,
        "bootstrap_median": float(np.median(values)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
    }


def summarize_global_shap(
    paired: pd.DataFrame, features: list[str], config: dict[str, Any], selected_model: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_repeat = paired.groupby(["repeat", "Geological Group ID"], as_index=False).agg({
        "CV Block ID": "first",
        **{f"SHAP::{feature}": "median" for feature in features},
        **{f"COMPLETED::{feature}": "median" for feature in features},
        **{f"IMPUTED::{feature}": "mean" for feature in features},
    })
    rank_parts = []
    direction_rows = []
    for repeat, subset in group_repeat.groupby("repeat"):
        importance = subset[[f"SHAP::{feature}" for feature in features]].abs().mean().sort_values(ascending=False)
        for rank, (column, value) in enumerate(importance.items(), 1):
            feature = column.replace("SHAP::", "", 1)
            rank_parts.append({"repeat": repeat, "feature": feature, "rank": rank, "mean_abs_shap": value})
        repeat_records = paired[paired["repeat"].eq(repeat)]
        for feature in features:
            correlation = spearmanr(
                subset[f"COMPLETED::{feature}"], subset[f"SHAP::{feature}"], nan_policy="omit"
            ).statistic
            observed_records = repeat_records[repeat_records[f"IMPUTED::{feature}"].eq(0)]
            observed_groups = observed_records.groupby("Geological Group ID", as_index=False).agg({
                f"OBSERVED::{feature}": "median", f"SHAP::{feature}": "median"
            })
            if len(observed_groups) >= 5:
                observed_correlation = spearmanr(
                    observed_groups[f"OBSERVED::{feature}"], observed_groups[f"SHAP::{feature}"], nan_policy="omit"
                ).statistic
            else:
                observed_correlation = np.nan
            direction_rows.append({
                "repeat": repeat, "feature": feature,
                "signed_direction_spearman": float(correlation) if np.isfinite(correlation) else 0.0,
                "observed_only_direction_spearman": float(observed_correlation) if np.isfinite(observed_correlation) else np.nan,
                "direction_sign": int(np.sign(correlation)) if np.isfinite(correlation) else 0,
            })
    ranks = pd.DataFrame(rank_parts).merge(pd.DataFrame(direction_rows), on=["repeat", "feature"])
    group_overall = group_repeat.groupby("Geological Group ID", as_index=False).agg({
        "CV Block ID": "first",
        **{f"SHAP::{feature}": "median" for feature in features},
        **{f"COMPLETED::{feature}": "median" for feature in features},
        **{f"IMPUTED::{feature}": "mean" for feature in features},
    })
    rows = []
    iterations = int(config["shap"]["bootstrap_iterations"])
    for feature in features:
        feature_ranks = ranks[ranks["feature"].eq(feature)]
        direction_counts = feature_ranks["direction_sign"].value_counts()
        agreement = float(direction_counts.max() / max(len(feature_ranks), 1))
        ci_low, ci_high = _bootstrap_mean_abs(
            group_overall, f"SHAP::{feature}", iterations,
            int(config["validation"]["base_seed"]) + features.index(feature),
        )
        completed_direction = _bootstrap_spearman_direction(
            group_overall, f"COMPLETED::{feature}", f"SHAP::{feature}", iterations,
            int(config["validation"]["base_seed"]) + 10000 + features.index(feature),
        )
        completed_block_direction = _bootstrap_spearman_direction(
            group_overall, f"COMPLETED::{feature}", f"SHAP::{feature}",
            int(config["shap"]["cluster_bootstrap_iterations"]),
            int(config["validation"]["base_seed"]) + 20000 + features.index(feature),
            cluster_column="CV Block ID",
        )
        observed_records = paired[paired[f"IMPUTED::{feature}"].eq(0)]
        observed_groups = observed_records.groupby("Geological Group ID", as_index=False).agg({
            "CV Block ID": "first", f"OBSERVED::{feature}": "median", f"SHAP::{feature}": "median"
        })
        observed_direction = _bootstrap_spearman_direction(
            observed_groups, f"OBSERVED::{feature}", f"SHAP::{feature}", iterations,
            int(config["validation"]["base_seed"]) + 30000 + features.index(feature),
        )
        completed_ci_excludes_zero = bool(
            np.isfinite(completed_direction["ci_low"])
            and (completed_direction["ci_low"] > 0 or completed_direction["ci_high"] < 0)
        )
        observed_ci_excludes_zero = bool(
            np.isfinite(observed_direction["ci_low"])
            and (observed_direction["ci_low"] > 0 or observed_direction["ci_high"] < 0)
        )
        observed_completed_agreement = bool(
            np.sign(completed_direction["estimate"]) != 0
            and np.sign(completed_direction["estimate"]) == np.sign(observed_direction["estimate"])
        )
        rows.append({
            "model": selected_model, "feature": feature,
            "mean_abs_oof_shap": float(group_overall[f"SHAP::{feature}"].abs().mean()),
            "bootstrap_ci_low": ci_low, "bootstrap_ci_high": ci_high,
            "mean_rank": float(feature_ranks["rank"].mean()),
            "rank_sd": float(feature_ranks["rank"].std(ddof=1)),
            "direction_spearman_mean": float(feature_ranks["signed_direction_spearman"].mean()),
            "direction_spearman": completed_direction["estimate"],
            "direction_bootstrap_median": completed_direction["bootstrap_median"],
            "direction_bootstrap_ci_low": completed_direction["ci_low"],
            "direction_bootstrap_ci_high": completed_direction["ci_high"],
            "direction_block_bootstrap_ci_low": completed_block_direction["ci_low"],
            "direction_block_bootstrap_ci_high": completed_block_direction["ci_high"],
            "observed_only_direction_spearman": observed_direction["estimate"],
            "observed_only_direction_bootstrap_median": observed_direction["bootstrap_median"],
            "observed_only_direction_ci_low": observed_direction["ci_low"],
            "observed_only_direction_ci_high": observed_direction["ci_high"],
            "direction_sign": int(np.sign(completed_direction["estimate"])) if np.isfinite(completed_direction["estimate"]) else 0,
            "completed_direction_ci_excludes_zero": completed_ci_excludes_zero,
            "observed_direction_ci_excludes_zero": observed_ci_excludes_zero,
            "observed_completed_direction_agreement": observed_completed_agreement,
            "direction_agreement_fraction": agreement,
            "primary_missing_fraction": float(group_repeat[f"IMPUTED::{feature}"].mean()),
            "imputed_fraction": float(paired[f"IMPUTED::{feature}"].mean()),
            "observed_group_count": int(len(observed_groups)),
        })
    importance = pd.DataFrame(rows).sort_values("mean_abs_oof_shap", ascending=False).reset_index(drop=True)
    importance["new_rank"] = np.arange(1, len(importance) + 1)
    importance["normalized_importance"] = importance["mean_abs_oof_shap"] / importance["mean_abs_oof_shap"].sum()
    importance["top10_repeat_count"] = importance["feature"].map(
        ranks[ranks["rank"].le(10)].groupby("feature")["repeat"].nunique()
    ).fillna(0).astype(int)
    return importance, ranks


def build_global_attribution_gate(
    paired: pd.DataFrame,
    importance: pd.DataFrame,
    ranks: pd.DataFrame,
    reproduction: pd.DataFrame,
    quality_sensitivity: dict[str, Any],
    config: dict[str, Any],
    gate_hash: str,
    protocol_hash: str,
) -> dict[str, Any]:
    repeat_pivots = ranks.pivot(index="feature", columns="repeat", values="rank")
    correlations = []
    top10_overlaps = []
    columns = list(repeat_pivots.columns)
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1:]:
            correlations.append(float(spearmanr(repeat_pivots[left], repeat_pivots[right]).statistic))
            left_top = set(repeat_pivots[left].nsmallest(10).index)
            right_top = set(repeat_pivots[right].nsmallest(10).index)
            top10_overlaps.append(len(left_top.intersection(right_top)) / 10.0)
    top10 = importance.nsmallest(10, "new_rank")
    checks = {
        "oof_reproduction": float(reproduction["maximum_model_score_reproduction_error"].max()) <= float(config["shap"]["maximum_oof_reproduction_absolute_error"]),
        "tree_shap_additivity": float(reproduction["maximum_additivity_absolute_error"].max()) <= float(config["shap"]["maximum_additivity_absolute_error"]),
        "repeat_rank_spearman": bool(correlations) and min(correlations) >= float(config["shap"]["minimum_repeat_rank_spearman"]),
        "top10_overlap": bool(top10_overlaps) and min(top10_overlaps) >= float(config["shap"]["minimum_pairwise_top10_overlap"]),
        "direction_stability": bool(top10["direction_agreement_fraction"].ge(float(config["shap"]["minimum_direction_agreement_fraction"])).all()),
        "top10_observed_completed_direction_agreement": bool(top10["observed_completed_direction_agreement"].fillna(False).all()),
        "quality_panel_sensitivity": bool(quality_sensitivity.get("quality_panel_gate_passed", False)),
    }
    return {
        "analysis_revision": config["analysis_revision"],
        "analysis_protocol_hash": protocol_hash,
        "gate_configuration_hash": gate_hash,
        "thresholds": config["shap"],
        "checks": checks,
        "global_gate_passed": bool(all(checks.values())),
        "maximum_oof_score_reproduction_error": float(reproduction["maximum_model_score_reproduction_error"].max()),
        "maximum_additivity_absolute_error": float(reproduction["maximum_additivity_absolute_error"].max()),
        "repeat_rank_spearman_values": correlations,
        "pairwise_top10_overlap_values": top10_overlaps,
        "quality_panel_sensitivity": quality_sensitivity,
        "interpretation_level": "exploratory_model_attribution",
        "interpretation": "Passing supports exploratory attribution transfer; it does not establish geological causality.",
    }


def build_locked_feature_reproducibility(
    importance: pd.DataFrame,
    locked: list[str],
    config: dict[str, Any],
    quality_sensitivity: dict[str, Any],
) -> pd.DataFrame:
    lookup = importance.set_index("feature")
    quality_features = set(quality_sensitivity.get("quality_panel_features", []))
    rows = []
    for legacy_order, feature in enumerate(locked, 1):
        if feature not in lookup.index:
            rows.append({
                "feature": feature, "legacy_locked_order": legacy_order, "new_rank": np.nan,
                "status": "not_reproduced_under_optuna_revision", "reason": "feature_not_in_retained_panel",
            })
            continue
        row = lookup.loc[feature]
        candidate_rule = config["shap"]["candidate_rule"]
        stability_checks = {
            "rank_range": row["new_rank"] <= int(candidate_rule["maximum_new_rank"]),
            "rank_sd": row["rank_sd"] <= float(config["shap"]["maximum_rank_standard_deviation"]),
            "repeat_direction": row["direction_agreement_fraction"] >= float(config["shap"]["minimum_direction_agreement_fraction"]),
            "completed_direction_ci": bool(row["completed_direction_ci_excludes_zero"]),
            "observed_direction_ci": bool(row["observed_direction_ci_excludes_zero"]),
            "observed_completed_direction": bool(row["observed_completed_direction_agreement"]),
            "missing_fraction": row["primary_missing_fraction"] <= float(candidate_rule["maximum_missing_fraction"]),
            "label_missingness_gap": row["absolute_label_missingness_gap"] <= float(candidate_rule["maximum_label_missingness_gap"]),
            "quality_panel_feature": feature in quality_features,
            "quality_panel_global_gate": bool(quality_sensitivity.get("quality_panel_gate_passed", False)),
        }
        stable = bool(all(stability_checks.values()))
        failed = [name for name, passed in stability_checks.items() if not passed]
        rows.append({
            "feature": feature, "legacy_locked_order": legacy_order, "new_rank": int(row["new_rank"]),
            "mean_abs_oof_shap": row["mean_abs_oof_shap"],
            "mean_abs_oof_shap_ci_low": row["bootstrap_ci_low"],
            "mean_abs_oof_shap_ci_high": row["bootstrap_ci_high"],
            "rank_sd": row["rank_sd"],
            "direction_spearman": row["direction_spearman"],
            "direction_bootstrap_ci_low": row["direction_bootstrap_ci_low"],
            "direction_bootstrap_ci_high": row["direction_bootstrap_ci_high"],
            "direction_block_bootstrap_ci_low": row["direction_block_bootstrap_ci_low"],
            "direction_block_bootstrap_ci_high": row["direction_block_bootstrap_ci_high"],
            "direction_sign": row["direction_sign"],
            "direction_agreement_fraction": row["direction_agreement_fraction"],
            "observed_only_direction_spearman": row["observed_only_direction_spearman"],
            "observed_only_direction_ci_low": row["observed_only_direction_ci_low"],
            "observed_only_direction_ci_high": row["observed_only_direction_ci_high"],
            "observed_completed_direction_agreement": row["observed_completed_direction_agreement"],
            "primary_missing_fraction": row["primary_missing_fraction"],
            "imputed_fraction": row["imputed_fraction"],
            "absolute_label_missingness_gap": row["absolute_label_missingness_gap"],
            "included_in_quality_panel": feature in quality_features,
            "status": "reproduced_under_optuna_revision" if stable else "not_reproduced_under_optuna_revision",
            "reason": "all_predeclared_stability_checks_passed" if stable else ";".join(failed),
        })
    return pd.DataFrame(rows)


def run_quality_panel_sensitivity(
    frame: pd.DataFrame,
    features: list[str],
    paired: pd.DataFrame,
    importance: pd.DataFrame,
    fold_parameters: pd.DataFrame,
    selected_model: str,
    config: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    quality_file = paths["processed"] / "quality_controlled_feature_list.txt"
    quality_features = [feature for feature in read_lines(quality_file) if feature in features]
    rules = config["shap"]["quality_panel"]
    minimum_count = int(rules["minimum_feature_count"])
    if len(quality_features) < minimum_count:
        result = {
            "status": "failed_insufficient_quality_controlled_features",
            "quality_panel_feature_count": len(quality_features),
            "quality_panel_features": quality_features,
            "minimum_feature_count": minimum_count,
            "quality_panel_gate_passed": False,
        }
        save_json(paths["shap"] / "quality_panel_sensitivity_summary.json", result)
        return result

    split_registry = pd.read_csv(paths["oof"] / "outer_split_registry.csv", low_memory=False)
    columns = config["columns"]
    score_parts: list[pd.DataFrame] = []
    shap_parts: list[pd.DataFrame] = []
    for parameter_row in fold_parameters.itertuples(index=False):
        repeat = int(parameter_row.repeat)
        outer_fold = int(parameter_row.outer_fold)
        split = split_registry[
            split_registry["repeat"].eq(repeat) & split_registry["outer_fold"].eq(outer_fold)
        ]
        train_groups = set(split.loc[split["partition"].eq("train"), "Geological Group ID"])
        valid_groups = set(split.loc[split["partition"].eq("validation"), "Geological Group ID"])
        train = frame[frame[columns["group_id"]].isin(train_groups)].reset_index(drop=True)
        valid = frame[frame[columns["group_id"]].isin(valid_groups)].reset_index(drop=True)
        matrix = FoldModelMatrix(quality_features, config)
        x_train = matrix.fit_transform(train)
        x_valid = matrix.transform(valid)
        completed = matrix.completed_raw(valid)
        seed = int(config["validation"]["base_seed"]) + repeat * 10000 + outer_fold * 1000 + MODEL_INDEX[selected_model] * 100
        model = build_model(selected_model, parameter_row.parameters, seed)
        fit_model(selected_model, model, x_train, train, config, seed)
        scores = predict_score(selected_model, model, x_valid)
        values, _, _ = _tree_shap_values(selected_model, model, x_valid)
        score_part = valid[[columns["record_id"], columns["group_id"], columns["cv_block_id"], columns["label"]]].copy()
        score_part.columns = ["Record ID", "Geological Group ID", "CV Block ID", "target"]
        score_part["repeat"] = repeat
        score_part["outer_fold"] = outer_fold
        score_part["model_score"] = scores
        score_part["decision_threshold"] = float(parameter_row.threshold)
        score_part["prediction"] = (scores >= float(parameter_row.threshold)).astype(int)
        score_parts.append(score_part)
        shap_part = score_part[["Record ID", "Geological Group ID", "CV Block ID", "target", "repeat", "outer_fold"]].copy()
        for index, feature in enumerate(quality_features):
            shap_part[f"SHAP::{feature}"] = values[:, index]
            shap_part[f"COMPLETED::{feature}"] = completed[feature].to_numpy()
        shap_parts.append(shap_part)

    sample_scores = pd.concat(score_parts, ignore_index=True)
    sample_shap = pd.concat(shap_parts, ignore_index=True)
    grouped_scores = sample_scores.groupby(["repeat", "Geological Group ID"], as_index=False).agg(
        target=("target", "first"), cv_block=("CV Block ID", "first"),
        model_score=("model_score", "median"), decision_threshold=("decision_threshold", "first"),
    )
    grouped_scores["prediction"] = (grouped_scores["model_score"] >= grouped_scores["decision_threshold"]).astype(int)
    consensus = grouped_scores.groupby("Geological Group ID", as_index=False).agg(
        target=("target", "first"), cv_block=("cv_block", "first"),
        model_score=("model_score", "median"), positive_vote_fraction=("prediction", "mean"),
    )
    consensus["prediction"] = (consensus["positive_vote_fraction"] >= 0.5).astype(int)
    primary_consensus = pd.read_csv(paths["results"] / "geological_group_consensus_oof.csv", low_memory=False)
    primary_consensus = primary_consensus[primary_consensus["model"].eq(selected_model)]
    primary_auc = float(roc_auc_score(primary_consensus["target"], primary_consensus["model_score"]))
    primary_f1 = float(f1_score(primary_consensus["target"], primary_consensus["prediction"], zero_division=0))
    quality_auc = float(roc_auc_score(consensus["target"], consensus["model_score"]))
    quality_f1 = float(f1_score(consensus["target"], consensus["prediction"], zero_division=0))

    group_shap = sample_shap.groupby(["repeat", "Geological Group ID"], as_index=False).agg({
        **{f"SHAP::{feature}": "median" for feature in quality_features},
        **{f"COMPLETED::{feature}": "median" for feature in quality_features},
    })
    quality_rows = []
    for feature in quality_features:
        values = group_shap[f"SHAP::{feature}"].abs().groupby(group_shap["repeat"]).mean()
        directions = group_shap.groupby("repeat").apply(
            lambda part: spearmanr(part[f"COMPLETED::{feature}"], part[f"SHAP::{feature}"], nan_policy="omit").statistic
        )
        quality_rows.append({
            "feature": feature,
            "mean_abs_oof_shap": float(group_shap[f"SHAP::{feature}"].abs().mean()),
            "repeat_importance_mean": float(values.mean()),
            "direction_spearman_mean": float(np.nanmean(directions)),
            "direction_sign": int(np.sign(np.nanmean(directions))),
        })
    quality_importance = pd.DataFrame(quality_rows).sort_values("mean_abs_oof_shap", ascending=False).reset_index(drop=True)
    quality_importance["quality_rank"] = np.arange(1, len(quality_importance) + 1)
    primary_top10 = set(importance.nsmallest(10, "new_rank")["feature"])
    quality_top10 = set(quality_importance.head(10)["feature"])
    shared = primary_top10.intersection(quality_top10)
    overlap = len(shared) / 10.0
    primary_direction = importance.set_index("feature")["direction_sign"]
    quality_direction = quality_importance.set_index("feature")["direction_sign"]
    direction_comparable = [feature for feature in shared if primary_direction.get(feature, 0) != 0 and quality_direction.get(feature, 0) != 0]
    direction_agreement = (
        float(np.mean([primary_direction[feature] == quality_direction[feature] for feature in direction_comparable]))
        if direction_comparable else 0.0
    )
    checks = {
        "minimum_feature_count": len(quality_features) >= minimum_count,
        "auc_absolute_delta": abs(quality_auc - primary_auc) <= float(rules["maximum_auc_absolute_delta"]),
        "f1_absolute_delta": abs(quality_f1 - primary_f1) <= float(rules["maximum_f1_absolute_delta"]),
        "top10_overlap": overlap >= float(rules["minimum_top10_overlap"]),
        "direction_agreement": direction_agreement >= float(rules["minimum_direction_agreement_fraction"]),
    }
    result = {
        "status": "completed",
        "quality_panel_feature_count": len(quality_features),
        "quality_panel_features": quality_features,
        "primary_auc": primary_auc, "quality_panel_auc": quality_auc,
        "auc_absolute_delta": abs(quality_auc - primary_auc),
        "primary_f1": primary_f1, "quality_panel_f1": quality_f1,
        "f1_absolute_delta": abs(quality_f1 - primary_f1),
        "top10_overlap_fraction": overlap,
        "shared_top10_features": sorted(shared),
        "direction_agreement_fraction": direction_agreement,
        "checks": checks,
        "quality_panel_gate_passed": bool(all(checks.values())),
    }
    save_table(sample_scores, paths["shap"] / "quality_panel_sample_level_repeated_oof.csv")
    save_table(grouped_scores, paths["shap"] / "quality_panel_group_level_repeated_oof.csv")
    save_table(consensus, paths["shap"] / "quality_panel_consensus_group_oof.csv")
    save_table(quality_importance, paths["shap"] / "quality_panel_shap_importance.csv")
    save_json(paths["shap"] / "quality_panel_sensitivity_summary.json", result)
    return result


def build_candidate_stability_table(
    importance: pd.DataFrame,
    config: dict[str, Any],
    quality_sensitivity: dict[str, Any],
) -> pd.DataFrame:
    rule = config["shap"]["candidate_rule"]
    rule_hash = sha256_payload(rule)
    quality_features = set(quality_sensitivity.get("quality_panel_features", []))
    table = importance.copy()
    table["quality_panel_status"] = table["feature"].isin(quality_features)
    table["quality_panel_global_gate_passed"] = bool(quality_sensitivity.get("quality_panel_gate_passed", False))
    table["candidate_rule_hash"] = rule_hash
    table["passes_rank"] = table["new_rank"].le(int(rule["maximum_new_rank"]))
    table["passes_rank_sd"] = table["rank_sd"].le(float(rule["maximum_rank_standard_deviation"]))
    table["passes_direction_agreement"] = table["direction_agreement_fraction"].ge(float(rule["minimum_direction_agreement_fraction"]))
    table["passes_direction_ci"] = table["completed_direction_ci_excludes_zero"] & table["observed_direction_ci_excludes_zero"]
    table["passes_observed_completed_direction"] = table["observed_completed_direction_agreement"]
    table["passes_missingness"] = table["primary_missing_fraction"].le(float(rule["maximum_missing_fraction"]))
    if "absolute_label_missingness_gap" not in table.columns:
        table["absolute_label_missingness_gap"] = np.nan
    table["passes_label_missingness_gap"] = table["absolute_label_missingness_gap"].le(float(rule["maximum_label_missingness_gap"]))
    required = [
        "passes_rank", "passes_rank_sd", "passes_direction_agreement", "passes_direction_ci",
        "passes_observed_completed_direction", "passes_missingness", "passes_label_missingness_gap",
    ]
    if bool(rule["quality_panel_status_required"]):
        required.extend(["quality_panel_status", "quality_panel_global_gate_passed"])
    table["eligible_for_exploratory_coupling"] = table[required].all(axis=1)
    table["eligibility_reason"] = table.apply(
        lambda row: "all_predeclared_rules_passed" if row["eligible_for_exploratory_coupling"]
        else ";".join([name.replace("passes_", "") for name in required if not bool(row[name])]), axis=1
    )
    return table


def build_record_bridge(
    paired: pd.DataFrame,
    features: list[str],
    locked: list[str],
    locked_table: pd.DataFrame,
    selected_model: str,
    gate: dict[str, Any],
    protocol_contract: dict[str, Any],
    candidates: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    aggregations: dict[str, Any] = {
        "Geological Group ID": "first", "CV Block ID": "first", "Reference ID": "first", "target": "first",
        "model_score": ["median", "mean", "std"], "decision_threshold": ["median", "min", "max"],
        "repeat": "nunique",
        "analysis_protocol_hash": "first", "score_type": "first", "additivity_absolute_error": "max",
    }
    for feature in locked:
        aggregations[f"SHAP::{feature}"] = ["mean", "std"]
        aggregations[f"OBSERVED::{feature}"] = "median"
        aggregations[f"COMPLETED::{feature}"] = "mean"
        aggregations[f"IMPUTED::{feature}"] = "mean"
    grouped = paired.groupby("Record ID", as_index=False).agg(aggregations)
    grouped.columns = [
        "::".join([str(value) for value in column if value]) if isinstance(column, tuple) else str(column)
        for column in grouped.columns
    ]
    rename = {
        "Record ID::": "Record ID", "Geological Group ID::first": "Geological Group ID",
        "CV Block ID::first": "CV Block ID", "Reference ID::first": "Reference ID", "target::first": "target",
        "model_score::median": "PROSPECTIVITY_MODEL_SCORE_MEDIAN",
        "model_score::mean": "PROSPECTIVITY_MODEL_SCORE_MEAN_SENSITIVITY",
        "model_score::std": "PROSPECTIVITY_MODEL_SCORE_SD",
        "decision_threshold::median": "DECISION_THRESHOLD_MEDIAN",
        "decision_threshold::min": "DECISION_THRESHOLD_MIN",
        "decision_threshold::max": "DECISION_THRESHOLD_MAX",
        "repeat::nunique": "OOF_REPEAT_COUNT",
        "analysis_protocol_hash::first": "ANALYSIS_PROTOCOL_HASH",
        "score_type::first": "MODEL_SCORE_TYPE",
        "additivity_absolute_error::max": "SHAP_ADDITIVITY_MAX_ERROR",
    }
    for feature in locked:
        rename[f"SHAP::{feature}::mean"] = f"PROSPECTIVITY_SHAP_MEAN::{feature}"
        rename[f"SHAP::{feature}::std"] = f"PROSPECTIVITY_SHAP_SD::{feature}"
        rename[f"OBSERVED::{feature}::median"] = f"PROSPECTIVITY_OBSERVED_MEDIAN::{feature}"
        rename[f"COMPLETED::{feature}::mean"] = f"PROSPECTIVITY_COMPLETED_MEAN::{feature}"
        rename[f"IMPUTED::{feature}::mean"] = f"PROSPECTIVITY_IMPUTED_FRACTION::{feature}"
    grouped = grouped.rename(columns=rename)
    if grouped["Record ID"].duplicated().any():
        raise RuntimeError("Record bridge is not one row per Record ID.")
    parameter_hashes = paired.groupby("Record ID")["parameter_hash"].apply(
        lambda values: ";".join(sorted(set(values.astype(str))))
    )
    grouped["OUTER_FOLD_PARAMETER_HASHES"] = grouped["Record ID"].map(parameter_hashes)
    grouped["FEATURE_LIST_HASH"] = protocol_contract["feature_list_hash"]
    contract = {
        "schema_version": "task-a-bridge-v7.1",
        "analysis_revision": config["analysis_revision"],
        "analysis_protocol_hash": protocol_contract["analysis_protocol_hash"],
        "selected_model": selected_model,
        "row_unit": "one row per Record ID",
        "locked_bridge_features": locked,
        "locked_bridge_feature_count": len(locked),
        "locked_feature_count": len(locked),
        "locked_features_reproduced_count": int(locked_table["status"].eq("reproduced_under_optuna_revision").sum()),
        "all_locked_features_reproduced": bool(locked_table["status"].eq("reproduced_under_optuna_revision").all()),
        "locked_bridge_feature_status": locked_table[["feature", "status", "reason"]].to_dict("records"),
        "eligible_candidate_features": candidates.loc[candidates["eligible_for_exploratory_coupling"], "feature"].tolist(),
        "candidate_rule_hash": str(candidates["candidate_rule_hash"].iloc[0]) if len(candidates) else sha256_payload(config["shap"]["candidate_rule"]),
        "global_attribution_gate_passed": gate["global_gate_passed"],
        "bridge_status": "eligible_for_exploratory_part4_coupling" if gate["global_gate_passed"] else "generated_but_gate_not_passed",
        "exploratory_only": True,
        "coupling_use": "exploratory_only",
        "comparison_rule": "Use standardized rank, contribution share, and direction agreement; do not compare raw SHAP magnitudes across models.",
        "causal_interpretation_prohibited": True,
    }
    return grouped, contract
