from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import squareform
from scipy.special import logit, softmax
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from classification_pipeline import (
    _json_hash,
    _tree_shap,
    class_complete_group_splits,
    make_prepared_folds,
    optimize_model,
    output_paths,
    prepare_data,
    validate_analysis_runtime,
)
from model_core import (
    CLASSES,
    aggregate_group_type_probabilities,
    build_estimator,
    fit_estimator,
    multiclass_metrics,
    predict_score_matrix,
    prepare_fold,
)
from source_blocks import assert_no_block_overlap


EPS = np.finfo(float).eps


def _robustness_dir(config: dict[str, Any]) -> Path:
    folder = output_paths(config)["root"] / "08_Robustness"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _safe_spearman(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> float:
    pair = pd.DataFrame({"left": left, "right": right}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or pair["left"].nunique() < 2 or pair["right"].nunique() < 2:
        return float("nan")
    return float(spearmanr(pair["left"], pair["right"]).statistic)


def _temperature_transform(probability: np.ndarray, temperature: float) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-12, 1.0)
    logits = np.log(probability)
    return softmax(logits / max(float(temperature), 1e-6), axis=1)


def _fit_temperature(y_true: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(probability).all(axis=1)
    y_valid = np.asarray(y_true, dtype=int)[valid]
    p_valid = np.asarray(probability, dtype=float)[valid]
    if len(y_valid) < 30 or len(np.unique(y_valid)) != len(CLASSES):
        return {
            "temperature": 1.0,
            "fitted": False,
            "reason": "insufficient_inner_oof_class_coverage",
            "inner_oof_records": int(len(y_valid)),
        }

    def objective(log_temperature: float) -> float:
        transformed = _temperature_transform(p_valid, float(np.exp(log_temperature)))
        return float(log_loss(y_valid, transformed, labels=[0, 1, 2]))

    result = minimize_scalar(
        objective,
        bounds=(np.log(0.25), np.log(4.0)),
        method="bounded",
        options={"xatol": 1e-4, "maxiter": 200},
    )
    temperature = float(np.exp(result.x)) if result.success else 1.0
    return {
        "temperature": temperature,
        "fitted": bool(result.success),
        "reason": "" if result.success else "temperature_optimization_failed",
        "inner_oof_records": int(len(y_valid)),
        "inner_oof_log_loss_before": objective(0.0),
        "inner_oof_log_loss_after": objective(float(np.log(temperature))),
    }


def _inner_oof_probability(
    model_name: str,
    params: dict[str, Any],
    folds: list[Any],
    y: pd.Series,
    groups: pd.Series,
    config: dict[str, Any],
    seed: int,
) -> np.ndarray:
    probability = np.full((len(y), len(CLASSES)), np.nan)
    for fold in folds:
        estimator = build_estimator(
            model_name,
            params,
            seed + int(fold.fold_id),
            int(config["optimization"]["model_n_jobs"]),
        )
        estimator = fit_estimator(
            model_name,
            estimator,
            fold.x_training,
            y.iloc[fold.training_positions],
            groups.iloc[fold.training_positions],
        )
        probability[fold.validation_positions] = predict_score_matrix(
            model_name, estimator, fold.x_validation
        )
    return probability


def _ece_binary(y_true: np.ndarray, probability: np.ndarray, n_bins: int) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(probability, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for bin_id in range(n_bins):
        selected = bins == bin_id
        if selected.any():
            ece += selected.mean() * abs(float(probability[selected].mean()) - float(y_true[selected].mean()))
    return float(ece)


def _calibration_slope_intercept(y_binary: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y_binary)) < 2:
        return float("nan"), float("nan")
    predictor = logit(np.clip(probability, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    try:
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        model.fit(predictor, y_binary)
        return float(model.coef_[0, 0]), float(model.intercept_[0])
    except Exception:
        return float("nan"), float("nan")


def probability_diagnostics(
    y_true: np.ndarray,
    probability: np.ndarray,
    unit: str,
    probability_version: str,
    n_bins: int,
) -> pd.DataFrame:
    valid = np.isfinite(probability).all(axis=1)
    y_valid = np.asarray(y_true, dtype=int)[valid]
    p_valid = np.asarray(probability, dtype=float)[valid]
    rows: list[dict[str, Any]] = []
    if not len(y_valid):
        return pd.DataFrame()
    one_hot = np.eye(len(CLASSES))[y_valid]
    rows.extend([
        {
            "evaluation_unit": unit,
            "probability_version": probability_version,
            "class": "multiclass",
            "metric": "log_loss",
            "value": float(log_loss(y_valid, p_valid, labels=[0, 1, 2])),
            "n": int(len(y_valid)),
        },
        {
            "evaluation_unit": unit,
            "probability_version": probability_version,
            "class": "multiclass",
            "metric": "brier_score",
            "value": float(np.mean(np.sum((p_valid - one_hot) ** 2, axis=1))),
            "n": int(len(y_valid)),
        },
    ])
    confidence = p_valid.max(axis=1)
    correct = (p_valid.argmax(axis=1) == y_valid).astype(float)
    rows.append({
        "evaluation_unit": unit,
        "probability_version": probability_version,
        "class": "top_label",
        "metric": "ece",
        "value": _ece_binary(correct, confidence, n_bins),
        "n": int(len(y_valid)),
    })
    for class_index, class_name in enumerate(CLASSES):
        binary = (y_valid == class_index).astype(int)
        class_probability = p_valid[:, class_index]
        slope, intercept = _calibration_slope_intercept(binary, class_probability)
        for metric, value in (
            ("ovr_brier_score", np.mean((class_probability - binary) ** 2)),
            ("ovr_ece", _ece_binary(binary, class_probability, n_bins)),
            ("calibration_slope", slope),
            ("calibration_intercept", intercept),
        ):
            rows.append({
                "evaluation_unit": unit,
                "probability_version": probability_version,
                "class": class_name,
                "metric": metric,
                "value": float(value),
                "n": int(len(y_valid)),
                "positive_support": int(binary.sum()),
            })
    return pd.DataFrame(rows)


def run_label_source_audit(config: dict[str, Any]) -> dict[str, Any]:
    context = prepare_data(config)
    metadata = context["metadata"].copy()
    output = _robustness_dir(config)
    rows: list[dict[str, Any]] = []
    blocks = metadata["Reference-connected block"].astype(str)
    groups = metadata["Geological Group ID"].astype(str)
    labels = metadata["Granite type"].astype(str)
    for class_name in ["ALL", *CLASSES]:
        selected = np.ones(len(metadata), dtype=bool) if class_name == "ALL" else labels.eq(class_name).to_numpy()
        rows.append({
            "audit_level": "class_summary",
            "reported_type": class_name,
            "records": int(selected.sum()),
            "geological_groups": int(groups[selected].nunique()),
            "source_blocks": int(blocks[selected].nunique()),
            "label_field": "Granite type",
            "label_source": "reported granite type in Supplementary Table S1",
            "predictor_derived_label": "not asserted by code",
            "manual_provenance_verification_required": True,
        })
    group_type = metadata.assign(_group=groups, _label=labels).groupby("_group")["_label"].nunique()
    mixed_groups = set(group_type[group_type > 1].index)
    for block_id, subset in metadata.assign(_block=blocks, _group=groups, _label=labels).groupby("_block"):
        coverage = sorted(subset["_label"].unique())
        rows.append({
            "audit_level": "source_block",
            "source_block_id": block_id,
            "records": int(len(subset)),
            "geological_groups": int(subset["_group"].nunique()),
            "source_blocks": 1,
            "type_coverage": ";".join(coverage),
            "n_types": int(len(coverage)),
            "mixed_type_groups_in_block": int(subset["_group"].isin(mixed_groups).groupby(subset["_group"]).max().sum()),
            "manual_provenance_verification_required": True,
        })
    audit = pd.DataFrame(rows)
    audit["analysis_protocol_hash"] = context["analysis_protocol_hash"]
    audit.to_csv(output / "taskB_label_and_source_audit.csv", index=False, encoding="utf-8-sig")
    return {
        "records": int(len(metadata)),
        "geological_groups": int(groups.nunique()),
        "source_blocks": int(blocks.nunique()),
        "mixed_type_groups": int(len(mixed_groups)),
        "label_source_independence_verified": bool(
            config.get("label_audit", {}).get("source_independence_verified", False)
        ),
    }


def _prediction_rows(
    meta: pd.DataFrame,
    groups: pd.Series,
    blocks: pd.Series,
    y: pd.Series,
    outer_validation: np.ndarray,
    valid_positions: np.ndarray,
    raw_probability: np.ndarray,
    calibrated_probability: np.ndarray,
    repeat_id: int,
    repeat_seed: int,
    outer_fold: int,
    temperature: dict[str, Any],
    protocol_hash: str,
) -> list[dict[str, Any]]:
    lookup = {int(position): row for row, position in enumerate(valid_positions)}
    rows: list[dict[str, Any]] = []
    for position in outer_validation:
        raw = raw_probability[lookup[int(position)]] if int(position) in lookup else np.full(3, np.nan)
        calibrated = calibrated_probability[lookup[int(position)]] if int(position) in lookup else np.full(3, np.nan)
        valid = bool(np.isfinite(raw).all())
        rows.append({
            "Record ID": meta.iloc[position]["Record ID"],
            "Geological Group ID": groups.iloc[position],
            "Reference-connected block": blocks.iloc[position],
            "reported_type": meta.iloc[position]["Granite type"],
            "true_code": int(y.iloc[position]),
            "repeat_id": repeat_id,
            "repeat_seed": repeat_seed,
            "outer_fold": outer_fold,
            "valid_oof": valid,
            "abstain_reason": "" if valid else "row_missingness_at_or_above_threshold",
            "raw_P_I": raw[0], "raw_P_A": raw[1], "raw_P_S": raw[2],
            "calibrated_P_I": calibrated[0], "calibrated_P_A": calibrated[1], "calibrated_P_S": calibrated[2],
            "temperature": float(temperature["temperature"]),
            "temperature_fitted": bool(temperature["fitted"]),
            "analysis_protocol_hash": protocol_hash,
        })
    return rows


def _group_strata_from_records(records: pd.DataFrame, prefix: str) -> pd.DataFrame:
    valid = records[records["valid_oof"]].copy()
    if valid.empty:
        return pd.DataFrame()
    probability = valid[[f"{prefix}_P_I", f"{prefix}_P_A", f"{prefix}_P_S"]].to_numpy(float)
    strata = aggregate_group_type_probabilities(
        valid["true_code"].astype(int),
        probability,
        valid["Geological Group ID"].astype(str),
        valid["Reference-connected block"].astype(str),
    )
    return strata


def _block_bootstrap_class_metrics(
    strata: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    n_bootstrap = int(config["repeated_validation"]["source_block_bootstrap_replicates"])
    rng = np.random.default_rng(seed)
    block_frames = {
        block: subset.copy()
        for block, subset in strata.groupby("Reference-connected block", dropna=False)
    }
    unique_blocks = np.asarray(list(block_frames), dtype=object)
    observed = multiclass_metrics(
        strata["true_code"].to_numpy(int),
        strata[["score_I", "score_A", "score_S"]].to_numpy(float),
    )
    boot: dict[str, list[float]] = {
        "macro_f1": [], "balanced_accuracy": [],
        **{f"precision_{name}": [] for name in CLASSES},
        **{f"recall_{name}": [] for name in CLASSES},
        **{f"f1_{name}": [] for name in CLASSES},
    }
    for _ in range(n_bootstrap):
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        sampled = pd.concat([block_frames[block] for block in sampled_blocks], ignore_index=True)
        metrics = multiclass_metrics(
            sampled["true_code"].to_numpy(int),
            sampled[["score_I", "score_A", "score_S"]].to_numpy(float),
        )
        for metric in boot:
            boot[metric].append(float(metrics[metric]))
    rows = []
    for metric, values in boot.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        low, high = np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
        class_name = metric.rsplit("_", 1)[-1] if metric.startswith(("precision_", "recall_", "f1_")) else "multiclass"
        rows.append({
            "metric": metric,
            "class": class_name,
            "estimate": float(observed[metric]),
            "ci_low": float(low),
            "ci_high": float(high),
            "bootstrap_unit": "Reference-connected block",
            "bootstrap_replicates": n_bootstrap,
            "group_type_strata": int(len(strata)),
            "source_blocks": int(strata["Reference-connected block"].nunique()),
        })
    return pd.DataFrame(rows)


def run_repeated_development_validation(config: dict[str, Any]) -> dict[str, Any]:
    validate_analysis_runtime(config)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    context = prepare_data(config)
    paths = output_paths(config)
    output = _robustness_dir(config)
    selection = json.loads((paths["comparison"] / "model_selection.json").read_text(encoding="utf-8"))
    model_name = str(selection["best_ranked_model"])
    dev = context["development"]
    x = context["chemistry"].iloc[dev].reset_index(drop=True)
    y = context["y"].iloc[dev].reset_index(drop=True)
    groups = context["groups"].iloc[dev].reset_index(drop=True)
    blocks = context["blocks"].iloc[dev].reset_index(drop=True)
    meta = context["metadata"].iloc[dev].reset_index(drop=True)
    cfg = config["repeated_validation"]
    repeat_seeds = [int(value) for value in cfg["repeat_seeds"]]
    outer_splits = int(cfg["outer_folds"])
    inner_splits = int(cfg["inner_folds"])
    n_trials = int(cfg["trials_per_outer_fold"])
    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    trial_tables: list[pd.DataFrame] = []
    protocol_hash = context["analysis_protocol_hash"]
    protocol_contract = json.loads(
        (paths["audit"] / "analysis_protocol_contract.json").read_text(encoding="utf-8")
    )
    input_hashes_hash = _json_hash(protocol_contract["input_hashes"])

    used_effective_repeat_seeds: set[int] = set()
    for repeat_id, repeat_seed in enumerate(repeat_seeds, 1):
        repeat_splits, effective_repeat_seed = class_complete_group_splits(
            x, y, blocks, outer_splits, repeat_seed, config,
            forbidden_seeds=used_effective_repeat_seeds,
        )
        used_effective_repeat_seeds.add(effective_repeat_seed)
        for outer_fold, (training, validation) in enumerate(repeat_splits, 1):
            assert_no_block_overlap(training, validation, meta)
            x_inner = x.iloc[training].reset_index(drop=True)
            y_inner = y.iloc[training].reset_index(drop=True)
            g_inner = groups.iloc[training].reset_index(drop=True)
            b_inner = blocks.iloc[training].reset_index(drop=True)
            m_inner = meta.iloc[training].reset_index(drop=True)
            inner_folds, inner_registry = make_prepared_folds(
                x_inner, y_inner, b_inner, m_inner, inner_splits,
                repeat_seed + 1000 + outer_fold, config,
            )
            split_hash = _json_hash({
                "repeat_seed": repeat_seed,
                "training_ids": sorted(meta.iloc[training]["Record ID"].astype(str)),
                "validation_ids": sorted(meta.iloc[validation]["Record ID"].astype(str)),
                "inner_registry": inner_registry[["Record ID", "fold_id", "Reference-connected block"]]
                .sort_values(["fold_id", "Record ID"]).to_dict(orient="records"),
            })
            model_seed = repeat_seed + 10000 * outer_fold
            params, trials = optimize_model(
                model_name, inner_folds, y_inner, g_inner, config, model_seed, n_trials,
                study_name=(
                    f"{config['optimization']['study_revision']}_{protocol_hash[:12]}_"
                    f"confirmatory_r{repeat_id}_f{outer_fold}_{model_name}"
                ),
                study_context={
                    "analysis_protocol_hash": protocol_hash,
                    "input_hashes_hash": input_hashes_hash,
                    "block_registry_hash": context["block_registry_hash"],
                    "feature_rule_hash": _json_hash(config["preprocessing"]),
                    "split_hash": split_hash,
                    "analysis_role": "selected_algorithm_repeated_nested_confirmation",
                },
            )
            trials.insert(0, "repeat_id", repeat_id)
            trials.insert(1, "repeat_seed", repeat_seed)
            trials.insert(2, "outer_fold", outer_fold)
            trials.insert(3, "model", model_name)
            trial_tables.append(trials)

            inner_probability = _inner_oof_probability(
                model_name, params, inner_folds, y_inner, g_inner, config, model_seed + 5000
            )
            temperature = _fit_temperature(y_inner.to_numpy(), inner_probability)
            calibration_rows.append({
                "repeat_id": repeat_id, "repeat_seed": repeat_seed, "outer_fold": outer_fold,
                "effective_split_seed": effective_repeat_seed,
                "split_seed_offset": effective_repeat_seed - repeat_seed,
                "model": model_name, **temperature, "analysis_protocol_hash": protocol_hash,
            })
            prepared = prepare_fold(
                x, training, validation, outer_fold,
                float(config["preprocessing"]["feature_missingness_threshold"]),
                float(config["preprocessing"]["row_missingness_threshold"]),
                list(config["preprocessing"]["excluded_features"]),
                int(config["preprocessing"]["knn_neighbors"]),
            )
            estimator = build_estimator(
                model_name, params, model_seed + 6000,
                int(config["optimization"]["model_n_jobs"]),
            )
            estimator = fit_estimator(
                model_name, estimator, prepared.x_training,
                y.iloc[prepared.training_positions], groups.iloc[prepared.training_positions],
            )
            raw = predict_score_matrix(model_name, estimator, prepared.x_validation)
            calibrated = _temperature_transform(raw, temperature["temperature"])
            fold_rows = _prediction_rows(
                meta, groups, blocks, y, validation, prepared.validation_positions,
                raw, calibrated, repeat_id, repeat_seed, outer_fold, temperature, protocol_hash,
            )
            for row in fold_rows:
                row["effective_split_seed"] = int(effective_repeat_seed)
                row["split_seed_offset"] = int(effective_repeat_seed - repeat_seed)
            prediction_rows.extend(fold_rows)
            fold_frame = pd.DataFrame(fold_rows)
            for version in ("raw", "calibrated"):
                strata = _group_strata_from_records(fold_frame, version)
                if strata.empty:
                    continue
                metrics = multiclass_metrics(
                    strata["true_code"].to_numpy(int),
                    strata[["score_I", "score_A", "score_S"]].to_numpy(float),
                )
                metric_rows.append({
                    "scope": "outer_fold", "repeat_id": repeat_id, "repeat_seed": repeat_seed,
                    "effective_split_seed": effective_repeat_seed,
                    "split_seed_offset": effective_repeat_seed - repeat_seed,
                    "outer_fold": outer_fold, "probability_version": version,
                    "evaluation_unit": "geological_group_by_reported_type_stratum",
                    "group_type_strata": int(len(strata)),
                    "source_blocks": int(strata["Reference-connected block"].nunique()),
                    **metrics, "analysis_protocol_hash": protocol_hash,
                })

    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(output / "taskB_repeated_outer_oof_record_predictions.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(output / "taskB_temperature_calibration_fits.csv", index=False)
    if trial_tables:
        pd.concat(trial_tables, ignore_index=True).to_csv(
            output / "taskB_repeated_nested_optuna_trials.csv", index=False
        )

    for (repeat_id, repeat_seed), repeat_frame in predictions.groupby(["repeat_id", "repeat_seed"]):
        for version in ("raw", "calibrated"):
            strata = _group_strata_from_records(repeat_frame, version)
            metrics = multiclass_metrics(
                strata["true_code"].to_numpy(int),
                strata[["score_I", "score_A", "score_S"]].to_numpy(float),
            )
            metric_rows.append({
                "scope": "pooled_repeat", "repeat_id": int(repeat_id), "repeat_seed": int(repeat_seed),
                "outer_fold": 0, "probability_version": version,
                "evaluation_unit": "geological_group_by_reported_type_stratum",
                "group_type_strata": int(len(strata)),
                "source_blocks": int(strata["Reference-connected block"].nunique()),
                **metrics, "analysis_protocol_hash": protocol_hash,
            })
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(output / "taskB_repeated_oof_metrics.csv", index=False)

    valid_predictions = predictions[predictions["valid_oof"]].copy()
    averaged = (
        valid_predictions.groupby(
            ["Record ID", "Geological Group ID", "Reference-connected block", "reported_type", "true_code"],
            as_index=False,
        )
        .agg(
            repeats_available=("repeat_id", "nunique"),
            **{f"mean_{version}_P_{name}": (f"{version}_P_{name}", "mean")
               for version in ("raw", "calibrated") for name in CLASSES},
        )
    )
    group_tables = []
    diagnostic_tables = []
    class_ci_tables = []
    for version in ("raw", "calibrated"):
        probability = averaged[[f"mean_{version}_P_{name}" for name in CLASSES]].to_numpy(float)
        strata = aggregate_group_type_probabilities(
            averaged["true_code"].astype(int), probability,
            averaged["Geological Group ID"].astype(str),
            averaged["Reference-connected block"].astype(str),
        )
        strata["probability_version"] = version
        group_tables.append(strata)
        diagnostic_tables.append(probability_diagnostics(
            averaged["true_code"].to_numpy(int), probability,
            "record_mean_across_repeats", version, int(config["probability_audit"]["n_bins"]),
        ))
        diagnostic_tables.append(probability_diagnostics(
            strata["true_code"].to_numpy(int),
            strata[["score_I", "score_A", "score_S"]].to_numpy(float),
            "geological_group_by_reported_type_stratum", version,
            int(config["probability_audit"]["n_bins"]),
        ))
        class_ci = _block_bootstrap_class_metrics(
            strata, config, int(config["seed"]) + (0 if version == "raw" else 1)
        )
        class_ci["probability_version"] = version
        class_ci_tables.append(class_ci)
    group_frame = pd.concat(group_tables, ignore_index=True)
    group_frame.to_csv(output / "taskB_group_type_oof_predictions.csv", index=False)
    calibration = pd.concat(diagnostic_tables, ignore_index=True)
    calibration["analysis_protocol_hash"] = protocol_hash
    calibration.to_csv(output / "taskB_calibration_metrics.csv", index=False)
    class_ci = pd.concat(class_ci_tables, ignore_index=True)
    class_ci["analysis_protocol_hash"] = protocol_hash
    class_ci.to_csv(output / "taskB_classwise_metrics_with_group_bootstrap_ci.csv", index=False)
    return {
        "selected_algorithm": model_name,
        "repeats": len(repeat_seeds),
        "outer_folds": outer_splits,
        "development_records": int(len(x)),
        "valid_record_repeat_predictions": int(predictions["valid_oof"].sum()),
    }


def _full_data_repeat_shap(
    config: dict[str, Any],
    repeat_id: int,
    repeat_seed: int,
    context: dict[str, Any],
    model_name: str,
    params: dict[str, Any],
    forbidden_seeds: set[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x, y = context["chemistry"], context["y"]
    groups, blocks, meta = context["groups"], context["blocks"], context["metadata"]
    n_splits = int(config["repeated_shap"]["folds"])
    repeat_splits, effective_repeat_seed = class_complete_group_splits(
        x, y, blocks, n_splits, repeat_seed, config,
        forbidden_seeds=forbidden_seeds,
    )
    prediction_rows: list[dict[str, Any]] = []
    shap_rows: list[pd.DataFrame] = []
    closure_rows: list[dict[str, Any]] = []
    for outer_fold, (training, validation) in enumerate(repeat_splits, 1):
        assert_no_block_overlap(training, validation, meta)
        prepared = prepare_fold(
            x, training, validation, outer_fold,
            float(config["preprocessing"]["feature_missingness_threshold"]),
            float(config["preprocessing"]["row_missingness_threshold"]),
            list(config["preprocessing"]["excluded_features"]),
            int(config["preprocessing"]["knn_neighbors"]),
        )
        x_inner = x.iloc[training].reset_index(drop=True)
        y_inner = y.iloc[training].reset_index(drop=True)
        g_inner = groups.iloc[training].reset_index(drop=True)
        b_inner = blocks.iloc[training].reset_index(drop=True)
        m_inner = meta.iloc[training].reset_index(drop=True)
        calibration_folds, _ = make_prepared_folds(
            x_inner, y_inner, b_inner, m_inner,
            int(config["repeated_shap"]["calibration_inner_folds"]),
            repeat_seed + 2000 + outer_fold, config,
        )
        model_seed = int(config["seed"]) + 9100 + outer_fold + (repeat_id - 1) * 10000
        inner_probability = _inner_oof_probability(
            model_name, params, calibration_folds, y_inner, g_inner, config, model_seed + 3000
        )
        temperature = _fit_temperature(y_inner.to_numpy(), inner_probability)
        estimator = build_estimator(
            model_name, params, model_seed,
            int(config["optimization"]["model_n_jobs"]),
        )
        estimator = fit_estimator(
            model_name, estimator, prepared.x_training,
            y.iloc[prepared.training_positions], groups.iloc[prepared.training_positions],
        )
        raw = predict_score_matrix(model_name, estimator, prepared.x_validation)
        calibrated = _temperature_transform(raw, temperature["temperature"])
        fold_prediction_rows = _prediction_rows(
            meta, groups, blocks, y, validation, prepared.validation_positions,
            raw, calibrated, repeat_id, repeat_seed, outer_fold, temperature,
            context["analysis_protocol_hash"],
        )
        for row in fold_prediction_rows:
            row["effective_split_seed"] = int(effective_repeat_seed)
            row["split_seed_offset"] = int(effective_repeat_seed - repeat_seed)
        prediction_rows.extend(fold_prediction_rows)
        values, expected, model_output, closure_error, output_scale = _tree_shap(
            model_name, estimator, prepared.x_validation
        )
        closure_rows.append({
            "repeat_id": repeat_id, "repeat_seed": repeat_seed, "outer_fold": outer_fold,
            "effective_split_seed": effective_repeat_seed,
            "split_seed_offset": effective_repeat_seed - repeat_seed,
            "model": model_name, "shap_output_scale": output_scale,
            "max_abs_model_output_closure_error": closure_error,
            "valid_records": int(len(prepared.validation_positions)),
            "retained_features": int(len(prepared.features)),
            "analysis_protocol_hash": context["analysis_protocol_hash"],
        })
        ids = meta.iloc[prepared.validation_positions]["Record ID"].to_numpy()
        group_values = groups.iloc[prepared.validation_positions].to_numpy()
        block_values = blocks.iloc[prepared.validation_positions].to_numpy()
        reported = meta.iloc[prepared.validation_positions]["Granite type"].to_numpy()
        completed = prepared.x_validation[prepared.features].to_numpy(float)
        observed = x.iloc[prepared.validation_positions][prepared.features].to_numpy(float)
        n_records, n_features = completed.shape
        for class_index, class_name in enumerate(CLASSES):
            shap_rows.append(pd.DataFrame({
                "Record ID": np.repeat(ids, n_features),
                "Geological Group ID": np.repeat(group_values, n_features),
                "Reference-connected block": np.repeat(block_values, n_features),
                "reported_type": np.repeat(reported, n_features),
                "explained_class": class_name,
                "feature": np.tile(np.asarray(prepared.features, dtype=object), n_records),
                "SHAP value": values[:, :, class_index].reshape(-1),
                "imputed feature value": completed.reshape(-1),
                "observed feature value": observed.reshape(-1),
                "was_imputed": np.isnan(observed.reshape(-1)),
                "repeat_id": repeat_id,
                "repeat_seed": repeat_seed,
                "effective_split_seed": effective_repeat_seed,
                "split_seed_offset": effective_repeat_seed - repeat_seed,
                "outer_fold": outer_fold,
                "model_seed": model_seed,
                "base_value": np.repeat(expected[:, class_index], n_features),
                "model_output": np.repeat(model_output[:, class_index], n_features),
                "analysis_protocol_hash": context["analysis_protocol_hash"],
            }))
    return pd.DataFrame(prediction_rows), pd.concat(shap_rows, ignore_index=True), pd.DataFrame(closure_rows)


def _rank_stability(shap_long: pd.DataFrame) -> pd.DataFrame:
    importance = (
        shap_long.groupby(["repeat_id", "outer_fold", "explained_class", "feature"], as_index=False)["SHAP value"]
        .agg(mean_abs_shap=lambda values: float(np.mean(np.abs(values))))
    )
    rows = []
    for class_name in CLASSES:
        subset = importance[importance["explained_class"].eq(class_name)]
        partitions = sorted(set(zip(subset["repeat_id"], subset["outer_fold"])))
        correlations, overlaps = [], []
        for left, right in combinations(partitions, 2):
            left_values = subset[(subset["repeat_id"].eq(left[0])) & (subset["outer_fold"].eq(left[1]))].set_index("feature")["mean_abs_shap"]
            right_values = subset[(subset["repeat_id"].eq(right[0])) & (subset["outer_fold"].eq(right[1]))].set_index("feature")["mean_abs_shap"]
            joint = pd.concat([left_values, right_values], axis=1).fillna(0.0)
            correlations.append(_safe_spearman(joint.iloc[:, 0], joint.iloc[:, 1]))
            top_left, top_right = set(left_values.nlargest(10).index), set(right_values.nlargest(10).index)
            overlaps.append(len(top_left & top_right) / max(len(top_left | top_right), 1))
        finite_correlations = np.asarray(correlations, dtype=float)
        finite_correlations = finite_correlations[np.isfinite(finite_correlations)]
        rows.append({
            "class": class_name,
            "partitions": int(len(partitions)),
            "pairwise_comparisons": int(len(correlations)),
            "median_rank_spearman": float(np.median(finite_correlations)) if len(finite_correlations) else np.nan,
            "rank_spearman_q025": float(np.quantile(finite_correlations, 0.025)) if len(finite_correlations) else np.nan,
            "rank_spearman_q975": float(np.quantile(finite_correlations, 0.975)) if len(finite_correlations) else np.nan,
            "mean_top10_jaccard": float(np.mean(overlaps)) if overlaps else np.nan,
            "minimum_top10_jaccard": float(np.min(overlaps)) if overlaps else np.nan,
        })
    return pd.DataFrame(rows)


def _bridge_direction_stability(shap_long: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    primary = list(config["bridge_features"]["primary_reproduced_in_task_a"])
    legacy = list(config["bridge_features"]["legacy_sensitivity_only"])
    requested = primary + legacy
    selected = shap_long[shap_long["feature"].isin(requested)].copy()
    group_partition = (
        selected.groupby(
            ["repeat_id", "outer_fold", "explained_class", "feature", "Geological Group ID", "Reference-connected block"],
            as_index=False,
        )
        .agg(feature_value=("imputed feature value", "median"), shap_value=("SHAP value", "median"))
    )
    group_global = (
        group_partition.groupby(
            ["explained_class", "feature", "Geological Group ID", "Reference-connected block"], as_index=False
        )
        .agg(feature_value=("feature_value", "median"), shap_value=("shap_value", "mean"))
    )
    rng = np.random.default_rng(int(config["seed"]) + 19001)
    n_bootstrap = int(config["repeated_shap"]["direction_block_bootstrap_replicates"])
    rows = []
    for class_name in CLASSES:
        for feature in requested:
            partition_rhos = []
            feature_partition = group_partition[
                group_partition["explained_class"].eq(class_name) & group_partition["feature"].eq(feature)
            ]
            for _, partition in feature_partition.groupby(["repeat_id", "outer_fold"]):
                partition_rhos.append(_safe_spearman(partition["feature_value"], partition["shap_value"]))
            finite = np.asarray(partition_rhos, dtype=float)
            finite = finite[np.isfinite(finite) & (finite != 0)]
            sign_consistency = (
                float(max((finite > 0).mean(), (finite < 0).mean())) if len(finite) else 0.0
            )
            global_frame = group_global[
                group_global["explained_class"].eq(class_name) & group_global["feature"].eq(feature)
            ]
            global_rho = _safe_spearman(global_frame["feature_value"], global_frame["shap_value"])
            block_frames = {
                block: subset for block, subset in global_frame.groupby("Reference-connected block", dropna=False)
            }
            blocks = np.asarray(list(block_frames), dtype=object)
            boot = []
            if len(blocks) >= 2:
                for _ in range(n_bootstrap):
                    sampled = rng.choice(blocks, size=len(blocks), replace=True)
                    frame = pd.concat([block_frames[block] for block in sampled], ignore_index=True)
                    rho = _safe_spearman(frame["feature_value"], frame["shap_value"])
                    if np.isfinite(rho):
                        boot.append(rho)
            low, high = np.quantile(boot, [0.025, 0.975]) if boot else (np.nan, np.nan)
            expected_partitions = int(config["repeated_shap"]["folds"]) * len(config["repeated_shap"]["repeat_seeds"])
            rows.append({
                "class": class_name, "feature": feature,
                "bridge_role": "primary" if feature in primary else "legacy_sensitivity_only",
                "global_group_level_spearman": global_rho,
                "block_bootstrap_ci_low": float(low), "block_bootstrap_ci_high": float(high),
                "partition_direction_sign_consistency": sign_consistency,
                "partitions_with_finite_direction": int(len(finite)),
                "expected_partitions": expected_partitions,
                "partition_availability": float(len(finite) / max(expected_partitions, 1)),
                "geological_groups": int(global_frame["Geological Group ID"].nunique()),
                "source_blocks": int(global_frame["Reference-connected block"].nunique()),
                "bootstrap_replicates_valid": int(len(boot)),
            })
    return pd.DataFrame(rows)


def _leave_one_block_sensitivity(shap_long: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    primary = list(config["bridge_features"]["primary_reproduced_in_task_a"])
    grouped = (
        shap_long.groupby(
            ["explained_class", "feature", "Geological Group ID", "Reference-connected block"], as_index=False
        )
        .agg(mean_abs_shap=("SHAP value", lambda values: float(np.mean(np.abs(values)))),
             feature_value=("imputed feature value", "median"),
             shap_value=("SHAP value", "mean"))
    )
    rows = []
    for class_name in CLASSES:
        class_frame = grouped[grouped["explained_class"].eq(class_name)]
        baseline_importance = class_frame.groupby("feature")["mean_abs_shap"].mean()
        baseline_top = set(baseline_importance.nlargest(10).index)
        baseline_direction = {
            feature: _safe_spearman(
                class_frame[class_frame["feature"].eq(feature)]["feature_value"],
                class_frame[class_frame["feature"].eq(feature)]["shap_value"],
            ) for feature in primary
        }
        for block in sorted(class_frame["Reference-connected block"].astype(str).unique()):
            reduced = class_frame[~class_frame["Reference-connected block"].astype(str).eq(block)]
            reduced_importance = reduced.groupby("feature")["mean_abs_shap"].mean()
            joint = pd.concat([baseline_importance, reduced_importance], axis=1).fillna(0.0)
            rank_rho = _safe_spearman(joint.iloc[:, 0], joint.iloc[:, 1])
            reduced_top = set(reduced_importance.nlargest(10).index)
            direction_changes, sign_flips = [], 0
            for feature in primary:
                subset = reduced[reduced["feature"].eq(feature)]
                reduced_rho = _safe_spearman(subset["feature_value"], subset["shap_value"])
                base_rho = baseline_direction[feature]
                if np.isfinite(base_rho) and np.isfinite(reduced_rho):
                    direction_changes.append(abs(reduced_rho - base_rho))
                    sign_flips += int(base_rho != 0 and reduced_rho != 0 and np.sign(base_rho) != np.sign(reduced_rho))
            rows.append({
                "class": class_name, "omitted_source_block": block,
                "remaining_groups": int(reduced["Geological Group ID"].nunique()),
                "rank_spearman_vs_full": rank_rho,
                "top10_jaccard_vs_full": len(baseline_top & reduced_top) / max(len(baseline_top | reduced_top), 1),
                "maximum_primary_bridge_direction_change": max(direction_changes) if direction_changes else np.nan,
                "primary_bridge_direction_sign_flips": sign_flips,
                "sensitivity_definition": "delete-one-source-block aggregation influence; models are not refitted",
            })
    return pd.DataFrame(rows)


def _correlation_cluster_sensitivity(
    shap_long: pd.DataFrame, chemistry: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    features = sorted(set(shap_long["feature"]) & set(chemistry.columns))
    correlation = chemistry[features].corr(method="spearman", min_periods=20).fillna(0.0).abs()
    np.fill_diagonal(correlation.values, 1.0)
    distance = (1.0 - correlation).clip(0.0, 1.0)
    np.fill_diagonal(distance.values, 0.0)
    threshold = float(config["correlation_sensitivity"]["absolute_spearman_threshold"])
    if len(features) > 1:
        clusters = fcluster(
            linkage(squareform(distance.to_numpy(), checks=False), method="average"),
            t=1.0 - threshold, criterion="distance",
        )
    else:
        clusters = np.ones(len(features), dtype=int)
    cluster_map = dict(zip(features, clusters))
    importance = (
        shap_long.groupby(["explained_class", "feature"], as_index=False)["SHAP value"]
        .agg(mean_abs_shap=lambda values: float(np.mean(np.abs(values))))
    )
    importance["correlation_cluster"] = importance["feature"].map(cluster_map)
    importance["cluster_mean_abs_shap"] = importance.groupby(
        ["explained_class", "correlation_cluster"]
    )["mean_abs_shap"].transform("sum")
    primary = set(config["bridge_features"]["primary_reproduced_in_task_a"])
    legacy = set(config["bridge_features"]["legacy_sensitivity_only"])
    max_other, neighbours = {}, {}
    for feature in features:
        others = correlation.loc[feature].drop(feature)
        max_other[feature] = float(others.max()) if len(others) else np.nan
        neighbours[feature] = ";".join(others[others >= threshold].sort_values(ascending=False).index)
    importance["maximum_absolute_spearman_with_other_feature"] = importance["feature"].map(max_other)
    importance["features_at_or_above_threshold"] = importance["feature"].map(neighbours)
    importance["bridge_role"] = importance["feature"].map(
        lambda feature: "primary" if feature in primary else ("legacy_sensitivity_only" if feature in legacy else "non_bridge")
    )
    importance["cluster_method"] = "average-linkage on 1-|Spearman rho|"
    importance["absolute_spearman_threshold"] = threshold
    return importance


def run_repeated_full_data_shap(config: dict[str, Any]) -> dict[str, Any]:
    context = prepare_data(config)
    paths = output_paths(config)
    output = _robustness_dir(config)
    final_info = json.loads((paths["final"] / "selected_model_hyperparameters.json").read_text(encoding="utf-8"))
    model_name = str(final_info["model"])
    if model_name not in {"RF", "XGBoost"}:
        raise RuntimeError("Repeated exact OOF SHAP requires the selected interpretation model to be RF or XGBoost.")
    params = final_info["parameters"]
    all_predictions, all_shap, all_closure = [], [], []
    used_effective_repeat_seeds: set[int] = set()
    for repeat_id, repeat_seed in enumerate(config["repeated_shap"]["repeat_seeds"], 1):
        predictions, shap_long, closure = _full_data_repeat_shap(
            config, repeat_id, int(repeat_seed), context, model_name, params,
            forbidden_seeds=used_effective_repeat_seeds,
        )
        used_effective_repeat_seeds.add(int(predictions["effective_split_seed"].iloc[0]))
        all_predictions.append(predictions)
        all_shap.append(shap_long)
        all_closure.append(closure)
    predictions = pd.concat(all_predictions, ignore_index=True)
    shap_long = pd.concat(all_shap, ignore_index=True)
    closure = pd.concat(all_closure, ignore_index=True)
    predictions.to_csv(output / "taskB_repeated_full_data_oof_record_predictions.csv", index=False)
    shap_long.to_csv(output / "taskB_oof_class_specific_shap_long.csv", index=False)
    closure.to_csv(output / "taskB_repeated_shap_additivity.csv", index=False)

    canonical = predictions[predictions["repeat_id"].eq(1)].copy()
    if not canonical["Record ID"].is_unique:
        raise RuntimeError("Canonical repeated-SHAP OOF prediction is not one row per Record ID.")
    canonical.to_csv(output / "taskB_canonical_oof_record_predictions.csv", index=False)
    canonical_shap = shap_long[shap_long["repeat_id"].eq(1)].copy()
    for class_name in CLASSES:
        class_long = canonical_shap[canonical_shap["explained_class"].eq(class_name)]
        shap_wide = class_long.pivot(index="Record ID", columns="feature", values="SHAP value")
        shap_wide = shap_wide.rename(columns=lambda feature: f"SHAP::{feature}")
        prefix = canonical.set_index("Record ID")
        prefix.join(shap_wide, how="left").reset_index().to_csv(
            output / f"taskB_canonical_oof_{class_name}_shap_with_ids_v6.csv", index=False
        )

    rank_stability = _rank_stability(shap_long)
    rank_stability["analysis_protocol_hash"] = context["analysis_protocol_hash"]
    rank_stability.to_csv(output / "taskB_shap_rank_stability_by_class.csv", index=False)
    direction = _bridge_direction_stability(shap_long, config)
    direction["analysis_protocol_hash"] = context["analysis_protocol_hash"]
    direction.to_csv(output / "taskB_bridge_feature_direction_stability.csv", index=False)
    leave_one = _leave_one_block_sensitivity(shap_long, config)
    leave_one["analysis_protocol_hash"] = context["analysis_protocol_hash"]
    leave_one.to_csv(output / "taskB_leave_one_source_block_shap_sensitivity.csv", index=False)
    correlation = _correlation_cluster_sensitivity(shap_long, context["chemistry"], config)
    correlation["analysis_protocol_hash"] = context["analysis_protocol_hash"]
    correlation.to_csv(output / "taskB_correlation_cluster_shap_sensitivity.csv", index=False)
    return {
        "model": model_name,
        "repeats": len(config["repeated_shap"]["repeat_seeds"]),
        "valid_canonical_oof": int(canonical["valid_oof"].sum()),
        "shap_long_rows": int(len(shap_long)),
        "maximum_additivity_error": float(closure["max_abs_model_output_closure_error"].max()),
    }


def _build_v6_readiness(config: dict[str, Any], label_audit: dict[str, Any]) -> dict[str, Any]:
    output = _robustness_dir(config)
    protocol_hash = json.loads(
        (output_paths(config)["audit"] / "analysis_protocol_contract.json").read_text(encoding="utf-8")
    )["analysis_protocol_hash"]
    group_predictions = pd.read_csv(output / "taskB_group_type_oof_predictions.csv")
    performance_frame = group_predictions[group_predictions["probability_version"].eq("raw")]
    performance = multiclass_metrics(
        performance_frame["true_code"].to_numpy(int),
        performance_frame[["score_I", "score_A", "score_S"]].to_numpy(float),
    )
    rank = pd.read_csv(output / "taskB_shap_rank_stability_by_class.csv")
    direction = pd.read_csv(output / "taskB_bridge_feature_direction_stability.csv")
    direction_primary = direction[direction["bridge_role"].eq("primary")]
    closure = pd.read_csv(output / "taskB_repeated_shap_additivity.csv")
    primary_expected = len(config["bridge_features"]["primary_reproduced_in_task_a"]) * len(CLASSES)
    all_primary_exist = len(direction_primary) == primary_expected
    minimum_class_f1 = min(float(performance[f"f1_{name}"]) for name in CLASSES)
    minimum_class_recall = min(float(performance[f"recall_{name}"]) for name in CLASSES)
    minimum_rank = float(rank["median_rank_spearman"].min())
    minimum_overlap = float(rank["mean_top10_jaccard"].min())
    minimum_direction = float(direction_primary["partition_direction_sign_consistency"].min())
    minimum_availability = float(direction_primary["partition_availability"].min())
    maximum_closure = float(closure["max_abs_model_output_closure_error"].max())
    gate = config["selection"]["interpretation_gate"]
    predictive_gate = (
        float(performance["macro_f1"]) >= float(gate["minimum_macro_f1"])
        and float(performance["balanced_accuracy"]) >= float(gate["minimum_balanced_accuracy"])
        and minimum_class_f1 >= float(gate["minimum_class_f1"])
        and minimum_class_recall >= float(gate["minimum_class_recall"])
    )
    exploratory = config["coupling_readiness"]["exploratory"]
    exploratory_ready = (
        predictive_gate
        and minimum_rank >= float(exploratory["minimum_median_rank_spearman"])
        and minimum_overlap >= float(exploratory["minimum_mean_top10_overlap"])
        and minimum_direction >= float(exploratory["minimum_locked_direction_sign_consistency"])
        and minimum_availability >= float(exploratory["minimum_locked_feature_availability"])
        and maximum_closure <= float(exploratory["maximum_additivity_closure_error"])
        and (all_primary_exist or not bool(exploratory["require_all_locked_features"]))
    )
    robust = config["coupling_readiness"]["internally_robust"]
    robust_statistical = (
        float(performance["macro_f1"]) >= float(robust["minimum_macro_f1"])
        and float(performance["balanced_accuracy"]) >= float(robust["minimum_balanced_accuracy"])
        and minimum_class_f1 >= float(robust["minimum_class_f1"])
        and minimum_rank >= float(robust["minimum_median_rank_spearman"])
        and minimum_overlap >= float(robust["minimum_mean_top10_overlap"])
        and minimum_direction >= float(robust["minimum_locked_direction_sign_consistency"])
        and minimum_availability >= float(robust["minimum_locked_feature_availability"])
        and maximum_closure <= float(robust["maximum_additivity_closure_error"])
        and (all_primary_exist or not bool(robust["require_all_locked_features"]))
    )
    label_verified = bool(label_audit["label_source_independence_verified"])
    internally_robust = robust_statistical and label_verified
    level = "internally_robust_for_linkage" if internally_robust else (
        "exploratory_ready" if exploratory_ready else "not_ready"
    )
    heldout_metrics_path = output_paths(config)["final"] / "heldout_metrics_evaluable_records.csv"
    heldout = pd.read_csv(heldout_metrics_path).iloc[0].to_dict() if heldout_metrics_path.exists() else {}
    readiness = {
        "analysis_protocol_hash": protocol_hash,
        "readiness_level": level,
        "exploratory_ready": bool(exploratory_ready),
        "internally_robust_for_linkage": bool(internally_robust),
        "statistical_robustness_gate_passed": bool(robust_statistical),
        "label_source_independence_verified": label_verified,
        "label_verification_effect": (
            "manual label-source verification is required before an internally robust petrogenetic claim"
            if not label_verified else "verified in configuration by the analyst after source audit"
        ),
        "gating_evidence_layer": "repeated development source-connected nested OOF plus repeated full-data OOF SHAP",
        "fixed_heldout_role": "reported separately; not used to determine v6 readiness",
        "repeated_development_oof_metrics": performance,
        "fixed_heldout_metrics_non_gating": heldout,
        "minimum_class_f1": minimum_class_f1,
        "minimum_class_recall": minimum_class_recall,
        "minimum_classwise_median_rank_spearman": minimum_rank,
        "minimum_classwise_mean_top10_jaccard": minimum_overlap,
        "minimum_primary_bridge_direction_sign_consistency": minimum_direction,
        "minimum_primary_bridge_partition_availability": minimum_availability,
        "maximum_additivity_closure_error": maximum_closure,
        "all_primary_bridge_features_exist": bool(all_primary_exist),
        "primary_bridge_features": list(config["bridge_features"]["primary_reproduced_in_task_a"]),
        "legacy_sensitivity_features": list(config["bridge_features"]["legacy_sensitivity_only"]),
        "external_validation_claim_permitted": False,
        "causal_interpretation_permitted": False,
        "part4_rule": "Part 4 must use valid canonical OOF rows, six primary bridge features, and group/source-block inference.",
    }
    (output / "taskB_interpretation_readiness_v6.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    return readiness


def _write_bridge_contract(config: dict[str, Any], readiness: dict[str, Any]) -> dict[str, Any]:
    output = _robustness_dir(config)
    canonical = pd.read_csv(output / "taskB_canonical_oof_record_predictions.csv")
    contract = {
        "analysis_revision": config["analysis_revision"],
        "analysis_protocol_hash": readiness["analysis_protocol_hash"],
        "readiness_level": readiness["readiness_level"],
        "evidence_layer": "canonical repeat of full-data source-connected OOF prediction and attribution",
        "performance_evidence_layer": "repeated development source-connected nested OOF",
        "canonical_repeat_id": 1,
        "canonical_repeat_seed": int(config["repeated_shap"]["repeat_seeds"][0]),
        "validity_column": "valid_oof",
        "record_id_column": "Record ID",
        "group_id_column": "Geological Group ID",
        "source_block_column": "Reference-connected block",
        "preferred_probability_columns": {name: f"calibrated_P_{name}" for name in CLASSES},
        "uncalibrated_sensitivity_columns": {name: f"raw_P_{name}" for name in CLASSES},
        "primary_bridge_features": list(config["bridge_features"]["primary_reproduced_in_task_a"]),
        "legacy_sensitivity_features": list(config["bridge_features"]["legacy_sensitivity_only"]),
        "primary_shap_files": {
            name: f"taskB_canonical_oof_{name}_shap_with_ids_v6.csv" for name in CLASSES
        },
        "prediction_file": "taskB_canonical_oof_record_predictions.csv",
        "valid_canonical_oof_records": int(canonical["valid_oof"].sum()),
        "total_records": int(len(canonical)),
        "record_id_unique": bool(canonical["Record ID"].is_unique),
        "part4_eligibility_determined_here": False,
        "prohibited_claims": [
            "coupling improves predictive performance without a prospective comparison",
            "SHAP proves petrogenetic or metallogenic causality",
            "fixed heldout S-type F1 alone establishes stable attribution",
        ],
    }
    (output / "taskB_bridge_contract_v6.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return contract


def validate_robustness_outputs(config: dict[str, Any]) -> dict[str, Any]:
    output = _robustness_dir(config)
    required = [
        "taskB_label_and_source_audit.csv",
        "taskB_repeated_outer_oof_record_predictions.csv",
        "taskB_canonical_oof_record_predictions.csv",
        "taskB_group_type_oof_predictions.csv",
        "taskB_repeated_oof_metrics.csv",
        "taskB_classwise_metrics_with_group_bootstrap_ci.csv",
        "taskB_calibration_metrics.csv",
        "taskB_oof_class_specific_shap_long.csv",
        "taskB_shap_rank_stability_by_class.csv",
        "taskB_bridge_feature_direction_stability.csv",
        "taskB_leave_one_source_block_shap_sensitivity.csv",
        "taskB_correlation_cluster_shap_sensitivity.csv",
        "taskB_interpretation_readiness_v6.json",
        "taskB_bridge_contract_v6.json",
    ]
    missing = [name for name in required if not (output / name).exists()]
    if missing:
        raise RuntimeError(f"Task B v6 robustness outputs are incomplete: {missing}")
    canonical = pd.read_csv(output / "taskB_canonical_oof_record_predictions.csv")
    if not canonical["Record ID"].is_unique:
        raise RuntimeError("Task B v6 canonical OOF Record IDs are not unique.")
    contract = json.loads((output / "taskB_bridge_contract_v6.json").read_text(encoding="utf-8"))
    readiness = json.loads((output / "taskB_interpretation_readiness_v6.json").read_text(encoding="utf-8"))
    if contract["analysis_protocol_hash"] != readiness["analysis_protocol_hash"]:
        raise RuntimeError("Task B v6 readiness and bridge contracts use different protocol hashes.")
    return {
        "status": "PASS",
        "required_files_checked": len(required),
        "canonical_record_id_unique": True,
        "readiness_level": readiness["readiness_level"],
    }


def run_robustness_audit(config: dict[str, Any]) -> dict[str, Any]:
    label_audit = run_label_source_audit(config)
    development = run_repeated_development_validation(config)
    repeated_shap = run_repeated_full_data_shap(config)
    readiness = _build_v6_readiness(config, label_audit)
    bridge_contract = _write_bridge_contract(config, readiness)
    validation = validate_robustness_outputs(config)
    return {
        "label_source_audit": label_audit,
        "repeated_development_validation": development,
        "repeated_full_data_shap": repeated_shap,
        "readiness": readiness,
        "bridge_contract": bridge_contract,
        "validation": validation,
    }
