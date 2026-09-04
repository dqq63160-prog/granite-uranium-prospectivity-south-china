"""Label-blind mixed-group challenge prediction for uranium prospectivity modelling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

from export_utils import IntegrityLedger, load_config, save_json, save_table, sha256_file, stage_complete
from model_core import (
    MODEL_INDEX,
    FoldModelMatrix,
    build_model,
    fit_model,
    predict_score,
    score_type_for_model,
    standardize_for_model,
)
from nested_tuning import load_primary_input


def _median_absolute_deviation(values: pd.Series) -> float:
    array = values.to_numpy(float)
    median = np.median(array)
    return float(np.median(np.abs(array - median)))


def run_mixed_group_challenge(config_path: str | Path) -> dict[str, Any]:
    config, root, paths, primary, features = load_primary_input(config_path)
    protocol = json.loads((paths["audit"] / "analysis_protocol_contract.json").read_text(encoding="utf-8"))
    protocol_hash = str(protocol["analysis_protocol_hash"])
    run_id = f"{config['analysis_revision']}::{protocol_hash[:12]}"
    ledger = IntegrityLedger(paths["audit"] / "preflight_and_integrity_checks.json", run_id=run_id)
    decision_path = paths["results"] / "model_selection_decision.json"
    if not decision_path.exists():
        raise FileNotFoundError("Run repeated nested model evaluation before the mixed-group challenge.")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    selected_model = str(decision["selected_model"])
    challenge = pd.read_csv(paths["processed"] / "mixed_group_challenge_cohort_with_nan.csv", low_memory=False)
    if challenge.empty:
        status = {"status": "not_applicable_no_mixed_groups", "selected_model": selected_model}
        save_json(paths["challenge"] / "challenge_set_interpretation.json", status)
        return status
    columns = config["columns"]
    audit_labels = challenge[[columns["record_id"], columns["label"]]].copy()
    challenge_predictors = challenge.drop(columns=[columns["label"]]).copy()
    ledger.check("challenge_prediction_frame_masks_label", columns["label"] not in challenge_predictors.columns, None)
    split_registry = pd.read_csv(paths["oof"] / "outer_split_registry.csv", low_memory=False)
    selected_table = pd.read_csv(paths["optuna"] / "best_trial_by_outer_fold.csv", low_memory=False)
    selected_table = selected_table[selected_table["model"].eq(selected_model)].copy()
    predictions = []
    for row in selected_table.itertuples(index=False):
        repeat, outer_fold = int(row.repeat), int(row.outer_fold)
        params = json.loads(row.parameters_json)
        split = split_registry[
            split_registry["repeat"].eq(repeat) & split_registry["outer_fold"].eq(outer_fold)
        ]
        training_groups = set(split.loc[split["partition"].eq("train"), "Geological Group ID"])
        train = primary[primary[columns["group_id"]].isin(training_groups)].reset_index(drop=True)
        training_blocks = set(train[columns["cv_block_id"]].astype(str))
        matrix = FoldModelMatrix(features, config)
        x_train = matrix.fit_transform(train)
        x_challenge = matrix.transform(challenge_predictors)
        if selected_model in {"SVM", "MLP"}:
            x_train, x_challenge, _ = standardize_for_model(selected_model, x_train, x_challenge)
        seed = int(config["validation"]["base_seed"]) + repeat * 10000 + outer_fold * 1000 + MODEL_INDEX[selected_model] * 100
        model = build_model(selected_model, params, seed)
        fit_model(selected_model, model, x_train, train, config, seed)
        scores = predict_score(selected_model, model, x_challenge)
        part = challenge_predictors[[
            columns["record_id"], columns["group_id"], columns["cv_block_id"], columns["reference_id"]
        ]].copy()
        part.columns = ["Record ID", "Geological Group ID", "CV Block ID", "Reference ID"]
        part["repeat"] = repeat
        part["outer_fold"] = outer_fold
        part["model"] = selected_model
        part["parameter_hash"] = row.parameter_hash
        part["model_score"] = scores
        part["score_type"] = score_type_for_model(selected_model)
        part["analysis_protocol_hash"] = protocol_hash
        part["source_block_overlap_with_training"] = part["CV Block ID"].astype(str).isin(training_blocks)
        part["valid_prediction"] = np.isfinite(scores)
        predictions.append(part)
    prediction_table = pd.concat(predictions, ignore_index=True)
    expected_models = int(config["validation"]["outer_folds"]) * int(config["validation"]["outer_repeats"])
    counts = prediction_table.groupby("Record ID").size()
    ledger.check("challenge_each_record_has_all_outer_model_predictions", counts.eq(expected_models).all(), {
        "expected": expected_models, "minimum": int(counts.min()), "maximum": int(counts.max())
    })
    save_table(prediction_table, paths["challenge"] / "mixed_group_sample_predictions.csv")
    sample_ensemble = prediction_table.groupby(
        ["Record ID", "Geological Group ID", "CV Block ID"], as_index=False
    ).agg(
        model_score_median=("model_score", "median"),
        model_score_mean=("model_score", "mean"),
        model_score_sd=("model_score", "std"),
        model_score_q25=("model_score", lambda values: values.quantile(0.25)),
        model_score_q75=("model_score", lambda values: values.quantile(0.75)),
        source_overlap_fraction=("source_block_overlap_with_training", "mean"),
    )
    sample_ensemble["model_score_iqr"] = sample_ensemble["model_score_q75"] - sample_ensemble["model_score_q25"]
    group_summary = sample_ensemble.groupby("Geological Group ID", as_index=False).agg(
        ensemble_model_score_median=("model_score_median", "median"),
        ensemble_model_score_q25=("model_score_median", lambda values: values.quantile(0.25)),
        ensemble_model_score_q75=("model_score_median", lambda values: values.quantile(0.75)),
        ensemble_model_score_mad=("model_score_median", _median_absolute_deviation),
        model_score_sd_median=("model_score_sd", "median"),
        model_score_iqr_median=("model_score_iqr", "median"),
        source_overlap_fraction=("source_overlap_fraction", "mean"),
        record_count=("Record ID", "size"),
    )
    group_summary["ensemble_model_score_iqr"] = (
        group_summary["ensemble_model_score_q75"] - group_summary["ensemble_model_score_q25"]
    )
    stable = pd.read_csv(paths["results"] / "geological_group_consensus_oof.csv", low_memory=False)
    stable = stable[stable["model"].eq(selected_model)].copy()
    stable_negative = stable.loc[stable["target"].eq(0), "model_score"].to_numpy(float)
    stable_positive = stable.loc[stable["target"].eq(1), "model_score"].to_numpy(float)
    stable_all = stable["model_score"].to_numpy(float)
    negative_median = float(np.median(stable_negative))
    positive_median = float(np.median(stable_positive))
    denominator = positive_median - negative_median
    group_summary["percentile_among_stable_groups"] = group_summary["ensemble_model_score_median"].map(
        lambda value: percentileofscore(stable_all, value, kind="mean")
    )
    group_summary["relative_location_between_stable_class_medians"] = group_summary["ensemble_model_score_median"].map(
        lambda value: (value - negative_median) / denominator if denominator != 0 else np.nan
    )
    save_table(group_summary, paths["challenge"] / "mixed_group_summary_all_models.csv")
    nonoverlap_predictions = prediction_table[~prediction_table["source_block_overlap_with_training"]].copy()
    nonoverlap_sample = nonoverlap_predictions.groupby(
        ["Record ID", "Geological Group ID", "CV Block ID"], as_index=False
    ).agg(
        nonoverlap_model_score_median=("model_score", "median"),
        nonoverlap_model_score_mean=("model_score", "mean"),
        nonoverlap_model_score_sd=("model_score", "std"),
        nonoverlap_outer_model_count=("model_score", "size"),
    )
    nonoverlap_group = nonoverlap_sample.groupby("Geological Group ID", as_index=False).agg(
        nonoverlap_ensemble_model_score_median=("nonoverlap_model_score_median", "median"),
        nonoverlap_ensemble_model_score_mad=("nonoverlap_model_score_median", _median_absolute_deviation),
        nonoverlap_record_count=("Record ID", "size"),
        nonoverlap_outer_model_count_min=("nonoverlap_outer_model_count", "min"),
    )
    comparison = group_summary.merge(nonoverlap_group, on="Geological Group ID", how="left", validate="one_to_one")
    comparison["nonoverlap_score_available"] = comparison["nonoverlap_ensemble_model_score_median"].notna()
    comparison["absolute_all_vs_nonoverlap_score_difference"] = (
        comparison["ensemble_model_score_median"] - comparison["nonoverlap_ensemble_model_score_median"]
    ).abs()
    record_overlap = sample_ensemble.merge(
        nonoverlap_sample, on=["Record ID", "Geological Group ID", "CV Block ID"],
        how="left", validate="one_to_one",
    )
    record_overlap = record_overlap.rename(columns={
        "model_score_median": "all_outer_models_score_median",
        "model_score_mean": "all_outer_models_score_mean",
        "model_score_sd": "all_outer_models_score_sd",
        "nonoverlap_model_score_median": "nonoverlap_models_score_median",
        "nonoverlap_model_score_mean": "nonoverlap_models_score_mean",
        "nonoverlap_outer_model_count": "nonoverlap_models_available",
    })
    record_overlap["nonoverlap_models_available"] = record_overlap["nonoverlap_models_available"].fillna(0).astype(int)
    record_overlap["nonoverlap_models_fraction"] = record_overlap["nonoverlap_models_available"] / expected_models
    record_overlap["absolute_all_vs_nonoverlap_score_difference"] = (
        record_overlap["all_outer_models_score_median"] - record_overlap["nonoverlap_models_score_median"]
    ).abs()
    save_table(record_overlap, paths["challenge"] / "mixed_group_sample_overlap_sensitivity.csv")
    save_table(record_overlap, paths["challenge"] / "mixed_group_sample_ensemble_summary.csv")
    save_table(nonoverlap_group, paths["challenge"] / "mixed_group_summary_nonoverlap_only.csv")
    save_table(comparison, paths["challenge"] / "mixed_group_overlap_sensitivity.csv")
    leave_one_out = []
    for omitted in group_summary["Geological Group ID"]:
        retained = group_summary[group_summary["Geological Group ID"].ne(omitted)]
        leave_one_out.append({
            "omitted_mixed_group": omitted,
            "remaining_mixed_groups": len(retained),
            "remaining_model_score_median": float(retained["ensemble_model_score_median"].median()) if len(retained) else np.nan,
            "remaining_model_score_iqr": float(retained["ensemble_model_score_median"].quantile(0.75) - retained["ensemble_model_score_median"].quantile(0.25)) if len(retained) else np.nan,
        })
    save_table(pd.DataFrame(leave_one_out), paths["challenge"] / "mixed_group_leave_one_out.csv")
    no_overlap = comparison[comparison["nonoverlap_score_available"]]
    interpretation = {
        "status": "completed_without_binary_performance_scoring",
        "selected_model": selected_model,
        "challenge_records": int(challenge_predictors[columns["record_id"]].nunique()),
        "challenge_groups": int(challenge_predictors[columns["group_id"]].nunique()),
        "outer_models_per_record": expected_models,
        "source_overlap_flagged_records": int(
            sample_ensemble["source_overlap_fraction"].gt(0).sum()
        ),
        "nonoverlapping_group_sensitivity_count": len(no_overlap),
        "stable_negative_model_score_median": negative_median,
        "stable_positive_model_score_median": positive_median,
        "model_score_type": score_type_for_model(selected_model),
        "performance_metrics_calculated_for_mixed_groups": False,
        "label_audit_retained_separately_but_not_used_for_prediction": len(audit_labels),
        "auc_closeness_reference_not_applied_to_unlabelled_challenge": config["challenge"]["auc_closeness_reference"],
        "average_precision_closeness_reference_not_applied_to_unlabelled_challenge": config["challenge"]["average_precision_closeness_reference"],
        "interpretive_limit": "Challenge scores characterize prediction behaviour and uncertainty; they are not validation metrics.",
    }
    save_json(paths["challenge"] / "challenge_set_interpretation.json", interpretation)
    stage_complete(
        paths["logs"] / "stage_04_mixed_challenge.complete.json",
        "mixed_group_challenge", {"selection": sha256_file(decision_path)},
        [
            paths["challenge"] / "mixed_group_sample_predictions.csv",
            paths["challenge"] / "mixed_group_sample_overlap_sensitivity.csv",
            paths["challenge"] / "mixed_group_summary_all_models.csv",
            paths["challenge"] / "mixed_group_summary_nonoverlap_only.csv",
            paths["challenge"] / "mixed_group_overlap_sensitivity.csv",
            paths["challenge"] / "challenge_set_interpretation.json",
        ],
        analysis_revision=config["analysis_revision"],
        analysis_protocol_hash=protocol_hash,
    )
    return interpretation
