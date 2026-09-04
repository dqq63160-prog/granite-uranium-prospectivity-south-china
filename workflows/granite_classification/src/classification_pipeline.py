from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import shap
from optuna.trial import TrialState
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from model_core import (
    CLASSES,
    CLASS_TO_INT,
    INT_TO_CLASS,
    MODEL_NAMES,
    MODEL_SEARCH_SPACE_PROTOCOL,
    FoldPreprocessor,
    GraniteModelBundle,
    aggregate_group_type_probabilities,
    build_estimator,
    fit_estimator,
    group_bootstrap_macro_f1_difference,
    load_config,
    load_granite_dataset,
    load_runtime_lock,
    multiclass_metrics,
    optimization_score,
    predict_score_matrix,
    model_score_type,
    prepare_fold,
    resolve_path,
    save_bundle,
    select_features_and_rows,
    sha256_file,
    shap_direction,
    software_versions,
    suggest_parameters,
    validate_runtime,
)
from source_blocks import (
    REFERENCE_NORMALIZATION_RULE,
    assert_no_block_overlap,
    audit_reference_block_graph,
    build_reference_connected_blocks,
    reference_block_registry_hash,
)
from data_filter import (
    cohort_attrition_audit,
    excluded_group_ids,
    reconcile_group_type_map,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "granite_classification_config.json"


def validate_analysis_runtime(config: dict[str, Any]) -> dict[str, Any]:
    """Apply the runtime contract declared by the formal Task B configuration."""
    runtime = config.get("runtime", {})
    lock_file = runtime.get("lock_file", "config/runtime_lock.json")
    return validate_runtime(
        strict=bool(runtime.get("strict_environment", True)),
        lock_path=resolve_path(PROJECT_ROOT, lock_file),
    )


def output_paths(config: dict[str, Any]) -> dict[str, Path]:
    root = resolve_path(PROJECT_ROOT, config["paths"]["output_root"])
    paths = {
        "root": root,
        "audit": root / "00_Audit",
        "processed": root / "01_Processed_Data",
        "comparison": root / "02_Model_Comparison",
        "final": root / "03_Final_Model",
        "shap": root / "04_SHAP",
        "sensitivity": root / "05_Sensitivity",
        "figures": root / "06_Figures",
        "logs": root / "07_Logs",
        "robustness": root / "08_Robustness",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def read_inputs(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s1 = resolve_path(PROJECT_ROOT, config["paths"]["s1"])
    s2 = resolve_path(PROJECT_ROOT, config["paths"]["s2"])
    exclude_groups = excluded_group_ids(s1, s2, config)
    reconcile_types = reconcile_group_type_map(s1, s2, config)
    metadata, chemistry, audit = load_granite_dataset(
        s1, s2, exclude_groups=exclude_groups, reconcile_types=reconcile_types
    )
    s2_groups = pd.read_excel(s2, sheet_name="Geological Groups", header=1)
    blocks, normalized_references, registry = build_reference_connected_blocks(
        metadata, s2_groups
    )
    metadata = metadata.copy()
    metadata["Reference-connected block"] = blocks.to_numpy()
    metadata["Reference ID normalized"] = normalized_references.to_numpy()
    audit = audit.copy()
    audit["Reference-connected block"] = blocks.to_numpy()
    audit["Reference ID normalized"] = normalized_references.to_numpy()
    if "Mineralization label" in chemistry.columns:
        raise RuntimeError("Mineralization label entered the Task B chemistry matrix.")
    return metadata, chemistry, audit, registry, s2_groups


def validate_inputs(config: dict[str, Any]) -> dict[str, Any]:
    metadata, chemistry, audit, registry, _ = read_inputs(config)
    s1 = resolve_path(PROJECT_ROOT, config["paths"]["s1"])
    s2 = resolve_path(PROJECT_ROOT, config["paths"]["s2"])
    _, cohort_summary = cohort_attrition_audit(s1, s2, config)
    report = {
        "records": int(len(metadata)),
        "classes": metadata["Granite type"].value_counts().to_dict(),
        "geological_groups": int(metadata["Geological group"].nunique()),
        "references": int(registry["Reference ID normalized"].str.split("; ").explode().replace("", np.nan).nunique()),
        "reference_connected_blocks": int(metadata["Reference-connected block"].nunique()),
        "chemistry_variables_after_FeOT_removal": int(chemistry.shape[1]),
        "record_id_unique": bool(metadata["Record ID"].is_unique),
        "contains_mineralization_predictor": bool("Mineralization label" in chemistry.columns),
        "s2_records_merged": int(len(audit)),
        "cohort_flow": cohort_summary,
    }
    required_folds = max(
        int(config["holdout"]["candidate_folds"]),
        int(config["validation"]["outer_folds"]),
        int(config["shap"]["crossfit_folds"]),
        int(config.get("repeated_validation", {}).get("outer_folds", 2)),
        int(config.get("repeated_shap", {}).get("folds", 2)),
    )
    class_block_counts = {
        class_name: int(metadata.loc[
            metadata["Granite type"].eq(class_name), "Reference-connected block"
        ].nunique()) for class_name in CLASSES
    }
    report["source_blocks_by_class"] = class_block_counts
    report["required_outer_block_folds"] = required_folds
    if min(class_block_counts.values()) < required_folds:
        raise RuntimeError(
            "Source-connected blocking leaves fewer class-specific blocks than the requested "
            f"{required_folds}-fold split: {class_block_counts}. Reduce the prespecified fold count "
            "before model fitting and record the revised analysis version."
        )
    if report["contains_mineralization_predictor"]:
        raise RuntimeError("Mineralization label entered the feature matrix.")
    return report


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def audit_protocol(config: dict[str, Any]) -> dict[str, Any]:
    """Freeze the Task B protocol identity before any split or model fitting."""
    paths = output_paths(config)
    metadata, chemistry, audit, registry, _ = read_inputs(config)
    s1 = resolve_path(PROJECT_ROOT, config["paths"]["s1"])
    s2 = resolve_path(PROJECT_ROOT, config["paths"]["s2"])
    input_hashes = {"S1": sha256_file(s1), "S2": sha256_file(s2)}
    block_hash = reference_block_registry_hash(registry)
    implementation_files = {
        "source_blocks.py": PROJECT_ROOT / "src" / "source_blocks.py",
        "model_core.py": PROJECT_ROOT / "src" / "model_core.py",
        "classification_pipeline.py": PROJECT_ROOT / "src" / "classification_pipeline.py",
        "plot_classification.py": PROJECT_ROOT / "src" / "plot_classification.py",
        "data_audit.py": PROJECT_ROOT / "src" / "data_audit.py",
        "data_filter.py": PROJECT_ROOT / "src" / "data_filter.py",
        "robustness_audit.py": PROJECT_ROOT / "src" / "robustness_audit.py",
    }
    implementation_hashes = {
        name: sha256_file(path) for name, path in implementation_files.items()
    }
    runtime_lock_path = resolve_path(
        PROJECT_ROOT,
        config.get("runtime", {}).get(
            "lock_file", "config/runtime_lock.json"
        ),
    )
    runtime_lock = load_runtime_lock(runtime_lock_path)

    registry.to_csv(
        paths["audit"] / "reference_connected_block_registry.csv",
        index=False, encoding="utf-8-sig",
    )
    graph_audit = audit_reference_block_graph(
        metadata, metadata["Reference-connected block"], metadata["Reference ID normalized"]
    )
    graph_audit.to_csv(paths["audit"] / "reference_graph_audit.csv", index=False)
    (paths["audit"] / "input_hashes.json").write_text(
        json.dumps(input_hashes, indent=2), encoding="utf-8"
    )

    protocol_payload = {
        "analysis_revision": config["analysis_revision"],
        "protocol_status": config["protocol_status"],
        "runtime_protocol": config["runtime"],
        "runtime_lock": runtime_lock,
        "runtime_lock_sha256": sha256_file(runtime_lock_path),
        "input_hashes": input_hashes,
        "class_order": list(CLASSES),
        "excluded_features": list(config["preprocessing"]["excluded_features"]),
        "preprocessing": config["preprocessing"],
        "data_filter_protocol": config.get("data_filter", {}),
        "reference_id_normalization_rule": REFERENCE_NORMALIZATION_RULE,
        "block_registry_hash": block_hash,
        "holdout_protocol": config["holdout"],
        "validation_protocol": config["validation"],
        "split_integrity_protocol": config.get("split_integrity", {}),
        "model_search_spaces": MODEL_SEARCH_SPACE_PROTOCOL,
        "optimization_protocol": config["optimization"],
        "model_selection_protocol": config["selection"],
        "shap_protocol": config["shap"],
        "probability_audit_protocol": config["probability_audit"],
        "repeated_validation_protocol": config.get("repeated_validation", {}),
        "repeated_shap_protocol": config.get("repeated_shap", {}),
        "correlation_sensitivity_protocol": config.get("correlation_sensitivity", {}),
        "label_audit_protocol": config.get("label_audit", {}),
        "coupling_readiness_protocol": config["coupling_readiness"],
        "locked_bridge_features": list(config["locked_bridge_features"]),
        "bridge_feature_roles": config.get("bridge_features", {}),
    }
    protocol_hash = _json_hash(protocol_payload)
    contract_path = paths["audit"] / "analysis_protocol_contract.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing.get("analysis_protocol_hash") != protocol_hash:
            changed_fields = sorted(
                key for key, value in protocol_payload.items()
                if existing.get(key) != value
            )
            raise RuntimeError(
                "The formal output directory contains a different scientific Task B protocol. "
                f"Changed fields: {changed_fields}. Create a new analysis_revision and "
                "output_root; do not combine results from different protocols."
            )
        original_implementation = existing.get(
            "implementation_source_hashes_at_contract_creation", {}
        )
        changed_sources = sorted(
            name for name, current_hash in implementation_hashes.items()
            if original_implementation.get(name) != current_hash
        )
        if changed_sources:
            raise RuntimeError(
                "Task B source files changed after the protocol contract was created: "
                f"{changed_sources}. Start a clean analysis revision before model fitting."
            )
        contract = existing
    else:
        contract = {
            **protocol_payload,
            "implementation_source_hashes_at_contract_creation": implementation_hashes,
            "analysis_protocol_hash": protocol_hash,
            "contract_created_utc": datetime.now(timezone.utc).isoformat(),
        }
        contract_path.write_text(
            json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return {
        "metadata": metadata, "chemistry": chemistry, "audit": audit,
        "block_registry": registry, "input_hashes": input_hashes,
        "block_registry_hash": block_hash,
        "analysis_protocol_hash": protocol_hash,
        "protocol_contract": contract,
    }


def _group_counts_by_class(metadata: pd.DataFrame, positions: np.ndarray) -> dict[str, int]:
    subset = metadata.iloc[positions]
    return {
        class_name: int(subset.loc[subset["Granite type"].eq(class_name), "Geological Group ID"].nunique())
        for class_name in CLASSES
    }


def choose_holdout(
    y: pd.Series,
    metadata: pd.DataFrame,
    config: dict[str, Any],
    paths: dict[str, Path],
    input_hashes: dict[str, str],
    block_registry_hash: str,
    analysis_protocol_hash: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Freeze a performance-blind source-block holdout selected from preset candidates."""
    contract_path = paths["processed"] / "holdout_selection_contract.json"
    block_labels = metadata["Reference-connected block"].astype(str)
    if contract_path.exists():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        expected_contract_fields = {
            "analysis_protocol_hash": analysis_protocol_hash,
            "input_hashes": input_hashes,
            "block_registry_hash": block_registry_hash,
            "candidate_seeds": config["holdout"]["candidate_seeds"],
            "target_fraction": config["holdout"]["target_fraction"],
            "candidate_folds": config["holdout"]["candidate_folds"],
            "minimum_geological_groups_per_class_per_partition": config["holdout"][
                "minimum_geological_groups_per_class_per_partition"
            ],
            "objective_weights": config["holdout"]["objective_weights"],
        }
        mismatched = [
            key for key, expected in expected_contract_fields.items()
            if contract.get(key) != expected
        ]
        selected_hash = _json_hash(sorted(contract.get("selected_heldout_blocks", [])))
        if contract.get("selected_heldout_blocks_hash") != selected_hash:
            mismatched.append("selected_heldout_blocks_hash")
        if mismatched:
            raise RuntimeError(
                "The frozen held-out contract does not match the formal Task B protocol: "
                f"{sorted(set(mismatched))}. Create a new analysis revision; do not reuse or reselect it."
            )
        heldout_blocks = set(contract["selected_heldout_blocks"])
        heldout = np.flatnonzero(block_labels.isin(heldout_blocks).to_numpy())
        development = np.flatnonzero(~block_labels.isin(heldout_blocks).to_numpy())
        assert_no_block_overlap(development, heldout, metadata)
        candidates = pd.read_csv(paths["processed"] / "holdout_candidate_partitions.csv")
        return development, heldout, candidates, contract

    holdout_cfg = config["holdout"]
    target = float(holdout_cfg["target_fraction"])
    overall = np.bincount(y, minlength=3) / len(y)
    minimum_groups = int(holdout_cfg["minimum_geological_groups_per_class_per_partition"])
    rows: list[dict[str, Any]] = []
    candidates: list[tuple[float, int, int, np.ndarray, np.ndarray]] = []
    dummy = np.zeros((len(y), 1))
    for seed in [int(value) for value in holdout_cfg["candidate_seeds"]]:
        splitter = StratifiedGroupKFold(
            n_splits=int(holdout_cfg["candidate_folds"]), shuffle=True, random_state=seed
        )
        for fold_id, (development, heldout) in enumerate(
            splitter.split(dummy, y, block_labels), 1
        ):
            dev_counts = _group_counts_by_class(metadata, development)
            held_counts = _group_counts_by_class(metadata, heldout)
            class_complete = len(np.unique(y.iloc[development])) == 3 and len(np.unique(y.iloc[heldout])) == 3
            group_minimum_passed = all(
                dev_counts[name] >= minimum_groups and held_counts[name] >= minimum_groups
                for name in CLASSES
            )
            held_proportion = np.bincount(y.iloc[heldout], minlength=3) / len(heldout)
            size_deviation = abs(len(heldout) / len(y) - target)
            class_deviation = float(np.abs(held_proportion - overall).sum())
            objective = (
                float(holdout_cfg["objective_weights"]["size_deviation"]) * size_deviation
                + float(holdout_cfg["objective_weights"]["class_proportion_deviation"]) * class_deviation
            )
            eligible = bool(class_complete and group_minimum_passed)
            row = {
                "seed": seed, "candidate_fold": fold_id, "eligible": eligible,
                "objective": objective, "size_deviation": size_deviation,
                "class_proportion_deviation": class_deviation,
                "development_n": int(len(development)), "heldout_n": int(len(heldout)),
                "development_blocks": int(block_labels.iloc[development].nunique()),
                "heldout_blocks": int(block_labels.iloc[heldout].nunique()),
                **{f"development_groups_{name}": dev_counts[name] for name in CLASSES},
                **{f"heldout_groups_{name}": held_counts[name] for name in CLASSES},
                "heldout_block_ids": "; ".join(sorted(block_labels.iloc[heldout].unique())),
            }
            rows.append(row)
            if eligible:
                candidates.append((objective, seed, fold_id, development, heldout))
    candidate_frame = pd.DataFrame(rows).sort_values(
        ["eligible", "objective", "seed", "candidate_fold"], ascending=[False, True, True, True]
    )
    candidate_frame.to_csv(paths["processed"] / "holdout_candidate_partitions.csv", index=False)
    if not candidates:
        raise RuntimeError("No preset source-block holdout candidate meets the class/group constraints.")
    _, selected_seed, selected_fold, development, heldout = min(
        candidates, key=lambda item: (item[0], item[1], item[2])
    )
    assert_no_block_overlap(development, heldout, metadata)
    selected_blocks = sorted(block_labels.iloc[heldout].unique())
    contract = {
        "analysis_revision": config["analysis_revision"],
        "analysis_protocol_hash": analysis_protocol_hash,
        "selection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selection_basis": "size and class-composition balance only; no chemistry, model performance, or SHAP",
        "input_hashes": input_hashes,
        "block_registry_hash": block_registry_hash,
        "candidate_seeds": holdout_cfg["candidate_seeds"],
        "target_fraction": holdout_cfg["target_fraction"],
        "candidate_folds": holdout_cfg["candidate_folds"],
        "minimum_geological_groups_per_class_per_partition": holdout_cfg[
            "minimum_geological_groups_per_class_per_partition"
        ],
        "objective_weights": holdout_cfg["objective_weights"],
        "selected_seed": int(selected_seed),
        "selected_candidate_fold": int(selected_fold),
        "selected_heldout_blocks": selected_blocks,
        "selected_heldout_blocks_hash": _json_hash(selected_blocks),
        "final_evaluation_timestamp_utc": None,
        "final_evaluation_completed": False,
        "heldout_access_count": 0,
        "heldout_used_for_selection": False,
        "heldout_used_for_tuning": False,
        "heldout_used_for_feature_selection": False,
    }
    contract_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
    return development, heldout, candidate_frame, contract


def prepare_data(config: dict[str, Any]) -> dict[str, Any]:
    paths = output_paths(config)
    protocol = audit_protocol(config)
    metadata = protocol["metadata"]
    chemistry = protocol["chemistry"]
    audit = protocol["audit"]
    registry = protocol["block_registry"]
    s1 = resolve_path(PROJECT_ROOT, config["paths"]["s1"])
    s2 = resolve_path(PROJECT_ROOT, config["paths"]["s2"])
    cohort_audit, cohort_summary = cohort_attrition_audit(s1, s2, config)
    cohort_audit.to_csv(
        paths["audit"] / "taskB_cohort_attrition_record_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (paths["audit"] / "taskB_cohort_attrition_summary.json").write_text(
        json.dumps(cohort_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    y = metadata["Granite type"].map(CLASS_TO_INT).astype(int)
    groups = metadata["Geological group"].astype(str)
    blocks = metadata["Reference-connected block"].astype(str)
    development, heldout, _, contract = choose_holdout(
        y, metadata, config, paths, protocol["input_hashes"],
        protocol["block_registry_hash"], protocol["analysis_protocol_hash"]
    )

    metadata.to_csv(paths["processed"] / "granite_metadata_no_mineralization.csv", index=False, encoding="utf-8-sig")
    chemistry.to_csv(paths["processed"] / "granite_raw_chemistry.csv", index=False)
    audit.to_csv(paths["processed"] / "geological_group_audit.csv", index=False, encoding="utf-8-sig")
    missing = chemistry.isna().mean().rename("missing_fraction").reset_index(name="missing_fraction")
    missing = missing.rename(columns={"index": "feature"})
    missing.to_csv(paths["processed"] / "feature_missingness.csv", index=False)
    exclusion_set = set(config["preprocessing"]["excluded_features"])
    pd.DataFrame({
        "feature": chemistry.columns,
        "excluded_from_all_Task_B_models": [feature in exclusion_set for feature in chemistry.columns],
        "exclusion_policy": [
            "frozen_before_model fitting" if feature in exclusion_set else "eligible_for_fold-wise_screening"
            for feature in chemistry.columns
        ],
    }).to_csv(paths["audit"] / "model_feature_exclusion_audit.csv", index=False)

    partition = pd.DataFrame({
        "Record ID": metadata["Record ID"],
        "reported_type": metadata["Granite type"],
        "Geological Group ID": groups,
        "Reference-connected block": blocks,
        "partition": np.where(np.isin(np.arange(len(metadata)), heldout), "heldout", "development"),
    })
    partition.to_csv(paths["processed"] / "fixed_source_block_holdout_registry.csv", index=False)
    assert_no_block_overlap(development, heldout, metadata)
    dev_groups = set(groups.iloc[development])
    test_groups = set(groups.iloc[heldout])
    dev_blocks = set(blocks.iloc[development])
    test_blocks = set(blocks.iloc[heldout])

    summary = {
        "raw_S1_n": cohort_summary["raw_S1_records"],
        "explicit_IAS_n": cohort_summary["explicit_IAS_records"],
        "post_protocol_filter_n": cohort_summary["retained_analysis_records"],
        "development_n": int(len(development)),
        "heldout_n": int(len(heldout)),
        "development_groups": int(len(dev_groups)),
        "heldout_groups": int(len(test_groups)),
        "development_source_blocks": int(len(dev_blocks)),
        "heldout_source_blocks": int(len(test_blocks)),
        "selected_holdout_seed": int(contract["selected_seed"]),
        "selected_holdout_candidate_fold": int(contract["selected_candidate_fold"]),
        "record_overlap": 0, "geological_group_overlap": 0, "source_block_overlap": 0,
        "evidence_layer": "fixed source-connected heldout reserved for one final evaluation",
        "analysis_protocol_hash": protocol["analysis_protocol_hash"],
        "block_registry_hash": protocol["block_registry_hash"],
    }
    (paths["processed"] / "data_preparation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"metadata": metadata, "chemistry": chemistry, "y": y, "groups": groups,
            "blocks": blocks,
            "development": development, "heldout": heldout, "summary": summary,
            "analysis_protocol_hash": protocol["analysis_protocol_hash"],
            "block_registry_hash": protocol["block_registry_hash"],
            "holdout_contract": contract}


def class_complete_group_splits(
    x: pd.DataFrame,
    y: pd.Series,
    blocks: pd.Series,
    n_splits: int,
    requested_seed: int,
    config: dict[str, Any],
    forbidden_seeds: set[int] | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    """Build source-connected folds with complete I/A/S coverage.

    A deterministic seed search is allowed only to satisfy the prespecified
    class-coverage constraint. No model is fitted and no performance metric is
    consulted, so this cannot select a favourable predictive result.
    """
    attempts = int(config.get("split_integrity", {}).get("maximum_seed_attempts", 100))
    expected = set(range(len(CLASSES)))
    forbidden = forbidden_seeds or set()
    for offset in range(attempts):
        effective_seed = int(requested_seed) + offset
        if effective_seed in forbidden:
            continue
        splitter = StratifiedGroupKFold(
            n_splits=int(n_splits), shuffle=True, random_state=effective_seed
        )
        splits = [
            (np.asarray(training, dtype=int), np.asarray(validation, dtype=int))
            for training, validation in splitter.split(x, y, blocks)
        ]
        complete = all(
            set(y.iloc[training].unique()) == expected
            and set(y.iloc[validation].unique()) == expected
            for training, validation in splits
        )
        if complete:
            return splits, effective_seed
    raise RuntimeError(
        f"No {n_splits}-fold source-connected partition with complete I/A/S coverage "
        f"was found in {attempts} deterministic seeds starting at {requested_seed}."
    )


def make_prepared_folds(
    x: pd.DataFrame,
    y: pd.Series,
    blocks: pd.Series,
    metadata: pd.DataFrame,
    n_splits: int,
    seed: int,
    config: dict[str, Any],
) -> tuple[list[Any], pd.DataFrame]:
    prep = config["preprocessing"]
    splits, effective_seed = class_complete_group_splits(
        x, y, blocks, n_splits, seed, config
    )
    folds = []
    registry_rows = []
    for fold_id, (training, validation) in enumerate(splits, 1):
        assert_no_block_overlap(training, validation, metadata)
        fold = prepare_fold(
            x, training, validation, fold_id,
            missing_threshold=float(prep["feature_missingness_threshold"]),
            row_missing_threshold=float(prep["row_missingness_threshold"]),
            excluded_features=list(prep["excluded_features"]),
            n_neighbors=int(prep["knn_neighbors"]),
        )
        folds.append(fold)
        retained_set = set(fold.validation_positions.tolist())
        for position in validation:
            row_missingness = float(x.iloc[position][fold.features].isna().mean())
            registry_rows.append({
                "position": int(position), "fold_id": fold_id, "role": "validation",
                "Record ID": metadata.iloc[position]["Record ID"],
                "Geological Group ID": metadata.iloc[position]["Geological Group ID"],
                "Reference-connected block": metadata.iloc[position]["Reference-connected block"],
                "reported_type": metadata.iloc[position]["Granite type"],
                "valid_for_fold": bool(position in retained_set),
                "abstain_reason": "" if position in retained_set else "row_missingness_at_or_above_threshold",
                "row_missing_fraction": row_missingness,
                "n_retained_features": int(len(fold.features)),
                "retained_features": "; ".join(fold.features),
                "requested_split_seed": int(seed),
                "effective_split_seed": int(effective_seed),
                "split_seed_offset": int(effective_seed - seed),
            })
    return folds, pd.DataFrame(registry_rows)


def optimize_model(
    model_name: str,
    folds: list[Any],
    y: pd.Series,
    groups: pd.Series,
    config: dict[str, Any],
    seed: int,
    n_trials: int,
    study_name: str,
    study_context: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    n_jobs = int(config["optimization"]["model_n_jobs"])

    def objective(trial: optuna.Trial) -> float:
        params = suggest_parameters(model_name, trial)
        probability = np.full((len(y), 3), np.nan)
        convergence_warnings = 0
        for fold_number, fold in enumerate(folds, 1):
            estimator = build_estimator(model_name, params, seed + fold.fold_id, n_jobs)
            estimator = fit_estimator(
                model_name, estimator, fold.x_training, y.iloc[fold.training_positions],
                groups.iloc[fold.training_positions],
            )
            convergence_warnings += int(
                getattr(estimator, "_task_b_convergence_warning_count", 0)
            )
            probability[fold.validation_positions] = predict_score_matrix(
                model_name, estimator, fold.x_validation
            )
            interim_valid = np.isfinite(probability).all(axis=1)
            interim_f1 = multiclass_metrics(
                y.iloc[interim_valid].to_numpy(), probability[interim_valid]
            )["macro_f1"]
            trial.report(float(interim_f1), step=fold_number)
            trial.set_user_attr("inner_fold_fits_completed", fold_number)
            if bool(config["optimization"]["pruning"]) and trial.should_prune():
                raise optuna.TrialPruned()
        valid = np.isfinite(probability).all(axis=1)
        record_metrics = multiclass_metrics(y.iloc[valid].to_numpy(), probability[valid])
        strata = aggregate_group_type_probabilities(
            y.iloc[valid], probability[valid], groups.iloc[valid]
        )
        group_metrics = multiclass_metrics(
            strata["true_code"].to_numpy(),
            strata[["score_I", "score_A", "score_S"]].to_numpy(float),
        )
        for key in ("macro_f1", "balanced_accuracy", "macro_ovr_auc"):
            trial.set_user_attr(f"group_stratum_{key}", group_metrics[key])
            trial.set_user_attr(f"record_{key}", record_metrics[key])
        trial.set_user_attr("convergence_warning_count", convergence_warnings)
        return optimization_score(group_metrics)

    optimization = config["optimization"]
    storage_path = resolve_path(PROJECT_ROOT, optimization["persistent_storage"])
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_path.as_posix()}"
    pruning_enabled = bool(optimization["pruning"])
    pruner = (
        optuna.pruners.MedianPruner()
        if pruning_enabled else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=pruner,
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
    )
    expected_attrs = {
        **study_context,
        "model": model_name,
        "study_name": study_name,
        "search_space_hash": _json_hash(MODEL_SEARCH_SPACE_PROTOCOL[model_name]),
        "trial_budget": int(n_trials),
        "budget_definition": "COMPLETE trials" if not pruning_enabled else "initiated trials",
        "pruning": pruning_enabled,
    }
    for key, expected in expected_attrs.items():
        existing = study.user_attrs.get(key)
        if existing is not None and existing != expected:
            raise RuntimeError(
                f"Optuna study {study_name} has an incompatible {key}. "
                "Do not resume a study created under another protocol or split."
            )
        if existing is None:
            study.set_user_attr(key, expected)

    if pruning_enabled:
        finished = sum(
            trial.state in {TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL}
            for trial in study.trials
        )
        remaining = max(0, n_trials - finished)
    else:
        completed = sum(trial.state == TrialState.COMPLETE for trial in study.trials)
        remaining = max(0, n_trials - completed)
    if remaining > 0:
        study.optimize(
            objective, n_trials=remaining, n_jobs=1,
            show_progress_bar=False, gc_after_trial=True,
        )
    complete = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    if not pruning_enabled and len(complete) != n_trials:
        raise RuntimeError(
            f"Formal study {study_name} requires exactly {n_trials} COMPLETE trials; "
            f"found {len(complete)}. Resolve failed trials without changing the protocol."
        )
    if not complete:
        raise RuntimeError(
            f"Study {study_name} has no completed trial. Re-run the same cell to resume the "
            "persistent Optuna study."
        )
    params = study.best_trial.params.copy()
    if model_name == "MLP":
        params["hidden_layer_sizes"] = (int(params.pop("hidden_1")), int(params.pop("hidden_2")))
    trials = study.trials_dataframe()
    trials.insert(0, "study_name_frozen", study_name)
    trials.insert(1, "analysis_protocol_hash", study_context["analysis_protocol_hash"])
    return params, trials


def _select_interpretation_model(
    pooled: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    config: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    """Apply a class-neutral, group-stratum ranking rule to every model."""
    protocol_hash = json.loads(
        (paths["audit"] / "analysis_protocol_contract.json").read_text(encoding="utf-8")
    )["analysis_protocol_hash"]
    ranking = pooled.copy()
    stability = (
        fold_metrics.groupby("model")
        .agg(
            macro_f1_fold_mean=("macro_f1", "mean"),
            macro_f1_fold_sd=("macro_f1", lambda values: values.std(ddof=0)),
            s_f1_fold_mean=("f1_S", "mean"),
            s_f1_fold_sd=("f1_S", lambda values: values.std(ddof=0)),
            s_recall_fold_mean=("recall_S", "mean"),
            s_recall_fold_sd=("recall_S", lambda values: values.std(ddof=0)),
        )
        .reset_index()
    )
    fold_winners = (
        fold_metrics.loc[fold_metrics.groupby("outer_fold_id")["macro_f1"].idxmax(), "model"]
        .value_counts().rename("macro_f1_fold_wins")
    )
    ranking = ranking.merge(stability, on="model", how="left")
    ranking["macro_f1_fold_wins"] = ranking["model"].map(fold_winners).fillna(0).astype(int)
    ranking["macro_ovr_auc"] = ranking["macro_ovr_auc"].fillna(0.0)
    ranking["minimum_class_recall"] = ranking[[
        "recall_I", "recall_A", "recall_S"
    ]].min(axis=1)
    ranking["minimum_class_f1"] = ranking[["f1_I", "f1_A", "f1_S"]].min(axis=1)
    ranking["selection_score"] = ranking["macro_f1"]
    best_macro_f1 = float(ranking["macro_f1"].max())
    macro_f1_tolerance = float(config["selection"]["macro_f1_tolerance"])
    ranking["within_primary_tolerance"] = (
        ranking["macro_f1"] >= best_macro_f1 - macro_f1_tolerance
    )
    ranking = ranking.sort_values(
        [
            "within_primary_tolerance", "balanced_accuracy", "minimum_class_recall",
            "macro_ovr_auc", "macro_f1_fold_sd", "macro_f1",
        ],
        ascending=[False, False, False, False, True, False],
    ).reset_index(drop=True)
    ranking.insert(0, "comprehensive_rank", np.arange(1, len(ranking) + 1))
    ranking.to_csv(paths["comparison"] / "model_comprehensive_ranking.csv", index=False)

    best_model = str(ranking.iloc[0]["model"])
    pairwise_rows = []
    all_unique = True
    valid_long = predictions[predictions["valid_oof"]].copy()
    seed = int(config["seed"])
    for comparator in MODEL_NAMES:
        if comparator == best_model:
            continue
        observed, low, high = group_bootstrap_macro_f1_difference(
            valid_long, best_model, comparator,
            int(config["selection"]["source_block_bootstrap_replicates"]),
            seed + 500 + MODEL_NAMES.index(comparator),
        )
        supports_unique = low > float(config["selection"]["minimum_macro_f1_gain"])
        pairwise_rows.append({
            "best_model": best_model, "comparator": comparator,
            "macro_f1_difference": observed, "ci_low": low, "ci_high": high,
            "supports_unique_best": bool(supports_unique),
        })
        all_unique &= supports_unique
    pairwise_frame = pd.DataFrame(pairwise_rows)
    pairwise_frame.to_csv(
        paths["comparison"] / "source_block_bootstrap_model_differences.csv", index=False
    )
    (paths["comparison"] / "model_selection_uncertainty.json").write_text(
        json.dumps({
            "resampling_unit": "Reference-connected block",
            "bootstrap_replicates": int(config["selection"]["source_block_bootstrap_replicates"]),
            "pairwise_macro_f1_differences": pairwise_rows,
        }, indent=2), encoding="utf-8"
    )

    selected = ranking.iloc[0]
    gate = config["selection"]["interpretation_gate"]
    minimum_class_f1 = min(float(selected[f"f1_{class_name}"]) for class_name in CLASSES)
    minimum_class_recall = min(
        float(selected[f"recall_{class_name}"]) for class_name in CLASSES
    )
    predictive_gate = (
        float(selected["macro_f1"]) >= float(gate["minimum_macro_f1"])
        and float(selected["balanced_accuracy"]) >= float(gate["minimum_balanced_accuracy"])
        and minimum_class_f1 >= float(gate["minimum_class_f1"])
        and minimum_class_recall >= float(gate["minimum_class_recall"])
    )
    tree_models = tuple(config["selection"].get(
        "tree_shap_eligible_models", ["RF", "XGBoost"]
    ))
    if best_model in tree_models and predictive_gate:
        status = f"{best_model}_selected_for_SHAP"
        interpretation_model = best_model
    elif not predictive_gate:
        status = f"{best_model}_predictive_gate_not_passed"
        interpretation_model = None
    else:
        status = f"{best_model}_selected_but_no_exact_tree_SHAP"
        interpretation_model = None

    selected_metrics = {
        key: float(selected[key])
        for key in [
            "macro_f1", "balanced_accuracy", "macro_ovr_auc",
            "precision_I", "recall_I", "f1_I",
            "precision_A", "recall_A", "f1_A",
            "precision_S", "recall_S", "f1_S",
        ]
    }
    allowed_claim_strength = (
        "selected model uniquely supported by the prespecified source-block bootstrap macro-F1 rule"
        if all_unique else
        "selected model ranked first by the frozen class-neutral rule; at least one pairwise interval included zero"
    )
    selection = {
        "analysis_protocol_hash": protocol_hash,
        "best_ranked_model": best_model,
        "interpretation_model": interpretation_model,
        "selection_status": status,
        "unique_best_supported": bool(all_unique),
        "allowed_claim_strength": allowed_claim_strength,
        "selection_rule_frozen": True,
        "predictive_interpretation_gate_passed": bool(predictive_gate),
        "selection_data": "development nested source-connected outer OOF only",
        "heldout_used_for_selection": False,
        "heldout_used_for_tuning": False,
        "heldout_used_for_feature_selection": False,
        "heldout_not_used": True,
        "primary_selection_metric": "group_type_stratum_macro_f1",
        "primary_metric_tolerance": macro_f1_tolerance,
        "tie_break_order": [
            "group_type_stratum_balanced_accuracy",
            "minimum_class_recall",
            "group_type_stratum_macro_ovr_auc",
            "macro_f1_fold_sd",
        ],
        "selected_selection_score": float(selected["selection_score"]),
        "selected_metrics": selected_metrics,
        "minimum_selected_class_f1": float(minimum_class_f1),
        "minimum_selected_class_recall": float(minimum_class_recall),
        "interpretation_basis": (
            "models within the prespecified tolerance of the highest geological-group-by-reported-type "
            "macro-F1, followed by class-neutral tie-breakers; exact multiclass TreeSHAP is "
            "generated only when the selected model is tree based and passes the same "
            "minimum F1 and recall requirements for all three classes"
        ),
    }
    (paths["comparison"] / "model_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return selection


def reassess_existing_model_selection(config: dict[str, Any]) -> dict[str, Any]:
    """Reapply the current selection rule without refitting any candidate model."""
    paths = output_paths(config)
    pooled = pd.read_csv(paths["comparison"] / "model_comparison_pooled_metrics.csv")
    fold_metrics = pd.read_csv(paths["comparison"] / "outer_fold_model_metrics.csv")
    predictions = pd.read_csv(
        paths["comparison"] / "all_models_nested_source_block_group_strata_oof.csv"
    )
    return _select_interpretation_model(pooled, fold_metrics, predictions, config, paths)


def _write_trial_budget_audit(
    trials: pd.DataFrame,
    grouping_columns: list[str],
    required_budget: int,
    output_path: Path,
    pruning_enabled: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, subset in trials.groupby(grouping_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        states = subset["state"].astype(str).str.upper().str.rsplit(".", n=1).str[-1]
        completed = int(states.eq("COMPLETE").sum())
        pruned = int(states.eq("PRUNED").sum())
        failed = int(states.eq("FAIL").sum())
        running = int(states.eq("RUNNING").sum())
        waiting = int(states.eq("WAITING").sum())
        initiated = int(len(subset))
        fit_column = "user_attrs_inner_fold_fits_completed"
        fit_values = (
            subset[fit_column] if fit_column in subset
            else pd.Series(0, index=subset.index, dtype=float)
        )
        inner_fits = int(pd.to_numeric(fit_values, errors="coerce").fillna(0).sum())
        budget_passed = (
            initiated == required_budget if pruning_enabled else completed == required_budget
        )
        rows.append({
            **dict(zip(grouping_columns, keys)),
            "budget_definition": "initiated_trials" if pruning_enabled else "complete_trials",
            "required_budget": int(required_budget),
            "initiated_trials": initiated, "complete_trials": completed,
            "pruned_trials": pruned, "failed_trials": failed,
            "running_trials": running, "waiting_trials": waiting,
            "inner_fold_fits_completed": inner_fits,
            "budget_passed": bool(budget_passed),
        })
    audit = pd.DataFrame(rows)
    audit.to_csv(output_path, index=False)
    if audit.empty or not audit["budget_passed"].all():
        raise RuntimeError(f"Optuna trial budget audit failed: {output_path}")
    return audit


def compare_models(config: dict[str, Any]) -> dict[str, Any]:
    validate_analysis_runtime(config)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    paths = output_paths(config)
    context = prepare_data(config)
    protocol_hash = context["analysis_protocol_hash"]
    protocol_contract = json.loads(
        (paths["audit"] / "analysis_protocol_contract.json").read_text(encoding="utf-8")
    )
    shared_study_context = {
        "analysis_protocol_hash": protocol_hash,
        "input_hashes_hash": _json_hash(protocol_contract["input_hashes"]),
        "block_registry_hash": context["block_registry_hash"],
        "feature_rule_hash": _json_hash(config["preprocessing"]),
    }
    dev = context["development"]
    x = context["chemistry"].iloc[dev].reset_index(drop=True)
    y = context["y"].iloc[dev].reset_index(drop=True)
    groups = context["groups"].iloc[dev].reset_index(drop=True)
    blocks = context["blocks"].iloc[dev].reset_index(drop=True)
    meta = context["metadata"].iloc[dev].reset_index(drop=True)

    outer_splits = int(config["validation"]["outer_folds"])
    inner_splits = int(config["validation"]["inner_folds"])
    n_trials = int(config["optimization"]["trials_per_model"])
    revision = str(config["optimization"]["study_revision"])
    seed = int(config["seed"])
    requested_outer_seed = seed + 101
    outer_splits_list, effective_outer_seed = class_complete_group_splits(
        x, y, blocks, outer_splits, requested_outer_seed, config
    )
    probabilities = {model: np.full((len(y), 3), np.nan) for model in MODEL_NAMES}
    outer_fold = np.full(len(y), -1, dtype=int)
    trial_tables = []
    fold_metrics = []
    fold_registry = []

    coverage_tables = []
    for fold_id, (outer_training, outer_validation) in enumerate(outer_splits_list, 1):
        assert_no_block_overlap(outer_training, outer_validation, meta)
        outer_fold[outer_validation] = fold_id

        x_inner = x.iloc[outer_training].reset_index(drop=True)
        y_inner = y.iloc[outer_training].reset_index(drop=True)
        g_inner = groups.iloc[outer_training].reset_index(drop=True)
        b_inner = blocks.iloc[outer_training].reset_index(drop=True)
        m_inner = meta.iloc[outer_training].reset_index(drop=True)
        inner_folds, inner_coverage = make_prepared_folds(
            x_inner, y_inner, b_inner, m_inner, inner_splits, seed + 1000 + fold_id, config
        )
        inner_coverage.insert(0, "outer_fold_id", fold_id)
        coverage_tables.append(inner_coverage)
        outer_prepared = prepare_fold(
            x, outer_training, outer_validation, fold_id,
            float(config["preprocessing"]["feature_missingness_threshold"]),
            float(config["preprocessing"]["row_missingness_threshold"]),
            list(config["preprocessing"]["excluded_features"]),
            int(config["preprocessing"]["knn_neighbors"]),
        )
        outer_valid_set = set(outer_prepared.validation_positions.tolist())
        fold_registry.extend({
            "Record ID": meta.iloc[position]["Record ID"], "outer_fold_id": fold_id,
            "role": "validation", "Geological Group ID": groups.iloc[position],
            "Reference-connected block": blocks.iloc[position],
            "reported_type": meta.iloc[position]["Granite type"],
            "valid_oof": bool(position in outer_valid_set),
            "analysis_protocol_hash": protocol_hash,
            "abstain_reason": "" if position in outer_valid_set else "row_missingness_at_or_above_threshold",
            "row_missing_fraction": float(x.iloc[position][outer_prepared.features].isna().mean()),
            "n_retained_features": int(len(outer_prepared.features)),
            "retained_features": "; ".join(outer_prepared.features),
            "requested_split_seed": int(requested_outer_seed),
            "effective_split_seed": int(effective_outer_seed),
            "split_seed_offset": int(effective_outer_seed - requested_outer_seed),
        } for position in outer_validation)

        for model_index, model_name in enumerate(MODEL_NAMES):
            print(
                f"Outer fold {fold_id}/{outer_splits}: optimizing {model_name} "
                f"({n_trials} persistent trials).",
                flush=True,
            )
            model_seed = seed + 10000 * fold_id + 100 * model_index
            split_hash = _json_hash({
                "outer_training_record_ids": sorted(meta.iloc[outer_training]["Record ID"].astype(str)),
                "outer_validation_record_ids": sorted(meta.iloc[outer_validation]["Record ID"].astype(str)),
                "inner_fold_assignment": inner_coverage[[
                    "Record ID", "fold_id", "Reference-connected block"
                ]].sort_values(["fold_id", "Record ID"]).to_dict(orient="records"),
            })
            params, trials = optimize_model(
                model_name, inner_folds, y_inner, g_inner, config, model_seed, n_trials,
                study_name=f"{revision}_{protocol_hash[:12]}_outer{fold_id}_{model_name}",
                study_context={**shared_study_context, "split_hash": split_hash},
            )
            trials.insert(0, "model", model_name)
            trials.insert(1, "outer_fold_id", fold_id)
            trial_tables.append(trials)
            estimator = build_estimator(
                model_name, params, model_seed,
                int(config["optimization"]["model_n_jobs"]),
            )
            estimator = fit_estimator(
                model_name, estimator, outer_prepared.x_training,
                y.iloc[outer_prepared.training_positions],
                groups.iloc[outer_prepared.training_positions],
            )
            warning_count = int(getattr(estimator, "_task_b_convergence_warning_count", 0))
            fold_probability = predict_score_matrix(
                model_name, estimator, outer_prepared.x_validation
            )
            probabilities[model_name][outer_prepared.validation_positions] = fold_probability
            record_metrics = multiclass_metrics(
                y.iloc[outer_prepared.validation_positions].to_numpy(), fold_probability
            )
            strata = aggregate_group_type_probabilities(
                y.iloc[outer_prepared.validation_positions],
                fold_probability,
                groups.iloc[outer_prepared.validation_positions],
                blocks.iloc[outer_prepared.validation_positions],
            )
            group_metrics = multiclass_metrics(
                strata["true_code"].to_numpy(),
                strata[["score_I", "score_A", "score_S"]].to_numpy(float),
            )
            fold_metrics.append({
                "model": model_name, "outer_fold_id": fold_id,
                "analysis_protocol_hash": protocol_hash,
                "evaluation_unit": "geological_group_by_reported_type_stratum",
                "convergence_warning_count": warning_count,
                **group_metrics,
                **{f"record_{key}": value for key, value in record_metrics.items()},
            })

    all_trials = pd.concat(trial_tables, ignore_index=True)
    all_trials.to_csv(
        paths["comparison"] / "all_models_equal_budget_optuna_trials.csv", index=False
    )
    _write_trial_budget_audit(
        all_trials, ["model", "outer_fold_id"], n_trials,
        paths["comparison"] / "optuna_trial_budget_audit.csv",
        pruning_enabled=bool(config["optimization"]["pruning"]),
    )
    fold_metrics_frame = pd.DataFrame(fold_metrics)
    fold_metrics_frame.to_csv(
        paths["comparison"] / "outer_fold_model_metrics.csv", index=False
    )
    pd.DataFrame(fold_registry).to_csv(
        paths["comparison"] / "outer_fold_registry.csv", index=False
    )
    if coverage_tables:
        pd.concat(coverage_tables, ignore_index=True).to_csv(
            paths["comparison"] / "inner_fold_coverage_audit.csv", index=False
        )

    prediction_rows = []
    stratum_prediction_rows = []
    pooled_rows = []
    for model_name in MODEL_NAMES:
        valid = np.isfinite(probabilities[model_name]).all(axis=1)
        record_metrics = multiclass_metrics(
            y.iloc[valid].to_numpy(), probabilities[model_name][valid]
        )
        strata = aggregate_group_type_probabilities(
            y.iloc[valid], probabilities[model_name][valid],
            groups.iloc[valid], blocks.iloc[valid],
        )
        group_metrics = multiclass_metrics(
            strata["true_code"].to_numpy(),
            strata[["score_I", "score_A", "score_S"]].to_numpy(float),
        )
        pooled_rows.append({
            "model": model_name,
            "evaluation": "development_nested_source_connected_outer_oof_group_type_strata",
            "evaluation_unit": "geological_group_by_reported_type_stratum",
            "score_type": model_score_type(model_name),
            **group_metrics,
            **{f"record_{key}": value for key, value in record_metrics.items()},
            "analysis_protocol_hash": protocol_hash,
        })
        for _, stratum in strata.iterrows():
            stratum_prediction_rows.append({
                "model": model_name,
                "Record ID": stratum["stratum_id"],
                "stratum_id": stratum["stratum_id"],
                "Geological group": stratum["Geological Group ID"],
                "Geological Group ID": stratum["Geological Group ID"],
                "Reference-connected block": stratum["Reference-connected block"],
                "true_code": int(stratum["true_code"]),
                "predicted_code": int(stratum["predicted_code"]),
                "reported_type": INT_TO_CLASS[int(stratum["true_code"])],
                "predicted_type": INT_TO_CLASS[int(stratum["predicted_code"])],
                "score_I": float(stratum["score_I"]),
                "score_A": float(stratum["score_A"]),
                "score_S": float(stratum["score_S"]),
                "n_records": int(stratum["n_records"]),
                "valid_oof": True,
                "analysis_protocol_hash": protocol_hash,
            })
        predicted = np.full(len(y), None, dtype=object)
        predicted[valid] = np.asarray(CLASSES)[np.argmax(probabilities[model_name][valid], axis=1)]
        for position in range(len(y)):
            prediction_rows.append({
                "model": model_name,
                "Record ID": meta.iloc[position]["Record ID"],
                "outer_fold_id": int(outer_fold[position]),
                "Geological group": groups.iloc[position],
                "Geological Group ID": groups.iloc[position],
                "Reference-connected block": blocks.iloc[position],
                "reported_type": meta.iloc[position]["Granite type"],
                "true_code": int(y.iloc[position]),
                "predicted_type": predicted[position] if valid[position] else None,
                "predicted_code": int(np.argmax(probabilities[model_name][position])) if valid[position] else None,
                "score_I": probabilities[model_name][position, 0],
                "score_A": probabilities[model_name][position, 1],
                "score_S": probabilities[model_name][position, 2],
                "score_type": model_score_type(model_name),
                "valid_oof": bool(valid[position]),
                "analysis_protocol_hash": protocol_hash,
            })
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(paths["comparison"] / "all_models_nested_source_block_oof.csv", index=False)
    stratum_predictions = pd.DataFrame(stratum_prediction_rows)
    stratum_predictions.to_csv(
        paths["comparison"] / "all_models_nested_source_block_group_strata_oof.csv",
        index=False,
    )
    s_audit = predictions[
        predictions["valid_oof"] & predictions["reported_type"].eq("S")
    ].copy()
    if "Geological Group Name" in meta:
        name_map = meta.set_index("Record ID")["Geological Group Name"]
        s_audit["Geological Group Name"] = s_audit["Record ID"].map(name_map)
    else:
        s_audit["Geological Group Name"] = ""
    s_audit["correct_S"] = s_audit["predicted_type"].eq("S").astype(int)
    for predicted_class in CLASSES:
        s_audit[f"predicted_as_{predicted_class}"] = (
            s_audit["predicted_type"].eq(predicted_class).astype(int)
        )
    s_group_audit = (
        s_audit.groupby([
            "model", "Geological Group ID", "Geological Group Name",
            "Reference-connected block",
        ], dropna=False)
        .agg(
            n_S=("Record ID", "size"),
            correctly_classified_S=("correct_S", "sum"),
            predicted_as_I=("predicted_as_I", "sum"),
            predicted_as_A=("predicted_as_A", "sum"),
            predicted_as_S=("predicted_as_S", "sum"),
        )
        .reset_index()
    )
    s_group_audit["S_recall_within_group"] = (
        s_group_audit["correctly_classified_S"] / s_group_audit["n_S"]
    )
    s_group_audit["analysis_protocol_hash"] = protocol_hash
    s_group_audit.sort_values(
        ["model", "S_recall_within_group", "n_S"], ascending=[True, True, False]
    ).to_csv(paths["comparison"] / "S_type_source_block_error_audit.csv", index=False)
    pooled = pd.DataFrame(pooled_rows).sort_values(
        ["macro_f1", "balanced_accuracy", "macro_ovr_auc"], ascending=False
    )
    pooled.to_csv(paths["comparison"] / "model_comparison_pooled_metrics.csv", index=False)

    selection = _select_interpretation_model(
        pooled, fold_metrics_frame, stratum_predictions, config, paths
    )
    pooled.merge(
        pd.read_csv(paths["comparison"] / "model_comprehensive_ranking.csv")[[
            "model", "comprehensive_rank", "selection_score",
            "minimum_class_recall", "minimum_class_f1",
        ]], on="model", how="left"
    ).to_csv(paths["figures"] / "Fig_granite_model_comparison_source_data.csv", index=False)
    return {"selection": selection, "pooled": pooled, "predictions": predictions}


def _write_heldout_figure_source(paths: dict[str, Path], predictions: pd.DataFrame) -> None:
    valid = predictions[predictions["evaluable"]].copy()
    matrix = pd.crosstab(valid["reported_type"], valid["predicted_type"], dropna=False)
    matrix = matrix.reindex(index=CLASSES, columns=CLASSES, fill_value=0)
    rows = []
    for true_type in CLASSES:
        denominator = max(int(matrix.loc[true_type].sum()), 1)
        for predicted_type in CLASSES:
            count = int(matrix.loc[true_type, predicted_type])
            rows.append({
                "reported_type": true_type, "predicted_type": predicted_type,
                "count": count, "row_proportion": count / denominator,
                "analysis_protocol_hash": valid["analysis_protocol_hash"].iloc[0],
            })
    pd.DataFrame(rows).to_csv(
        paths["figures"] / "Fig_granite_heldout_confusion_source_data.csv", index=False
    )


def _write_core_panel_sensitivity(
    config: dict[str, Any],
    paths: dict[str, Path],
    model_name: str,
    params: dict[str, Any],
    x_dev: pd.DataFrame,
    y_dev: pd.Series,
    g_dev: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    meta_test: pd.DataFrame,
    groups_test: pd.Series,
    blocks_test: pd.Series,
) -> None:
    """Evaluate the predeclared nine-variable panel without using it for selection."""
    panel = [feature for feature in config["locked_bridge_features"] if feature in x_dev.columns]
    excluded = set(config["preprocessing"]["excluded_features"])
    if excluded & set(panel):
        raise RuntimeError("The locked sensitivity panel violates the frozen exclusion policy.")
    if not panel:
        pd.DataFrame([{"status": "no_locked_panel_feature_available"}]).to_csv(
            paths["sensitivity"] / "core_panel_coverage_performance.csv", index=False
        )
        return
    threshold = float(config["preprocessing"]["row_missingness_threshold"])
    dev_keep = x_dev[panel].isna().mean(axis=1) < threshold
    test_keep = x_test[panel].isna().mean(axis=1) < threshold
    development_classes_complete = set(y_dev.loc[dev_keep].unique()) == {0, 1, 2}
    if dev_keep.sum() < 3 or test_keep.sum() == 0 or not development_classes_complete:
        pd.DataFrame([{
            "status": "insufficient_evaluable_records", "n_panel_features": len(panel),
            "development_evaluable": int(dev_keep.sum()), "heldout_evaluable": int(test_keep.sum()),
            "development_contains_all_classes": bool(development_classes_complete),
        }]).to_csv(paths["sensitivity"] / "core_panel_coverage_performance.csv", index=False)
        return
    processor = FoldPreprocessor(panel, int(config["preprocessing"]["knn_neighbors"]))
    train = processor.fit_transform(x_dev.loc[dev_keep, panel])
    test = processor.transform(x_test.loc[test_keep, panel])
    estimator = build_estimator(
        model_name, params, int(config["seed"]) + 7301,
        int(config["optimization"]["model_n_jobs"]),
    )
    estimator = fit_estimator(
        model_name, estimator, train, y_dev.loc[dev_keep], g_dev.loc[dev_keep]
    )
    score = predict_score_matrix(model_name, estimator, test)
    metrics = multiclass_metrics(y_test.loc[test_keep].to_numpy(), score)
    pd.DataFrame([{
        "analysis": "predeclared_locked_nine_feature_panel_sensitivity",
        "used_for_model_selection": False,
        "used_for_feature_selection": False,
        "model": model_name,
        "score_type": model_score_type(model_name),
        "panel_features": "; ".join(panel), "n_panel_features": len(panel),
        "heldout_total_records": int(len(x_test)),
        "heldout_evaluable_records": int(test_keep.sum()),
        "heldout_coverage": float(test_keep.mean()),
        **metrics,
    }]).to_csv(paths["sensitivity"] / "core_panel_coverage_performance.csv", index=False)


def final_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    validate_analysis_runtime(config)
    paths = output_paths(config)
    context = prepare_data(config)
    selection_file = paths["comparison"] / "model_selection.json"
    if not selection_file.exists():
        raise FileNotFoundError("Run compare_models before final_evaluation.")
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    if selection.get("analysis_protocol_hash") != context["analysis_protocol_hash"]:
        raise RuntimeError("Model selection belongs to a different Task B protocol.")
    model_name = selection["best_ranked_model"]
    holdout_contract_path = paths["processed"] / "holdout_selection_contract.json"
    holdout_contract = json.loads(holdout_contract_path.read_text(encoding="utf-8"))
    if holdout_contract.get("analysis_protocol_hash") != context["analysis_protocol_hash"]:
        raise RuntimeError("Held-out contract belongs to a different Task B protocol.")
    if bool(holdout_contract.get("final_evaluation_completed", False)):
        required_existing = [
            paths["final"] / "heldout_predictions_all_records.csv",
            paths["final"] / "heldout_metrics_evaluable_records.csv",
            paths["final"] / "selected_model_hyperparameters.json",
            paths["final"] / "granite_type_model_complete_bundle.joblib",
        ]
        missing_existing = [str(path) for path in required_existing if not path.exists()]
        if missing_existing:
            raise RuntimeError(
                "The held-out contract is marked complete but formal outputs are missing: "
                f"{missing_existing}"
            )
        final_info = json.loads(required_existing[2].read_text(encoding="utf-8"))
        metrics = pd.read_csv(required_existing[1]).iloc[0].to_dict()
        return {
            "model": final_info["model"], "params": final_info["parameters"],
            "metrics": metrics, "features": final_info["retained_features"],
            "heldout_evaluation_reused_without_reaccess": True,
        }
    if int(holdout_contract.get("heldout_access_count", 0)) != 0:
        raise RuntimeError(
            "The formal held-out set was already accessed but evaluation did not complete. "
            "Do not rerun under the same analysis revision; audit the failed run first."
        )

    dev, heldout = context["development"], context["heldout"]
    x_dev = context["chemistry"].iloc[dev].reset_index(drop=True)
    y_dev = context["y"].iloc[dev].reset_index(drop=True)
    g_dev = context["groups"].iloc[dev].reset_index(drop=True)
    b_dev = context["blocks"].iloc[dev].reset_index(drop=True)
    m_dev = context["metadata"].iloc[dev].reset_index(drop=True)

    folds, final_tuning_coverage = make_prepared_folds(
        x_dev, y_dev, b_dev, m_dev, int(config["validation"]["final_tuning_folds"]),
        int(config["seed"]) + 7001, config,
    )
    final_tuning_coverage.to_csv(
        paths["final"] / "selected_model_final_tuning_coverage.csv", index=False
    )
    params, trials = optimize_model(
        model_name, folds, y_dev, g_dev, config, int(config["seed"]) + 7101,
        int(config["optimization"]["trials_per_model"]),
        study_name=(
            f"{config['analysis_revision']}_{context['analysis_protocol_hash'][:12]}_final_{model_name}"
        ),
        study_context={
            "analysis_protocol_hash": context["analysis_protocol_hash"],
            "input_hashes_hash": _json_hash(audit_protocol(config)["input_hashes"]),
            "block_registry_hash": context["block_registry_hash"],
            "feature_rule_hash": _json_hash(config["preprocessing"]),
            "split_hash": _json_hash(final_tuning_coverage[[
                "Record ID", "fold_id", "Reference-connected block"
            ]].sort_values(["fold_id", "Record ID"]).to_dict(orient="records")),
        },
    )
    trials.insert(0, "model", model_name)
    trials.to_csv(paths["final"] / "selected_model_final_tuning_trials.csv", index=False)
    _write_trial_budget_audit(
        trials, ["model"], int(config["optimization"]["trials_per_model"]),
        paths["final"] / "final_tuning_trial_budget_audit.csv",
        pruning_enabled=bool(config["optimization"]["pruning"]),
    )

    prep_cfg = config["preprocessing"]
    features, dev_keep = select_features_and_rows(
        x_dev, float(prep_cfg["feature_missingness_threshold"]),
        float(prep_cfg["row_missingness_threshold"]), list(prep_cfg["excluded_features"]),
    )
    preprocessor = FoldPreprocessor(features, int(prep_cfg["knn_neighbors"]))
    completed_dev = preprocessor.fit_transform(x_dev.loc[dev_keep, features])
    estimator = build_estimator(
        model_name, params, int(config["seed"]) + 7201,
        int(config["optimization"]["model_n_jobs"]),
    )
    estimator = fit_estimator(
        model_name, estimator, completed_dev, y_dev.loc[dev_keep], g_dev.loc[dev_keep]
    )
    final_parameter_record = {
        "analysis_protocol_hash": context["analysis_protocol_hash"],
        "model": model_name, "parameters": params,
        "retained_features": features,
        "feature_set_hash": _json_hash(sorted(features)),
        "feature_rule_hash": _json_hash(config["preprocessing"]),
        "score_type": model_score_type(model_name),
    }
    (paths["final"] / "selected_model_hyperparameters.json").write_text(
        json.dumps(final_parameter_record, indent=2), encoding="utf-8"
    )

    # The model program is frozen before the first formal access to held-out chemistry.
    holdout_contract["heldout_access_count"] = 1
    holdout_contract["final_evaluation_started_utc"] = datetime.now(timezone.utc).isoformat()
    holdout_contract_path.write_text(
        json.dumps(holdout_contract, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    x_test = context["chemistry"].iloc[heldout].reset_index(drop=True)
    y_test = context["y"].iloc[heldout].reset_index(drop=True)
    meta_test = context["metadata"].iloc[heldout].reset_index(drop=True)
    groups_test = context["groups"].iloc[heldout].reset_index(drop=True)
    blocks_test = context["blocks"].iloc[heldout].reset_index(drop=True)
    test_keep = x_test[features].isna().mean(axis=1) < float(prep_cfg["row_missingness_threshold"])
    completed_test = preprocessor.transform(x_test.loc[test_keep, features])
    probability = predict_score_matrix(model_name, estimator, completed_test)
    record_metrics = multiclass_metrics(y_test.loc[test_keep].to_numpy(), probability)
    heldout_strata = aggregate_group_type_probabilities(
        y_test.loc[test_keep], probability,
        groups_test.loc[test_keep], blocks_test.loc[test_keep],
    )
    metrics = multiclass_metrics(
        heldout_strata["true_code"].to_numpy(),
        heldout_strata[["score_I", "score_A", "score_S"]].to_numpy(float),
    )
    heldout_strata["reported_type"] = heldout_strata["true_code"].map(INT_TO_CLASS)
    heldout_strata["predicted_type"] = heldout_strata["predicted_code"].map(INT_TO_CLASS)
    heldout_strata["analysis_protocol_hash"] = context["analysis_protocol_hash"]
    heldout_strata.to_csv(
        paths["final"] / "heldout_group_type_stratum_predictions.csv", index=False
    )
    total_heldout = int(len(x_test))
    evaluable_heldout = int(test_keep.sum())
    metrics_row = {
        "analysis_protocol_hash": context["analysis_protocol_hash"],
        "model": model_name,
        "evaluation": "fixed_source_connected_heldout_group_type_strata",
        "evaluation_unit": "geological_group_by_reported_type_stratum",
        "score_type": model_score_type(model_name),
        "heldout_total_records": total_heldout,
        "heldout_evaluable_records": evaluable_heldout,
        "heldout_coverage": evaluable_heldout / max(total_heldout, 1),
        "heldout_used_for_selection": False,
        "heldout_used_for_tuning": False,
        "heldout_used_for_feature_selection": False,
        **metrics,
        **{f"record_{key}": value for key, value in record_metrics.items()},
    }
    pd.DataFrame([metrics_row]).to_csv(
        paths["final"] / "heldout_metrics_evaluable_records.csv", index=False
    )
    pd.DataFrame([{
        "analysis_protocol_hash": context["analysis_protocol_hash"],
        "model": model_name,
        "evaluation": "fixed_source_connected_heldout_evaluable_records",
        "evaluation_unit": "whole_rock_analysis",
        **record_metrics,
    }]).to_csv(
        paths["final"] / "heldout_record_level_metrics.csv", index=False
    )
    matrix = confusion_matrix(y_test.loc[test_keep], np.argmax(probability, axis=1), labels=[0, 1, 2])
    pd.DataFrame(matrix, index=[f"true_{c}" for c in CLASSES], columns=[f"pred_{c}" for c in CLASSES]).to_csv(
        paths["final"] / "heldout_confusion_matrix.csv"
    )
    rows = []
    kept_positions = np.flatnonzero(test_keep.to_numpy())
    for local_position in range(len(x_test)):
        if not test_keep.iloc[local_position]:
            rows.append({
                "Record ID": meta_test.iloc[local_position]["Record ID"],
                "Geological group": groups_test.iloc[local_position],
                "Geological Group ID": groups_test.iloc[local_position],
                "Reference-connected block": blocks_test.iloc[local_position],
                "analysis_protocol_hash": context["analysis_protocol_hash"],
                "reported_type": meta_test.iloc[local_position]["Granite type"],
                "predicted_type": None, "score_I": np.nan, "score_A": np.nan, "score_S": np.nan,
                "score_type": model_score_type(model_name),
                "evaluable": False, "abstained": True,
                "abstain_reason": "row_missingness_at_or_above_threshold",
                "n_retained_features": int(len(features)),
                "row_missing_fraction": float(x_test.iloc[local_position][features].isna().mean()),
            })
            continue
        probability_position = int(np.flatnonzero(kept_positions == local_position)[0])
        p = probability[probability_position]
        rows.append({
            "Record ID": meta_test.iloc[local_position]["Record ID"],
            "Geological group": groups_test.iloc[local_position],
            "Geological Group ID": groups_test.iloc[local_position],
            "Reference-connected block": blocks_test.iloc[local_position],
            "analysis_protocol_hash": context["analysis_protocol_hash"],
            "reported_type": meta_test.iloc[local_position]["Granite type"],
            "predicted_type": CLASSES[int(np.argmax(p))],
            "score_I": p[0], "score_A": p[1], "score_S": p[2],
            "score_type": model_score_type(model_name),
            "evaluable": True, "abstained": False, "abstain_reason": "",
            "n_retained_features": int(len(features)),
            "row_missing_fraction": float(x_test.iloc[local_position][features].isna().mean()),
        })
    heldout_predictions = pd.DataFrame(rows)
    heldout_predictions.to_csv(paths["final"] / "heldout_predictions_all_records.csv", index=False)
    heldout_s = heldout_predictions[
        heldout_predictions["evaluable"] & heldout_predictions["reported_type"].eq("S")
    ].copy()
    if "Geological Group Name" in meta_test:
        name_map = meta_test.set_index("Record ID")["Geological Group Name"]
        heldout_s["Geological Group Name"] = heldout_s["Record ID"].map(name_map)
    else:
        heldout_s["Geological Group Name"] = ""
    heldout_s["correct_S"] = heldout_s["predicted_type"].eq("S").astype(int)
    for predicted_class in CLASSES:
        heldout_s[f"predicted_as_{predicted_class}"] = (
            heldout_s["predicted_type"].eq(predicted_class).astype(int)
        )
    heldout_s_group_audit = (
        heldout_s.groupby([
            "Geological Group ID", "Geological Group Name",
            "Reference-connected block",
        ], dropna=False)
        .agg(
            n_S=("Record ID", "size"),
            correctly_classified_S=("correct_S", "sum"),
            predicted_as_I=("predicted_as_I", "sum"),
            predicted_as_A=("predicted_as_A", "sum"),
            predicted_as_S=("predicted_as_S", "sum"),
        )
        .reset_index()
    )
    heldout_s_group_audit["S_recall_within_group"] = (
        heldout_s_group_audit["correctly_classified_S"] / heldout_s_group_audit["n_S"]
    )
    heldout_s_group_audit["selected_model"] = model_name
    heldout_s_group_audit["analysis_protocol_hash"] = context["analysis_protocol_hash"]
    heldout_s_group_audit.sort_values(
        ["S_recall_within_group", "n_S"], ascending=[True, False]
    ).to_csv(paths["final"] / "heldout_S_type_group_error_audit.csv", index=False)
    coverage_by_class = (
        heldout_predictions.groupby("reported_type", dropna=False)["evaluable"]
        .agg(total_records="size", evaluable_records="sum").reset_index()
    )
    coverage_by_class["coverage"] = (
        coverage_by_class["evaluable_records"] / coverage_by_class["total_records"]
    )
    coverage_by_class.to_csv(paths["final"] / "heldout_coverage_by_class.csv", index=False)
    coverage_by_block = (
        heldout_predictions.groupby("Reference-connected block", dropna=False)["evaluable"]
        .agg(total_records="size", evaluable_records="sum").reset_index()
    )
    coverage_by_block["coverage"] = (
        coverage_by_block["evaluable_records"] / coverage_by_block["total_records"]
    )
    coverage_by_block.to_csv(paths["final"] / "heldout_coverage_by_block.csv", index=False)
    heldout_predictions.loc[heldout_predictions["abstained"], [
        "Record ID", "Geological Group ID", "Reference-connected block",
        "reported_type", "abstain_reason", "row_missing_fraction", "n_retained_features",
    ]].to_csv(paths["final"] / "heldout_abstention_reasons.csv", index=False)
    bundle = GraniteModelBundle(
        model_name=model_name, classes=CLASSES, features=features,
        preprocessor=preprocessor, estimator=estimator,
        metadata={
            "seed": int(config["seed"]),
            "analysis_revision": config["analysis_revision"],
            "analysis_protocol_hash": context["analysis_protocol_hash"],
            "block_registry_hash": context["block_registry_hash"],
            "selected_heldout_blocks_hash": holdout_contract["selected_heldout_blocks_hash"],
            "selection_status": selection["selection_status"],
            "selection_rule_frozen": True,
            "class_order": list(CLASSES),
            "score_type": model_score_type(model_name),
            "heldout_used_for_selection": False,
        },
    )
    save_bundle(bundle, paths["final"] / "granite_type_model_complete_bundle.joblib")
    _write_core_panel_sensitivity(
        config, paths, model_name, params, x_dev, y_dev, g_dev,
        x_test, y_test, meta_test, groups_test, blocks_test,
    )
    _write_heldout_figure_source(paths, heldout_predictions)
    holdout_contract["final_evaluation_completed"] = True
    holdout_contract["final_evaluation_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    holdout_contract_path.write_text(
        json.dumps(holdout_contract, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    from plot_classification import generate_performance_figures
    generate_performance_figures(PROJECT_ROOT, config)
    return {
        "model": model_name, "params": params, "metrics": metrics,
        "features": features, "heldout_evaluation_reused_without_reaccess": False,
    }


def _tree_shap(
    model_name: str, estimator: Any, x: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    if model_name not in {"RF", "XGBoost"}:
        raise ValueError("Exact multiclass TreeSHAP is restricted to RF and XGBoost.")
    explainer = shap.TreeExplainer(estimator)
    explanation = explainer(x, check_additivity=False)
    values = np.asarray(explanation.values)
    if values.ndim != 3:
        raise RuntimeError(f"Unexpected multiclass SHAP dimensions: {values.shape}")
    if values.shape[0] == 3 and values.shape[1] == len(x):
        values = np.moveaxis(values, 0, -1)
    if values.shape != (len(x), x.shape[1], 3):
        raise RuntimeError(f"Unexpected multiclass SHAP shape: {values.shape}")
    expected = np.asarray(explanation.base_values, dtype=float)
    if expected.ndim == 1:
        expected = np.repeat(expected.reshape(1, 3), len(x), axis=0)
    elif expected.shape != (len(x), 3):
        expected = np.repeat(
            np.asarray(explainer.expected_value, dtype=float).reshape(1, 3),
            len(x), axis=0,
        )
    if model_name == "XGBoost":
        model_output = np.asarray(estimator.predict(x, output_margin=True), dtype=float)
        output_scale = "raw_margin"
    else:
        model_output = np.asarray(estimator.predict_proba(x), dtype=float)
        output_scale = "probability"
    reconstructed = values.sum(axis=1) + expected
    closure_error = float(np.max(np.abs(reconstructed - model_output)))
    return values, expected, model_output, closure_error, output_scale


def _locked_feature_direction_stability(
    config: dict[str, Any],
    metadata: pd.DataFrame,
    raw_chemistry: pd.DataFrame,
    completed_values: pd.DataFrame,
    shap_frames: dict[str, pd.DataFrame],
    fold_ids: np.ndarray,
    valid_oof: np.ndarray,
) -> pd.DataFrame:
    """Audit locked-feature direction across source-blocked SHAP folds."""
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(config["seed"]) + 9701)
    n_bootstrap = int(config["shap"]["direction_block_bootstrap_replicates"])
    group_ids = metadata["Geological Group ID"].astype(str).reset_index(drop=True)
    block_ids = metadata["Reference-connected block"].astype(str).reset_index(drop=True)

    for class_name in CLASSES:
        for feature in config["locked_bridge_features"]:
            available = feature in completed_values and feature in shap_frames[class_name]
            if not available:
                rows.append({
                    "class": class_name, "feature": feature, "feature_exists": False,
                    "usable_records": 0, "geological_groups": 0, "source_blocks": 0,
                    "availability_fraction": 0.0,
                    "completed_direction_spearman": np.nan,
                    "observed_only_direction_spearman": np.nan,
                    "fold_direction_sign_consistency": 0.0,
                    "observed_completed_fold_sign_agreement": 0.0,
                    "block_bootstrap_direction_ci_low": np.nan,
                    "block_bootstrap_direction_ci_high": np.nan,
                })
                continue

            completed = completed_values[feature]
            attribution = shap_frames[class_name][feature]
            usable = valid_oof & completed.notna().to_numpy() & attribution.notna().to_numpy()
            observed = (
                raw_chemistry[feature] if feature in raw_chemistry
                else pd.Series(np.nan, index=raw_chemistry.index)
            )
            observed_usable = usable & observed.notna().to_numpy()
            completed_rho = shap_direction(completed[usable], attribution[usable])
            observed_rho = shap_direction(observed[observed_usable], attribution[observed_usable])

            fold_completed: list[float] = []
            fold_observed: list[float] = []
            for fold_id in sorted(set(fold_ids[fold_ids > 0])):
                fold_mask = usable & (fold_ids == fold_id)
                observed_fold_mask = observed_usable & (fold_ids == fold_id)
                fold_completed.append(shap_direction(completed[fold_mask], attribution[fold_mask]))
                fold_observed.append(shap_direction(observed[observed_fold_mask], attribution[observed_fold_mask]))
            finite_completed = np.asarray([value for value in fold_completed if np.isfinite(value)])
            nonzero_signs = np.sign(finite_completed[finite_completed != 0])
            sign_consistency = (
                float(max((nonzero_signs > 0).mean(), (nonzero_signs < 0).mean()))
                if len(nonzero_signs) else 0.0
            )
            agreements = [
                np.sign(left) == np.sign(right)
                for left, right in zip(fold_completed, fold_observed)
                if np.isfinite(left) and np.isfinite(right) and left != 0 and right != 0
            ]
            observed_completed_agreement = float(np.mean(agreements)) if agreements else 0.0

            pair = pd.DataFrame({
                "feature_value": completed[usable].to_numpy(),
                "shap_value": attribution[usable].to_numpy(),
                "block": block_ids[usable].to_numpy(),
            })
            unique_blocks = pair["block"].unique()
            bootstrap_rhos: list[float] = []
            if len(unique_blocks) >= 2:
                block_frames = {block: pair[pair["block"].eq(block)] for block in unique_blocks}
                for _ in range(n_bootstrap):
                    sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
                    sampled = pd.concat([block_frames[block] for block in sampled_blocks], ignore_index=True)
                    rho = shap_direction(sampled["feature_value"], sampled["shap_value"])
                    if np.isfinite(rho):
                        bootstrap_rhos.append(rho)
            ci_low, ci_high = (
                np.quantile(bootstrap_rhos, [0.025, 0.975])
                if bootstrap_rhos else (np.nan, np.nan)
            )
            rows.append({
                "class": class_name, "feature": feature, "feature_exists": True,
                "usable_records": int(usable.sum()),
                "geological_groups": int(group_ids[usable].nunique()),
                "source_blocks": int(block_ids[usable].nunique()),
                "availability_fraction": float(usable.sum() / max(int(valid_oof.sum()), 1)),
                "completed_direction_spearman": completed_rho,
                "observed_only_direction_spearman": observed_rho,
                "fold_direction_sign_consistency": sign_consistency,
                "observed_completed_fold_sign_agreement": observed_completed_agreement,
                "block_bootstrap_direction_ci_low": float(ci_low),
                "block_bootstrap_direction_ci_high": float(ci_high),
                "fold_completed_spearman": json.dumps([
                    None if not np.isfinite(value) else float(value) for value in fold_completed
                ]),
                "fold_observed_only_spearman": json.dumps([
                    None if not np.isfinite(value) else float(value) for value in fold_observed
                ]),
                "bootstrap_replicates_requested": n_bootstrap,
                "bootstrap_replicates_valid": int(len(bootstrap_rhos)),
            })
    return pd.DataFrame(rows)


def _coupling_readiness(
    config: dict[str, Any],
    selection: dict[str, Any],
    heldout_metrics: dict[str, Any],
    stability: pd.DataFrame,
    closure: pd.DataFrame,
    availability: pd.DataFrame,
    direction_stability: pd.DataFrame,
    analysis_protocol_hash: str,
) -> dict[str, Any]:
    development_metrics = selection["selected_metrics"]
    minimum_class_f1 = min(float(development_metrics[f"f1_{class_name}"]) for class_name in CLASSES)
    minimum_class_recall = min(
        float(development_metrics[f"recall_{class_name}"]) for class_name in CLASSES
    )
    minimum_rank_stability = float(stability["median_rank_spearman"].min())
    minimum_top10_overlap = float(stability["mean_top10_overlap"].min())
    maximum_closure_error = float(closure["max_abs_model_output_closure_error"].max())
    all_locked_features_exist = bool(availability["available_in_crossfit"].all())
    minimum_locked_availability = float(availability["availability_fraction_of_valid_oof"].min())
    minimum_locked_direction_consistency = float(
        direction_stability["fold_direction_sign_consistency"].min()
    )

    exploratory = config["coupling_readiness"]["exploratory"]
    interpretation_gate = config["selection"]["interpretation_gate"]
    development_predictive_gate = (
        float(development_metrics["macro_f1"]) >= float(interpretation_gate["minimum_macro_f1"])
        and float(development_metrics["balanced_accuracy"]) >= float(interpretation_gate["minimum_balanced_accuracy"])
        and minimum_class_f1 >= float(interpretation_gate["minimum_class_f1"])
        and minimum_class_recall >= float(interpretation_gate["minimum_class_recall"])
    )
    heldout_minimum_class_f1 = min(
        float(heldout_metrics[f"f1_{class_name}"]) for class_name in CLASSES
    )
    heldout_predictive_gate = (
        float(heldout_metrics["macro_f1"]) >= float(interpretation_gate["minimum_macro_f1"])
        and float(heldout_metrics["balanced_accuracy"]) >= float(interpretation_gate["minimum_balanced_accuracy"])
        and heldout_minimum_class_f1 >= float(interpretation_gate["minimum_class_f1"])
    )
    exploratory_ready = (
        bool(selection["predictive_interpretation_gate_passed"])
        and development_predictive_gate
        and minimum_rank_stability >= float(exploratory["minimum_median_rank_spearman"])
        and minimum_top10_overlap >= float(exploratory["minimum_mean_top10_overlap"])
        and maximum_closure_error <= float(exploratory["maximum_additivity_closure_error"])
        and minimum_locked_availability >= float(exploratory["minimum_locked_feature_availability"])
        and minimum_locked_direction_consistency >= float(
            exploratory["minimum_locked_direction_sign_consistency"]
        )
        and (all_locked_features_exist or not bool(exploratory["require_all_locked_features"]))
    )
    robust = config["coupling_readiness"]["internally_robust"]
    internally_robust = (
        float(development_metrics["macro_f1"]) >= float(robust["minimum_macro_f1"])
        and float(development_metrics["balanced_accuracy"]) >= float(robust["minimum_balanced_accuracy"])
        and minimum_class_f1 >= float(robust["minimum_class_f1"])
        and minimum_rank_stability >= float(robust["minimum_median_rank_spearman"])
        and minimum_top10_overlap >= float(robust["minimum_mean_top10_overlap"])
        and maximum_closure_error <= float(robust["maximum_additivity_closure_error"])
        and minimum_locked_availability >= float(robust["minimum_locked_feature_availability"])
        and minimum_locked_direction_consistency >= float(
            robust["minimum_locked_direction_sign_consistency"]
        )
        and (all_locked_features_exist or not bool(robust["require_all_locked_features"]))
    )
    if internally_robust:
        level = "internally_robust_for_linkage"
    elif exploratory_ready:
        level = "exploratory_ready"
    else:
        level = "not_ready"
    return {
        "analysis_protocol_hash": analysis_protocol_hash,
        "readiness_level": level,
        "exploratory_ready": bool(exploratory_ready),
        "internally_robust_for_linkage": bool(internally_robust),
        "development_oof_predictive_metrics": development_metrics,
        "fixed_heldout_predictive_metrics": heldout_metrics,
        "readiness_predictive_gate_evidence": "development nested source-connected OOF",
        "fixed_heldout_role": "small-support internal check reported separately; not a readiness gate",
        "development_predictive_gate_passed": bool(development_predictive_gate),
        "fixed_heldout_predictive_gate_passed": bool(heldout_predictive_gate),
        "minimum_class_f1": minimum_class_f1,
        "minimum_class_recall": minimum_class_recall,
        "minimum_classwise_median_rank_spearman": minimum_rank_stability,
        "minimum_classwise_mean_top10_overlap": minimum_top10_overlap,
        "maximum_additivity_closure_error": maximum_closure_error,
        "all_locked_bridge_features_exist": all_locked_features_exist,
        "minimum_locked_feature_availability": minimum_locked_availability,
        "minimum_locked_direction_sign_consistency": minimum_locked_direction_consistency,
        "external_validation_claim_permitted": False,
        "joint_part4_eligibility_determined_here": False,
        "interpretation": (
            "Task B readiness concerns internal model interpretation and linkage inputs only. "
            "Part 4 must combine the independent Task A and Task B contracts before judging joint eligibility."
        ),
    }


def _write_revision_sensitivity(
    config: dict[str, Any],
    paths: dict[str, Path],
    selection: dict[str, Any],
    readiness: dict[str, Any],
) -> None:
    """Record the v2 result and compare with an old group-only run when supplied."""
    rows: list[dict[str, Any]] = [{
        "revision": config["analysis_revision"],
        "blocking_unit": "Reference-connected block",
        "comparison_status": "current_source_block_result",
        "selected_model": selection["best_ranked_model"],
        **{f"selected_{key}": value for key, value in selection["selected_metrics"].items()},
        "taskB_interpretation_readiness": readiness["readiness_level"],
    }]
    previous_value = config.get("paths", {}).get("previous_group_only_results")
    if previous_value:
        previous_root = resolve_path(PROJECT_ROOT, previous_value)
        old_metrics = previous_root / "02_Model_Comparison" / "model_comparison_pooled_metrics.csv"
        old_selection = previous_root / "02_Model_Comparison" / "model_selection.json"
        if old_metrics.exists() and old_selection.exists():
            old = json.loads(old_selection.read_text(encoding="utf-8"))
            rows.append({
                "revision": "previous_group_only",
                "blocking_unit": "Geological Group ID",
                "comparison_status": "available",
                "selected_model": old.get("best_ranked_model"),
                **{f"selected_{key}": value for key, value in old.get("selected_metrics", {}).items()},
            })
        else:
            rows.append({
                "revision": "previous_group_only", "blocking_unit": "Geological Group ID",
                "comparison_status": "configured_path_missing_required_files",
            })
    else:
        rows.append({
            "revision": "previous_group_only", "blocking_unit": "Geological Group ID",
            "comparison_status": "not_provided; no retrospective comparison performed",
        })
    pd.DataFrame(rows).to_csv(
        paths["sensitivity"] / "reference_block_revision_comparison.csv", index=False
    )


def generate_shap(config: dict[str, Any]) -> dict[str, Any]:
    paths = output_paths(config)
    context = prepare_data(config)
    selection = json.loads((paths["comparison"] / "model_selection.json").read_text(encoding="utf-8"))
    if selection.get("analysis_protocol_hash") != context["analysis_protocol_hash"]:
        raise RuntimeError("Model selection and SHAP generation use different protocol hashes.")
    interpretation_model = selection.get("interpretation_model")
    if interpretation_model not in {"RF", "XGBoost"}:
        message = (
            "SHAP gate not passed. The model ranked first under the class-neutral, group-stratum "
            "rule must be RF or XGBoost and must pass the same minimum F1 and recall gates for "
            "I-, A-, and S-type granites."
        )
        (paths["shap"] / "SHAP_GATE_NOT_PASSED.md").write_text(message, encoding="utf-8")
        return {"generated": False, "reason": message}
    gate_marker = paths["shap"] / "SHAP_GATE_NOT_PASSED.md"
    if gate_marker.exists():
        gate_marker.unlink()

    final_info = json.loads((paths["final"] / "selected_model_hyperparameters.json").read_text(encoding="utf-8"))
    if final_info.get("analysis_protocol_hash") != context["analysis_protocol_hash"]:
        raise RuntimeError("Final parameters and SHAP generation use different protocol hashes.")
    holdout_contract_path = paths["processed"] / "holdout_selection_contract.json"
    holdout_contract = json.loads(holdout_contract_path.read_text(encoding="utf-8"))
    if not bool(holdout_contract.get("final_evaluation_completed", False)):
        raise RuntimeError("Complete the one-time fixed held-out evaluation before SHAP cross-fitting.")
    params = final_info["parameters"]
    model_name = str(final_info["model"])
    if model_name != interpretation_model:
        raise RuntimeError("Final model and frozen interpretation model are inconsistent.")
    metadata, x, y = context["metadata"], context["chemistry"], context["y"]
    groups, blocks = context["groups"], context["blocks"]
    requested_shap_seed = int(config["seed"]) + 9001
    shap_splits, effective_shap_seed = class_complete_group_splits(
        x, y, blocks, int(config["shap"]["crossfit_folds"]),
        requested_shap_seed, config,
    )
    probability = np.full((len(y), 3), np.nan)
    model_output = np.full((len(y), 3), np.nan)
    base_value = np.full((len(y), 3), np.nan)
    fold_ids = np.full(len(y), -1, dtype=int)
    feature_set_hash = np.full(len(y), None, dtype=object)
    feature_values = pd.DataFrame(index=np.arange(len(y)))
    imputation_flags = pd.DataFrame(index=np.arange(len(y)))
    abstain_reason = np.full(len(y), "", dtype=object)
    shap_frames = {class_name: pd.DataFrame(index=np.arange(len(y))) for class_name in CLASSES}
    fold_importance = []
    closure_rows = []
    prep_cfg = config["preprocessing"]

    parameter_hash = _json_hash(params)
    fold_registry_rows = []
    for fold_id, (training, validation) in enumerate(shap_splits, 1):
        assert_no_block_overlap(training, validation, metadata)
        prepared = prepare_fold(
            x, training, validation, fold_id,
            float(prep_cfg["feature_missingness_threshold"]),
            float(prep_cfg["row_missingness_threshold"]), list(prep_cfg["excluded_features"]),
            int(prep_cfg["knn_neighbors"]),
        )
        fold_ids[validation] = fold_id
        estimator = build_estimator(
            model_name, params, int(config["seed"]) + 9100 + fold_id,
            int(config["optimization"]["model_n_jobs"]),
        )
        estimator = fit_estimator(
            model_name, estimator, prepared.x_training, y.iloc[prepared.training_positions],
            groups.iloc[prepared.training_positions],
        )
        p = predict_score_matrix(model_name, estimator, prepared.x_validation)
        probability[prepared.validation_positions] = p
        values, expected, outputs, closure_error, output_scale = _tree_shap(
            model_name, estimator, prepared.x_validation
        )
        model_output[prepared.validation_positions] = outputs
        base_value[prepared.validation_positions] = expected
        fold_feature_hash = _json_hash(sorted(prepared.features))
        feature_set_hash[validation] = fold_feature_hash
        closure_rows.append({
            "fold_id": fold_id,
            "model": model_name,
            "shap_output_scale": output_scale,
            "max_abs_model_output_closure_error": closure_error,
        })
        retained = set(prepared.validation_positions.tolist())
        for position in validation:
            if position not in retained:
                abstain_reason[position] = "row_missingness_at_or_above_threshold"
        fold_registry_rows.extend({
            "Record ID": metadata.iloc[position]["Record ID"],
            "Geological Group ID": metadata.iloc[position]["Geological Group ID"],
            "Reference-connected block": blocks.iloc[position],
            "crossfit_fold": fold_id,
            "requested_split_seed": int(requested_shap_seed),
            "effective_split_seed": int(effective_shap_seed),
            "split_seed_offset": int(effective_shap_seed - requested_shap_seed),
            "valid_oof": bool(position in retained),
            "abstain_reason": "" if position in retained else "row_missingness_at_or_above_threshold",
            "n_retained_features": int(len(prepared.features)),
            "feature_set_hash": fold_feature_hash,
        } for position in validation)
        for feature in prepared.features:
            if feature not in feature_values:
                feature_values[feature] = np.nan
                imputation_flags[feature] = pd.Series(pd.NA, index=feature_values.index, dtype="boolean")
            feature_values.loc[prepared.validation_positions, feature] = prepared.x_validation[feature].to_numpy()
            imputation_flags.loc[prepared.validation_positions, feature] = (
                x.loc[prepared.validation_positions, feature].isna().to_numpy()
            )
        for class_index, class_name in enumerate(CLASSES):
            for feature in prepared.features:
                if feature not in shap_frames[class_name]:
                    shap_frames[class_name][feature] = np.nan
            shap_frames[class_name].loc[prepared.validation_positions, prepared.features] = values[:, :, class_index]
            fold_importance.extend({
                "fold_id": fold_id, "class": class_name, "feature": feature,
                "mean_abs_shap": float(np.abs(values[:, feature_index, class_index]).mean()),
            } for feature_index, feature in enumerate(prepared.features))

    valid = np.isfinite(probability).all(axis=1)
    predicted = np.full(len(y), None, dtype=object)
    predicted[valid] = np.asarray(CLASSES)[np.argmax(probability[valid], axis=1)]
    prediction_frame = pd.DataFrame({
        "Record ID": metadata["Record ID"], "Geological Group ID": groups,
        "Reference-connected block": blocks,
        "crossfit_fold": np.where(fold_ids >= 0, fold_ids, np.nan),
        "evidence_layer": "full-data source-connected cross-fit attribution; not independent heldout performance",
        "analysis_protocol_hash": context["analysis_protocol_hash"],
        "reported_granite_type": metadata["Granite type"],
        "predicted_type": predicted, "P_I": probability[:, 0], "P_A": probability[:, 1],
        "P_S": probability[:, 2],
        "predicted_probability_I": probability[:, 0],
        "predicted_probability_A": probability[:, 1],
        "predicted_probability_S": probability[:, 2],
        "model_output_I": model_output[:, 0], "model_output_A": model_output[:, 1],
        "model_output_S": model_output[:, 2], "base_value_I": base_value[:, 0],
        "base_value_A": base_value[:, 1], "base_value_S": base_value[:, 2],
        "feature_set_hash": feature_set_hash, "model_parameter_hash": parameter_hash,
        "score_type": model_score_type(model_name), "class_order": "I; A; S",
        "interpretation_model": model_name, "shap_output_scale": output_scale,
        "valid_oof": valid, "abstain_reason": abstain_reason,
    })
    prediction_frame.to_csv(paths["shap"] / "granite_type_full_oof_predictions_with_ids.csv", index=False)

    prefix = pd.DataFrame({
        "Record ID": metadata["Record ID"],
        "Geological Group ID": groups,
        "Reference-connected block": blocks,
        "crossfit_fold": np.where(fold_ids >= 0, fold_ids, np.nan),
        "valid_oof": valid,
        "abstain_reason": abstain_reason,
        "reported_granite_type": metadata["Granite type"],
        "predicted_probability_I": probability[:, 0],
        "predicted_probability_A": probability[:, 1],
        "predicted_probability_S": probability[:, 2],
        "model_output_I": model_output[:, 0], "model_output_A": model_output[:, 1],
        "model_output_S": model_output[:, 2], "base_value_I": base_value[:, 0],
        "base_value_A": base_value[:, 1], "base_value_S": base_value[:, 2],
        "feature_set_hash": feature_set_hash, "model_parameter_hash": parameter_hash,
        "analysis_protocol_hash": context["analysis_protocol_hash"],
        "score_type": model_score_type(model_name), "shap_output_scale": output_scale,
        "interpretation_model": model_name,
    })
    imputed_output = feature_values.rename(columns=lambda name: f"imputed::{name}")
    imputation_output = imputation_flags.rename(columns=lambda name: f"was_imputed::{name}")
    pd.concat([prefix, imputed_output, imputation_output], axis=1).to_csv(
        paths["shap"] / "granite_type_oof_imputed_feature_values.csv", index=False
    )
    summary_rows = []
    for class_name in CLASSES:
        class_shap_output = shap_frames[class_name].rename(columns=lambda name: f"SHAP::{name}")
        class_prefix = prefix.copy()
        class_prefix.insert(
            class_prefix.columns.get_loc("predicted_probability_I"),
            "class_score", probability[:, CLASSES.index(class_name)],
        )
        class_prefix.insert(
            class_prefix.columns.get_loc("class_score") + 1,
            "explained_class", class_name,
        )
        pd.concat([class_prefix, class_shap_output, imputed_output, imputation_output], axis=1).to_csv(
            paths["shap"] / f"granite_type_{class_name}_oof_shap_with_ids.csv", index=False
        )
        for feature in shap_frames[class_name].columns:
            summary_rows.append({
                "class": class_name, "feature": feature,
                "mean_abs_shap": float(shap_frames[class_name][feature].abs().mean()),
                "median_signed_shap": float(shap_frames[class_name][feature].median()),
                "feature_shap_spearman": shap_direction(
                    feature_values[feature], shap_frames[class_name][feature]
                ),
                "n": int(pd.concat([feature_values[feature], shap_frames[class_name][feature]], axis=1).dropna().shape[0]),
            })
    summary = pd.DataFrame(summary_rows)
    summary["analysis_protocol_hash"] = context["analysis_protocol_hash"]
    summary.to_csv(paths["shap"] / "granite_type_class_specific_shap_summary.csv", index=False)
    fold_table = pd.DataFrame(fold_importance)
    fold_table["analysis_protocol_hash"] = context["analysis_protocol_hash"]
    fold_table.to_csv(paths["shap"] / "granite_type_shap_fold_importance.csv", index=False)
    stability_rows = []
    for class_name in CLASSES:
        wide = fold_table[fold_table["class"] == class_name].pivot_table(
            index="feature", columns="fold_id", values="mean_abs_shap", fill_value=0
        )
        correlations, overlaps = [], []
        for left in range(wide.shape[1]):
            for right in range(left + 1, wide.shape[1]):
                correlations.append(spearmanr(wide.iloc[:, left], wide.iloc[:, right]).statistic)
                top_left = set(wide.iloc[:, left].nlargest(10).index)
                top_right = set(wide.iloc[:, right].nlargest(10).index)
                overlaps.append(len(top_left & top_right) / 10)
        stability_rows.append({
            "class": class_name, "median_rank_spearman": float(np.nanmedian(correlations)),
            "mean_top10_overlap": float(np.nanmean(overlaps)),
        })
    stability = pd.DataFrame(stability_rows)
    closure = pd.DataFrame(closure_rows)
    stability["analysis_protocol_hash"] = context["analysis_protocol_hash"]
    closure["analysis_protocol_hash"] = context["analysis_protocol_hash"]
    stability.to_csv(paths["shap"] / "granite_type_shap_stability.csv", index=False)
    closure.to_csv(paths["shap"] / "granite_type_shap_additivity_check.csv", index=False)
    crossfit_registry = pd.DataFrame(fold_registry_rows)
    crossfit_registry["analysis_protocol_hash"] = context["analysis_protocol_hash"]
    crossfit_registry_path = paths["shap"] / "granite_type_shap_crossfit_fold_registry.csv"
    crossfit_registry.to_csv(crossfit_registry_path, index=False)
    crossfit_registry_hash = sha256_file(crossfit_registry_path)
    heldout_row = pd.read_csv(
        paths["final"] / "heldout_metrics_evaluable_records.csv"
    ).iloc[0]
    heldout_metric_names = [
        "macro_f1", "balanced_accuracy", "macro_ovr_auc",
        *[f"f1_{class_name}" for class_name in CLASSES],
    ]
    heldout_metrics = {
        name: float(heldout_row[name]) for name in heldout_metric_names
    }
    locked_availability = []
    for class_name in CLASSES:
        for feature in config["locked_bridge_features"]:
            available = (
                feature in shap_frames[class_name]
                and feature in feature_values
            )
            usable = int(
                (shap_frames[class_name][feature].notna() & feature_values[feature].notna()).sum()
            ) if available else 0
            locked_availability.append({
                "class": class_name, "feature": feature,
                "available_in_crossfit": bool(available),
                "usable_records": usable,
                "availability_fraction_of_valid_oof": usable / max(int(valid.sum()), 1),
            })
    locked_availability_frame = pd.DataFrame(locked_availability)
    locked_availability_frame["analysis_protocol_hash"] = context["analysis_protocol_hash"]
    locked_availability_frame.to_csv(
        paths["shap"] / "locked_bridge_feature_availability.csv", index=False
    )
    direction_stability = _locked_feature_direction_stability(
        config, metadata, x, feature_values, shap_frames, fold_ids, valid
    )
    direction_stability["analysis_protocol_hash"] = context["analysis_protocol_hash"]
    direction_stability.to_csv(
        paths["shap"] / "locked_bridge_feature_direction_stability.csv", index=False
    )
    locked_availability_frame = locked_availability_frame.merge(
        direction_stability[[
            "class", "feature", "geological_groups", "source_blocks"
        ]], on=["class", "feature"], how="left", validate="one_to_one"
    )
    locked_availability_frame.to_csv(
        paths["shap"] / "locked_bridge_feature_availability.csv", index=False
    )
    readiness = _coupling_readiness(
        config, selection, heldout_metrics, stability, closure,
        locked_availability_frame, direction_stability,
        context["analysis_protocol_hash"],
    )
    readiness["locked_bridge_feature_availability_file"] = "locked_bridge_feature_availability.csv"
    readiness["locked_bridge_feature_direction_stability_file"] = (
        "locked_bridge_feature_direction_stability.csv"
    )
    readiness_path = paths["shap"] / "taskB_interpretation_readiness.json"
    readiness_path.write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if not prediction_frame["Record ID"].is_unique:
        raise RuntimeError("Task B bridge predictions must contain one row per Record ID.")
    holdout_contract_hash = sha256_file(holdout_contract_path)
    required_bridge_files = [
        paths["shap"] / "granite_type_full_oof_predictions_with_ids.csv",
        paths["shap"] / "granite_type_I_oof_shap_with_ids.csv",
        paths["shap"] / "granite_type_A_oof_shap_with_ids.csv",
        paths["shap"] / "granite_type_S_oof_shap_with_ids.csv",
        paths["shap"] / "granite_type_oof_imputed_feature_values.csv",
        readiness_path,
        paths["shap"] / "locked_bridge_feature_availability.csv",
        paths["shap"] / "locked_bridge_feature_direction_stability.csv",
    ]
    task_b_output_complete = all(path.exists() for path in required_bridge_files)
    if not task_b_output_complete:
        missing_bridge = [str(path) for path in required_bridge_files if not path.exists()]
        raise RuntimeError(f"Task B bridge files are incomplete: {missing_bridge}")
    bridge_contract = {
        "analysis_revision": config["analysis_revision"],
        "analysis_protocol_hash": context["analysis_protocol_hash"],
        "holdout_contract_hash": holdout_contract_hash,
        "block_registry_hash": context["block_registry_hash"],
        "crossfit_registry_hash": crossfit_registry_hash,
        "evidence_layer": "full-data source-connected cross-fit attribution",
        "independent_heldout_performance": False,
        "intended_use": "interpretation_and_part4_linkage_only",
        "causal_interpretation_prohibited": True,
        "taskB_output_complete": bool(task_b_output_complete),
        "joint_part4_eligibility_determined_here": False,
        "taskB_interpretation_readiness": readiness,
        "record_id_hash": _json_hash(metadata["Record ID"].astype(str).tolist()),
        "geological_group_hash": _json_hash(groups.astype(str).tolist()),
        "reference_connected_block_hash": _json_hash(blocks.astype(str).tolist()),
        "feature_set_hashes": sorted(pd.Series(feature_set_hash).dropna().unique().tolist()),
        "model_parameter_hash": parameter_hash,
        "locked_bridge_features": config["locked_bridge_features"],
        "locked_bridge_feature_availability": json.loads(
            locked_availability_frame.to_json(orient="records")
        ),
        "class_order": list(CLASSES),
        "class_score_columns": {"I": "P_I", "A": "P_A", "S": "P_S"},
        "class_score_semantics": (
            f"{model_name} predicted probabilities from source-connected-block cross-fitting"
        ),
        "interpretation_model": model_name,
        "shap_output_scale": output_scale,
        "validity_column": "valid_oof",
        "abstention_reason_column": "abstain_reason",
        "valid_oof_records": int(valid.sum()),
        "total_records": int(len(valid)),
        "bridge_files": [
            "granite_type_full_oof_predictions_with_ids.csv",
            "granite_type_I_oof_shap_with_ids.csv",
            "granite_type_A_oof_shap_with_ids.csv",
            "granite_type_S_oof_shap_with_ids.csv",
            "granite_type_oof_imputed_feature_values.csv",
            "taskB_interpretation_readiness.json",
            "locked_bridge_feature_availability.csv",
            "locked_bridge_feature_direction_stability.csv"
        ],
    }
    (paths["shap"] / "taskB_bridge_contract.json").write_text(
        json.dumps(bridge_contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    figure_source_rows = []
    top_n = int(config["shap"]["top_features_per_class"])
    for class_name in CLASSES:
        top_features = (
            summary[summary["class"].eq(class_name)]
            .sort_values("mean_abs_shap", ascending=False).head(top_n)["feature"].tolist()
        )
        for feature_rank, feature in enumerate(top_features, 1):
            available = feature_values[feature].notna() & shap_frames[class_name][feature].notna()
            figure_source_rows.append(pd.DataFrame({
                "Record ID": metadata.loc[available, "Record ID"].to_numpy(),
                "class": class_name, "feature": feature,
                "feature_rank": feature_rank,
                "SHAP value": shap_frames[class_name].loc[available, feature].to_numpy(),
                "imputed feature value": feature_values.loc[available, feature].to_numpy(),
                "analysis_protocol_hash": context["analysis_protocol_hash"],
            }))
    if figure_source_rows:
        pd.concat(figure_source_rows, ignore_index=True).to_csv(
            paths["figures"] / "Fig_granite_SHAP_source_data.csv", index=False
        )
    _write_revision_sensitivity(config, paths, selection, readiness)

    from plot_classification import generate_all_figures
    generate_all_figures(PROJECT_ROOT, config)
    return {
        "generated": True,
        "valid_oof": int(valid.sum()),
        "summary_rows": int(len(summary)),
        "taskB_interpretation_readiness": readiness,
    }


def write_manifest(config: dict[str, Any], config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    paths = output_paths(config)
    manifest_path = paths["logs"] / "run_manifest.json"
    inputs = {
        "S1": resolve_path(PROJECT_ROOT, config["paths"]["s1"]),
        "S2": resolve_path(PROJECT_ROOT, config["paths"]["s2"]),
        "config": config_path.resolve(),
    }
    outputs = sorted(
        path for path in paths["root"].rglob("*")
        if path.is_file() and path.resolve() != manifest_path.resolve()
    )
    reproducibility_roots = [
        PROJECT_ROOT / "config",
        PROJECT_ROOT / "notebooks",
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "03_Documentation",
    ]
    package_files = [
        PROJECT_ROOT / "environment.yml",
        PROJECT_ROOT / "requirements-lock.txt",
        PROJECT_ROOT / "README.md",
    ]
    for root in reproducibility_roots:
        package_files.extend(
            path for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and ".ipynb_checkpoints" not in path.parts
        )
    package_files = sorted({path.resolve() for path in package_files if path.exists()})
    protocol_contract = json.loads(
        (paths["audit"] / "analysis_protocol_contract.json").read_text(encoding="utf-8")
    )
    contract_files = {
        "analysis_protocol_contract": paths["audit"] / "analysis_protocol_contract.json",
        "block_registry": paths["audit"] / "reference_connected_block_registry.csv",
        "heldout_contract": paths["processed"] / "holdout_selection_contract.json",
        "fixed_holdout_registry": paths["processed"] / "fixed_source_block_holdout_registry.csv",
        "outer_fold_registry": paths["comparison"] / "outer_fold_registry.csv",
        "shap_crossfit_registry": paths["shap"] / "granite_type_shap_crossfit_fold_registry.csv",
        "v6_readiness": paths["robustness"] / "taskB_interpretation_readiness_v6.json",
        "v6_bridge_contract": paths["robustness"] / "taskB_bridge_contract_v6.json",
    }
    manifest = {
        "analysis_revision": config["analysis_revision"],
        "analysis_protocol_hash": protocol_contract["analysis_protocol_hash"],
        "pipeline_scope": "Task B granite-type classification and source-connected cross-fit SHAP only; no prospectivity coupling",
        "evidence_layers": {
            "development_nested_oof": "model tuning, comparison, and selection only",
            "fixed_heldout": "one final internal generalization assessment only",
            "full_data_crossfit_shap": "canonical interpretation output; not heldout performance",
            "repeated_development_nested_oof": "selected-algorithm performance and probability uncertainty; readiness gate",
            "repeated_full_data_crossfit_shap": "attribution stability and Part 4 bridge sensitivity"
        },
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in inputs.items()},
        "software": software_versions(),
        "config": config,
        "contract_and_split_hashes": {
            name: sha256_file(path) if path.exists() else None
            for name, path in contract_files.items()
        },
        "reproducibility_files": [
            {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(path)}
            for path in package_files
        ],
        "outputs": [{"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(path)} for path in outputs],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def validate_final_outputs(config: dict[str, Any]) -> dict[str, Any]:
    """Stop if the formal Task B package is incomplete or contracts diverge."""
    validate_analysis_runtime(config)
    paths = output_paths(config)
    required = [
        paths["audit"] / "analysis_protocol_contract.json",
        paths["audit"] / "input_hashes.json",
        paths["audit"] / "taskB_cohort_attrition_record_audit.csv",
        paths["audit"] / "taskB_cohort_attrition_summary.json",
        paths["audit"] / "reference_connected_block_registry.csv",
        paths["audit"] / "reference_graph_audit.csv",
        paths["processed"] / "fixed_source_block_holdout_registry.csv",
        paths["processed"] / "holdout_candidate_partitions.csv",
        paths["processed"] / "holdout_selection_contract.json",
        paths["comparison"] / "all_models_nested_source_block_oof.csv",
        paths["comparison"] / "all_models_nested_source_block_group_strata_oof.csv",
        paths["comparison"] / "outer_fold_registry.csv",
        paths["comparison"] / "inner_fold_coverage_audit.csv",
        paths["comparison"] / "optuna_trial_budget_audit.csv",
        paths["comparison"] / "model_comprehensive_ranking.csv",
        paths["comparison"] / "S_type_source_block_error_audit.csv",
        paths["comparison"] / "source_block_bootstrap_model_differences.csv",
        paths["comparison"] / "model_selection.json",
        paths["final"] / "final_tuning_trial_budget_audit.csv",
        paths["final"] / "selected_model_hyperparameters.json",
        paths["final"] / "heldout_predictions_all_records.csv",
        paths["final"] / "heldout_metrics_evaluable_records.csv",
        paths["final"] / "heldout_record_level_metrics.csv",
        paths["final"] / "heldout_group_type_stratum_predictions.csv",
        paths["final"] / "heldout_confusion_matrix.csv",
        paths["final"] / "heldout_coverage_by_class.csv",
        paths["final"] / "heldout_coverage_by_block.csv",
        paths["final"] / "heldout_abstention_reasons.csv",
        paths["final"] / "heldout_S_type_group_error_audit.csv",
        paths["shap"] / "granite_type_full_oof_predictions_with_ids.csv",
        paths["shap"] / "granite_type_I_oof_shap_with_ids.csv",
        paths["shap"] / "granite_type_A_oof_shap_with_ids.csv",
        paths["shap"] / "granite_type_S_oof_shap_with_ids.csv",
        paths["shap"] / "granite_type_oof_imputed_feature_values.csv",
        paths["shap"] / "granite_type_shap_stability.csv",
        paths["shap"] / "granite_type_shap_additivity_check.csv",
        paths["shap"] / "locked_bridge_feature_availability.csv",
        paths["shap"] / "locked_bridge_feature_direction_stability.csv",
        paths["shap"] / "taskB_interpretation_readiness.json",
        paths["shap"] / "taskB_bridge_contract.json",
        paths["sensitivity"] / "reference_block_revision_comparison.csv",
        paths["sensitivity"] / "core_panel_coverage_performance.csv",
        paths["logs"] / "run_manifest.json",
    ]
    if bool(config.get("repeated_validation", {}).get("enabled", False)):
        required.extend([
            paths["robustness"] / "taskB_label_and_source_audit.csv",
            paths["robustness"] / "taskB_repeated_outer_oof_record_predictions.csv",
            paths["robustness"] / "taskB_canonical_oof_record_predictions.csv",
            paths["robustness"] / "taskB_group_type_oof_predictions.csv",
            paths["robustness"] / "taskB_repeated_oof_metrics.csv",
            paths["robustness"] / "taskB_classwise_metrics_with_group_bootstrap_ci.csv",
            paths["robustness"] / "taskB_calibration_metrics.csv",
            paths["robustness"] / "taskB_oof_class_specific_shap_long.csv",
            paths["robustness"] / "taskB_shap_rank_stability_by_class.csv",
            paths["robustness"] / "taskB_bridge_feature_direction_stability.csv",
            paths["robustness"] / "taskB_leave_one_source_block_shap_sensitivity.csv",
            paths["robustness"] / "taskB_correlation_cluster_shap_sensitivity.csv",
            paths["robustness"] / "taskB_interpretation_readiness_v6.json",
            paths["robustness"] / "taskB_bridge_contract_v6.json",
        ])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Formal Task B output package is incomplete: {missing}")

    protocol = json.loads(required[0].read_text(encoding="utf-8"))
    protocol_hash = protocol["analysis_protocol_hash"]
    holdout = json.loads((paths["processed"] / "holdout_selection_contract.json").read_text(encoding="utf-8"))
    selection = json.loads((paths["comparison"] / "model_selection.json").read_text(encoding="utf-8"))
    bridge = json.loads((paths["shap"] / "taskB_bridge_contract.json").read_text(encoding="utf-8"))
    if any(item.get("analysis_protocol_hash") != protocol_hash for item in (holdout, selection, bridge)):
        raise RuntimeError("Protocol hash mismatch among held-out, selection, and bridge contracts.")
    if not holdout.get("final_evaluation_completed") or int(holdout.get("heldout_access_count", 0)) != 1:
        raise RuntimeError("Formal held-out evaluation must be completed exactly once.")
    for audit_path in (
        paths["comparison"] / "optuna_trial_budget_audit.csv",
        paths["final"] / "final_tuning_trial_budget_audit.csv",
    ):
        audit = pd.read_csv(audit_path)
        if audit.empty or not audit["budget_passed"].astype(bool).all():
            raise RuntimeError(f"Trial budget audit failed: {audit_path}")
    prediction = pd.read_csv(paths["shap"] / "granite_type_full_oof_predictions_with_ids.csv")
    if not prediction["Record ID"].is_unique:
        raise RuntimeError("Task B cross-fit bridge is not one row per Record ID.")
    readiness_level = bridge["taskB_interpretation_readiness"]["readiness_level"]
    if bool(config.get("repeated_validation", {}).get("enabled", False)):
        v6_readiness_path = paths["robustness"] / "taskB_interpretation_readiness_v6.json"
        readiness_level = json.loads(
            v6_readiness_path.read_text(encoding="utf-8")
        )["readiness_level"]
    result = {
        "status": "PASS",
        "analysis_protocol_hash": protocol_hash,
        "formal_heldout_access_count": 1,
        "taskB_interpretation_readiness": readiness_level,
        "joint_part4_eligibility_determined_here": False,
        "required_files_checked": len(required),
    }
    if bool(config.get("repeated_validation", {}).get("enabled", False)):
        from robustness_audit import validate_robustness_outputs
        result["v6_robustness"] = validate_robustness_outputs(config)
    return result


def run_all(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    preparation = prepare_data(config)
    comparison = compare_models(config)
    final = final_evaluation(config)
    shap_result = generate_shap(config)
    robustness_result = None
    if bool(config.get("repeated_validation", {}).get("enabled", False)):
        from robustness_audit import run_robustness_audit
        robustness_result = run_robustness_audit(config)
    manifest = write_manifest(config, config_path)
    acceptance = validate_final_outputs(config)
    return {
        "preparation": preparation["summary"],
        "selection": comparison["selection"],
        "final": final,
        "shap": shap_result,
        "robustness": robustness_result,
        "manifest_outputs": len(manifest["outputs"]),
        "acceptance": acceptance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Granite-type model comparison and SHAP pipeline")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--stage", choices=[
            "protocol", "prepare", "compare", "final", "shap", "robust", "manifest", "validate", "all"
        ], default="all"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.check_only:
        print(json.dumps(validate_inputs(config), ensure_ascii=False, indent=2))
        return
    if args.stage == "protocol":
        protocol = audit_protocol(config)
        result = {
            "analysis_protocol_hash": protocol["analysis_protocol_hash"],
            "block_registry_hash": protocol["block_registry_hash"],
        }
    elif args.stage == "prepare":
        result = prepare_data(config)["summary"]
    elif args.stage == "compare":
        result = compare_models(config)["selection"]
    elif args.stage == "final":
        result = final_evaluation(config)
    elif args.stage == "shap":
        result = generate_shap(config)
    elif args.stage == "robust":
        from robustness_audit import run_robustness_audit
        result = run_robustness_audit(config)
    elif args.stage == "manifest":
        result = write_manifest(config, args.config)
    elif args.stage == "validate":
        result = validate_final_outputs(config)
    else:
        result = run_all(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
