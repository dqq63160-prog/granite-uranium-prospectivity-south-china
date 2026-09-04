"""Repeated nested source-connected grouped OOF evaluation with Optuna."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from export_utils import (
    build_analysis_protocol_contract,
    IntegrityLedger,
    load_config,
    save_json,
    save_table,
    sha256_file,
    sha256_payload,
    stage_complete,
)
from model_core import (
    MODEL_INDEX,
    FoldModelMatrix,
    aggregate_to_geological_groups,
    build_model,
    fit_model,
    make_grouped_splits,
    mean_confidence_interval,
    predict_score,
    probability_metrics,
    score_type_for_model,
    select_threshold,
    standardize_for_model,
    threshold_metrics,
)


@dataclass
class PreparedFold:
    fold: int
    train: pd.DataFrame
    valid: pd.DataFrame
    x_train: np.ndarray
    x_valid: np.ndarray
    x_train_standardized: np.ndarray
    x_valid_standardized: np.ndarray


def require_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna is required. Create the environment from environment.yml and select that Jupyter kernel."
        ) from exc
    return optuna


def load_primary_input(config_path: str | Path) -> tuple[dict[str, Any], Path, dict[str, Path], pd.DataFrame, list[str]]:
    config, root, paths = load_config(config_path)
    frame_path = paths["processed"] / "primary_model_cohort_with_nan.csv"
    feature_path = paths["processed"] / "primary_feature_list.txt"
    if not frame_path.exists() or not feature_path.exists():
        raise FileNotFoundError("Run run_data_pipeline(config_path) before nested model evaluation.")
    frame = pd.read_csv(frame_path, low_memory=False)
    features = [line.strip() for line in feature_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = sorted(set(features).difference(frame.columns))
    if missing:
        raise KeyError(f"Frozen predictors missing from primary cohort: {missing}")
    return config, root, paths, frame, features


def prepare_inner_folds(
    outer_train: pd.DataFrame,
    features: list[str],
    config: dict[str, Any],
    seed: int,
) -> tuple[list[PreparedFold], int, int]:
    requested = int(config["validation"]["inner_folds"])
    minimum = int(config["validation"].get("minimum_inner_folds", 3))
    last_error: Exception | None = None
    for n_splits in range(requested, minimum - 1, -1):
        try:
            splits, used_seed = make_grouped_splits(outer_train, config, n_splits, seed)
            prepared: list[PreparedFold] = []
            for fold, (train_idx, valid_idx) in enumerate(splits, 1):
                train = outer_train.iloc[train_idx].reset_index(drop=True)
                valid = outer_train.iloc[valid_idx].reset_index(drop=True)
                matrix = FoldModelMatrix(features, config)
                x_train = matrix.fit_transform(train)
                x_valid = matrix.transform(valid)
                x_train_scaled, x_valid_scaled, _ = standardize_for_model("SVM", x_train, x_valid)
                prepared.append(PreparedFold(
                    fold=fold, train=train, valid=valid,
                    x_train=x_train, x_valid=x_valid,
                    x_train_standardized=x_train_scaled, x_valid_standardized=x_valid_scaled,
                ))
            return prepared, used_seed, n_splits
        except RuntimeError as exc:
            last_error = exc
    raise RuntimeError(f"Unable to construct valid inner grouped folds: {last_error}")


def suggest_task_a_parameters(model_name: str, trial: Any, config: dict[str, Any]) -> dict[str, Any]:
    space = config["optuna"]["search_spaces"][model_name]
    if model_name == "RF":
        unlimited = trial.suggest_categorical("unlimited_depth", [False, True]) if space["allow_unlimited_depth"] else False
        return {
            "n_estimators": trial.suggest_int("n_estimators", *space["n_estimators"]),
            "max_depth": None if unlimited else trial.suggest_int("max_depth", *space["max_depth"]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", *space["min_samples_leaf"]),
            "max_features": trial.suggest_float("max_features", *space["max_features"]),
        }
    if model_name == "SVM":
        return {
            "C": trial.suggest_float("C", *space["C"], log=True),
            "gamma": trial.suggest_float("gamma", *space["gamma"], log=True),
        }
    if model_name == "MLP":
        layers = trial.suggest_int("hidden_layers", *space["hidden_layers"])
        units_1 = trial.suggest_int("hidden_units_1", *space["hidden_units"])
        units_2 = trial.suggest_int("hidden_units_2", *space["hidden_units"]) if layers == 2 else None
        return {
            "hidden_layer_sizes": [units_1] if layers == 1 else [units_1, units_2],
            "alpha": trial.suggest_float("alpha", *space["alpha"], log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", *space["learning_rate_init"], log=True),
        }
    if model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", *space["n_estimators"]),
            "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
            "max_depth": trial.suggest_int("max_depth", *space["max_depth"]),
            "min_child_weight": trial.suggest_float("min_child_weight", *space["min_child_weight"], log=True),
            "subsample": trial.suggest_float("subsample", *space["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *space["colsample_bytree"]),
            "gamma": trial.suggest_float("gamma", *space["gamma"]),
            "reg_alpha": trial.suggest_float("reg_alpha", *space["reg_alpha"], log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", *space["reg_lambda"], log=True),
        }
    raise KeyError(model_name)


def parameter_complexity(model_name: str, params: dict[str, Any]) -> float:
    if model_name == "RF":
        depth = 30 if params.get("max_depth") is None else float(params["max_depth"])
        return float(params["n_estimators"]) * depth / max(float(params["min_samples_leaf"]), 1.0)
    if model_name == "SVM":
        return math.log10(float(params["C"])) + math.log10(float(params["gamma"]) * 1e5 + 1.0)
    if model_name == "MLP":
        return float(sum(params["hidden_layer_sizes"]))
    if model_name == "XGBoost":
        return float(params["n_estimators"]) * (2 ** int(params["max_depth"]))
    return math.inf


def evaluate_parameters(
    model_name: str,
    params: dict[str, Any],
    prepared: list[PreparedFold],
    config: dict[str, Any],
    seed: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    parts = []
    fold_aucs = []
    optimizer_convergence = []
    optimizer_iterations = []
    optimizer_fit_cycles = []
    for item in prepared:
        fold_seed = seed + item.fold
        if model_name in {"SVM", "MLP"}:
            x_train, x_valid = item.x_train_standardized, item.x_valid_standardized
        else:
            x_train, x_valid = item.x_train, item.x_valid
        model = build_model(model_name, params, fold_seed)
        fit_model(model_name, model, x_train, item.train, config, fold_seed)
        optimizer_convergence.append(bool(getattr(model, "_task_a_optimizer_converged", True)))
        optimizer_iterations.append(int(getattr(model, "_task_a_optimizer_iterations", 0)))
        optimizer_fit_cycles.append(int(getattr(model, "_task_a_fit_cycles", 1)))
        model_score = predict_score(model_name, model, x_valid)
        grouped = aggregate_to_geological_groups(item.valid, model_score, config)
        grouped["inner_fold"] = item.fold
        parts.append(grouped)
        fold_aucs.append(roc_auc_score(grouped["target"], grouped["model_score"]))
    oof = pd.concat(parts, ignore_index=True)
    probability_summary = probability_metrics(oof["target"].to_numpy(), oof["model_score"].to_numpy())
    prediction = (oof["model_score"].to_numpy() >= 0.5).astype(int)
    metrics = {
        **probability_summary,
        "fold_auc_sd": float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0,
        "f1_at_0_5": float(f1_score(oof["target"], prediction, zero_division=0)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(oof["target"], prediction)),
        "complexity": parameter_complexity(model_name, params),
        "optimizer_converged_fraction": float(np.mean(optimizer_convergence)),
        "optimizer_iterations_max": float(max(optimizer_iterations, default=0)),
        "optimizer_fit_cycles_max": float(max(optimizer_fit_cycles, default=1)),
    }
    return metrics, oof


def select_completed_trial(study: Any, model_name: str, config: dict[str, Any]) -> Any:
    optuna = require_optuna()
    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise RuntimeError(f"No completed Optuna trials for {model_name}.")
    if model_name == "MLP":
        converged = [
            trial for trial in complete
            if float(trial.user_attrs.get("optimizer_converged_fraction", 0.0)) >= 1.0
        ]
        if converged:
            complete = converged
    best_auc = max(float(trial.user_attrs["roc_auc"]) for trial in complete)
    tolerance = float(config["validation"]["trial_auc_tolerance"])
    eligible = [trial for trial in complete if float(trial.user_attrs["roc_auc"]) >= best_auc - tolerance]
    return sorted(
        eligible,
        key=lambda trial: (
            -float(trial.user_attrs["average_precision"]),
            float(trial.user_attrs["fold_auc_sd"]),
            float(trial.user_attrs["complexity"]),
            int(trial.number),
        ),
    )[0]


def _params_from_trial(model_name: str, trial: Any) -> dict[str, Any]:
    raw = dict(trial.params)
    if model_name == "RF":
        return {
            "n_estimators": raw["n_estimators"],
            "max_depth": None if raw.get("unlimited_depth", False) else raw["max_depth"],
            "min_samples_leaf": raw["min_samples_leaf"],
            "max_features": raw["max_features"],
        }
    if model_name == "MLP":
        layers = raw["hidden_layers"]
        return {
            "hidden_layer_sizes": [raw["hidden_units_1"]] if layers == 1 else [raw["hidden_units_1"], raw["hidden_units_2"]],
            "alpha": raw["alpha"], "learning_rate_init": raw["learning_rate_init"],
        }
    return raw


def optimize_one_model_in_outer_training(
    model_name: str,
    prepared: list[PreparedFold],
    config: dict[str, Any],
    paths: dict[str, Path],
    protocol_contract: dict[str, Any],
    outer_split_hash: str,
    repeat: int,
    outer_fold: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    optuna = require_optuna()
    seed = int(config["validation"]["base_seed"]) + repeat * 10000 + outer_fold * 1000 + MODEL_INDEX[model_name] * 100
    storage_path = paths["optuna"] / config["optuna"]["storage_filename"]
    storage = f"sqlite:///{storage_path.as_posix()}"
    protocol_hash = str(protocol_contract["analysis_protocol_hash"])
    revision = str(protocol_contract["analysis_revision"])
    study_name = f"{revision}_{protocol_hash[:12]}_{model_name}_r{repeat}_f{outer_fold}"
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=study_name, direction="maximize", sampler=sampler,
        storage=storage, load_if_exists=bool(config["optuna"].get("load_if_exists", True)),
    )
    required_attributes = {
        "analysis_protocol_hash": protocol_hash,
        "input_hash_s1": protocol_contract["input_hashes"]["inputs"][0]["sha256"],
        "input_hash_s2": protocol_contract["input_hashes"]["inputs"][1]["sha256"],
        "feature_list_hash": protocol_contract["feature_list_hash"],
        "outer_split_hash": outer_split_hash,
        "search_space_hash": sha256_payload(config["optuna"]["search_spaces"][model_name]),
        "requested_complete_trials": int(config["validation"]["trials_per_model_per_outer_fold"]),
    }
    if study.user_attrs:
        mismatches = {
            key: {"stored": study.user_attrs.get(key), "current": value}
            for key, value in required_attributes.items()
            if study.user_attrs.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"Optuna study identity mismatch for {study_name}. Start a new analysis revision. {mismatches}"
            )
    else:
        for key, value in required_attributes.items():
            study.set_user_attr(key, value)

    def objective(trial: Any) -> float:
        params = suggest_task_a_parameters(model_name, trial, config)
        retry_limit = int(config["runtime"].get("keyboard_interrupt_retries_per_trial", 0))
        interrupt_count = 0
        while True:
            try:
                metrics, _ = evaluate_parameters(
                    model_name, params, prepared, config, seed + trial.number * 10
                )
                break
            except KeyboardInterrupt:
                interrupt_count += 1
                trial.set_user_attr("keyboard_interrupt_retries", interrupt_count)
                save_json(paths["logs"] / "transient_interrupt_status.json", {
                    "analysis_revision": config["analysis_revision"],
                    "analysis_protocol_hash": protocol_hash,
                    "study_name": study_name,
                    "model": model_name,
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "trial_number": int(trial.number),
                    "interrupt_count_for_current_trial": interrupt_count,
                    "retry_limit": retry_limit,
                    "action": "retry_same_trial_with_identical_parameters_and_seed",
                })
                if interrupt_count > retry_limit:
                    raise
        for key, value in metrics.items():
            trial.set_user_attr(key, float(value))
        trial.set_user_attr("keyboard_interrupt_retries", interrupt_count)
        trial.set_user_attr("parameter_hash", sha256_payload(params))
        return float(metrics["roc_auc"])

    budget = int(config["validation"]["trials_per_model_per_outer_fold"])
    initial_state_counts = pd.Series([trial.state.name for trial in study.trials]).value_counts()
    if int(initial_state_counts.get("PRUNED", 0)) or int(initial_state_counts.get("FAIL", 0)):
        raise RuntimeError(
            f"Study {study_name} contains an interrupted or failed trial: {initial_state_counts.to_dict()}. "
            "It cannot be part of the formal zero-FAIL protocol. Preserve the failed run for audit, "
            "then start from a clean ../../results/prospectivity directory under a new uninterrupted formal run."
        )
    complete_count = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    if complete_count > budget:
        raise RuntimeError(
            f"Study {study_name} already contains {complete_count} COMPLETE trials, exceeding the frozen budget of {budget}. "
            "Use a new analysis revision; do not truncate or silently reuse the incompatible study."
        )
    while complete_count < budget:
        try:
            study.optimize(
                objective,
                n_trials=budget - complete_count,
                n_jobs=int(config["validation"]["study_parallel_jobs"]),
                show_progress_bar=False,
                catch=(FloatingPointError,),
            )
        except KeyboardInterrupt:
            save_json(paths["logs"] / "interrupted_run_status.json", {
                "analysis_revision": config["analysis_revision"],
                "analysis_protocol_hash": protocol_hash,
                "study_name": study_name,
                "model": model_name,
                "repeat": repeat,
                "outer_fold": outer_fold,
                "completed_trials_before_interruption": complete_count,
                "status": "manually_interrupted_trial_recorded_as_FAIL",
                "required_action": "Do not use this SQLite database for the formal zero-FAIL run.",
            })
            raise
        new_count = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
        if new_count == complete_count:
            raise RuntimeError(f"Optuna could not complete the required trial budget for {study_name}.")
        complete_count = new_count
    state_counts = pd.Series([trial.state.name for trial in study.trials]).value_counts()
    if int(state_counts.get("PRUNED", 0)) or int(state_counts.get("FAIL", 0)):
        raise RuntimeError(
            f"Study {study_name} contains PRUNED or FAIL trials under a no-pruning protocol: "
            f"{state_counts.to_dict()}. Correct the cause and start a new analysis revision."
        )
    selected = select_completed_trial(study, model_name, config)
    params = _params_from_trial(model_name, selected)
    selected_metrics, selected_inner_oof = evaluate_parameters(
        model_name, params, prepared, config, seed + selected.number * 10
    )
    threshold, threshold_status, threshold_table = select_threshold(
        selected_inner_oof["target"].to_numpy(), selected_inner_oof["model_score"].to_numpy(), config
    )
    result = {
        "study_name": study_name,
        "model": model_name,
        "repeat": repeat,
        "outer_fold": outer_fold,
        "selected_trial_number": int(selected.number),
        "parameters": params,
        "parameter_hash": sha256_payload(params),
        "threshold": threshold,
        "threshold_status": threshold_status,
        "completed_trial_count": complete_count,
        **selected_metrics,
    }
    trial_rows = []
    for trial in study.trials:
        row = {
            "study_name": study_name, "model": model_name, "repeat": repeat, "outer_fold": outer_fold,
            "trial_number": trial.number, "state": trial.state.name, "objective": trial.value,
            "duration_seconds": trial.duration.total_seconds() if trial.duration is not None else np.nan,
            "parameters_json": json.dumps(_params_from_trial(model_name, trial), sort_keys=True) if trial.state == optuna.trial.TrialState.COMPLETE else None,
        }
        row.update({f"metric::{key}": value for key, value in trial.user_attrs.items()})
        trial_rows.append(row)
    return result, pd.DataFrame(trial_rows), threshold_table


def _append_table(rows: pd.DataFrame, path: Path) -> None:
    if path.exists():
        prior = pd.read_csv(path, low_memory=False)
        keys = [column for column in ["study_name", "trial_number"] if column in rows.columns]
        rows = pd.concat([prior, rows], ignore_index=True)
        if keys:
            rows = rows.drop_duplicates(keys, keep="last")
    save_table(rows, path)


def run_repeated_nested_evaluation(config_path: str | Path) -> dict[str, Any]:
    config, root, paths, frame, features = load_primary_input(config_path)
    data_summary = json.loads((paths["processed"] / "data_pipeline_summary.json").read_text(encoding="utf-8"))
    cohort_hash = data_summary["cohort_hash"]
    models = list(config["validation"]["models"])
    outer_rows: list[dict[str, Any]] = []
    outer_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, int]] = {}
    for repeat in range(1, int(config["validation"]["outer_repeats"]) + 1):
        outer_seed = int(config["validation"]["base_seed"]) + repeat * 10000
        outer_splits, used_outer_seed = make_grouped_splits(
            frame, config, int(config["validation"]["outer_folds"]), outer_seed
        )
        for outer_fold, (train_idx, valid_idx) in enumerate(outer_splits, 1):
            outer_cache[(repeat, outer_fold)] = (train_idx, valid_idx, used_outer_seed)
            for partition, indices in [("train", train_idx), ("validation", valid_idx)]:
                part = frame.iloc[indices]
                group_block_pairs = part[
                    [config["columns"]["group_id"], config["columns"]["cv_block_id"]]
                ].drop_duplicates()
                for group_id, block_id in group_block_pairs.itertuples(index=False, name=None):
                    outer_rows.append({
                        "repeat": repeat, "outer_fold": outer_fold, "outer_seed": used_outer_seed,
                        "partition": partition, "Geological Group ID": group_id, "CV Block ID": block_id,
                    })
    outer_registry = pd.DataFrame(outer_rows)
    save_table(outer_registry, paths["oof"] / "outer_split_registry.csv")
    protocol_contract = build_analysis_protocol_contract(config_path, frame, features, outer_registry)
    protocol_hash = str(protocol_contract["analysis_protocol_hash"])
    stage_complete(
        paths["logs"] / "stage_01_data_pipeline.complete.json",
        "data_pipeline",
        {"config": sha256_file(config_path), "cohort_hash": cohort_hash},
        [
            paths["processed"] / "primary_model_cohort_with_nan.csv",
            paths["processed"] / "mixed_group_challenge_cohort_with_nan.csv",
            paths["processed"] / "primary_feature_list.txt",
            paths["processed"] / "data_pipeline_summary.json",
            paths["audit"] / "analysis_protocol_contract.json",
        ],
        analysis_revision=config["analysis_revision"],
        analysis_protocol_hash=protocol_hash,
    )
    run_id = f"{config['analysis_revision']}::{protocol_hash[:12]}"
    ledger = IntegrityLedger(paths["audit"] / "preflight_and_integrity_checks.json", run_id=run_id)
    ledger_payload = json.loads((paths["audit"] / "preflight_and_integrity_checks.json").read_text(encoding="utf-8"))
    data_run_id = f"{config['analysis_revision']}::data_contract"
    data_run = ledger_payload.get("runs", {}).get(data_run_id, {})
    ledger.check(
        "data_contract_run_passed_before_model_evaluation",
        bool(data_run.get("all_fatal_checks_passed", False)),
        {"data_run_id": data_run_id},
    )
    sample_rows, group_rows, selected_rows, completed_oof_rows = [], [], [], []
    all_trials_path = paths["optuna"] / "all_trials.csv"
    completed_trial_counts: list[tuple[int, int, str, int]] = []
    for repeat in range(1, int(config["validation"]["outer_repeats"]) + 1):
        for outer_fold in range(1, int(config["validation"]["outer_folds"]) + 1):
            train_idx, valid_idx, used_outer_seed = outer_cache[(repeat, outer_fold)]
            outer_seed = int(config["validation"]["base_seed"]) + repeat * 10000
            outer_train = frame.iloc[train_idx].reset_index(drop=True)
            outer_valid = frame.iloc[valid_idx].reset_index(drop=True)
            train_blocks = set(outer_train[config["columns"]["cv_block_id"]])
            valid_blocks = set(outer_valid[config["columns"]["cv_block_id"]])
            ledger.check(
                f"outer_block_disjoint_r{repeat}_f{outer_fold}", train_blocks.isdisjoint(valid_blocks),
                {"overlap": sorted(train_blocks.intersection(valid_blocks))},
            )
            fold_registry = outer_registry[
                outer_registry["repeat"].eq(repeat) & outer_registry["outer_fold"].eq(outer_fold)
            ].sort_values(["partition", "Geological Group ID"])
            outer_split_hash = sha256_payload(fold_registry.to_dict("records"))
            inner_seed = outer_seed + outer_fold * 100
            prepared, used_inner_seed, actual_inner_folds = prepare_inner_folds(
                outer_train, features, config, inner_seed
            )
            outer_matrix = FoldModelMatrix(features, config)
            x_outer_train = outer_matrix.fit_transform(outer_train)
            x_outer_valid = outer_matrix.transform(outer_valid)
            completed_outer = outer_matrix.completed_raw(outer_valid).reset_index(drop=True)
            completion_part = outer_valid[[
                config["columns"]["record_id"], config["columns"]["group_id"],
                config["columns"]["cv_block_id"], config["columns"]["label"],
            ]].reset_index(drop=True).copy()
            completion_part.columns = ["Record ID", "Geological Group ID", "CV Block ID", "target"]
            completed_values = completed_outer[features].reset_index(drop=True)
            imputation_flags = pd.DataFrame({
                f"IMPUTED::{feature}": outer_valid[feature].isna().to_numpy(dtype=int)
                for feature in features
            })
            completion_metadata = pd.DataFrame({
                "repeat": np.full(len(outer_valid), repeat, dtype=int),
                "outer_fold": np.full(len(outer_valid), outer_fold, dtype=int),
                "analysis_protocol_hash": np.full(len(outer_valid), protocol_hash, dtype=object),
            })
            completion_part = pd.concat(
                [completion_part, completed_values, imputation_flags, completion_metadata], axis=1
            )
            completed_oof_rows.append(completion_part)
            for model_name in models:
                selected, trials, threshold_table = optimize_one_model_in_outer_training(
                    model_name, prepared, config, paths, protocol_contract, outer_split_hash, repeat, outer_fold
                )
                selected["outer_seed"] = used_outer_seed
                selected["inner_seed"] = used_inner_seed
                selected["actual_inner_folds"] = actual_inner_folds
                selected_rows.append(selected)
                completed_trial_counts.append((repeat, outer_fold, model_name, selected["completed_trial_count"]))
                _append_table(trials, all_trials_path)
                save_table(
                    threshold_table,
                    paths["optuna"] / f"threshold_grid_{model_name}_repeat{repeat}_outerfold{outer_fold}.csv",
                )
                if model_name in {"SVM", "MLP"}:
                    x_train, x_valid, _ = standardize_for_model(model_name, x_outer_train, x_outer_valid)
                else:
                    x_train, x_valid = x_outer_train, x_outer_valid
                model_seed = int(config["validation"]["base_seed"]) + repeat * 10000 + outer_fold * 1000 + MODEL_INDEX[model_name] * 100
                model = build_model(model_name, selected["parameters"], model_seed)
                fit_model(model_name, model, x_train, outer_train, config, model_seed)
                optimizer_converged = bool(getattr(model, "_task_a_optimizer_converged", True))
                optimizer_iterations = int(getattr(model, "_task_a_optimizer_iterations", 0))
                optimizer_fit_cycles = int(getattr(model, "_task_a_fit_cycles", 1))
                model_score = predict_score(model_name, model, x_valid)
                sample = outer_valid[[
                    config["columns"]["record_id"], config["columns"]["group_id"],
                    config["columns"]["cv_block_id"], config["columns"]["label"],
                ]].copy()
                sample.columns = ["Record ID", "Geological Group ID", "CV Block ID", "target"]
                sample["model_score"] = model_score
                sample["score_type"] = score_type_for_model(model_name)
                sample["prediction"] = (model_score >= selected["threshold"]).astype(int)
                sample["threshold"] = selected["threshold"]
                sample["threshold_status"] = selected["threshold_status"]
                sample["repeat"] = repeat
                sample["outer_fold"] = outer_fold
                sample["model"] = model_name
                sample["parameter_hash"] = selected["parameter_hash"]
                sample["analysis_protocol_hash"] = protocol_hash
                sample["optimizer_converged"] = optimizer_converged
                sample["optimizer_iterations"] = optimizer_iterations
                sample["optimizer_fit_cycles"] = optimizer_fit_cycles
                sample_rows.append(sample)
                grouped = aggregate_to_geological_groups(outer_valid, model_score, config)
                grouped = grouped.rename(columns={config["columns"]["group_id"]: "Geological Group ID"})
                grouped["prediction"] = (grouped["model_score"] >= selected["threshold"]).astype(int)
                grouped["score_type"] = score_type_for_model(model_name)
                grouped["threshold"] = selected["threshold"]
                grouped["threshold_status"] = selected["threshold_status"]
                grouped["repeat"] = repeat
                grouped["outer_fold"] = outer_fold
                grouped["model"] = model_name
                grouped["parameter_hash"] = selected["parameter_hash"]
                grouped["analysis_protocol_hash"] = protocol_hash
                grouped["optimizer_converged"] = optimizer_converged
                grouped["optimizer_iterations"] = optimizer_iterations
                grouped["optimizer_fit_cycles"] = optimizer_fit_cycles
                group_rows.append(grouped)
    sample_oof = pd.concat(sample_rows, ignore_index=True)
    group_oof = pd.concat(group_rows, ignore_index=True)
    completed_oof = pd.concat(completed_oof_rows, ignore_index=True)
    selected_table = pd.DataFrame([
        {**{key: value for key, value in row.items() if key != "parameters"},
         "parameters_json": json.dumps(row["parameters"], sort_keys=True)}
        for row in selected_rows
    ])
    duplicate_count = int(group_oof.duplicated(["repeat", "model", "Geological Group ID"]).sum())
    ledger.check("one_group_oof_record_per_repeat_and_model", duplicate_count == 0, {"duplicates": duplicate_count})
    mixed_groups = pd.read_csv(paths["processed"] / "mixed_group_challenge_cohort_with_nan.csv")[config["columns"]["group_id"]].unique()
    ledger.check("mixed_groups_absent_from_supervised_oof", not group_oof["Geological Group ID"].isin(mixed_groups).any(), None)
    counts = pd.DataFrame(completed_trial_counts, columns=["repeat", "outer_fold", "model", "completed_trials"])
    ledger.check("equal_completed_optuna_trial_budget", counts["completed_trials"].eq(int(config["validation"]["trials_per_model_per_outer_fold"])).all(), counts.to_dict("records"))
    save_table(sample_oof, paths["oof"] / "sample_level_repeated_oof.csv")
    save_table(group_oof, paths["oof"] / "geological_group_repeated_oof.csv")
    save_table(completed_oof, paths["processed"] / "fold_local_oof_completed_geochemistry.csv")
    save_table(selected_table, paths["optuna"] / "best_trial_by_outer_fold.csv")
    all_trials_current = pd.read_csv(all_trials_path, low_memory=False)
    current_prefix = f"{config['analysis_revision']}_{protocol_hash[:12]}_"
    all_trials_current = all_trials_current[all_trials_current["study_name"].str.startswith(current_prefix)].copy()
    trial_budget = all_trials_current.groupby(
        ["model", "repeat", "outer_fold"], as_index=False
    ).agg(
        COMPLETE=("state", lambda values: int((values == "COMPLETE").sum())),
        PRUNED=("state", lambda values: int((values == "PRUNED").sum())),
        FAIL=("state", lambda values: int((values == "FAIL").sum())),
        total_trials=("state", "size"),
        total_duration_seconds=("duration_seconds", "sum"),
    )
    requested = int(config["validation"]["trials_per_model_per_outer_fold"])
    trial_budget["passed"] = (
        trial_budget["COMPLETE"].eq(requested)
        & trial_budget["PRUNED"].eq(0)
        & trial_budget["FAIL"].eq(0)
    )
    save_table(trial_budget, paths["optuna"] / "trial_budget_audit.csv")
    ledger.check("optuna_exact_complete_budget_without_pruning_or_failure", bool(trial_budget["passed"].all()), trial_budget.to_dict("records"))
    search_record = {
        "cohort_hash": cohort_hash,
        "analysis_revision": config["analysis_revision"],
        "analysis_protocol_hash": protocol_hash,
        "search_spaces": config["optuna"]["search_spaces"],
        "trials_per_model_per_outer_fold": config["validation"]["trials_per_model_per_outer_fold"],
        "models": models,
        "sampler": config["optuna"]["sampler"],
        "pruning": config["validation"]["pruning"],
    }
    save_json(paths["optuna"] / "search_space_and_budget.json", search_record)
    summary = summarize_and_select_models(group_oof, config, paths)
    stage_complete(
        paths["logs"] / "stage_02_nested_models.complete.json",
        "nested_models", {"cohort_hash": cohort_hash, "config": sha256_file(config_path)},
        [
            paths["oof"] / "sample_level_repeated_oof.csv",
            paths["oof"] / "geological_group_repeated_oof.csv",
            paths["processed"] / "fold_local_oof_completed_geochemistry.csv",
            paths["optuna"] / "best_trial_by_outer_fold.csv",
            paths["results"] / "model_selection_decision.json",
        ],
        analysis_revision=config["analysis_revision"],
        analysis_protocol_hash=protocol_hash,
    )
    return summary


def build_consensus_oof(group_oof: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    expected_repeats = int(config["validation"]["outer_repeats"])
    counts = group_oof.groupby(["model", "Geological Group ID"])["repeat"].nunique()
    if not counts.eq(expected_repeats).all():
        raise RuntimeError("Consensus OOF cannot be formed because some model-group pairs lack repeats.")
    consensus = group_oof.groupby(["model", "Geological Group ID"], as_index=False).agg(
        target=("target", "first"), cv_block=("cv_block", "first"),
        model_score=("model_score", "median"), model_score_mean_sensitivity=("model_score", "mean"),
        model_score_sd=("model_score", "std"), score_type=("score_type", "first"),
        positive_vote_fraction=("prediction", "mean"),
    )
    consensus["prediction"] = (consensus["positive_vote_fraction"] >= 0.5).astype(int)
    return consensus


def _repeat_metrics(group_oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, repeat), subset in group_oof.groupby(["model", "repeat"]):
        probability = probability_metrics(subset["target"].to_numpy(), subset["model_score"].to_numpy())
        pred = subset["prediction"].to_numpy()
        target = subset["target"].to_numpy()
        rows.append({
            "model": model, "repeat": repeat, **probability,
            "f1": float(f1_score(target, pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(target, pred)),
        })
    return pd.DataFrame(rows)


def _model_summary(repeated: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    confidence = float(config["validation"]["confidence_level"])
    for model, subset in repeated.groupby("model"):
        row: dict[str, Any] = {"model": model}
        for metric in ["roc_auc", "average_precision", "f1", "balanced_accuracy"]:
            mean, low, high = mean_confidence_interval(subset[metric].to_numpy(), confidence)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
            row[f"{metric}_sd"] = float(subset[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _consensus_metrics(consensus: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, subset in consensus.groupby("model"):
        metrics = probability_metrics(subset["target"].to_numpy(), subset["model_score"].to_numpy())
        classification = threshold_metrics(subset["target"].to_numpy(), subset["positive_vote_fraction"].to_numpy(), 0.5)
        rows.append({"model": model, **metrics, **{key: value for key, value in classification.items() if key != "threshold"}})
    return pd.DataFrame(rows)


def _selection(consensus_metrics: pd.DataFrame, summary: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    table = consensus_metrics.merge(summary[["model", "roc_auc_sd"]], on="model", how="left")
    maximum = table["roc_auc"].max()
    tolerance = float(config["validation"]["trial_auc_tolerance"])
    eligible = table[table["roc_auc"] >= maximum - tolerance].copy()
    selected = eligible.sort_values(
        ["average_precision", "roc_auc_sd", "f1", "model"], ascending=[False, True, False, True]
    ).iloc[0]
    return {
        "selected_model": selected["model"],
        "selected_model_is_tree_based": bool(selected["model"] in {"RF", "XGBoost"}),
        "selection_rule": config["validation"]["model_selection_rule"],
        "auc_tolerance": tolerance,
        "eligible_models": eligible["model"].tolist(),
        "consensus_metrics": table.to_dict("records"),
        "note": "Model selection was completed before SHAP attribution and did not prespecify XGBoost.",
    }


def _pairwise_model_bootstrap(
    consensus: pd.DataFrame, config: dict[str, Any], resampling_unit: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    models = sorted(consensus["model"].unique())
    iterations = int(config["validation"]["group_bootstrap_iterations"])
    seed = int(config["validation"]["base_seed"]) + 909
    rng = np.random.default_rng(seed)
    wide_score = consensus.pivot(index="Geological Group ID", columns="model", values="model_score")
    group_meta = consensus.drop_duplicates("Geological Group ID").set_index("Geological Group ID")
    target = group_meta["target"].reindex(wide_score.index)
    block = group_meta["cv_block"].reindex(wide_score.index)
    rows = []
    for left_index, left in enumerate(models):
        for right in models[left_index + 1:]:
            differences = []
            observed = float(
                roc_auc_score(target, wide_score[left]) - roc_auc_score(target, wide_score[right])
            )
            for _ in range(iterations):
                if resampling_unit == "Geological Group ID":
                    indices = rng.integers(0, len(wide_score), len(wide_score))
                elif resampling_unit == "CV Block ID":
                    unique_blocks = block.drop_duplicates().to_numpy()
                    sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
                    indices = np.concatenate([
                        np.flatnonzero(block.to_numpy() == sampled_block) for sampled_block in sampled_blocks
                    ])
                else:
                    raise ValueError(resampling_unit)
                sampled_y = target.to_numpy()[indices]
                if np.unique(sampled_y).size < 2:
                    continue
                left_auc = roc_auc_score(sampled_y, wide_score[left].to_numpy()[indices])
                right_auc = roc_auc_score(sampled_y, wide_score[right].to_numpy()[indices])
                differences.append(left_auc - right_auc)
            values = np.asarray(differences)
            rows.append({
                "model_left": left, "model_right": right,
                "resampling_unit": resampling_unit,
                "observed_auc_difference_left_minus_right": observed,
                "bootstrap_mean_auc_difference_left_minus_right": float(values.mean()),
                "ci_low": float(np.quantile(values, 0.025)), "ci_high": float(np.quantile(values, 0.975)),
                "bootstrap_iterations_valid": int(values.size),
            })
    table = pd.DataFrame(rows)
    intervals_include_zero = (
        int(((table["ci_low"] <= 0) & (table["ci_high"] >= 0)).sum())
        if not table.empty else 0
    )
    uncertainty = {
        "bootstrap_unit": resampling_unit,
        "iterations_requested": iterations,
        "pairwise_auc_intervals_include_zero": intervals_include_zero,
        "pair_count": len(table),
    }
    return table, uncertainty


def summarize_and_select_models(
    group_oof: pd.DataFrame, config: dict[str, Any], paths: dict[str, Path]
) -> dict[str, Any]:
    repeated = _repeat_metrics(group_oof)
    summary = _model_summary(repeated, config)
    consensus = build_consensus_oof(group_oof, config)
    consensus_metrics = _consensus_metrics(consensus)
    decision = _selection(consensus_metrics, summary, config)
    protocol_contract = json.loads(
        (paths["audit"] / "analysis_protocol_contract.json").read_text(encoding="utf-8")
    )
    decision["analysis_revision"] = config["analysis_revision"]
    decision["analysis_protocol_hash"] = protocol_contract["analysis_protocol_hash"]
    group_bootstrap, group_uncertainty = _pairwise_model_bootstrap(
        consensus, config, "Geological Group ID"
    )
    block_bootstrap, block_uncertainty = _pairwise_model_bootstrap(
        consensus, config, "CV Block ID"
    )
    pairwise = pd.concat([group_bootstrap, block_bootstrap], ignore_index=True)
    uncertainty = {
        "primary_resampling_unit": "Geological Group ID",
        "sensitivity_resampling_unit": "CV Block ID",
        "group_bootstrap": group_uncertainty,
        "block_bootstrap": block_uncertainty,
    }
    repeat_winners = []
    tolerance = float(config["validation"]["trial_auc_tolerance"])
    for repeat, subset in repeated.groupby("repeat"):
        maximum = subset["roc_auc"].max()
        eligible = subset[subset["roc_auc"].ge(maximum - tolerance)]
        winner = eligible.sort_values(
            ["average_precision", "roc_auc", "f1", "model"], ascending=[False, False, False, True]
        ).iloc[0]
        repeat_winners.append({"repeat": int(repeat), "selected_model": winner["model"]})
    selection_frequency = pd.DataFrame(repeat_winners).groupby("selected_model", as_index=False).agg(
        selected_repeat_count=("repeat", "size")
    )
    selection_frequency["selected_repeat_fraction"] = (
        selection_frequency["selected_repeat_count"] / repeated["repeat"].nunique()
    )
    uncertainty["selected_model"] = decision["selected_model"]
    uncertainty["eligible_models_within_auc_tolerance"] = decision["eligible_models"]
    save_table(repeated, paths["results"] / "model_performance_by_repeat.csv")
    save_table(summary, paths["results"] / "model_performance_mean_ci.csv")
    save_table(consensus, paths["results"] / "geological_group_consensus_oof.csv")
    save_table(consensus_metrics, paths["results"] / "model_consensus_metrics.csv")
    save_table(group_bootstrap, paths["results"] / "pairwise_group_bootstrap_model_differences.csv")
    save_table(block_bootstrap, paths["results"] / "pairwise_block_bootstrap_model_differences.csv")
    save_table(pairwise, paths["results"] / "pairwise_group_and_block_bootstrap_model_differences.csv")
    save_table(selection_frequency, paths["results"] / "model_family_selection_frequency.csv")
    save_json(paths["results"] / "model_selection_decision.json", decision)
    save_json(paths["results"] / "model_selection_uncertainty.json", uncertainty)
    return decision
