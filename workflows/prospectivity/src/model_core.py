"""Model construction, grouped splitting, weighting, and metric utilities for uranium prospectivity modelling."""

from __future__ import annotations

import inspect
import math
import signal
import threading
import warnings
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data_pipeline import GeochemicalKNNPreprocessor


MODEL_INDEX = {"RF": 1, "SVM": 2, "MLP": 3, "XGBoost": 4}


@contextmanager
def defer_sigint_during_model_fit(enabled: bool = True) -> Iterator[dict[str, int]]:
    """Defer SIGINT while an estimator is fitting and restore the prior handler afterwards.

    Jupyter may deliver a pending interrupt while joblib is collecting worker results. That
    operational interrupt does not indicate an invalid parameter set, but Optuna would record
    the trial as FAIL. The guard is limited to the estimator-fit call and is active only on the
    Python main thread. A kernel restart remains available when a formal run must be stopped.
    """

    state = {"deferred_sigint_count": 0}
    can_install = (
        enabled
        and threading.current_thread() is threading.main_thread()
        and hasattr(signal, "SIGINT")
    )
    if not can_install:
        yield state
        return

    previous_handler = signal.getsignal(signal.SIGINT)

    def _deferred_handler(signum: int, frame: Any) -> None:
        del signum, frame
        state["deferred_sigint_count"] += 1

    signal.signal(signal.SIGINT, _deferred_handler)
    try:
        yield state
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def group_registry(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    columns = config["columns"]
    result = frame.groupby(columns["group_id"], as_index=False).agg(
        target=(columns["label"], "first"),
        label_nunique=(columns["label"], "nunique"),
        cv_block=(columns["cv_block_id"], "first"),
        block_nunique=(columns["cv_block_id"], "nunique"),
        record_count=(columns["record_id"], "size"),
    )
    if not result["label_nunique"].eq(1).all():
        raise ValueError("Mixed-label geological groups cannot enter supervised model evaluation.")
    if not result["block_nunique"].eq(1).all():
        raise ValueError("A geological group maps to more than one CV block.")
    return result


def _split_is_valid(registry: pd.DataFrame, train_idx: np.ndarray, valid_idx: np.ndarray) -> bool:
    train = registry.iloc[train_idx]
    valid = registry.iloc[valid_idx]
    return (
        train["target"].nunique() == 2
        and valid["target"].nunique() == 2
        and set(train["cv_block"]).isdisjoint(set(valid["cv_block"]))
    )


def make_grouped_splits(
    frame: pd.DataFrame,
    config: dict[str, Any],
    n_splits: int,
    seed: int,
    maximum_seed_attempts: int = 200,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    columns = config["columns"]
    registry = group_registry(frame, config)
    for offset in range(maximum_seed_attempts):
        used_seed = seed + offset
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=used_seed)
        candidate = list(splitter.split(registry, registry["target"], groups=registry["cv_block"]))
        if all(_split_is_valid(registry, train_idx, valid_idx) for train_idx, valid_idx in candidate):
            splits = []
            for train_group_idx, valid_group_idx in candidate:
                train_groups = set(registry.iloc[train_group_idx][columns["group_id"]])
                valid_groups = set(registry.iloc[valid_group_idx][columns["group_id"]])
                train_rows = np.flatnonzero(frame[columns["group_id"]].isin(train_groups).to_numpy())
                valid_rows = np.flatnonzero(frame[columns["group_id"]].isin(valid_groups).to_numpy())
                if not set(frame.iloc[train_rows][columns["cv_block_id"]]).isdisjoint(
                    set(frame.iloc[valid_rows][columns["cv_block_id"]])
                ):
                    raise RuntimeError("CV block leakage detected while mapping grouped folds to records.")
                splits.append((train_rows, valid_rows))
            return splits, used_seed
    raise RuntimeError(f"No valid {n_splits}-fold grouped split was found after {maximum_seed_attempts} seeds.")


class FoldModelMatrix:
    """Preprocessing object fitted once on a current training partition."""

    def __init__(self, features: list[str], config: dict[str, Any]):
        self.features = list(features)
        self.preprocessor = GeochemicalKNNPreprocessor(
            self.features, n_neighbors=int(config["data_rules"]["knn_neighbors"])
        )

    def fit(self, frame: pd.DataFrame) -> "FoldModelMatrix":
        self.preprocessor.fit(frame)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.preprocessor.transform(frame)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.preprocessor.fit_transform(frame)

    def completed_raw(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.preprocessor.completed_raw(frame)


def standardize_for_model(
    model_name: str, x_train: np.ndarray, x_valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, StandardScaler | None]:
    if model_name not in {"SVM", "MLP"}:
        return x_train, x_valid, None
    scaler = StandardScaler()
    return scaler.fit_transform(x_train), scaler.transform(x_valid), scaler


def equal_geological_group_weights(frame: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    group_column = config["columns"]["group_id"]
    sizes = frame.groupby(group_column)[group_column].transform("size").to_numpy(float)
    weights = 1.0 / sizes
    return weights / weights.mean()


def build_model(model_name: str, params: dict[str, Any], seed: int) -> Any:
    if model_name == "RF":
        return RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=params.get("max_depth"),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=float(params["max_features"]),
            random_state=seed,
            n_jobs=-1,
            class_weight=None,
        )
    if model_name == "SVM":
        return SVC(
            C=float(params["C"]), gamma=float(params["gamma"]), kernel="rbf",
            random_state=seed, class_weight=None,
        )
    if model_name == "MLP":
        hidden = tuple(int(value) for value in params["hidden_layer_sizes"])
        return MLPClassifier(
            hidden_layer_sizes=hidden, activation="relu", solver="adam",
            alpha=float(params["alpha"]), learning_rate_init=float(params["learning_rate_init"]),
            max_iter=600, early_stopping=False, random_state=seed,
        )
    if model_name == "XGBoost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("XGBoost is required. Create the environment from environment.yml and select that Jupyter kernel.") from exc
        return XGBClassifier(
            n_estimators=int(params["n_estimators"]), learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]), min_child_weight=float(params["min_child_weight"]),
            subsample=float(params["subsample"]), colsample_bytree=float(params["colsample_bytree"]),
            gamma=float(params["gamma"]), reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]), objective="binary:logistic",
            eval_metric="logloss", random_state=seed, n_jobs=-1, tree_method="hist",
        )
    raise KeyError(f"Unsupported model: {model_name}")


def _weighted_resample(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    probabilities = weights / weights.sum()
    indices = rng.choice(np.arange(len(y)), size=len(y), replace=True, p=probabilities)
    return x[indices], y[indices]


def fit_model(
    model_name: str,
    model: Any,
    x_train: np.ndarray,
    train_frame: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
) -> Any:
    label_column = config["columns"]["label"]
    target = train_frame[label_column].to_numpy(int)
    weights = equal_geological_group_weights(train_frame, config)
    fit_signature = inspect.signature(model.fit)
    defer_sigint = bool(config.get("runtime", {}).get("defer_sigint_during_model_fit", True))
    if "sample_weight" in fit_signature.parameters:
        x_fit, y_fit = x_train, target
        fit_kwargs = {"sample_weight": weights}
    else:
        x_fit, y_fit = _weighted_resample(x_train, target, weights, seed)
        fit_kwargs = {}

    total_deferred_sigint = 0
    total_optimizer_iterations = 0
    fit_cycles = 1
    optimizer_converged = True

    if model_name == "MLP":
        runtime = config.get("runtime", {})
        iterations_per_cycle = int(runtime.get("mlp_max_iter_per_cycle", 600))
        maximum_cycles = int(runtime.get("mlp_max_fit_cycles", 3))
        if iterations_per_cycle < 1 or maximum_cycles < 1:
            raise ValueError("MLP iteration and cycle limits must be positive integers.")
        model.set_params(
            max_iter=iterations_per_cycle,
            tol=float(runtime.get("mlp_tolerance", 1e-4)),
            n_iter_no_change=int(runtime.get("mlp_n_iter_no_change", 20)),
            warm_start=False,
        )
        optimizer_converged = False
        for cycle in range(1, maximum_cycles + 1):
            fit_cycles = cycle
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                with defer_sigint_during_model_fit(defer_sigint) as interrupt_state:
                    model.fit(x_fit, y_fit, **fit_kwargs)
            total_deferred_sigint += int(interrupt_state["deferred_sigint_count"])
            total_optimizer_iterations += int(getattr(model, "n_iter_", 0))
            reached_iteration_cap = any(
                issubclass(item.category, ConvergenceWarning) for item in caught
            )
            if not reached_iteration_cap:
                optimizer_converged = True
                break
            if cycle < maximum_cycles:
                model.set_params(warm_start=True)
    else:
        with defer_sigint_during_model_fit(defer_sigint) as interrupt_state:
            model.fit(x_fit, y_fit, **fit_kwargs)
        total_deferred_sigint = int(interrupt_state["deferred_sigint_count"])

    model._task_a_deferred_sigint_count = total_deferred_sigint
    model._task_a_optimizer_converged = bool(optimizer_converged)
    model._task_a_optimizer_iterations = int(total_optimizer_iterations)
    model._task_a_fit_cycles = int(fit_cycles)
    return model


def predict_score(model_name: str, model: Any, matrix: np.ndarray) -> np.ndarray:
    if model_name == "SVM":
        return expit(np.asarray(model.decision_function(matrix), dtype=float))
    probability = np.asarray(model.predict_proba(matrix), dtype=float)
    return probability[:, 1]


def score_type_for_model(model_name: str) -> str:
    if model_name == "SVM":
        return "monotonic_transformed_decision_score_not_calibrated"
    if model_name in {"RF", "MLP", "XGBoost"}:
        return "predict_proba"
    raise KeyError(model_name)


def aggregate_to_geological_groups(
    frame: pd.DataFrame, model_scores: np.ndarray, config: dict[str, Any]
) -> pd.DataFrame:
    columns = config["columns"]
    work = frame[[columns["group_id"], columns["cv_block_id"], columns["label"]]].copy()
    work["model_score"] = np.asarray(model_scores, dtype=float)
    aggregation = config["validation"].get("geological_group_score_aggregation", "median")
    if aggregation not in {"median", "mean"}:
        raise ValueError(f"Unsupported geological-group score aggregation: {aggregation}")
    return work.groupby(columns["group_id"], as_index=False).agg(
        target=(columns["label"], "first"),
        target_nunique=(columns["label"], "nunique"),
        cv_block=(columns["cv_block_id"], "first"),
        model_score=("model_score", aggregation),
        model_score_mean_sensitivity=("model_score", "mean"),
        record_count=("model_score", "size"),
    )


def threshold_metrics(target: np.ndarray, model_score: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (np.asarray(model_score) >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(target, prediction, labels=[0, 1]).ravel()
    specificity = tn / max(tn + fp, 1)
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def probability_metrics(target: np.ndarray, model_score: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(target, model_score)),
        "average_precision": float(average_precision_score(target, model_score)),
    }


def select_threshold(
    target: np.ndarray, model_score: np.ndarray, config: dict[str, Any]
) -> tuple[float, str, pd.DataFrame]:
    rules = config["validation"]
    grid = np.linspace(
        float(rules["threshold_grid_minimum"]),
        float(rules["threshold_grid_maximum"]),
        int(rules["threshold_grid_size"]),
    )
    rows = [threshold_metrics(target, model_score, threshold) for threshold in grid]
    table = pd.DataFrame(rows)
    feasible = table[
        table["recall"].ge(float(rules["threshold_minimum_recall"]))
        & table["specificity"].ge(float(rules["threshold_minimum_specificity"]))
    ].copy()
    status = "constraints_met"
    candidates = feasible
    if candidates.empty:
        candidates = table.copy()
        status = "constraint_not_met"
    candidates["distance_from_half"] = (candidates["threshold"] - 0.5).abs()
    chosen = candidates.sort_values(
        ["f1", "balanced_accuracy", "distance_from_half", "threshold"],
        ascending=[False, False, True, True],
    ).iloc[0]
    return float(chosen["threshold"]), status, table


def mean_confidence_interval(values: np.ndarray, confidence_level: float = 0.95) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan, math.nan
    mean = float(values.mean())
    if values.size == 1:
        return mean, mean, mean
    from scipy.stats import t
    half = float(t.ppf((1 + confidence_level) / 2, values.size - 1) * values.std(ddof=1) / math.sqrt(values.size))
    return mean, mean - half, mean + half
