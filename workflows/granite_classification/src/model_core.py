from __future__ import annotations

import hashlib
import json
import platform
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


CLASSES = ("I", "A", "S")
CLASS_TO_INT = {name: index for index, name in enumerate(CLASSES)}
INT_TO_CLASS = {index: name for name, index in CLASS_TO_INT.items()}
MODEL_NAMES = ("RF", "SVM", "MLP", "XGBoost")
MODEL_SEARCH_SPACE_PROTOCOL = {
    "RF": {
        "n_estimators": [300, 1200, "integer"], "max_depth": [5, 32, "integer"],
        "min_samples_leaf": [1, 12, "integer"], "min_samples_split": [2, 24, "integer"],
        "max_features": [0.20, 0.95, "continuous"],
    },
    "SVM": {
        "C": [1e-2, 1e3, "log-continuous"],
        "gamma": [1e-5, 1.0, "log-continuous"],
        "kernel": "rbf", "probability_calibration": "none",
    },
    "MLP": {
        "hidden_1": [64, 192, 32, "integer-step"],
        "hidden_2": [32, 96, 32, "integer-step"],
        "alpha": [1e-6, 1e-2, "log-continuous"],
        "learning_rate_init": [1e-4, 2e-2, "log-continuous"],
        "batch_size": [64, 128, 256], "max_iter": 600, "early_stopping": False,
    },
    "XGBoost": {
        "n_estimators": [150, 500, "integer"],
        "learning_rate": [0.015, 0.15, "log-continuous"],
        "max_depth": [3, 8, "integer"],
        "min_child_weight": [1.0, 18.0, "log-continuous"],
        "gamma": [0.0, 3.0, "continuous"], "subsample": [0.60, 1.0, "continuous"],
        "colsample_bytree": [0.55, 1.0, "continuous"],
        "reg_alpha": [1e-6, 4.0, "log-continuous"],
        "reg_lambda": [1e-3, 12.0, "log-continuous"],
    },
}
DEFAULT_RUNTIME_LOCK = (
    Path(__file__).resolve().parents[1] / "config" / "runtime_lock.json"
)


def clean_feature_name(value: object) -> str:
    return str(value).replace(" (wt.%)", "").replace(" (ppm)", "").strip()


RATIO_FEATURES = {
    "A/CNK", "A/NK", "Fe*", "K2O/Na2O", "Rb/Sr", "Zr/Hf", "Eu/Eu*", "(La/Yb)N",
}
_MOLAR_MASS = {
    "Al2O3": 101.9612, "CaO": 56.0774, "Na2O": 61.9789,
    "K2O": 94.196, "FeO": 71.8444, "Fe2O3": 159.6882, "MgO": 40.3044,
}
_CHONDRITE_PPM = {"La": 0.237, "Sm": 0.153, "Eu": 0.0580, "Gd": 0.2055, "Yb": 0.170}


def add_petrogenetic_ratios(chemistry: pd.DataFrame) -> pd.DataFrame:
    """Append the frozen v5 petrogenetic ratios before fold-wise imputation."""
    frame = chemistry.copy()
    molar = {
        oxide: frame[oxide] / _MOLAR_MASS[oxide]
        for oxide in _MOLAR_MASS if oxide in frame.columns
    }
    if all(name in molar for name in ("Al2O3", "CaO", "Na2O", "K2O")):
        frame["A/CNK"] = molar["Al2O3"] / (molar["CaO"] + molar["Na2O"] + molar["K2O"])
        frame["A/NK"] = molar["Al2O3"] / (molar["Na2O"] + molar["K2O"])
    if all(name in molar for name in ("FeO", "Fe2O3", "MgO")):
        feot_molar = (frame["FeO"] + 0.8998 * frame["Fe2O3"]) / _MOLAR_MASS["FeO"]
        frame["Fe*"] = feot_molar / (feot_molar + molar["MgO"])
    if {"K2O", "Na2O"}.issubset(frame.columns):
        frame["K2O/Na2O"] = frame["K2O"] / frame["Na2O"]
    if {"Rb", "Sr"}.issubset(frame.columns):
        frame["Rb/Sr"] = frame["Rb"] / frame["Sr"]
    if {"Zr", "Hf"}.issubset(frame.columns):
        frame["Zr/Hf"] = frame["Zr"] / frame["Hf"]
    if {"Sm", "Eu", "Gd"}.issubset(frame.columns):
        frame["Eu/Eu*"] = (frame["Eu"] / _CHONDRITE_PPM["Eu"]) / np.sqrt(
            (frame["Sm"] / _CHONDRITE_PPM["Sm"]) * (frame["Gd"] / _CHONDRITE_PPM["Gd"])
        )
    if {"La", "Yb"}.issubset(frame.columns):
        frame["(La/Yb)N"] = (frame["La"] / _CHONDRITE_PPM["La"]) / (
            frame["Yb"] / _CHONDRITE_PPM["Yb"]
        )
    for column in RATIO_FEATURES & set(frame.columns):
        values = frame[column].to_numpy(dtype=float, na_value=np.nan)
        frame[column] = np.where(np.isfinite(values), np.clip(values, 1e-6, 1e6), np.nan)
    return frame


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_runtime_lock(path: Path | None = None) -> dict[str, Any]:
    """Load the single version contract used by code and submission metadata."""
    lock_path = DEFAULT_RUNTIME_LOCK if path is None else Path(path)
    if not lock_path.exists():
        raise FileNotFoundError(f"Missing runtime lock: {lock_path}")
    with lock_path.open("r", encoding="utf-8") as stream:
        lock = json.load(stream)
    required = {"environment_name", "python_exact", "packages"}
    missing = required - set(lock)
    if missing:
        raise ValueError(f"Runtime lock is missing fields: {sorted(missing)}")
    return lock


def resolve_path(project_root: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (project_root / candidate).resolve()


def software_versions() -> dict[str, str]:
    import matplotlib
    import optuna
    import openpyxl
    import scipy
    import seaborn
    import shap
    import sklearn
    import xgboost

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "shap": shap.__version__,
        "optuna": optuna.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "seaborn": seaborn.__version__,
        "joblib": joblib.__version__,
        "openpyxl": openpyxl.__version__,
    }


def validate_runtime(strict: bool = True, lock_path: Path | None = None) -> dict[str, Any]:
    """Audit the active interpreter before a long model-comparison run."""
    lock = load_runtime_lock(lock_path)
    versions = software_versions()
    mismatches: list[str] = []
    if versions["python"] != lock["python_exact"]:
        mismatches.append(f"python={versions['python']} (expected {lock['python_exact']})")
    for package, expected in lock["packages"].items():
        actual = versions.get(package)
        if actual != expected:
            mismatches.append(f"{package}={actual} (expected {expected})")
    report: dict[str, Any] = {
        "executable": sys.executable,
        "versions": versions,
        "runtime_lock": str((DEFAULT_RUNTIME_LOCK if lock_path is None else Path(lock_path)).resolve()),
        "environment_name": lock["environment_name"],
        "compatible": not mismatches,
        "mismatches": mismatches,
    }
    if strict and mismatches:
        details = "; ".join(mismatches)
        raise RuntimeError(
            "The active Jupyter kernel does not match the locked analysis environment. "
            f"{details}. Use the environment defined in environment.yml (name: "
            f"'{lock['environment_name']}') before running model comparison."
        )
    return report


def load_granite_dataset(
    s1_path: Path,
    s2_path: Path,
    exclude_groups: set[str] | None = None,
    reconcile_types: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not s1_path.exists():
        raise FileNotFoundError(f"Missing S1 workbook: {s1_path}")
    if not s2_path.exists():
        raise FileNotFoundError(f"Missing Supplementary Table S1 workbook: {s2_path}")

    raw = pd.read_excel(s1_path, sheet_name="Dataset", header=1)
    required = {"Record ID", "Sample ID", "Granite type", "Geological Group ID"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"S1 is missing required columns: {sorted(missing)}")

    chemistry_start = next(
        (index for index, column in enumerate(raw.columns)
         if str(column).endswith("(wt.%)") or str(column).endswith("(ppm)")),
        None,
    )
    if chemistry_start is None:
        raise ValueError("No chemistry columns ending in (wt.%) or (ppm) were found in S1.")

    raw["Granite type"] = raw["Granite type"].astype(str).str.strip().str.upper()
    explicit = raw["Granite type"].isin(CLASSES)
    raw = raw.loc[explicit].reset_index(drop=True)
    if exclude_groups:
        raw = raw.loc[
            ~raw["Geological Group ID"].astype(str).str.strip().isin(exclude_groups)
        ].reset_index(drop=True)
    if reconcile_types:
        group_column = raw["Geological Group ID"].astype(str).str.strip()
        keep_mask = pd.Series(True, index=raw.index, dtype=bool)
        for group, majority_type in reconcile_types.items():
            mask = group_column.eq(group)
            keep_mask = keep_mask & (~mask | raw["Granite type"].eq(majority_type))
        raw = raw.loc[keep_mask].reset_index(drop=True)
    if raw["Record ID"].isna().any() or raw["Record ID"].duplicated().any():
        raise ValueError("Record ID must be non-missing and unique after I/A/S filtering.")

    safe_metadata = [
        "Record ID", "Sample ID", "Age reported (Ma)", "Granite type",
        "Geological Group ID", "Geological Group Name", "Reference ID", "Citation key",
    ]
    safe_metadata = [column for column in safe_metadata if column in raw.columns]
    metadata = raw[safe_metadata].copy()
    if "Reference ID" not in metadata:
        raise ValueError("S1 must contain Reference ID for source-connected blocking.")
    if "Citation key" not in metadata and "Reference ID" in metadata:
        metadata["Citation key"] = metadata["Reference ID"]
    metadata["Geological group"] = metadata["Geological Group ID"].astype(str).str.strip()
    if metadata["Geological group"].isin(["", "nan", "None"]).any():
        raise ValueError("Every main-analysis record must have a Geological Group ID.")

    chemistry = raw.iloc[:, chemistry_start:].copy()
    chemistry.columns = [clean_feature_name(column) for column in chemistry.columns]
    chemistry = chemistry.apply(pd.to_numeric, errors="coerce")
    chemistry[chemistry < 0] = np.nan
    if "FeOT" in chemistry.columns:
        chemistry = chemistry.drop(columns="FeOT")
    chemistry = add_petrogenetic_ratios(chemistry)

    forbidden = {
        "Mineralization label", "Granite type", "Record ID", "Sample ID",
        "Geological Group ID", "Geological group", "Citation key", "Reference ID",
    }
    leaked = forbidden & set(chemistry.columns)
    if leaked:
        raise RuntimeError(f"Metadata leakage detected in chemistry matrix: {sorted(leaked)}")

    groups = pd.read_excel(s2_path, sheet_name="Geological Groups", header=1)
    if "Geological Group ID" not in groups.columns:
        raise ValueError("Supplementary Table S1 sheet 'Geological Groups' lacks Geological Group ID.")
    groups = groups.drop_duplicates("Geological Group ID").copy()
    missing_groups = set(metadata["Geological group"]) - set(groups["Geological Group ID"].astype(str))
    if missing_groups:
        raise ValueError(f"Supplementary Table S1 lacks {len(missing_groups)} S1 geological groups.")

    audit_columns = [
        "Geological Group ID", "Geological Group Name", "Province / Region",
        "Parent Geological Unit or Included Subunit(s)", "Group Type",
        "Grouping Rationale", "Confidence Level", "Number of Samples", "Reference ID(s)",
    ]
    audit_columns = [column for column in audit_columns if column in groups.columns]
    audit = metadata.merge(groups[audit_columns], on="Geological Group ID", how="left", suffixes=("", "_S2"))
    if "Geological Group Name_S2" in audit:
        audit["Geological Group Name"] = audit.get("Geological Group Name").fillna(audit["Geological Group Name_S2"])
    return metadata.reset_index(drop=True), chemistry.reset_index(drop=True), audit.reset_index(drop=True)


class FoldPreprocessor:
    """Fold-fitted log-standardization and distance-weighted KNN imputation."""

    def __init__(self, features: list[str], n_neighbors: int = 10):
        self.features = list(features)
        self.n_neighbors = int(n_neighbors)

    def fit(self, frame: pd.DataFrame) -> "FoldPreprocessor":
        values = np.log1p(np.asarray(frame[self.features], dtype=float))
        self.mean_ = np.nanmean(values, axis=0)
        self.std_ = np.nanstd(values, axis=0)
        self.std_[~np.isfinite(self.std_) | (self.std_ == 0)] = 1.0
        standardized = (values - self.mean_) / self.std_
        self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors, weights="distance")
        self.imputer_.fit(standardized)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        values = np.log1p(np.asarray(frame[self.features], dtype=float))
        standardized = (values - self.mean_) / self.std_
        completed = self.imputer_.transform(standardized) * self.std_ + self.mean_
        completed = np.maximum(np.expm1(completed), 0.0)
        return pd.DataFrame(completed, columns=self.features, index=frame.index)

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)


def select_features_and_rows(
    training: pd.DataFrame,
    missing_threshold: float,
    row_missing_threshold: float,
    excluded_features: list[str],
) -> tuple[list[str], pd.Series]:
    missingness = training.isna().mean()
    features = missingness.index[missingness <= missing_threshold].tolist()
    features = [feature for feature in features if feature not in set(excluded_features)]
    if not features:
        raise RuntimeError("No features remain after fold-wise missingness screening.")
    keep = training[features].isna().mean(axis=1) < row_missing_threshold
    return features, keep


@dataclass
class PreparedFold:
    fold_id: int
    training_positions: np.ndarray
    validation_positions: np.ndarray
    features: list[str]
    preprocessor: FoldPreprocessor
    x_training: pd.DataFrame
    x_validation: pd.DataFrame


def prepare_fold(
    x: pd.DataFrame,
    training_positions: np.ndarray,
    validation_positions: np.ndarray,
    fold_id: int,
    missing_threshold: float,
    row_missing_threshold: float,
    excluded_features: list[str],
    n_neighbors: int,
) -> PreparedFold:
    train_raw = x.iloc[training_positions]
    valid_raw = x.iloc[validation_positions]
    features, train_keep = select_features_and_rows(
        train_raw, missing_threshold, row_missing_threshold, excluded_features
    )
    valid_keep = valid_raw[features].isna().mean(axis=1) < row_missing_threshold
    kept_training = training_positions[np.flatnonzero(train_keep.to_numpy())]
    kept_validation = validation_positions[np.flatnonzero(valid_keep.to_numpy())]
    preprocessor = FoldPreprocessor(features, n_neighbors=n_neighbors)
    x_training = preprocessor.fit_transform(x.iloc[kept_training][features])
    x_validation = preprocessor.transform(x.iloc[kept_validation][features])
    return PreparedFold(
        fold_id=fold_id,
        training_positions=kept_training,
        validation_positions=kept_validation,
        features=features,
        preprocessor=preprocessor,
        x_training=x_training,
        x_validation=x_validation,
    )


def suggest_parameters(model_name: str, trial: Any) -> dict[str, Any]:
    if model_name == "RF":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "max_depth": trial.suggest_int("max_depth", 5, 32),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 24),
            "max_features": trial.suggest_float("max_features", 0.20, 0.95),
        }
    if model_name == "SVM":
        return {
            "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
            "gamma": trial.suggest_float("gamma", 1e-5, 1.0, log=True),
        }
    if model_name == "MLP":
        return {
            "hidden_layer_sizes": (
                trial.suggest_int("hidden_1", 64, 192, step=32),
                trial.suggest_int("hidden_2", 32, 96, step=32),
            ),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 2e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        }
    if model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 18.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
            "subsample": trial.suggest_float("subsample", 0.60, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 4.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 12.0, log=True),
        }
    raise ValueError(f"Unknown model: {model_name}")


def build_estimator(model_name: str, params: dict[str, Any], seed: int, n_jobs: int) -> Any:
    params = dict(params)
    if model_name == "RF":
        estimator = RandomForestClassifier(
            **params, criterion="entropy",
            n_jobs=n_jobs, random_state=seed,
        )
    elif model_name == "SVM":
        estimator = Pipeline([
            ("scale", RobustScaler()),
            ("model", SVC(
                **params, kernel="rbf", probability=False,
                decision_function_shape="ovr", random_state=seed,
            )),
        ])
    elif model_name == "MLP":
        estimator = Pipeline([
            ("scale", RobustScaler()),
            ("model", MLPClassifier(
                **params, activation="relu", max_iter=600, early_stopping=False,
                random_state=seed,
            )),
        ])
    elif model_name == "XGBoost":
        estimator = XGBClassifier(
            **params, objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", tree_method="hist", n_jobs=n_jobs,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return estimator


def balanced_group_weights(y: pd.Series, groups: pd.Series | None = None) -> np.ndarray:
    """Combine class balancing with equal total weight per geological group."""
    weights = compute_sample_weight(class_weight="balanced", y=y).astype(float)
    if groups is not None:
        aligned_groups = pd.Series(groups).reset_index(drop=True).astype(str)
        if len(aligned_groups) != len(y):
            raise ValueError("Group labels and training targets must have the same length.")
        counts = aligned_groups.value_counts()
        group_weight = aligned_groups.map(1.0 / counts).to_numpy(float)
        weights *= group_weight
    return weights / np.mean(weights)


def fit_estimator(
    model_name: str,
    estimator: Any,
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None = None,
) -> Any:
    weights = balanced_group_weights(pd.Series(y).reset_index(drop=True), groups)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        if model_name == "MLP":
            try:
                estimator.fit(x, y, model__sample_weight=weights)
            except TypeError as error:
                if "sample_weight" not in str(error):
                    raise
                model = estimator.named_steps["model"]
                seed = int(model.random_state) if model.random_state is not None else 0
                rng = np.random.default_rng(seed)
                probability = weights / weights.sum()
                sampled = rng.choice(len(y), size=len(y), replace=True, p=probability)
                estimator.fit(x.iloc[sampled], pd.Series(y).iloc[sampled])
        elif model_name == "SVM":
            estimator.fit(x, y, model__sample_weight=weights)
        elif model_name == "XGBoost":
            estimator.fit(x, y, sample_weight=weights)
        else:
            estimator.fit(x, y, sample_weight=weights)
    estimator._task_b_convergence_warning_count = sum(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )
    return estimator


def predict_score_matrix(model_name: str, estimator: Any, x: pd.DataFrame) -> np.ndarray:
    """Return a three-column comparison score matrix in the fixed I/A/S order.

    RF, MLP and XGBoost return probabilities.  SVM returns OVR decision scores;
    a row-wise softmax is used only to place those scores on a finite simplex for
    multiclass OVR-AUC.  These SVM values are normalized decision scores, not
    calibrated probabilities.
    """
    if model_name == "SVM":
        raw = np.asarray(estimator.decision_function(x), dtype=float)
        raw -= np.max(raw, axis=1, keepdims=True)
        exp = np.exp(raw)
        return exp / exp.sum(axis=1, keepdims=True)
    return np.asarray(estimator.predict_proba(x), dtype=float)


def model_score_type(model_name: str) -> str:
    return "normalized_decision_score" if model_name == "SVM" else "probability"


def multiclass_ovr_auc(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Macro OVR-AUC that also accepts independently aggregated score columns."""
    per_class: list[float] = []
    probability = np.asarray(probability, dtype=float)
    y_array = np.asarray(y_true)
    for class_index in range(probability.shape[1]):
        binary_y = (y_array == class_index).astype(int)
        if np.unique(binary_y).size < 2:
            continue
        try:
            per_class.append(float(roc_auc_score(binary_y, probability[:, class_index])))
        except ValueError:
            continue
    return float(np.mean(per_class)) if per_class else float("nan")


def multiclass_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    prediction = np.argmax(probability, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, prediction, labels=[0, 1, 2], zero_division=0
    )
    auc = multiclass_ovr_auc(y_true, probability)
    result: dict[str, float | int] = {
        "n": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_ovr_auc": auc,
    }
    for index, class_name in enumerate(CLASSES):
        result.update({
            f"precision_{class_name}": float(precision[index]),
            f"recall_{class_name}": float(recall[index]),
            f"f1_{class_name}": float(f1[index]),
            f"support_{class_name}": int(support[index]),
        })
    return result


def optimization_score(metrics: dict[str, float | int]) -> float:
    """Class-neutral tuning objective; all three classes contribute equally."""
    return float(metrics["macro_f1"])


def aggregate_group_type_probabilities(
    y_true: np.ndarray | pd.Series,
    probability: np.ndarray,
    geological_groups: np.ndarray | pd.Series,
    source_blocks: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """Aggregate predictions to geological-group by reported-type strata.

    Mixed-type geological groups remain in one source-connected fold, while each
    reported type forms a separate evaluation stratum. Every stratum receives one
    vote regardless of its number of whole-rock analyses.
    """
    frame = pd.DataFrame({
        "true_code": np.asarray(y_true, dtype=int),
        "Geological Group ID": pd.Series(geological_groups).astype(str).to_numpy(),
        "score_I": np.asarray(probability, dtype=float)[:, 0],
        "score_A": np.asarray(probability, dtype=float)[:, 1],
        "score_S": np.asarray(probability, dtype=float)[:, 2],
    })
    if source_blocks is not None:
        frame["Reference-connected block"] = (
            pd.Series(source_blocks).astype(str).to_numpy()
        )
    valid = np.isfinite(frame[["score_I", "score_A", "score_S"]]).all(axis=1)
    frame = frame.loc[valid].copy()
    keys = ["Geological Group ID", "true_code"]
    aggregation = {
        "score_I": "median", "score_A": "median", "score_S": "median"
    }
    if "Reference-connected block" in frame:
        aggregation["Reference-connected block"] = "first"
    grouped = frame.groupby(keys, dropna=False).agg(aggregation).reset_index()
    counts = frame.groupby(keys, dropna=False).size().reset_index(name="n_records")
    grouped = grouped.merge(counts, on=keys, how="left", validate="one_to_one")
    grouped["predicted_code"] = np.argmax(
        grouped[["score_I", "score_A", "score_S"]].to_numpy(float), axis=1
    )
    grouped["stratum_id"] = (
        grouped["Geological Group ID"] + "::" + grouped["true_code"].astype(str)
    )
    return grouped


def group_bootstrap_macro_f1_difference(
    frame: pd.DataFrame,
    best_model: str,
    comparator: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    shared = frame[frame["model"].isin([best_model, comparator])].copy()
    wide = shared.pivot(index="Record ID", columns="model", values="predicted_code")
    grouping_column = (
        "Reference-connected block"
        if "Reference-connected block" in shared.columns
        else "Geological group"
    )
    meta = shared.drop_duplicates("Record ID").set_index("Record ID")[["true_code", grouping_column]]
    joined = meta.join(wide, how="inner").dropna()
    groups = joined[grouping_column].unique()
    best_matrices = []
    comparator_matrices = []
    for group in groups:
        subset = joined[joined[grouping_column] == group]
        best_matrices.append(confusion_matrix(
            subset["true_code"], subset[best_model], labels=[0, 1, 2]
        ))
        comparator_matrices.append(confusion_matrix(
            subset["true_code"], subset[comparator], labels=[0, 1, 2]
        ))
    best_matrices = np.asarray(best_matrices, dtype=float)
    comparator_matrices = np.asarray(comparator_matrices, dtype=float)

    def macro_f1_from_matrix(matrix: np.ndarray) -> float:
        true_positive = np.diag(matrix)
        false_positive = matrix.sum(axis=0) - true_positive
        false_negative = matrix.sum(axis=1) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        per_class = np.divide(
            2 * true_positive, denominator,
            out=np.zeros_like(true_positive, dtype=float), where=denominator > 0,
        )
        return float(per_class.mean())

    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(n_bootstrap):
        sampled = rng.integers(0, len(groups), size=len(groups))
        best = macro_f1_from_matrix(best_matrices[sampled].sum(axis=0))
        other = macro_f1_from_matrix(comparator_matrices[sampled].sum(axis=0))
        differences.append(best - other)
    observed = macro_f1_from_matrix(best_matrices.sum(axis=0)) - macro_f1_from_matrix(
        comparator_matrices.sum(axis=0)
    )
    low, high = np.quantile(differences, [0.025, 0.975])
    return float(observed), float(low), float(high)


@dataclass
class GraniteModelBundle:
    model_name: str
    classes: tuple[str, ...]
    features: list[str]
    preprocessor: FoldPreprocessor
    estimator: Any
    metadata: dict[str, Any]

    def predict_proba(self, raw_chemistry: pd.DataFrame) -> np.ndarray:
        completed = self.preprocessor.transform(raw_chemistry[self.features])
        return predict_score_matrix(self.model_name, self.estimator, completed)

    def predict(self, raw_chemistry: pd.DataFrame) -> np.ndarray:
        probability = self.predict_proba(raw_chemistry)
        return np.asarray(self.classes)[np.argmax(probability, axis=1)]


def save_bundle(bundle: GraniteModelBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path) -> GraniteModelBundle:
    return joblib.load(path)


def shap_direction(feature_values: pd.Series, shap_values: pd.Series) -> float:
    pair = pd.concat([feature_values, shap_values], axis=1).dropna()
    if len(pair) < 20:
        return np.nan
    x = pair.iloc[:, 0]
    y = pair.iloc[:, 1]
    if x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        try:
            return float(spearmanr(x, y).statistic)
        except Exception:
            return np.nan
