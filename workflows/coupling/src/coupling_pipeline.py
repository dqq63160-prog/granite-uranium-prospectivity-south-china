"""JGE Part 4: guarded coupling of prospectivity and granite-type attribution.

This module consumes only saved out-of-fold (OOF) outputs from Task A and
Task B. It does not refit either model. The workflow is deliberately gated:

1. audit contracts, schemas, identifiers and cohort attrition;
2. build a record-level matched OOF cohort;
3. aggregate inference to geological groups and connected dependency blocks;
4. compare six locked bridge features using direction, rank and contribution
   share rather than raw SHAP magnitudes;
5. quantify conditional probability associations with block-aware bootstrap
   and structure-preserving permutation;
6. export JGE-ready source data, figures and a decision manifest.

SHAP outputs are treated as model attribution, not geological causality.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from scipy.stats import spearmanr


# Mandatory editable-vector settings. These must precede figure creation.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "DejaVu Sans",
    "Liberation Sans",
]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


PALETTE = {
    "task_a": "#0F4D92",
    "task_b_s": "#9A4D8E",
    "task_b_i": "#42949E",
    "task_b_a": "#D99A2B",
    "agreement": "#3A8D5D",
    "conflict": "#B64342",
    "neutral_dark": "#3F3F3F",
    "neutral_mid": "#767676",
    "neutral_light": "#D9D9D9",
    "neutral_pale": "#F2F2F2",
    "literature": "#D8C3A5",
    "untested": "#FFFFFF",
}


FEATURE_SPECS: dict[str, dict[str, str]] = {
    "Rb": {
        "task_a": "Rb (ppm)",
        "task_b": "Rb",
        "display": "Rb",
        "unit": "ppm",
    },
    "CaO": {
        "task_a": "CaO (wt.%)",
        "task_b": "CaO",
        "display": "CaO",
        "unit": "wt.%",
    },
    "Nb": {
        "task_a": "Nb (ppm)",
        "task_b": "Nb",
        "display": "Nb",
        "unit": "ppm",
    },
    "Zr": {
        "task_a": "Zr (ppm)",
        "task_b": "Zr",
        "display": "Zr",
        "unit": "ppm",
    },
    "P2O5": {
        "task_a": "P2O5 (wt.%)",
        "task_b": "P2O5",
        "display": r"P$_2$O$_5$",
        "unit": "wt.%",
    },
    "Ba": {
        "task_a": "Ba (ppm)",
        "task_b": "Ba",
        "display": "Ba",
        "unit": "ppm",
    },
}


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    audit: Path
    tables: Path
    source_data: Path
    main_figures: Path
    supplementary_figures: Path
    logs: Path


class UnionFind:
    """Small union-find implementation for connected source-block components."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, item: str) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item: str) -> str:
        self.add(item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        root_l = self.find(left)
        root_r = self.find(right)
        if root_l == root_r:
            return
        if self.rank[root_l] < self.rank[root_r]:
            root_l, root_r = root_r, root_l
        self.parent[root_r] = root_l
        if self.rank[root_l] == self.rank[root_r]:
            self.rank[root_l] += 1


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def numeric_frame(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def require_columns(df: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def require_unique(df: pd.DataFrame, key: str, label: str) -> None:
    duplicated = df[key].duplicated(keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, key].astype(str).head(10).tolist()
        raise ValueError(f"{label} has duplicated {key}; examples: {examples}")


def make_output_paths(root: Path) -> OutputPaths:
    paths = OutputPaths(
        root=root,
        audit=root / "00_Audit",
        tables=root / "01_Tables",
        source_data=root / "02_Figure_Source_Data",
        main_figures=root / "03_Figures" / "Main_Text",
        supplementary_figures=root / "03_Figures" / "Supplementary",
        logs=root / "04_Logs",
    )
    for path in paths.__dict__.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return paths


def load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(config_path).expanduser().resolve()
    config = read_json(config_path)
    return config, config_path


def resolve_configured_path(project_root: Path, configured_path: str | Path) -> Path:
    """Resolve a configured path against the coupling workflow root."""
    path = Path(configured_path).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def resolve_input_paths(
    config: Mapping[str, Any], project_root: Path
) -> dict[str, Path]:
    """Locate the canonical result bundle used by the cross-task analysis."""
    prospectivity = resolve_configured_path(project_root, config["paths"]["prospectivity_results"])
    granite = resolve_configured_path(project_root, config["paths"]["granite_results"])
    files = {
        "task_a_contract": prospectivity / "08_Coupling_Bridge" / "taskA_bridge_contract.json",
        "task_a_bridge": prospectivity / "08_Coupling_Bridge" / "taskA_oof_bridge_one_row_per_Record_ID.csv",
        "task_a_locked_features": prospectivity / "08_Coupling_Bridge" / "taskA_locked_bridge_features_v1.csv",
        "task_a_performance_by_repeat": prospectivity / "05_Model_Results" / "model_performance_by_repeat.csv",
        "task_a_performance_ci": prospectivity / "05_Model_Results" / "model_performance_mean_ci.csv",
        "task_b_contract": granite / "08_Robustness" / "taskB_bridge_contract_v6.json",
        "task_b_readiness": granite / "08_Robustness" / "taskB_interpretation_readiness_v6.json",
        "task_b_predictions": granite / "08_Robustness" / "taskB_canonical_oof_record_predictions.csv",
        "task_b_s_shap": granite / "08_Robustness" / "taskB_canonical_oof_S_shap_with_ids_v6.csv",
        "task_b_feature_values": granite / "04_SHAP" / "granite_type_oof_imputed_feature_values.csv",
        "task_b_direction": granite / "08_Robustness" / "taskB_bridge_feature_direction_stability.csv",
        "task_b_repeated_metrics": granite / "08_Robustness" / "taskB_repeated_oof_metrics.csv",
        "task_b_classwise_metrics": granite / "08_Robustness" / "taskB_classwise_metrics_with_group_bootstrap_ci.csv",
        "task_b_rank_stability": granite / "08_Robustness" / "taskB_shap_rank_stability_by_class.csv",
        "task_b_calibration": granite / "08_Robustness" / "taskB_calibration_metrics.csv",
        "task_b_leave_one_source": granite / "08_Robustness" / "taskB_leave_one_source_block_shap_sensitivity.csv",
        "task_b_cluster_sensitivity": granite / "08_Robustness" / "taskB_correlation_cluster_shap_sensitivity.csv",
        "task_b_class_shap_summary": granite / "04_SHAP" / "granite_type_class_specific_shap_summary.csv",
        "task_b_heldout_confusion": granite / "03_Final_Model" / "heldout_confusion_matrix.csv",
    }
    missing = [f"{name}: {path}" for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required cross-task inputs are missing:\n" + "\n".join(missing))
    return {name: path.resolve() for name, path in files.items()}

def audit_inputs(files: Mapping[str, Path], paths: OutputPaths) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, path in files.items():
        rows.append(
            {
                "input_key": name,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "modified_time": path.stat().st_mtime,
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(paths.audit / "input_file_audit.csv", index=False, encoding="utf-8-sig")
    return audit


def normalize_feature_name(value: str) -> str:
    text = str(value).strip()
    for suffix in (" (ppm)", " (wt.%)"):
        text = text.replace(suffix, "")
    return text.replace("₂", "2").replace("₅", "5")


def determine_bridge_features(
    config: Mapping[str, Any],
    contract_a: Mapping[str, Any],
    contract_b: Mapping[str, Any],
    locked_a: pd.DataFrame,
) -> list[str]:
    requested = [normalize_feature_name(x) for x in config["analysis"]["bridge_features"]]
    primary_b = {
        normalize_feature_name(x) for x in contract_b.get("primary_bridge_features", [])
    }
    reproduced_a = {
        normalize_feature_name(row["feature"])
        for row in contract_a.get("locked_bridge_feature_status", [])
        if row.get("status") == "reproduced_under_optuna_revision"
    }
    if "status" in locked_a.columns:
        reproduced_a |= {
            normalize_feature_name(x)
            for x in locked_a.loc[
                locked_a["status"].eq("reproduced_under_optuna_revision"), "feature"
            ]
        }
    bridge = [
        feature
        for feature in requested
        if feature in primary_b and feature in reproduced_a and feature in FEATURE_SPECS
    ]
    if bridge != requested:
        missing = [feature for feature in requested if feature not in bridge]
        raise ValueError(
            "The requested primary bridge is not fully reproduced in both tasks. "
            f"Unavailable features: {missing}"
        )
    return bridge


def create_connected_dependency_blocks(
    frame: pd.DataFrame,
    a_block: str,
    b_block: str,
) -> pd.Series:
    uf = UnionFind()
    pairs = frame[[a_block, b_block]].drop_duplicates()
    for row in pairs.itertuples(index=False, name=None):
        left = f"A::{row[0]}"
        right = f"B::{row[1]}"
        uf.union(left, right)

    roots = sorted({uf.find(f"A::{value}") for value in frame[a_block].astype(str)})
    labels = {root: f"CCB{index:04d}" for index, root in enumerate(roots, start=1)}
    return frame[a_block].astype(str).map(lambda value: labels[uf.find(f"A::{value}")])


def build_matched_record_cohort(
    files: Mapping[str, Path],
    bridge_features: Sequence[str],
    paths: OutputPaths,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = pd.read_csv(files["task_a_bridge"], low_memory=False)
    b = pd.read_csv(files["task_b_predictions"], low_memory=False)
    s = pd.read_csv(files["task_b_s_shap"], low_memory=False)
    values = pd.read_csv(files["task_b_feature_values"], low_memory=False)

    a_required = [
        "Record ID",
        "Geological Group ID",
        "CV Block ID",
        "Reference ID",
        "target",
        "PROSPECTIVITY_MODEL_SCORE_MEDIAN",
        "PROSPECTIVITY_MODEL_SCORE_SD",
    ]
    b_required = [
        "Record ID",
        "Geological Group ID",
        "Reference-connected block",
        "valid_oof",
        "reported_type",
        "raw_P_S",
        "calibrated_P_S",
    ]
    require_columns(a, a_required, "Task A bridge")
    require_columns(b, b_required, "Task B canonical predictions")
    require_columns(s, ["Record ID"] + [f"SHAP::{x}" for x in bridge_features], "Task B S SHAP")
    require_columns(
        values,
        ["Record ID"] + [f"imputed::{x}" for x in bridge_features],
        "Task B OOF feature values",
    )

    b = b.loc[as_bool(b["valid_oof"])].copy()
    if "valid_oof" in s.columns:
        s = s.loc[as_bool(s["valid_oof"])].copy()
    if "valid_oof" in values.columns:
        values = values.loc[as_bool(values["valid_oof"])].copy()

    require_unique(a, "Record ID", "Task A bridge")
    require_unique(b, "Record ID", "Task B canonical valid OOF predictions")
    require_unique(s, "Record ID", "Task B canonical S SHAP")
    require_unique(values, "Record ID", "Task B OOF feature values")

    a_keep = a_required.copy()
    for feature in bridge_features:
        spec = FEATURE_SPECS[feature]
        a_keep.extend(
            [
                f"PROSPECTIVITY_SHAP_MEAN::{spec['task_a']}",
                f"PROSPECTIVITY_SHAP_SD::{spec['task_a']}",
                f"PROSPECTIVITY_OBSERVED_MEDIAN::{spec['task_a']}",
                f"PROSPECTIVITY_COMPLETED_MEAN::{spec['task_a']}",
                f"PROSPECTIVITY_IMPUTED_FRACTION::{spec['task_a']}",
            ]
        )
    require_columns(a, a_keep, "Task A bridge")
    a = a[a_keep].copy()
    a = a.rename(
        columns={
            "Geological Group ID": "TASK_A_GROUP_ID",
            "CV Block ID": "TASK_A_CV_BLOCK",
            "Reference ID": "TASK_A_REFERENCE_ID",
            "target": "MINERALIZATION_LABEL",
            "PROSPECTIVITY_MODEL_SCORE_MEDIAN": "P_U",
            "PROSPECTIVITY_MODEL_SCORE_SD": "P_U_SD",
        }
    )

    b = b[
        [
            "Record ID",
            "Geological Group ID",
            "Reference-connected block",
            "reported_type",
            "true_code",
            "raw_P_S",
            "calibrated_P_S",
        ]
    ].rename(
        columns={
            "Geological Group ID": "TASK_B_GROUP_ID",
            "Reference-connected block": "TASK_B_SOURCE_BLOCK",
            "reported_type": "REPORTED_GRANITE_TYPE",
            "raw_P_S": "P_S_RAW",
            "calibrated_P_S": "P_S_CALIBRATED",
        }
    )

    s_keep = ["Record ID"] + [f"SHAP::{x}" for x in bridge_features]
    s = s[s_keep].rename(
        columns={f"SHAP::{x}": f"TASK_B_S_SHAP::{x}" for x in bridge_features}
    )
    v_keep = ["Record ID"] + [f"imputed::{x}" for x in bridge_features]
    values = values[v_keep].rename(
        columns={f"imputed::{x}": f"TASK_B_VALUE::{x}" for x in bridge_features}
    )

    matched = (
        a.merge(b, on="Record ID", how="inner", validate="one_to_one")
        .merge(s, on="Record ID", how="inner", validate="one_to_one")
        .merge(values, on="Record ID", how="inner", validate="one_to_one")
    )
    if matched.empty:
        raise ValueError("No valid OOF records matched by Record ID.")

    group_mismatch = matched["TASK_A_GROUP_ID"].astype(str).ne(
        matched["TASK_B_GROUP_ID"].astype(str)
    )
    if group_mismatch.any():
        examples = matched.loc[
            group_mismatch, ["Record ID", "TASK_A_GROUP_ID", "TASK_B_GROUP_ID"]
        ].head(10)
        examples.to_csv(
            paths.audit / "group_id_mismatch_examples.csv",
            index=False,
            encoding="utf-8-sig",
        )
        raise ValueError(
            f"{int(group_mismatch.sum())} matched records have inconsistent geological group IDs."
        )

    numeric_columns = [
        "MINERALIZATION_LABEL",
        "P_U",
        "P_U_SD",
        "P_S_RAW",
        "P_S_CALIBRATED",
    ]
    for feature in bridge_features:
        spec = FEATURE_SPECS[feature]
        rename_map = {
            f"PROSPECTIVITY_SHAP_MEAN::{spec['task_a']}": f"TASK_A_SHAP::{feature}",
            f"PROSPECTIVITY_SHAP_SD::{spec['task_a']}": f"TASK_A_SHAP_SD::{feature}",
            f"PROSPECTIVITY_OBSERVED_MEDIAN::{spec['task_a']}": f"TASK_A_OBSERVED::{feature}",
            f"PROSPECTIVITY_COMPLETED_MEAN::{spec['task_a']}": f"TASK_A_COMPLETED::{feature}",
            f"PROSPECTIVITY_IMPUTED_FRACTION::{spec['task_a']}": f"TASK_A_IMPUTED_FRACTION::{feature}",
        }
        matched = matched.rename(columns=rename_map)
        numeric_columns.extend(rename_map.values())
        numeric_columns.extend([f"TASK_B_S_SHAP::{feature}", f"TASK_B_VALUE::{feature}"])
    matched = numeric_frame(matched, numeric_columns)

    matched["COUPLING_DEPENDENCY_BLOCK"] = create_connected_dependency_blocks(
        matched, "TASK_A_CV_BLOCK", "TASK_B_SOURCE_BLOCK"
    )
    matched["GEOLOGICAL_GROUP_ID"] = matched["TASK_A_GROUP_ID"].astype(str)

    for feature in bridge_features:
        observed = matched[f"TASK_A_OBSERVED::{feature}"]
        completed = matched[f"TASK_A_COMPLETED::{feature}"]
        b_value = matched[f"TASK_B_VALUE::{feature}"]
        matched[f"COMMON_VALUE::{feature}"] = observed.combine_first(b_value).combine_first(
            completed
        )
        both = observed.notna() & b_value.notna()
        matched[f"A_B_VALUE_ABS_DIFF::{feature}"] = np.where(
            both, np.abs(observed - b_value), np.nan
        )

    cohort_flow = pd.DataFrame(
        [
            {
                "stage": "Task A bridge rows",
                "records": len(a),
                "geological_groups": a["TASK_A_GROUP_ID"].nunique(),
            },
            {
                "stage": "Task B canonical valid OOF rows",
                "records": len(b),
                "geological_groups": b["TASK_B_GROUP_ID"].nunique(),
            },
            {
                "stage": "Exact Record ID intersection with S SHAP and feature values",
                "records": len(matched),
                "geological_groups": matched["GEOLOGICAL_GROUP_ID"].nunique(),
            },
        ]
    )
    cohort_flow["dependency_blocks"] = [
        a["TASK_A_CV_BLOCK"].nunique(),
        b["TASK_B_SOURCE_BLOCK"].nunique(),
        matched["COUPLING_DEPENDENCY_BLOCK"].nunique(),
    ]
    cohort_flow.to_csv(
        paths.audit / "coupling_cohort_flow.csv", index=False, encoding="utf-8-sig"
    )

    agreement_rows = []
    for feature in bridge_features:
        diff = matched[f"A_B_VALUE_ABS_DIFF::{feature}"]
        agreement_rows.append(
            {
                "feature": feature,
                "records_with_values_in_both_tasks": int(diff.notna().sum()),
                "median_absolute_difference": float(diff.median(skipna=True)),
                "maximum_absolute_difference": float(diff.max(skipna=True)),
            }
        )
    value_agreement = pd.DataFrame(agreement_rows)
    value_agreement.to_csv(
        paths.audit / "cross_task_feature_value_agreement.csv",
        index=False,
        encoding="utf-8-sig",
    )

    matched.to_csv(
        paths.tables / "Table_record_level_matched_oof_cohort.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return matched, cohort_flow


def _constant_or_raise(series: pd.Series, label: str) -> Any:
    values = series.dropna().unique()
    if len(values) > 1:
        raise ValueError(f"{label} is not constant within a geological group: {values[:5]}")
    return values[0] if len(values) else np.nan


def l1_signed_share(matrix: np.ndarray) -> np.ndarray:
    denom = np.nansum(np.abs(matrix), axis=1, keepdims=True)
    out = np.full_like(matrix, np.nan, dtype=float)
    valid = np.isfinite(denom[:, 0]) & (denom[:, 0] > 0)
    out[valid] = matrix[valid] / denom[valid]
    return out


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.nansum(left * right, axis=1)
    denom = np.sqrt(np.nansum(left**2, axis=1)) * np.sqrt(
        np.nansum(right**2, axis=1)
    )
    out = np.full(len(left), np.nan)
    valid = np.isfinite(denom) & (denom > 0)
    out[valid] = numerator[valid] / denom[valid]
    return out


def aggregate_to_geological_groups(
    matched: pd.DataFrame,
    bridge_features: Sequence[str],
    paths: OutputPaths,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_id, group in matched.groupby("GEOLOGICAL_GROUP_ID", sort=True):
        row: dict[str, Any] = {
            "GEOLOGICAL_GROUP_ID": group_id,
            "COUPLING_DEPENDENCY_BLOCK": _constant_or_raise(
                group["COUPLING_DEPENDENCY_BLOCK"],
                f"dependency block for {group_id}",
            ),
            "MINERALIZATION_LABEL": _constant_or_raise(
                group["MINERALIZATION_LABEL"], f"label for {group_id}"
            ),
            "N_RECORDS": len(group),
            "P_U": group["P_U"].median(),
            "P_U_SD_MEDIAN": group["P_U_SD"].median(),
            "P_S_RAW": group["P_S_RAW"].median(),
            "P_S_CALIBRATED": group["P_S_CALIBRATED"].median(),
            "TASK_A_REFERENCE_COUNT": group["TASK_A_REFERENCE_ID"].nunique(),
            "TASK_B_REPORTED_TYPE_MODE": (
                group["REPORTED_GRANITE_TYPE"].mode().iloc[0]
                if not group["REPORTED_GRANITE_TYPE"].mode().empty
                else np.nan
            ),
        }
        for feature in bridge_features:
            for prefix in (
                "TASK_A_SHAP",
                "TASK_B_S_SHAP",
                "COMMON_VALUE",
                "TASK_A_IMPUTED_FRACTION",
            ):
                row[f"{prefix}::{feature}"] = group[f"{prefix}::{feature}"].median()
        rows.append(row)
    groups = pd.DataFrame(rows)

    a_matrix = groups[[f"TASK_A_SHAP::{x}" for x in bridge_features]].to_numpy(
        dtype=float
    )
    b_matrix = groups[[f"TASK_B_S_SHAP::{x}" for x in bridge_features]].to_numpy(
        dtype=float
    )
    a_share = l1_signed_share(a_matrix)
    b_share = l1_signed_share(b_matrix)
    groups["SHAP_VECTOR_COSINE"] = row_cosine(a_share, b_share)
    for index, feature in enumerate(bridge_features):
        groups[f"TASK_A_SIGNED_SHARE::{feature}"] = a_share[:, index]
        groups[f"TASK_B_S_SIGNED_SHARE::{feature}"] = b_share[:, index]

    groups.to_csv(
        paths.tables / "Table_group_level_coupling.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return groups


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return np.nan
    result = spearmanr(frame["x"], frame["y"])
    return float(result.statistic)


def component_summary(
    frame: pd.DataFrame, x: str, y: str, block: str
) -> pd.DataFrame:
    data = frame[[block, x, y]].dropna()
    return (
        data.groupby(block, as_index=False)
        .agg({x: "median", y: "median"})
        .dropna()
        .reset_index(drop=True)
    )


def block_spearman_inference(
    frame: pd.DataFrame,
    x: str,
    y: str,
    block: str,
    bootstrap_replicates: int,
    permutation_replicates: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    component = component_summary(frame, x, y, block)
    if len(component) < 8:
        raise ValueError(
            f"Too few dependency components for inference on {x} and {y}: {len(component)}"
        )
    observed = safe_spearman(component[x], component[y])
    rng = np.random.default_rng(seed)

    boot_values: list[float] = []
    for _ in range(bootstrap_replicates):
        indices = rng.integers(0, len(component), size=len(component))
        rho = safe_spearman(component.iloc[indices][x], component.iloc[indices][y])
        if np.isfinite(rho):
            boot_values.append(rho)
    if not boot_values:
        ci_low = ci_high = np.nan
    else:
        ci_low, ci_high = np.quantile(boot_values, [0.025, 0.975])

    null_values: list[float] = []
    y_values = component[y].to_numpy()
    for _ in range(permutation_replicates):
        permuted = rng.permutation(y_values)
        rho = safe_spearman(component[x], permuted)
        if np.isfinite(rho):
            null_values.append(rho)
    if null_values and np.isfinite(observed):
        p_value = (1 + np.sum(np.abs(null_values) >= abs(observed))) / (
            1 + len(null_values)
        )
    else:
        p_value = np.nan

    result = {
        "x": x,
        "y": y,
        "estimate": observed,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "permutation_p": float(p_value),
        "inference_unit": "connected Task-A/Task-B dependency component",
        "dependency_components": len(component),
        "bootstrap_replicates_valid": len(boot_values),
        "permutation_replicates_valid": len(null_values),
    }
    boot = pd.DataFrame({"bootstrap_rho": boot_values})
    null = pd.DataFrame({"permutation_rho": null_values})
    return result, boot, null


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    result = np.full_like(values, np.nan)
    valid = np.isfinite(values)
    if not valid.any():
        return result
    valid_values = values[valid]
    order = np.argsort(valid_values)
    ranked = valid_values[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    restored = np.empty(n)
    restored[order] = adjusted
    result[valid] = restored
    return result


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    if not np.isfinite(denom) or denom == 0:
        return np.nan
    return float(np.dot(left, right) / denom)


def feature_concordance(
    files: Mapping[str, Path],
    groups: pd.DataFrame,
    bridge_features: Sequence[str],
    paths: OutputPaths,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    a = pd.read_csv(files["task_a_locked_features"])
    b = pd.read_csv(files["task_b_direction"])
    a["feature_key"] = a["feature"].map(normalize_feature_name)
    b["feature_key"] = b["feature"].map(normalize_feature_name)
    b = b.loc[b["class"].astype(str).eq("S")].copy()

    require_columns(
        a,
        [
            "feature_key",
            "mean_abs_oof_shap",
            "direction_spearman",
            "direction_block_bootstrap_ci_low",
            "direction_block_bootstrap_ci_high",
            "direction_agreement_fraction",
        ],
        "Task A locked feature table",
    )
    require_columns(
        b,
        [
            "feature_key",
            "global_group_level_spearman",
            "block_bootstrap_ci_low",
            "block_bootstrap_ci_high",
            "partition_direction_sign_consistency",
            "partition_availability",
        ],
        "Task B direction stability",
    )

    a = a.set_index("feature_key")
    b = b.set_index("feature_key")
    rows: list[dict[str, Any]] = []
    b_importance_raw = {
        feature: float(groups[f"TASK_B_S_SHAP::{feature}"].abs().mean())
        for feature in bridge_features
    }
    total_b = sum(b_importance_raw.values())
    # The locked Task A bridge table stores absolute OOF SHAP but, unlike the
    # global SHAP table, does not necessarily contain normalized_importance.
    # Normalize the six locked features locally so the cross-task comparison
    # remains within-task and never compares raw SHAP magnitudes.
    importance_column = (
        "normalized_importance"
        if "normalized_importance" in a.columns
        else "mean_abs_oof_shap"
    )
    a_importance_raw = {
        feature: float(pd.to_numeric(a.loc[feature, importance_column]))
        for feature in bridge_features
    }
    total_a = sum(a_importance_raw.values())

    for feature in bridge_features:
        a_rho = float(pd.to_numeric(a.loc[feature, "direction_spearman"]))
        b_rho = float(pd.to_numeric(b.loc[feature, "global_group_level_spearman"]))
        rows.append(
            {
                "feature": feature,
                "display": FEATURE_SPECS[feature]["display"],
                "task_a_direction_rho": a_rho,
                "task_a_ci_low": float(
                    pd.to_numeric(a.loc[feature, "direction_block_bootstrap_ci_low"])
                ),
                "task_a_ci_high": float(
                    pd.to_numeric(a.loc[feature, "direction_block_bootstrap_ci_high"])
                ),
                "task_a_direction_agreement": float(
                    pd.to_numeric(a.loc[feature, "direction_agreement_fraction"])
                ),
                "task_b_s_direction_rho": b_rho,
                "task_b_s_ci_low": float(
                    pd.to_numeric(b.loc[feature, "block_bootstrap_ci_low"])
                ),
                "task_b_s_ci_high": float(
                    pd.to_numeric(b.loc[feature, "block_bootstrap_ci_high"])
                ),
                "task_b_s_sign_consistency": float(
                    pd.to_numeric(
                        b.loc[feature, "partition_direction_sign_consistency"]
                    )
                ),
                "task_b_s_partition_availability": float(
                    pd.to_numeric(b.loc[feature, "partition_availability"])
                ),
                "task_a_contribution_share": a_importance_raw[feature] / total_a,
                "task_b_s_contribution_share": (
                    b_importance_raw[feature] / total_b if total_b > 0 else np.nan
                ),
                "direction_agreement": int(np.sign(a_rho) == np.sign(b_rho)),
            }
        )
    table = pd.DataFrame(rows)
    table["task_a_rank_within_bridge"] = table[
        "task_a_contribution_share"
    ].rank(ascending=False, method="min")
    table["task_b_s_rank_within_bridge"] = table[
        "task_b_s_contribution_share"
    ].rank(ascending=False, method="min")
    table["task_a_signed_share"] = np.sign(table["task_a_direction_rho"]) * table[
        "task_a_contribution_share"
    ]
    table["task_b_s_signed_share"] = np.sign(table["task_b_s_direction_rho"]) * table[
        "task_b_s_contribution_share"
    ]

    left = table["task_a_signed_share"].to_numpy()
    right = table["task_b_s_signed_share"].to_numpy()
    observed = cosine(left, right)
    null_rows: list[dict[str, Any]] = []
    for index, permutation in enumerate(itertools.permutations(range(len(table))), start=1):
        null_rows.append(
            {
                "permutation_id": index,
                "signed_contribution_cosine": cosine(left, right[list(permutation)]),
            }
        )
    null = pd.DataFrame(null_rows)
    exact_p = (1 + (null["signed_contribution_cosine"].abs() >= abs(observed)).sum()) / (
        1 + len(null)
    )
    summary = {
        "shared_features": len(table),
        "direction_agreements": int(table["direction_agreement"].sum()),
        "direction_agreement_fraction": float(table["direction_agreement"].mean()),
        "observed_signed_contribution_cosine": observed,
        "exact_feature_label_permutation_p": float(exact_p),
        "permutations": len(null),
        "comparison_rule": (
            "Cross-task comparison uses direction and within-task contribution share; "
            "raw SHAP magnitudes are not compared."
        ),
    }

    table.to_csv(
        paths.tables / "Table_shared_feature_concordance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    null.to_csv(
        paths.source_data / "Fig7c_feature_label_permutation_null.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_json(summary, paths.tables / "feature_concordance_summary.json")
    return table, null, summary


def robust_zscore(series: pd.Series) -> pd.Series:
    median = series.median(skipna=True)
    mad = (series - median).abs().median(skipna=True)
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        scale = series.std(skipna=True)
    if not np.isfinite(scale) or scale <= 0:
        return pd.Series(np.nan, index=series.index)
    return (series - median) / scale


def bootstrap_bin_medians(
    frame: pd.DataFrame,
    value_column: str,
    bin_column: str,
    block_column: str,
    bins: Sequence[Any],
    replicates: int,
    seed: int,
) -> dict[Any, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    block_ids = frame[block_column].dropna().unique()
    results: dict[Any, list[float]] = {bin_value: [] for bin_value in bins}
    for _ in range(replicates):
        sampled = rng.choice(block_ids, size=len(block_ids), replace=True)
        pieces = [
            frame.loc[frame[block_column].eq(block_id)] for block_id in sampled
        ]
        sample = pd.concat(pieces, ignore_index=True)
        for bin_value in bins:
            values = sample.loc[sample[bin_column].eq(bin_value), value_column].dropna()
            if len(values):
                results[bin_value].append(float(values.median()))
    intervals: dict[Any, tuple[float, float]] = {}
    for bin_value, values in results.items():
        if values:
            intervals[bin_value] = tuple(np.quantile(values, [0.025, 0.975]))
        else:
            intervals[bin_value] = (np.nan, np.nan)
    return intervals


def dependence_source_data(
    groups: pd.DataFrame,
    features: Sequence[str],
    bins: int,
    bootstrap_replicates: int,
    seed: int,
    paths: OutputPaths,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_index, feature in enumerate(features):
        base = groups[
            [
                "GEOLOGICAL_GROUP_ID",
                "COUPLING_DEPENDENCY_BLOCK",
                f"COMMON_VALUE::{feature}",
                f"TASK_A_SHAP::{feature}",
                f"TASK_B_S_SHAP::{feature}",
            ]
        ].copy()
        base = base.rename(columns={f"COMMON_VALUE::{feature}": "feature_value"})
        base = base.dropna(subset=["feature_value"])
        if base["feature_value"].nunique() < 4:
            continue
        q = min(bins, base["feature_value"].nunique())
        base["value_bin"] = pd.qcut(
            base["feature_value"], q=q, labels=False, duplicates="drop"
        )
        for task_index, (task, column) in enumerate(
            [
                ("Task A prospectivity", f"TASK_A_SHAP::{feature}"),
                ("Task B S-type", f"TASK_B_S_SHAP::{feature}"),
            ]
        ):
            working = base.dropna(subset=[column]).copy()
            working["standardized_shap"] = robust_zscore(working[column])
            bin_values = sorted(working["value_bin"].dropna().unique())
            intervals = bootstrap_bin_medians(
                working,
                "standardized_shap",
                "value_bin",
                "COUPLING_DEPENDENCY_BLOCK",
                bin_values,
                bootstrap_replicates,
                seed + feature_index * 100 + task_index,
            )
            for bin_value in bin_values:
                subset = working.loc[working["value_bin"].eq(bin_value)]
                lo, hi = intervals[bin_value]
                rows.append(
                    {
                        "feature": feature,
                        "task": task,
                        "bin": int(bin_value) + 1,
                        "x_median": subset["feature_value"].median(),
                        "y_median_standardized_shap": subset[
                            "standardized_shap"
                        ].median(),
                        "ci_low": lo,
                        "ci_high": hi,
                        "geological_groups": subset["GEOLOGICAL_GROUP_ID"].nunique(),
                        "dependency_blocks": subset[
                            "COUPLING_DEPENDENCY_BLOCK"
                        ].nunique(),
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(
        paths.source_data / "Fig7d_binned_dependence_source_data.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return result


def association_analysis(
    groups: pd.DataFrame,
    config: Mapping[str, Any],
    paths: OutputPaths,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    analysis = config["analysis"]
    comparisons = [
        ("S support (raw)", "P_S_RAW", "P_U"),
        ("S support (calibrated sensitivity)", "P_S_CALIBRATED", "P_U"),
        ("SHAP-vector correspondence", "SHAP_VECTOR_COSINE", "P_U"),
    ]
    results: list[dict[str, Any]] = []
    distributions: dict[str, pd.DataFrame] = {}
    for index, (label, x, y) in enumerate(comparisons):
        result, boot, null = block_spearman_inference(
            groups,
            x=x,
            y=y,
            block="COUPLING_DEPENDENCY_BLOCK",
            bootstrap_replicates=int(analysis["bootstrap_replicates"]),
            permutation_replicates=int(analysis["permutation_replicates"]),
            seed=int(analysis["random_seed"]) + index * 1000,
        )
        result["analysis"] = label
        results.append(result)
        distributions[f"{x}_bootstrap"] = boot
        distributions[f"{x}_null"] = null
        boot.to_csv(
            paths.source_data / f"association_{x}_bootstrap.csv",
            index=False,
            encoding="utf-8-sig",
        )
        null.to_csv(
            paths.source_data / f"association_{x}_permutation_null.csv",
            index=False,
            encoding="utf-8-sig",
        )
    table = pd.DataFrame(results)
    table["fdr_q"] = benjamini_hochberg(table["permutation_p"])
    table.to_csv(
        paths.tables / "Table_probability_and_attribution_associations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    q = int(analysis["s_probability_quantile_groups"])
    quartile = groups.dropna(subset=["P_S_RAW", "P_U"]).copy()
    quartile["_BIN_CODE"] = pd.qcut(
        quartile["P_S_RAW"], q=q, labels=False, duplicates="drop"
    )
    actual_codes = sorted(quartile["_BIN_CODE"].dropna().astype(int).unique())
    label_map = {code: f"Q{index}" for index, code in enumerate(actual_codes, start=1)}
    quartile["S_SUPPORT_BIN"] = quartile["_BIN_CODE"].map(label_map)
    bin_values = [label_map[code] for code in actual_codes]
    intervals = bootstrap_bin_medians(
        quartile,
        "P_U",
        "S_SUPPORT_BIN",
        "COUPLING_DEPENDENCY_BLOCK",
        bin_values,
        int(analysis["bootstrap_replicates"]),
        int(analysis["random_seed"]) + 4000,
    )
    quartile_rows = []
    for bin_value in bin_values:
        subset = quartile.loc[quartile["S_SUPPORT_BIN"].eq(bin_value)]
        lo, hi = intervals[bin_value]
        quartile_rows.append(
            {
                "s_support_bin": str(bin_value),
                "p_s_min": subset["P_S_RAW"].min(),
                "p_s_max": subset["P_S_RAW"].max(),
                "p_u_median": subset["P_U"].median(),
                "p_u_ci_low": lo,
                "p_u_ci_high": hi,
                "geological_groups": subset["GEOLOGICAL_GROUP_ID"].nunique(),
                "dependency_blocks": subset["COUPLING_DEPENDENCY_BLOCK"].nunique(),
            }
        )
    quartile_table = pd.DataFrame(quartile_rows)
    quartile_table.to_csv(
        paths.source_data / "Fig8b_s_support_quantile_source_data.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return table, distributions, quartile_table


def model_readiness_table(
    contract_a: Mapping[str, Any],
    readiness_b: Mapping[str, Any],
    cohort_flow: pd.DataFrame,
    feature_summary: Mapping[str, Any],
    paths: OutputPaths,
) -> pd.DataFrame:
    b_metrics = readiness_b["repeated_development_oof_metrics"]
    rows = [
        {
            "module": "Task A uranium prospectivity",
            "evidence_layer": "grouped OOF",
            "readiness": contract_a.get("bridge_status"),
            "coupling_use": contract_a.get("coupling_use"),
            "key_metric": "selected model",
            "value": contract_a.get("selected_model"),
            "claim_boundary": "exploratory attribution coupling; no causal claim",
        },
        {
            "module": "Task B complete I/A/S classification",
            "evidence_layer": readiness_b.get("gating_evidence_layer"),
            "readiness": readiness_b.get("readiness_level"),
            "coupling_use": "not eligible for formal full three-class linkage",
            "key_metric": "macro-F1",
            "value": b_metrics.get("macro_f1"),
            "claim_boundary": "performance and diagnostic attribution only",
        },
        {
            "module": "Task B S-type branch",
            "evidence_layer": "class-specific repeated OOF and SHAP stability",
            "readiness": "relative branch-level reproducibility",
            "coupling_use": "hypothesis-generating restricted analysis",
            "key_metric": "S-type F1",
            "value": b_metrics.get("f1_S"),
            "claim_boundary": "does not override overall Task B readiness",
        },
        {
            "module": "Matched Part 4 cohort",
            "evidence_layer": "exact Record ID intersection",
            "readiness": "audited at runtime",
            "coupling_use": "group/source-block inference",
            "key_metric": "matched records",
            "value": int(cohort_flow.iloc[-1]["records"]),
            "claim_boundary": (
                f"{int(cohort_flow.iloc[-1]['geological_groups'])} geological groups; "
                f"{int(cohort_flow.iloc[-1]['dependency_blocks'])} connected blocks"
            ),
        },
        {
            "module": "Shared geochemical attribution",
            "evidence_layer": "six locked features",
            "readiness": "diagnostic",
            "coupling_use": "direction/rank/contribution-share comparison",
            "key_metric": "direction agreements",
            "value": (
                f"{feature_summary['direction_agreements']}/"
                f"{feature_summary['shared_features']}"
            ),
            "claim_boundary": "raw SHAP magnitudes are not compared",
        },
    ]
    table = pd.DataFrame(rows)
    table.to_csv(
        paths.tables / "Table_model_readiness_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return table


def decision_gate(
    contract_a: Mapping[str, Any],
    readiness_b: Mapping[str, Any],
    rank_stability: pd.DataFrame,
    feature_summary: Mapping[str, Any],
    associations: pd.DataFrame,
    groups: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    thresholds = config["gates"]
    metrics_b = readiness_b["repeated_development_oof_metrics"]
    s_row = rank_stability.loc[rank_stability["class"].astype(str).eq("S")]
    if len(s_row) != 1:
        raise ValueError("Expected one S row in Task B SHAP rank stability table.")
    s_row = s_row.iloc[0]

    checks = {
        "task_a_exploratory_bridge_eligible": contract_a.get("bridge_status")
        == "eligible_for_exploratory_part4_coupling",
        "task_b_overall_formal_ready": readiness_b.get("readiness_level") == "ready",
        "s_f1_pass": float(metrics_b["f1_S"]) >= float(thresholds["minimum_s_f1"]),
        "s_rank_stability_pass": float(s_row["median_rank_spearman"])
        >= float(thresholds["minimum_s_rank_spearman"]),
        "s_top10_stability_pass": float(s_row["mean_top10_jaccard"])
        >= float(thresholds["minimum_s_top10_jaccard"]),
        "shared_direction_pass": float(feature_summary["direction_agreement_fraction"])
        >= float(thresholds["minimum_direction_agreement_fraction"]),
        "minimum_dependency_blocks_pass": groups[
            "COUPLING_DEPENDENCY_BLOCK"
        ].nunique()
        >= int(thresholds["minimum_dependency_blocks"]),
    }
    checks["restricted_s_diagnostic_eligible"] = all(
        checks[key]
        for key in (
            "task_a_exploratory_bridge_eligible",
            "s_f1_pass",
            "s_rank_stability_pass",
            "s_top10_stability_pass",
            "shared_direction_pass",
            "minimum_dependency_blocks_pass",
        )
    )

    raw = associations.loc[associations["analysis"].eq("S support (raw)")].iloc[0]
    calibrated = associations.loc[
        associations["analysis"].eq("S support (calibrated sensitivity)")
    ].iloc[0]
    checks["raw_probability_association_pass"] = bool(
        raw["estimate"] > 0
        and raw["ci_low"] > 0
        and raw["fdr_q"] <= float(thresholds["fdr_alpha"])
    )
    checks["calibrated_sensitivity_direction_agrees"] = bool(
        np.sign(raw["estimate"]) == np.sign(calibrated["estimate"])
    )
    checks["figure8_main_text_eligible"] = bool(
        checks["restricted_s_diagnostic_eligible"]
        and checks["raw_probability_association_pass"]
        and checks["calibrated_sensitivity_direction_agrees"]
    )
    checks["formal_full_three_class_coupling_eligible"] = bool(
        checks["task_a_exploratory_bridge_eligible"]
        and checks["task_b_overall_formal_ready"]
    )

    decision = {
        "gate_checks": checks,
        "task_b_readiness_level": readiness_b.get("readiness_level"),
        "figure8_placement": (
            "main_text"
            if checks["figure8_main_text_eligible"]
            else "supplementary"
        ),
        "permitted_primary_claim": (
            "Restricted, hypothesis-generating S-type correspondence."
            if checks["restricted_s_diagnostic_eligible"]
            else "No cross-task interpretive claim; report diagnostic outputs only."
        ),
        "prohibited_claims": [
            "Task B validates Task A.",
            "I/A weak coupling proves geological irrelevance.",
            "SHAP establishes petrogenetic or metallogenic causality.",
            "Coupling improves prediction without nested prospective comparison.",
        ],
    }
    return decision


def apply_publication_style(font_size: float = 7.5) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": font_size,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.03) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def export_figure(fig: plt.Figure, base: Path, dpi: int = 600) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for suffix, kwargs in (
        (".svg", {}),
        (".pdf", {}),
        (".tiff", {"dpi": dpi}),
        (".png", {"dpi": 300}),
    ):
        path = base.with_suffix(suffix)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        layout_failures = [
            str(item.message)
            for item in caught
            if "constrained_layout not applied" in str(item.message)
        ]
        if layout_failures:
            plt.close(fig)
            raise RuntimeError(
                f"Layout failed while exporting {base.name}: "
                f"{layout_failures[0]}"
            )
        saved.append(str(path))
    plt.close(fig)
    return saved


def plot_figure6(
    files: Mapping[str, Path],
    paths: OutputPaths,
    config: Mapping[str, Any],
) -> list[str]:
    apply_publication_style()
    repeated = numeric_frame(
        pd.read_csv(files["task_b_repeated_metrics"]),
        [
            "macro_f1",
            "balanced_accuracy",
            "macro_ovr_auc",
            "f1_I",
            "f1_A",
            "f1_S",
        ],
    )
    classwise = numeric_frame(
        pd.read_csv(files["task_b_classwise_metrics"]),
        ["estimate", "ci_low", "ci_high"],
    )
    stability = numeric_frame(
        pd.read_csv(files["task_b_rank_stability"]),
        [
            "median_rank_spearman",
            "rank_spearman_q025",
            "rank_spearman_q975",
            "mean_top10_jaccard",
        ],
    )
    calibration = numeric_frame(
        pd.read_csv(files["task_b_calibration"]), ["value", "n"]
    )

    repeated_source = repeated.loc[
        repeated["scope"].eq("pooled_repeat")
        & repeated["probability_version"].eq("raw")
    ].copy()
    repeated_source.to_csv(
        paths.source_data / "Fig6a_repeated_oof_source_data.csv",
        index=False,
        encoding="utf-8-sig",
    )
    classwise.to_csv(
        paths.source_data / "Fig6b_classwise_metrics_source_data.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stability.to_csv(
        paths.source_data / "Fig6c_shap_stability_source_data.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig = plt.figure(figsize=(7.25, 6.2))
    grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.38)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    metrics = [
        ("macro_f1", "Macro-F1"),
        ("balanced_accuracy", "Balanced\naccuracy"),
        ("macro_ovr_auc", "Macro-AUC"),
    ]
    x_positions = np.arange(len(metrics))
    rng = np.random.default_rng(int(config["analysis"]["random_seed"]))
    for index, (column, label) in enumerate(metrics):
        values = repeated_source[column].dropna().to_numpy()
        jitter = rng.normal(0, 0.035, size=len(values))
        ax_a.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=24,
            color=PALETTE["task_b_s"],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9,
            zorder=3,
        )
        ax_a.plot(
            [index - 0.15, index + 0.15],
            [np.median(values), np.median(values)],
            color=PALETTE["neutral_dark"],
            lw=1.6,
            zorder=4,
        )
    ax_a.set_xticks(x_positions, [label for _, label in metrics])
    ax_a.set_ylabel("Repeated grouped OOF score")
    ax_a.set_ylim(0, 1)
    ax_a.text(
        0.02,
        0.04,
        f"n = {len(repeated_source)} repeats",
        transform=ax_a.transAxes,
        color=PALETTE["neutral_mid"],
    )
    add_panel_label(ax_a, "a")

    f1 = classwise.loc[
        classwise["metric"].isin(["f1_I", "f1_A", "f1_S"])
        & classwise["probability_version"].eq("raw")
    ].copy()
    order = ["I", "A", "S"]
    f1["class"] = pd.Categorical(f1["class"], categories=order, ordered=True)
    f1 = f1.sort_values("class")
    colors = [PALETTE["task_b_i"], PALETTE["task_b_a"], PALETTE["task_b_s"]]
    y = np.arange(len(f1))[::-1]
    for yi, (_, row), color in zip(y, f1.iterrows(), colors):
        ax_b.plot([row["ci_low"], row["ci_high"]], [yi, yi], color=color, lw=1.6)
        ax_b.scatter(row["estimate"], yi, color=color, s=36, zorder=3)
    supports = {"I": 15, "A": 9, "S": 30}
    ax_b.set_yticks(
        y, [f"{row['class']} (n={supports[str(row['class'])]})" for _, row in f1.iterrows()]
    )
    ax_b.set_xlabel("Class-specific F1 (block-bootstrap 95% CI)")
    ax_b.set_xlim(0, 1)
    add_panel_label(ax_b, "b")

    stability = stability.set_index("class").loc[order].reset_index()
    y = np.arange(len(order))[::-1]
    for yi, (_, row), color in zip(y, stability.iterrows(), colors):
        ax_c.plot(
            [row["rank_spearman_q025"], row["rank_spearman_q975"]],
            [yi + 0.09, yi + 0.09],
            color=color,
            lw=1.3,
        )
        ax_c.scatter(
            row["median_rank_spearman"],
            yi + 0.09,
            color=color,
            marker="o",
            s=28,
            label="Rank Spearman" if yi == y[0] else None,
        )
        ax_c.scatter(
            row["mean_top10_jaccard"],
            yi - 0.09,
            facecolor="white",
            edgecolor=color,
            marker="s",
            s=28,
            label="Mean top-10 Jaccard" if yi == y[0] else None,
        )
    gate = config["gates"]
    ax_c.axvline(
        float(gate["minimum_s_rank_spearman"]),
        color=PALETTE["neutral_mid"],
        ls="--",
        lw=0.9,
    )
    ax_c.set_yticks(y, order)
    ax_c.set_xlim(0, 1)
    ax_c.set_xlabel("Attribution stability")
    ax_c.legend(loc="lower right", fontsize=6.5)
    add_panel_label(ax_c, "c")

    cal = calibration.loc[
        calibration["evaluation_unit"].eq(
            "geological_group_by_reported_type_stratum"
        )
        & (
            (
                calibration["class"].eq("multiclass")
                & calibration["metric"].isin(["log_loss", "brier_score"])
            )
            | (
                calibration["class"].eq("top_label")
                & calibration["metric"].eq("ece")
            )
        )
    ].copy()
    metric_order = ["log_loss", "brier_score", "ece"]
    labels = ["Log loss", "Brier score", "Top-label ECE"]
    width = 0.34
    for offset, probability_version, color, hatch in (
        (-width / 2, "raw", PALETTE["task_a"], ""),
        (width / 2, "calibrated", PALETTE["neutral_light"], "//"),
    ):
        values = [
            cal.loc[
                cal["probability_version"].eq(probability_version)
                & cal["metric"].eq(metric),
                "value",
            ].iloc[0]
            for metric in metric_order
        ]
        bars = ax_d.bar(
            np.arange(3) + offset,
            values,
            width=width,
            color=color,
            edgecolor=PALETTE["neutral_dark"],
            linewidth=0.6,
            hatch=hatch,
            label=probability_version.capitalize(),
        )
        for bar, value in zip(bars, values):
            ax_d.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.018,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=6,
            )
    ax_d.set_xticks(np.arange(3), labels, rotation=15, ha="right")
    ax_d.set_ylabel("Calibration/error score (lower is better)")
    ax_d.set_ylim(0, max(1.0, cal["value"].max() * 1.18))
    ax_d.legend(loc="upper right")
    add_panel_label(ax_d, "d")
    calibration.to_csv(
        paths.source_data / "Fig6d_calibration_source_data.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig.subplots_adjust(left=0.11, right=0.98, top=0.96, bottom=0.10)
    return export_figure(
        fig,
        paths.main_figures
        / "Fig6_granite_classification_performance_and_attribution_stability",
    )


def plot_figure7(
    concordance: pd.DataFrame,
    permutation_null: pd.DataFrame,
    feature_summary: Mapping[str, Any],
    dependence: pd.DataFrame,
    paths: OutputPaths,
) -> list[str]:
    apply_publication_style()
    order = concordance.sort_values(
        "task_a_contribution_share", ascending=False
    )["feature"].tolist()
    table = concordance.set_index("feature").loc[order].reset_index()
    table.to_csv(
        paths.source_data / "Fig7ab_shared_feature_source_data.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig = plt.figure(figsize=(7.25, 7.0))
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.05, 1.0],
        width_ratios=[1.35, 0.95, 1.0],
        hspace=0.48,
        wspace=0.48,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])
    dep_grid = grid[1, :].subgridspec(1, 3, wspace=0.38)
    dep_axes = [fig.add_subplot(dep_grid[0, index]) for index in range(3)]

    y = np.arange(len(table))[::-1]
    for yi, (_, row) in zip(y, table.iterrows()):
        ax_a.plot(
            [row["task_a_ci_low"], row["task_a_ci_high"]],
            [yi + 0.10, yi + 0.10],
            color=PALETTE["task_a"],
            lw=1.4,
        )
        ax_a.scatter(
            row["task_a_direction_rho"],
            yi + 0.10,
            color=PALETTE["task_a"],
            marker="o",
            s=28,
            zorder=3,
        )
        conflict = not bool(row["direction_agreement"])
        ax_a.plot(
            [row["task_b_s_ci_low"], row["task_b_s_ci_high"]],
            [yi - 0.10, yi - 0.10],
            color=PALETTE["conflict"] if conflict else PALETTE["task_b_s"],
            lw=1.4,
        )
        ax_a.scatter(
            row["task_b_s_direction_rho"],
            yi - 0.10,
            facecolor="white" if conflict else PALETTE["task_b_s"],
            edgecolor=PALETTE["conflict"] if conflict else PALETTE["task_b_s"],
            marker="s",
            s=28,
            zorder=3,
        )
    ax_a.axvline(0, color=PALETTE["neutral_mid"], lw=0.8, ls="--")
    ax_a.set_yticks(y, [FEATURE_SPECS[x]["display"] for x in order])
    ax_a.set_xlim(-1.05, 1.05)
    ax_a.set_xlabel(
        r"Feature value–SHAP Spearman $\rho$"
        "\n(block-bootstrap 95% CI)"
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=PALETTE["task_a"],
            marker="o",
            lw=1,
            label="Task A prospectivity",
        ),
        Line2D(
            [0],
            [0],
            color=PALETTE["task_b_s"],
            marker="s",
            lw=1,
            label="Task B S-type",
        ),
        Line2D(
            [0],
            [0],
            color=PALETTE["conflict"],
            marker="s",
            markerfacecolor="white",
            lw=1,
            label="Discordant direction",
        ),
    ]
    ax_a.legend(handles=handles, loc="lower left", fontsize=6.2)
    add_panel_label(ax_a, "a")

    matrix = np.column_stack(
        [
            table["task_a_direction_rho"],
            table["task_b_s_direction_rho"],
            table["direction_agreement"].map({1: 1.0, 0: -1.0}),
            table["task_b_s_sign_consistency"] * 2 - 1,
        ]
    )
    image = ax_b.imshow(
        matrix,
        cmap=mpl.colors.LinearSegmentedColormap.from_list(
            "evidence", [PALETTE["task_a"], "white", PALETTE["conflict"]]
        ),
        vmin=-1,
        vmax=1,
        aspect="auto",
    )
    ax_b.set_yticks(
        np.arange(len(table)), [FEATURE_SPECS[x]["display"] for x in order]
    )
    ax_b.set_xticks(
        np.arange(4),
        [
            "Task A\n" + r"$\rho$",
            "S-type\n" + r"$\rho$",
            "Direction\nmatch",
            "S sign\nstability",
        ],
        rotation=30,
        ha="right",
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text = (
                "Yes"
                if j == 2 and matrix[i, j] > 0
                else "No"
                if j == 2
                else f"{matrix[i, j]:.2f}"
            )
            ax_b.text(j, i, text, ha="center", va="center", fontsize=6.1)
    ax_b.tick_params(length=0)
    ax_b.set_frame_on(False)
    add_panel_label(ax_b, "b")

    null_values = permutation_null["signed_contribution_cosine"].dropna()
    ax_c.hist(
        null_values,
        bins=24,
        color=PALETTE["neutral_light"],
        edgecolor="white",
        linewidth=0.4,
    )
    observed = float(feature_summary["observed_signed_contribution_cosine"])
    ax_c.axvline(observed, color=PALETTE["conflict"], lw=1.8)
    ax_c.text(
        0.03,
        0.95,
        f"Observed = {observed:.2f}\nExact p = "
        f"{feature_summary['exact_feature_label_permutation_p']:.3f}",
        transform=ax_c.transAxes,
        ha="left",
        va="top",
    )
    ax_c.set_xlabel("Signed contribution-share cosine")
    ax_c.set_ylabel("Feature-label permutations")
    add_panel_label(ax_c, "c")

    dep_features = dependence["feature"].drop_duplicates().tolist()[:3]
    for axis_index, (ax, feature) in enumerate(zip(dep_axes, dep_features)):
        subset = dependence.loc[dependence["feature"].eq(feature)]
        for task, color, marker in (
            ("Task A prospectivity", PALETTE["task_a"], "o"),
            ("Task B S-type", PALETTE["task_b_s"], "s"),
        ):
            data = subset.loc[subset["task"].eq(task)].sort_values("bin")
            ax.fill_between(
                data["x_median"].to_numpy(dtype=float),
                data["ci_low"].to_numpy(dtype=float),
                data["ci_high"].to_numpy(dtype=float),
                color=color,
                alpha=0.13,
                linewidth=0,
            )
            ax.plot(
                data["x_median"],
                data["y_median_standardized_shap"],
                color=color,
                marker=marker,
                ms=3.5,
                lw=1.25,
                label=task,
            )
        ax.axhline(0, color=PALETTE["neutral_mid"], lw=0.7, ls="--")
        unit = FEATURE_SPECS[feature]["unit"]
        ax.set_xlabel(f"{FEATURE_SPECS[feature]['display']} ({unit})")
        if axis_index == 0:
            ax.set_ylabel("Within-task standardized SHAP")
        ax.text(
            0.03,
            0.96,
            f"{FEATURE_SPECS[feature]['display']}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
        if axis_index == 2:
            ax.legend(loc="best", fontsize=6.0)
    if dep_axes:
        add_panel_label(dep_axes[0], "d")

    fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.09)
    return export_figure(
        fig,
        paths.main_figures / "Fig7_cross_task_shared_geochemical_attribution",
    )


def plot_figure8(
    groups: pd.DataFrame,
    associations: pd.DataFrame,
    distributions: Mapping[str, pd.DataFrame],
    quartiles: pd.DataFrame,
    decision: Mapping[str, Any],
    paths: OutputPaths,
) -> list[str]:
    apply_publication_style()
    output_dir = (
        paths.main_figures
        if decision["figure8_placement"] == "main_text"
        else paths.supplementary_figures
    )
    groups.to_csv(
        paths.source_data / "Fig8a_group_level_scatter_source_data.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig = plt.figure(figsize=(7.25, 6.2))
    grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.38)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    scatter = groups.dropna(subset=["P_U", "P_S_RAW"]).copy()
    color_values = scatter["SHAP_VECTOR_COSINE"].to_numpy(dtype=float)
    image = ax_a.scatter(
        scatter["P_S_RAW"],
        scatter["P_U"],
        c=color_values,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        s=24 + 7 * np.sqrt(scatter["N_RECORDS"].clip(lower=1)),
        edgecolor="white",
        linewidth=0.45,
        alpha=0.88,
    )
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.set_xlabel("Task B raw OOF S-type support")
    ax_a.set_ylabel("Task A OOF uranium prospectivity")
    raw = associations.loc[associations["analysis"].eq("S support (raw)")].iloc[0]
    ax_a.text(
        0.03,
        0.97,
        r"Block-level $\rho$"
        f" = {raw['estimate']:.2f}\n"
        f"95% CI {raw['ci_low']:.2f} to {raw['ci_high']:.2f}\n"
        f"FDR q = {raw['fdr_q']:.3f}",
        transform=ax_a.transAxes,
        ha="left",
        va="top",
    )
    colorbar = fig.colorbar(image, ax=ax_a, fraction=0.046, pad=0.03)
    colorbar.set_label("Signed SHAP-vector cosine")
    add_panel_label(ax_a, "a")

    x = np.arange(len(quartiles))
    ax_b.errorbar(
        x,
        quartiles["p_u_median"],
        yerr=np.vstack(
            [
                quartiles["p_u_median"] - quartiles["p_u_ci_low"],
                quartiles["p_u_ci_high"] - quartiles["p_u_median"],
            ]
        ),
        fmt="o-",
        color=PALETTE["task_b_s"],
        ecolor=PALETTE["task_b_s"],
        capsize=3,
        lw=1.3,
        ms=4.5,
    )
    ax_b.set_xticks(
        x,
        [
            f"{row['s_support_bin']}\n(n={int(row['geological_groups'])})"
            for _, row in quartiles.iterrows()
        ],
    )
    ax_b.set_ylim(0, 1)
    ax_b.set_xlabel("Frozen S-support quantile")
    ax_b.set_ylabel("Median uranium prospectivity\n(block-bootstrap 95% CI)")
    add_panel_label(ax_b, "b")

    forest = associations.copy()
    forest_order = [
        "S support (raw)",
        "S support (calibrated sensitivity)",
        "SHAP-vector correspondence",
    ]
    forest["analysis"] = pd.Categorical(
        forest["analysis"], categories=forest_order, ordered=True
    )
    forest = forest.sort_values("analysis")
    y = np.arange(len(forest))[::-1]
    colors = [
        PALETTE["task_b_s"],
        PALETTE["neutral_mid"],
        PALETTE["agreement"],
    ]
    for yi, (_, row), color in zip(y, forest.iterrows(), colors):
        ax_c.plot([row["ci_low"], row["ci_high"]], [yi, yi], color=color, lw=1.5)
        ax_c.scatter(row["estimate"], yi, color=color, s=32, zorder=3)
        ax_c.text(
            1.02,
            yi,
            f"q={row['fdr_q']:.3f}",
            transform=ax_c.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=6.2,
        )
    ax_c.axvline(0, color=PALETTE["neutral_mid"], lw=0.8, ls="--")
    ax_c.set_yticks(y, forest_order)
    ax_c.set_xlim(-1, 1)
    ax_c.set_xlabel(r"Block-level Spearman $\rho$ (95% CI)")
    add_panel_label(ax_c, "c")

    null = distributions["P_S_RAW_null"]["permutation_rho"].dropna()
    ax_d.hist(
        null,
        bins=30,
        color=PALETTE["neutral_light"],
        edgecolor="white",
        linewidth=0.4,
    )
    ax_d.axvline(raw["estimate"], color=PALETTE["conflict"], lw=1.8)
    ax_d.set_xlabel(r"Permuted block-level Spearman $\rho$")
    ax_d.set_ylabel("Structure-preserving permutations")
    ax_d.text(
        0.03,
        0.95,
        f"Observed = {raw['estimate']:.2f}\np = {raw['permutation_p']:.3f}",
        transform=ax_d.transAxes,
        ha="left",
        va="top",
    )
    add_panel_label(ax_d, "d")

    fig.subplots_adjust(left=0.12, right=0.96, top=0.96, bottom=0.10)
    return export_figure(
        fig,
        output_dir / "Fig8_conditional_s_type_support_and_uranium_prospectivity",
    )


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str,
    linestyle: str = "-",
    hatch: str | None = None,
    fontsize: float = 6.8,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.012",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.0,
        linestyle=linestyle,
        hatch=hatch,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
    )
    return patch


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    linestyle: str = "-",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.0,
            color=color,
            linestyle=linestyle,
            connectionstyle="arc3,rad=0.0",
        )
    )


def plot_figure9(
    concordance: pd.DataFrame,
    decision: Mapping[str, Any],
    paths: OutputPaths,
) -> list[str]:
    apply_publication_style(font_size=7.2)
    concordance.to_csv(
        paths.source_data / "Fig9_model_evidence_source_data.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fig, ax = plt.subplots(figsize=(7.25, 4.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.955, "Model-derived evidence", fontweight="bold", fontsize=8.5)
    ax.text(
        0.36,
        0.955,
        "Literature-constrained petrogenetic interpretation",
        fontweight="bold",
        fontsize=8.5,
    )
    ax.text(
        0.70,
        0.955,
        "Mobilization and ore deposition\n(not tested by the present models)",
        fontweight="bold",
        fontsize=8.5,
        ha="center",
    )

    matrix_x = 0.02
    matrix_y = 0.29
    matrix_w = 0.29
    row_h = 0.082
    header_h = 0.065
    ax.add_patch(
        FancyBboxPatch(
            (matrix_x, matrix_y),
            matrix_w,
            header_h + row_h * len(concordance),
            boxstyle="round,pad=0.008",
            facecolor="white",
            edgecolor=PALETTE["neutral_dark"],
            linewidth=0.9,
        )
    )
    ax.text(matrix_x + 0.05, matrix_y + row_h * 6 + 0.035, "Feature", ha="center")
    ax.text(matrix_x + 0.15, matrix_y + row_h * 6 + 0.035, "Task A", ha="center")
    ax.text(matrix_x + 0.24, matrix_y + row_h * 6 + 0.035, "S-type", ha="center")

    ordered = concordance.set_index("feature").loc[
        ["Rb", "P2O5", "Nb", "Zr", "Ba", "CaO"]
    ].reset_index()
    for index, row in ordered.iterrows():
        y = matrix_y + row_h * (5 - index) + row_h / 2
        ax.text(matrix_x + 0.05, y, FEATURE_SPECS[row["feature"]]["display"], ha="center")
        a_sign = r"$\uparrow$" if row["task_a_direction_rho"] > 0 else r"$\downarrow$"
        b_sign = r"$\uparrow$" if row["task_b_s_direction_rho"] > 0 else r"$\downarrow$"
        color = (
            PALETTE["agreement"]
            if bool(row["direction_agreement"])
            else PALETTE["conflict"]
        )
        ax.text(matrix_x + 0.15, y, a_sign, ha="center", color=color, fontsize=10)
        ax.text(matrix_x + 0.24, y, b_sign, ha="center", color=color, fontsize=10)
        if index < len(ordered) - 1:
            ax.plot(
                [matrix_x + 0.01, matrix_x + matrix_w - 0.01],
                [y - row_h / 2, y - row_h / 2],
                color=PALETTE["neutral_light"],
                lw=0.5,
            )
    ax.text(
        0.165,
        0.20,
        "Five concordant directions; CaO remains discordant.\n"
        "Task B overall readiness: not ready.",
        ha="center",
        va="top",
        fontsize=6.6,
        color=PALETTE["neutral_dark"],
    )

    source = _box(
        ax,
        (0.37, 0.70),
        0.22,
        0.13,
        "Crustal/sedimentary source contribution\nand peraluminous affinity",
        PALETTE["literature"],
        PALETTE["neutral_dark"],
        hatch="//",
    )
    evolution = _box(
        ax,
        (0.37, 0.47),
        0.22,
        0.13,
        "Fractional crystallization and\nfeldspar/accessory-mineral control",
        PALETTE["literature"],
        PALETTE["neutral_dark"],
        hatch="//",
    )
    fertility = _box(
        ax,
        (0.37, 0.24),
        0.22,
        0.13,
        "U fertility and mineral-scale\navailability for later remobilization",
        PALETTE["literature"],
        PALETTE["neutral_dark"],
        hatch="//",
    )
    _arrow(ax, (0.48, 0.69), (0.48, 0.61), PALETTE["neutral_dark"], "--")
    _arrow(ax, (0.48, 0.46), (0.48, 0.38), PALETTE["neutral_dark"], "--")

    structure = _box(
        ax,
        (0.68, 0.70),
        0.27,
        0.13,
        "Faulting, extension and permeable\nfluid pathways",
        PALETTE["untested"],
        PALETTE["neutral_mid"],
        linestyle="--",
    )
    transport = _box(
        ax,
        (0.68, 0.47),
        0.27,
        0.13,
        "U(IV) oxidation, U(VI) complexing\nand fluid–rock interaction",
        PALETTE["untested"],
        PALETTE["neutral_mid"],
        linestyle="--",
    )
    deposition = _box(
        ax,
        (0.68, 0.24),
        0.27,
        0.13,
        "Reduction, cooling, fluid mixing or\n"
        r"CO$_2$ loss; precipitation and preservation",
        PALETTE["untested"],
        PALETTE["neutral_mid"],
        linestyle="--",
    )
    _arrow(ax, (0.815, 0.69), (0.815, 0.61), PALETTE["neutral_mid"], "--")
    _arrow(ax, (0.815, 0.46), (0.815, 0.38), PALETTE["neutral_mid"], "--")
    _arrow(ax, (0.60, 0.305), (0.67, 0.305), PALETTE["neutral_mid"], "--")
    _arrow(ax, (0.31, 0.54), (0.36, 0.54), PALETTE["neutral_dark"], "-")

    legend_handles = [
        Patch(
            facecolor="white",
            edgecolor=PALETTE["task_a"],
            label="Present OOF model evidence",
        ),
        Patch(
            facecolor=PALETTE["literature"],
            edgecolor=PALETTE["neutral_dark"],
            hatch="//",
            label="Literature-constrained interpretation",
        ),
        Patch(
            facecolor="white",
            edgecolor=PALETTE["neutral_mid"],
            linestyle="--",
            label="Mechanism not tested here",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="white",
            markeredgecolor=PALETTE["conflict"],
            color="none",
            label="Discordant/uncertain evidence",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        fontsize=6.3,
    )
    ax.text(
        0.5,
        0.11,
        "Petrogenetic affinity may condition the expression of uranium-favourable "
        "whole-rock patterns; ore formation additionally requires mobilization, "
        "structural focusing, precipitation and preservation.",
        ha="center",
        va="center",
        fontsize=6.8,
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.90,
        "a",
        fontweight="bold",
        fontsize=9,
    )
    ax.text(
        0.36,
        0.90,
        "b",
        fontweight="bold",
        fontsize=9,
    )
    ax.text(
        0.64,
        0.90,
        "c",
        fontweight="bold",
        fontsize=9,
    )
    return export_figure(
        fig,
        paths.main_figures / "Fig9_source_fertility_mobilization_framework",
    )


# -----------------------------------------------------------------------------
# Figure-audited v2 layouts
# -----------------------------------------------------------------------------

def _class_colors() -> dict[str, str]:
    return {"I": PALETTE["task_b_i"], "A": PALETTE["task_b_a"], "S": PALETTE["task_b_s"]}


def _plot_interval(
    ax: plt.Axes,
    estimate: float,
    low: float,
    high: float,
    y: float,
    color: str,
    marker: str = "o",
) -> None:
    ax.plot([low, high], [y, y], color=color, lw=1.5, solid_capstyle="round")
    ax.scatter(estimate, y, s=34, marker=marker, color=color, edgecolor="white", linewidth=0.45, zorder=3)


def plot_main_figure6_v2(
    files: Mapping[str, Path],
    paths: OutputPaths,
    config: Mapping[str, Any],
) -> list[str]:
    """Task-B readiness: one dominant performance panel plus two compact diagnostics."""
    apply_publication_style(7.4)
    classwise = numeric_frame(pd.read_csv(files["task_b_classwise_metrics"]), ["estimate", "ci_low", "ci_high"])
    repeated = numeric_frame(
        pd.read_csv(files["task_b_repeated_metrics"]),
        ["macro_f1", "balanced_accuracy", "macro_ovr_auc"],
    )
    stability = numeric_frame(
        pd.read_csv(files["task_b_rank_stability"]),
        ["median_rank_spearman", "rank_spearman_q025", "rank_spearman_q975", "mean_top10_jaccard"],
    )
    f1 = classwise.loc[
        classwise["metric"].isin(["f1_I", "f1_A", "f1_S"])
        & classwise["probability_version"].eq("raw")
    ].copy()
    repeat_pooled = repeated.loc[
        repeated["scope"].eq("pooled_repeat")
        & repeated["probability_version"].eq("raw")
    ].copy()
    f1.to_csv(paths.source_data / "Fig6a_classwise_F1.csv", index=False, encoding="utf-8-sig")
    repeat_pooled.to_csv(paths.source_data / "Fig6b_repeated_grouped_OOF.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(paths.source_data / "Fig6c_attribution_stability.csv", index=False, encoding="utf-8-sig")

    fig = plt.figure(figsize=(7.25, 4.45), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.28, 1.0], height_ratios=[1, 1], wspace=0.18, hspace=0.18)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])
    colors = _class_colors()
    order = ["I", "A", "S"]
    f1["class"] = pd.Categorical(f1["class"], order, ordered=True)
    f1 = f1.sort_values("class")
    y = np.arange(3)[::-1]
    for yi, (_, row) in zip(y, f1.iterrows()):
        cls = str(row["class"])
        _plot_interval(ax_a, row["estimate"], row["ci_low"], row["ci_high"], yi, colors[cls])
        ax_a.text(min(row["ci_high"] + 0.035, 0.96), yi, f"{row['estimate']:.2f}", va="center", color=colors[cls])
    ax_a.axvline(float(config["gates"]["minimum_s_f1"]), color=PALETTE["neutral_mid"], ls="--", lw=0.9)
    ax_a.set_yticks(y, ["I type", "A type", "S type"])
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(-0.65, 2.65)
    ax_a.set_xlabel("Class-specific F1 (block-bootstrap 95% CI)")
    ax_a.text(0.02, 0.03, "54 group-type strata; 52 source-connected blocks", transform=ax_a.transAxes, color=PALETTE["neutral_mid"])
    add_panel_label(ax_a, "a", x=-0.09)

    metrics = [("macro_f1", "Macro-F1"), ("balanced_accuracy", "Balanced accuracy"), ("macro_ovr_auc", "Macro-AUC")]
    rng = np.random.default_rng(int(config["analysis"]["random_seed"]))
    for i, (column, label) in enumerate(metrics):
        values = repeat_pooled[column].dropna().to_numpy(float)
        ax_b.scatter(np.full(len(values), i) + rng.normal(0, 0.035, len(values)), values, s=20,
                     color=PALETTE["task_b_s"], alpha=0.78, edgecolor="white", linewidth=0.35)
        ax_b.plot([i - 0.14, i + 0.14], [np.median(values)] * 2, color=PALETTE["neutral_dark"], lw=1.5)
    ax_b.set_xticks(range(3), [x[1] for x in metrics], rotation=12, ha="right")
    ax_b.set_ylim(0, 1)
    ax_b.set_ylabel("Repeated grouped OOF score")
    ax_b.text(0.02, 0.04, f"n = {len(repeat_pooled)} repeats", transform=ax_b.transAxes, color=PALETTE["neutral_mid"])
    add_panel_label(ax_b, "b", x=-0.12)

    st = stability.set_index("class").loc[order].reset_index()
    yy = np.arange(3)[::-1]
    for yi, (_, row) in zip(yy, st.iterrows()):
        cls = str(row["class"])
        _plot_interval(ax_c, row["median_rank_spearman"], row["rank_spearman_q025"], row["rank_spearman_q975"], yi + 0.10, colors[cls], "o")
        ax_c.scatter(row["mean_top10_jaccard"], yi - 0.10, s=30, facecolor="white", edgecolor=colors[cls], marker="s", linewidth=1.1)
    ax_c.axvline(float(config["gates"]["minimum_s_rank_spearman"]), color=PALETTE["neutral_mid"], ls="--", lw=0.9)
    ax_c.set_yticks(yy, order)
    ax_c.set_xlim(0, 1)
    ax_c.set_xlabel("SHAP stability")
    ax_c.legend(handles=[
        Line2D([0], [0], color=PALETTE["neutral_dark"], marker="o", label="Rank Spearman (95% range)"),
        Line2D([0], [0], color=PALETTE["neutral_dark"], marker="s", markerfacecolor="white", lw=0, label="Mean top-10 Jaccard"),
    ], loc="upper left", bbox_to_anchor=(0.0, -0.25), ncol=1, fontsize=6.2)
    add_panel_label(ax_c, "c", x=-0.12)
    return export_figure(fig, paths.main_figures / "Fig6_taskB_readiness_and_attribution_stability")


def plot_main_figure7_v2(
    concordance: pd.DataFrame,
    permutation_null: pd.DataFrame,
    feature_summary: Mapping[str, Any],
    paths: OutputPaths,
) -> list[str]:
    """Cross-task attribution evidence without dependence curves that could imply smooth effects."""
    apply_publication_style(7.4)
    order = concordance.sort_values("task_a_contribution_share", ascending=False)["feature"].tolist()
    table = concordance.set_index("feature").loc[order].reset_index()
    table.to_csv(paths.source_data / "Fig7ab_shared_feature_concordance.csv", index=False, encoding="utf-8-sig")
    permutation_null.to_csv(paths.source_data / "Fig7c_exact_feature_permutation.csv", index=False, encoding="utf-8-sig")
    fig = plt.figure(figsize=(7.25, 4.7), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.05, 0.95], wspace=0.18, hspace=0.20)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])
    y = np.arange(len(table))[::-1]
    for yi, (_, row) in zip(y, table.iterrows()):
        _plot_interval(ax_a, row["task_a_direction_rho"], row["task_a_ci_low"], row["task_a_ci_high"], yi + 0.11, PALETTE["task_a"], "o")
        b_color = PALETTE["task_b_s"] if bool(row["direction_agreement"]) else PALETTE["conflict"]
        ax_a.plot([row["task_b_s_ci_low"], row["task_b_s_ci_high"]], [yi - 0.11] * 2, color=b_color, lw=1.5)
        ax_a.scatter(row["task_b_s_direction_rho"], yi - 0.11, s=34, marker="s",
                     facecolor=b_color if bool(row["direction_agreement"]) else "white", edgecolor=b_color, linewidth=1.0, zorder=3)
    ax_a.axvline(0, color=PALETTE["neutral_mid"], ls="--", lw=0.8)
    ax_a.set_yticks(y, [FEATURE_SPECS[f]["display"] for f in order])
    ax_a.set_xlim(-1.05, 1.05)
    ax_a.set_xlabel(r"Feature value--SHAP Spearman $\rho$ (block-bootstrap 95% CI)")
    ax_a.legend(handles=[
        Line2D([0], [0], color=PALETTE["task_a"], marker="o", label="Task A: U prospectivity"),
        Line2D([0], [0], color=PALETTE["task_b_s"], marker="s", label="Task B: S-type support"),
        Line2D([0], [0], color=PALETTE["conflict"], marker="s", markerfacecolor="white", label="Discordant direction"),
    ], loc="lower left", fontsize=6.4)
    add_panel_label(ax_a, "a", x=-0.09)

    signed = table[["feature", "task_a_signed_share", "task_b_s_signed_share"]].copy()
    yy = np.arange(len(signed))[::-1]
    ax_b.axvline(0, color=PALETTE["neutral_mid"], lw=0.7)
    for yi, (_, row) in zip(yy, signed.iterrows()):
        ax_b.plot([row["task_a_signed_share"], row["task_b_s_signed_share"]], [yi, yi], color=PALETTE["neutral_light"], lw=1.0)
        ax_b.scatter(row["task_a_signed_share"], yi, s=25, color=PALETTE["task_a"], marker="o")
        ax_b.scatter(row["task_b_s_signed_share"], yi, s=25, color=PALETTE["task_b_s"], marker="s")
    ax_b.set_yticks(yy, [FEATURE_SPECS[f]["display"] for f in signed["feature"]])
    ax_b.set_xlabel("Signed within-task contribution share")
    add_panel_label(ax_b, "b", x=-0.17)

    null = permutation_null["signed_contribution_cosine"].dropna().to_numpy(float)
    ax_c.hist(null, bins=24, color=PALETTE["neutral_light"], edgecolor="white", linewidth=0.4)
    observed = float(feature_summary["observed_signed_contribution_cosine"])
    ax_c.axvline(observed, color=PALETTE["conflict"], lw=1.8)
    ax_c.text(0.03, 0.95, f"Observed = {observed:.2f}\nExact p = {feature_summary['exact_feature_label_permutation_p']:.3f}",
              transform=ax_c.transAxes, ha="left", va="top")
    ax_c.set_xlabel("Signed-share cosine under feature-label permutation")
    ax_c.set_ylabel("Permutations")
    add_panel_label(ax_c, "c", x=-0.17)
    return export_figure(fig, paths.main_figures / "Fig7_cross_task_attribution_concordance")


def plot_main_figure8_v2(
    groups: pd.DataFrame,
    associations: pd.DataFrame,
    quartiles: pd.DataFrame,
    decision: Mapping[str, Any],
    paths: OutputPaths,
) -> list[str]:
    """Conditional probability linkage; colour is mineralization label, never the null SHAP cosine."""
    apply_publication_style(7.4)
    output_dir = paths.main_figures if decision["figure8_placement"] == "main_text" else paths.supplementary_figures
    groups.to_csv(paths.source_data / "Fig8a_group_level_probability_link.csv", index=False, encoding="utf-8-sig")
    quartiles.to_csv(paths.source_data / "Fig8b_S_support_quantiles.csv", index=False, encoding="utf-8-sig")
    associations.to_csv(paths.source_data / "Fig8c_association_forest.csv", index=False, encoding="utf-8-sig")
    fig = plt.figure(figsize=(7.25, 4.75), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1, 1], wspace=0.18, hspace=0.20)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])
    scatter = groups.dropna(subset=["P_U", "P_S_RAW"]).copy()
    label_colors = {0: "#A6A6A6", 1: "#B64342"}
    for label, marker, legend in [(0, "o", "Non-mineralized"), (1, "^", "Mineralized")]:
        part = scatter.loc[scatter["MINERALIZATION_LABEL"].eq(label)]
        ax_a.scatter(part["P_S_RAW"], part["P_U"], s=20 + 5 * np.sqrt(part["N_RECORDS"].clip(lower=1)),
                     color=label_colors[label], marker=marker, alpha=0.72, edgecolor="white", linewidth=0.45, label=legend)
    raw = associations.loc[associations["analysis"].eq("S support (raw)")].iloc[0]
    ax_a.text(0.03, 0.97, rf"$\rho$ = {raw['estimate']:.2f} ({raw['ci_low']:.2f}, {raw['ci_high']:.2f})" + f"\npermutation p = {raw['permutation_p']:.3f}; FDR q = {raw['fdr_q']:.3f}",
              transform=ax_a.transAxes, ha="left", va="top", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2})
    ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1)
    ax_a.set_xlabel("Task B raw OOF S-type support")
    ax_a.set_ylabel("Task A OOF uranium prospectivity")
    ax_a.legend(loc="lower right", fontsize=6.4)
    add_panel_label(ax_a, "a", x=-0.09)

    x = np.arange(len(quartiles))
    ax_b.errorbar(x, quartiles["p_u_median"], yerr=np.vstack([quartiles["p_u_median"] - quartiles["p_u_ci_low"], quartiles["p_u_ci_high"] - quartiles["p_u_median"]]),
                  fmt="o", color=PALETTE["task_b_s"], ecolor=PALETTE["task_b_s"], capsize=3, ms=4.5, lw=1.3)
    ax_b.plot(x, quartiles["p_u_median"], color=PALETTE["neutral_mid"], lw=0.8, ls=":", zorder=0)
    ax_b.set_xticks(x, [f"{r['s_support_bin']}\n(n={int(r['geological_groups'])})" for _, r in quartiles.iterrows()])
    ax_b.set_ylim(0, 1)
    ax_b.set_ylabel("Median uranium prospectivity\n(block-bootstrap 95% CI)")
    ax_b.set_xlabel("Frozen S-support quantile")
    add_panel_label(ax_b, "b", x=-0.14)

    forest_order = ["S support (raw)", "S support (calibrated sensitivity)", "SHAP-vector correspondence"]
    forest = associations.set_index("analysis").loc[forest_order].reset_index()
    yy = np.arange(len(forest))[::-1]
    fcolors = [PALETTE["task_b_s"], PALETTE["neutral_mid"], PALETTE["neutral_light"]]
    for yi, (_, row), color in zip(yy, forest.iterrows(), fcolors):
        _plot_interval(ax_c, row["estimate"], row["ci_low"], row["ci_high"], yi, color)
        ax_c.text(0.98, yi, f"q={row['fdr_q']:.3f}", transform=ax_c.get_yaxis_transform(), ha="right", va="center", fontsize=6.2)
    ax_c.axvline(0, color=PALETTE["neutral_mid"], ls="--", lw=0.8)
    ax_c.set_yticks(yy, ["S support (raw)", "S support (calibrated)", "SHAP-vector cosine"])
    ax_c.set_xlim(-1, 1)
    ax_c.set_xlabel(r"Block-level Spearman $\rho$ (95% CI)")
    add_panel_label(ax_c, "c", x=-0.14)
    return export_figure(fig, output_dir / "Fig8_conditional_S_support_and_U_prospectivity")


def joint_geochemical_state(groups: pd.DataFrame, bridge_features: Sequence[str], paths: OutputPaths) -> pd.DataFrame:
    """Descriptive four-quadrant geochemical state; thresholds are medians, not geological cut-offs."""
    work = groups.dropna(subset=["P_U", "P_S_RAW"]).copy()
    u_cut = float(work["P_U"].median())
    s_cut = float(work["P_S_RAW"].median())
    work["U_STATE"] = np.where(work["P_U"] >= u_cut, "high U", "low U")
    work["S_STATE"] = np.where(work["P_S_RAW"] >= s_cut, "high S", "low S")
    work["JOINT_STATE"] = work["S_STATE"] + " / " + work["U_STATE"]
    order = ["low S / low U", "high S / low U", "low S / high U", "high S / high U"]
    rows: list[dict[str, Any]] = []
    for feature in bridge_features:
        work[f"ROBUST_Z::{feature}"] = robust_zscore(work[f"COMMON_VALUE::{feature}"])
    for state in order:
        subset = work.loc[work["JOINT_STATE"].eq(state)]
        for feature in bridge_features:
            rows.append({
                "joint_state": state,
                "feature": feature,
                "median_robust_z": subset[f"ROBUST_Z::{feature}"].median(),
                "geological_groups": subset["GEOLOGICAL_GROUP_ID"].nunique(),
                "dependency_blocks": subset["COUPLING_DEPENDENCY_BLOCK"].nunique(),
                "u_median_threshold": u_cut,
                "s_median_threshold": s_cut,
            })
    table = pd.DataFrame(rows)
    table.to_csv(paths.tables / "Table_joint_probability_geochemical_state.csv", index=False, encoding="utf-8-sig")
    table.to_csv(paths.source_data / "Fig9a_joint_geochemical_state.csv", index=False, encoding="utf-8-sig")
    return table


def plot_main_figure9_v2(
    groups: pd.DataFrame,
    concordance: pd.DataFrame,
    bridge_features: Sequence[str],
    paths: OutputPaths,
) -> list[str]:
    """Data-led synthesis plus explicitly bounded literature interpretation."""
    apply_publication_style(7.1)
    state = joint_geochemical_state(groups, bridge_features, paths)
    state_order = ["low S / low U", "high S / low U", "low S / high U", "high S / high U"]
    feature_order = ["Rb", "P2O5", "Nb", "Zr", "Ba", "CaO"]
    pivot = state.pivot(index="joint_state", columns="feature", values="median_robust_z").reindex(index=state_order, columns=feature_order)
    n_map = state.groupby("joint_state")["geological_groups"].first().to_dict()
    fig = plt.figure(figsize=(7.25, 4.35))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.12, 1.0],
        left=0.105,
        right=0.985,
        bottom=0.19,
        top=0.95,
        wspace=0.38,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    image = ax_a.imshow(pivot.clip(-2, 2).to_numpy(float), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax_a.set_xticks(
        np.arange(len(feature_order)),
        [FEATURE_SPECS[f]["display"] for f in feature_order],
        fontsize=6.5,
    )
    ax_a.set_yticks(
        np.arange(len(state_order)),
        [f"{s}\n(n={int(n_map.get(s, 0))})" for s in state_order],
        fontsize=6.5,
    )
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            ax_a.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=6.3,
                      color="white" if np.isfinite(value) and abs(value) > 1.05 else PALETTE["neutral_dark"])
    ax_a.tick_params(length=0)
    ax_a.set_xlabel("Shared whole-rock variables")
    cbar = fig.colorbar(image, ax=ax_a, fraction=0.050, pad=0.035)
    cbar.set_label("Median robust z score")
    ax_a.text(
        0.0,
        -0.16,
        "Median splits are descriptive; they are not geological thresholds.",
        transform=ax_a.transAxes,
        color=PALETTE["neutral_mid"],
        fontsize=6.0,
        ha="left",
        va="top",
    )
    add_panel_label(ax_a, "a", x=-0.10)

    ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 1); ax_b.axis("off")
    _box(ax_b, (0.08, 0.72), 0.84, 0.15,
         "Model-supported correspondence\n5/6 directions align\nS support covaries with U prospectivity",
         "#EAF1F7", PALETTE["task_a"], fontsize=6.4)
    _box(ax_b, (0.08, 0.44), 0.84, 0.15,
         "Literature-constrained context\nCrustal source and differentiation\nmay establish U-fertile states",
         PALETTE["literature"], PALETTE["neutral_dark"], hatch="//", fontsize=6.4)
    _box(ax_b, (0.08, 0.16), 0.84, 0.15,
         "Required but untested processes\nFluid access, structural focusing,\nand physicochemical precipitation",
         "white", PALETTE["neutral_mid"], linestyle="--", fontsize=6.4)
    _arrow(ax_b, (0.50, 0.71), (0.50, 0.60), PALETTE["neutral_dark"])
    _arrow(ax_b, (0.50, 0.43), (0.50, 0.32), PALETTE["neutral_mid"], "--")
    ax_b.text(
        0.50,
        0.045,
        "Petrogenetic affinity conditions favourability;\nit does not by itself create an ore deposit.",
        ha="center",
        va="center",
        fontsize=6.1,
        fontweight="bold",
    )
    add_panel_label(ax_b, "b", x=0.0)
    return export_figure(fig, paths.main_figures / "Fig9_data_literature_bounded_geological_synthesis")


def feature_subset_sensitivity(concordance: pd.DataFrame, paths: OutputPaths) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    features = concordance["feature"].tolist()
    for omitted in [None] + features:
        sub = concordance.copy() if omitted is None else concordance.loc[~concordance["feature"].eq(omitted)].copy()
        a_share = sub["task_a_contribution_share"] / sub["task_a_contribution_share"].sum()
        b_share = sub["task_b_s_contribution_share"] / sub["task_b_s_contribution_share"].sum()
        left = np.sign(sub["task_a_direction_rho"].to_numpy(float)) * a_share.to_numpy(float)
        right = np.sign(sub["task_b_s_direction_rho"].to_numpy(float)) * b_share.to_numpy(float)
        observed = cosine(left, right)
        null = [cosine(left, right[list(p)]) for p in itertools.permutations(range(len(sub)))]
        p_exact = (1 + np.sum(np.abs(null) >= abs(observed))) / (1 + len(null))
        rows.append({"omitted_feature": "none (locked six-feature primary)" if omitted is None else omitted,
                     "features_retained": len(sub), "signed_share_cosine": observed, "exact_permutation_p": p_exact,
                     "analysis_role": "primary" if omitted is None else "leave-one-feature sensitivity"})
    table = pd.DataFrame(rows)
    table.to_csv(paths.tables / "Table_leave_one_feature_coupling_sensitivity.csv", index=False, encoding="utf-8-sig")
    return table


def plot_supplementary_figures_v2(
    files: Mapping[str, Path],
    paths: OutputPaths,
    config: Mapping[str, Any],
    cohort_flow: pd.DataFrame,
    concordance: pd.DataFrame,
    dependence: pd.DataFrame,
    associations: pd.DataFrame,
    distributions: Mapping[str, pd.DataFrame],
) -> list[str]:
    """Always populate Supplementary with audit, diagnostics and robustness outputs."""
    apply_publication_style(7.2)
    saved: list[str] = []

    # Figure S1: cohort and cross-task value audit.
    agreement = pd.read_csv(paths.audit / "cross_task_feature_value_agreement.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.25), layout="constrained")
    y = np.arange(len(cohort_flow))[::-1]
    axes[0].barh(y, cohort_flow["records"], color=[PALETTE["task_a"], PALETTE["task_b_s"], PALETTE["agreement"]], height=0.55)
    axes[0].set_yticks(y, ["Task A eligible", "Task B valid OOF", "Exact matched cohort"])
    axes[0].set_xlabel("Records")
    for yi, value in zip(y, cohort_flow["records"]): axes[0].text(value, yi, f" {int(value):,}", va="center")
    add_panel_label(axes[0], "a", x=-0.13)
    metric_col = next((c for c in agreement.columns if "max" in c.lower() and "diff" in c.lower()), None)
    if metric_col is None:
        numeric_cols = [c for c in agreement.select_dtypes(include=[np.number]).columns if c != "feature"]
        metric_col = numeric_cols[-1]
    axes[1].barh(np.arange(len(agreement))[::-1], pd.to_numeric(agreement[metric_col], errors="coerce"), color=PALETTE["neutral_mid"])
    feature_col = "feature" if "feature" in agreement.columns else agreement.columns[0]
    axes[1].set_yticks(np.arange(len(agreement))[::-1], [FEATURE_SPECS.get(normalize_feature_name(x), {"display": str(x)})["display"] for x in agreement[feature_col]])
    axes[1].set_xlabel("Maximum absolute cross-task value difference")
    axes[1].ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
    add_panel_label(axes[1], "b", x=-0.13)
    saved.extend(export_figure(fig, paths.supplementary_figures / "FigS1_cohort_and_value_audit"))

    # Figure S2: I/A/S bridge-direction diagnostic, explicitly not a formal three-class result.
    direction = numeric_frame(pd.read_csv(files["task_b_direction"]), ["global_group_level_spearman", "block_bootstrap_ci_low", "block_bootstrap_ci_high", "partition_direction_sign_consistency"])
    direction = direction.loc[direction["feature"].map(normalize_feature_name).isin(FEATURE_SPECS)].copy()
    direction["feature"] = direction["feature"].map(normalize_feature_name)
    dmat = direction.pivot(index="feature", columns="class", values="global_group_level_spearman").reindex(index=["Rb", "CaO", "Nb", "Zr", "P2O5", "Ba"], columns=["I", "A", "S"])
    fig, ax = plt.subplots(figsize=(5.2, 4.1), layout="constrained")
    im = ax.imshow(dmat.to_numpy(float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(3), ["I type", "A type", "S type"])
    ax.set_yticks(range(6), [FEATURE_SPECS[f]["display"] for f in dmat.index])
    for i, feature in enumerate(dmat.index):
        for j, cls in enumerate(dmat.columns):
            row = direction.loc[direction["feature"].eq(feature) & direction["class"].eq(cls)].iloc[0]
            excludes_zero = row["block_bootstrap_ci_low"] * row["block_bootstrap_ci_high"] > 0
            ax.text(j, i, f"{dmat.loc[feature, cls]:.2f}" + ("*" if excludes_zero else ""), ha="center", va="center",
                    color="white" if abs(dmat.loc[feature, cls]) > 0.55 else PALETTE["neutral_dark"])
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03); cbar.set_label(r"Feature value--SHAP $\rho$")
    ax.text(0.0, -0.15, "* block-bootstrap 95% CI excludes zero; I/A panels are diagnostic only.", transform=ax.transAxes, fontsize=6.3)
    saved.extend(export_figure(fig, paths.supplementary_figures / "FigS2_multiclass_bridge_direction_diagnostic"))

    # Figure S3: all six binned dependence summaries; no threshold language.
    dep_features = [f for f in ["Rb", "CaO", "Nb", "Zr", "P2O5", "Ba"] if f in set(dependence["feature"])]
    fig, axes = plt.subplots(2, 3, figsize=(7.25, 5.0), layout="constrained")
    for ax, feature in zip(axes.flat, dep_features):
        subset = dependence.loc[dependence["feature"].eq(feature)]
        for task, color, marker in [("Task A prospectivity", PALETTE["task_a"], "o"), ("Task B S-type", PALETTE["task_b_s"], "s")]:
            d = subset.loc[subset["task"].eq(task)].sort_values("bin")
            ax.fill_between(d["x_median"].to_numpy(float), d["ci_low"].to_numpy(float), d["ci_high"].to_numpy(float), color=color, alpha=0.12, linewidth=0)
            ax.plot(d["x_median"], d["y_median_standardized_shap"], color=color, marker=marker, ms=3.2, lw=1.1, label=task)
        ax.axhline(0, color=PALETTE["neutral_mid"], ls="--", lw=0.7)
        ax.set_xlabel(f"{FEATURE_SPECS[feature]['display']} ({FEATURE_SPECS[feature]['unit']})")
        ax.set_ylabel("Standardized SHAP")
    if len(dep_features): axes.flat[0].legend(loc="best", fontsize=6.0)
    for i, ax in enumerate(axes.flat): add_panel_label(ax, chr(ord("a") + i), x=-0.14)
    saved.extend(export_figure(fig, paths.supplementary_figures / "FigS3_all_bridge_feature_dependence"))

    # Figure S4: Task-B delete-one-source robustness and locked-feature sensitivity.
    leave = numeric_frame(pd.read_csv(files["task_b_leave_one_source"]), ["rank_spearman_vs_full", "top10_jaccard_vs_full", "maximum_primary_bridge_direction_change"])
    leave_s = leave.loc[leave["class"].eq("S")].copy()
    subset_sens = feature_subset_sensitivity(concordance, paths)
    leave_s.to_csv(paths.source_data / "FigS4a_leave_one_source_S.csv", index=False, encoding="utf-8-sig")
    subset_sens.to_csv(paths.source_data / "FigS4b_leave_one_feature.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.25), layout="constrained")
    axes[0].scatter(leave_s["rank_spearman_vs_full"], leave_s["top10_jaccard_vs_full"], s=24, color=PALETTE["task_b_s"], alpha=0.75, edgecolor="white", linewidth=0.4)
    axes[0].set_xlim(0, 1.02); axes[0].set_ylim(0, 1.02)
    axes[0].set_xlabel("Rank Spearman vs full aggregation")
    axes[0].set_ylabel("Top-10 Jaccard vs full aggregation")
    axes[0].text(0.03, 0.05, f"delete-one source blocks; n={len(leave_s)}", transform=axes[0].transAxes, color=PALETTE["neutral_mid"])
    add_panel_label(axes[0], "a", x=-0.13)
    ss = subset_sens.iloc[::-1].reset_index(drop=True)
    yy = np.arange(len(ss))
    axes[1].scatter(ss["signed_share_cosine"], yy, s=32, color=[PALETTE["task_a"] if x == "primary" else PALETTE["neutral_mid"] for x in ss["analysis_role"]])
    axes[1].axvline(0, color=PALETTE["neutral_light"], lw=0.8)
    axes[1].set_yticks(yy, ["Primary six" if x.startswith("none") else f"omit {FEATURE_SPECS[x]['display']}" for x in ss["omitted_feature"]])
    axes[1].set_xlim(-1, 1)
    axes[1].set_xlabel("Signed contribution-share cosine")
    for yi, (_, row) in zip(yy, ss.iterrows()): axes[1].text(0.98, yi, f"p={row['exact_permutation_p']:.3f}", transform=axes[1].get_yaxis_transform(), ha="right", va="center", fontsize=6.0)
    add_panel_label(axes[1], "b", x=-0.13)
    saved.extend(export_figure(fig, paths.supplementary_figures / "FigS4_source_and_feature_set_sensitivity"))

    # Figure S5: calibration and all permutation nulls.
    calibration = numeric_frame(pd.read_csv(files["task_b_calibration"]), ["value"])
    cal = calibration.loc[calibration["evaluation_unit"].eq("geological_group_by_reported_type_stratum") & (((calibration["class"].eq("multiclass")) & calibration["metric"].isin(["log_loss", "brier_score"])) | (calibration["class"].eq("top_label") & calibration["metric"].eq("ece")))].copy()
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.0), layout="constrained")
    labels = [("log_loss", "Log loss"), ("brier_score", "Brier score"), ("ece", "Top-label ECE")]
    for i, (metric, label) in enumerate(labels):
        raw_v = cal.loc[cal["metric"].eq(metric) & cal["probability_version"].eq("raw"), "value"].iloc[0]
        cal_v = cal.loc[cal["metric"].eq(metric) & cal["probability_version"].eq("calibrated"), "value"].iloc[0]
        axes[0, 0].plot([raw_v, cal_v], [i, i], color=PALETTE["neutral_light"], lw=2)
        axes[0, 0].scatter(raw_v, i, color=PALETTE["task_a"], s=30, label="Raw" if i == 0 else None)
        axes[0, 0].scatter(cal_v, i, color=PALETTE["neutral_mid"], s=30, marker="s", label="Calibrated" if i == 0 else None)
    axes[0, 0].set_yticks(range(3), [x[1] for x in labels]); axes[0, 0].set_xlabel("Error score (lower is better)"); axes[0, 0].legend(loc="best")
    add_panel_label(axes[0, 0], "a", x=-0.14)
    for ax, key, label, panel in [
        (axes[0, 1], "P_S_RAW", "Raw S support", "b"),
        (axes[1, 0], "P_S_CALIBRATED", "Calibrated S support", "c"),
        (axes[1, 1], "SHAP_VECTOR_COSINE", "SHAP-vector cosine", "d"),
    ]:
        null = distributions[f"{key}_null"]["permutation_rho"].dropna()
        row = associations.loc[associations["x"].eq(key)].iloc[0]
        ax.hist(null, bins=28, color=PALETTE["neutral_light"], edgecolor="white", linewidth=0.35)
        ax.axvline(row["estimate"], color=PALETTE["conflict"] if row["permutation_p"] < 0.05 else PALETTE["neutral_mid"], lw=1.7)
        ax.text(0.03, 0.95, f"observed={row['estimate']:.2f}; p={row['permutation_p']:.3f}", transform=ax.transAxes, ha="left", va="top")
        ax.set_xlabel(rf"Permuted $\rho$: {label}"); ax.set_ylabel("Permutations")
        add_panel_label(ax, panel, x=-0.14)
    saved.extend(export_figure(fig, paths.supplementary_figures / "FigS5_calibration_and_permutation_diagnostics"))

    # Figure S6: single-heldout confusion is non-gating; bridge importance across all classes.
    confusion = pd.read_csv(files["task_b_heldout_confusion"], index_col=0)
    conf = confusion.to_numpy(float); conf_norm = conf / conf.sum(axis=1, keepdims=True)
    shap_summary = numeric_frame(pd.read_csv(files["task_b_class_shap_summary"]), ["mean_abs_shap"])
    shap_summary["feature"] = shap_summary["feature"].map(normalize_feature_name)
    total = shap_summary.groupby("class")["mean_abs_shap"].transform("sum")
    shap_summary["class_share"] = shap_summary["mean_abs_shap"] / total
    bridge_shap = shap_summary.loc[shap_summary["feature"].isin(FEATURE_SPECS)].copy()
    smat = bridge_shap.pivot(index="feature", columns="class", values="class_share").reindex(index=["Rb", "CaO", "Nb", "Zr", "P2O5", "Ba"], columns=["I", "A", "S"])
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.55), layout="constrained")
    im0 = axes[0].imshow(conf_norm, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_xticks(range(3), ["Pred I", "Pred A", "Pred S"]); axes[0].set_yticks(range(3), ["True I", "True A", "True S"])
    for i in range(3):
        for j in range(3): axes[0].text(j, i, f"{int(conf[i,j])}\n({conf_norm[i,j]:.0%})", ha="center", va="center", color="white" if conf_norm[i,j] > .55 else PALETTE["neutral_dark"])
    axes[0].text(0.0, -0.15, "Single held-out split; descriptive, not the gating evidence layer.", transform=axes[0].transAxes, fontsize=6.2)
    add_panel_label(axes[0], "a", x=-0.13)
    im1 = axes[1].imshow(smat.to_numpy(float), cmap="magma_r", aspect="auto")
    axes[1].set_xticks(range(3), ["I type", "A type", "S type"]); axes[1].set_yticks(range(6), [FEATURE_SPECS[f]["display"] for f in smat.index])
    for i in range(smat.shape[0]):
        for j in range(smat.shape[1]): axes[1].text(j, i, f"{smat.iloc[i,j]:.2f}", ha="center", va="center", fontsize=6.2)
    cbar = fig.colorbar(im1, ax=axes[1], fraction=0.045, pad=0.03); cbar.set_label("Share of class mean |SHAP|")
    add_panel_label(axes[1], "b", x=-0.13)
    saved.extend(export_figure(fig, paths.supplementary_figures / "FigS6_non_gating_confusion_and_multiclass_bridge_importance"))
    return saved


def figure_reference_manifest(paths: OutputPaths) -> pd.DataFrame:
    rows = [
        {
            "generated_figure": "Figure 6",
            "reference_paper": "Saha et al. (2021), Geochemistry, Geophysics, Geosystems",
            "reference_figure": "Figures 8–9",
            "borrowed_information_logic": "class-specific model/attribution organization only",
            "required_upgrade": "dominant classwise F1 forest, repeated grouped OOF and explicit stability gates",
            "url": "https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021GC010053",
        },
        {
            "generated_figure": "Figure 7",
            "reference_paper": "Kong et al. (2024), Minerals",
            "reference_figure": "Figures 16–17",
            "borrowed_information_logic": "signed SHAP distribution semantics",
            "required_upgrade": "locked cross-task directions, uncertainty, signed contribution shares and exact permutation",
            "url": "https://www.mdpi.com/2075-163X/14/2/128",
        },
        {
            "generated_figure": "Figure S3",
            "reference_paper": "Zhang et al. (2024), Minerals",
            "reference_figure": "Figures 4–5 and 8–9",
            "borrowed_information_logic": "nonlinear response/dependence presentation",
            "required_upgrade": "all six locked features, OOF group bins and dependency-block bootstrap bands; no threshold claim",
            "url": "https://www.mdpi.com/2075-163x/14/5/500",
        },
        {
            "generated_figure": "Figure 8",
            "reference_paper": "Pan-Canadian LCT pegmatite study, Natural Resources Research",
            "reference_figure": "Figure 8",
            "borrowed_information_logic": "two-dimensional decision-plane organization",
            "required_upgrade": "OOF S support versus OOF U prospectivity, mineralization-label encoding and block-aware uncertainty",
            "url": "https://link.springer.com/article/10.1007/s11053-024-10438-x",
        },
        {
            "generated_figure": "Figure 9",
            "reference_paper": "Zhang et al. (2021), Acta Petrologica Sinica",
            "reference_figure": "Figures 3 and 8",
            "borrowed_information_logic": "ore-bearing/barren geochemistry and two-stage metallogenic framework",
            "required_upgrade": "data-derived joint geochemical-state matrix plus an original evidence-bounded process chain; no copied artwork",
            "url": "https://html.rhhz.net/ysxb/20210904.htm",
        },
        {
            "generated_figure": "Figure 9b",
            "reference_paper": "Chi et al. (2020), Acta Petrologica Sinica",
            "reference_figure": "Figure 1",
            "borrowed_information_logic": "U mobilization/precipitation process constraints",
            "required_upgrade": "qualitative literature-constrained process only; no approximate thermodynamic fields",
            "url": "https://html.rhhz.net/ysxb/20200105.htm",
        },
        {
            "generated_figure": "Figures S1–S6",
            "reference_paper": "No single source; standard uncertainty and robustness diagnostics",
            "reference_figure": "not applicable",
            "borrowed_information_logic": "audit, sensitivity, calibration and non-gating diagnostics",
            "required_upgrade": "each panel is directly traceable to exported source data and a declared claim boundary",
            "url": "",
        },
    ]
    table = pd.DataFrame(rows)
    table.to_csv(
        paths.logs / "figure_reference_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return table


def figure_contract_manifest(paths: OutputPaths) -> pd.DataFrame:
    rows = [
        {"figure": "Figure 6", "section": "Results", "core_conclusion": "S is the only Task-B branch meeting the restricted coupling gate; overall I/A/S readiness remains insufficient.", "evidence": "classwise F1 CI + repeated grouped OOF + SHAP stability", "claim_boundary": "does not establish full three-class coupling"},
        {"figure": "Figure 7", "section": "Results", "core_conclusion": "Five of six locked features show concordant attribution directions and the signed-share pattern exceeds exact feature-label permutation expectation.", "evidence": "direction forest + signed contribution share + exact permutation", "claim_boundary": "attribution correspondence, not causal control"},
        {"figure": "Figure 8", "section": "Results", "core_conclusion": "S-type OOF support is positively associated with U prospectivity across connected dependency blocks; the SHAP-vector-cosine association is null.", "evidence": "group scatter + frozen quantiles + block inference forest", "claim_boundary": "association does not prove improved prediction or petrogenetic causality"},
        {"figure": "Figure 9", "section": "Discussion", "core_conclusion": "Petrogenetic affinity may condition a favourable whole-rock state, while ore formation still requires later mobilization, focusing and precipitation.", "evidence": "joint geochemical-state matrix + literature-bounded process chain", "claim_boundary": "late hydrothermal processes are not tested by either model"},
        {"figure": "Figure S1", "section": "Supplementary", "core_conclusion": "The exact matched cohort and cross-task feature values are auditable.", "evidence": "cohort flow + maximum value differences", "claim_boundary": "data integrity only"},
        {"figure": "Figure S2", "section": "Supplementary", "core_conclusion": "I/A/S bridge directions are shown diagnostically without promoting I/A to formal coupling.", "evidence": "class-feature direction matrix with bootstrap sign flag", "claim_boundary": "I/A are non-gating diagnostics"},
        {"figure": "Figure S3", "section": "Supplementary", "core_conclusion": "Binned OOF SHAP responses document nonlinearity without asserting numerical thresholds.", "evidence": "all six binned dependence summaries", "claim_boundary": "descriptive curves, not causal or threshold estimates"},
        {"figure": "Figure S4", "section": "Supplementary", "core_conclusion": "The restricted S interpretation is evaluated against source-block influence and leave-one-feature perturbations.", "evidence": "delete-one-source + leave-one-feature exact permutation", "claim_boundary": "models are not refitted in delete-one-source analysis"},
        {"figure": "Figure S5", "section": "Supplementary", "core_conclusion": "Calibration sensitivity preserves direction and permutation nulls distinguish supported and unsupported links.", "evidence": "calibration dumbbells + three permutation nulls", "claim_boundary": "raw probability remains primary"},
        {"figure": "Figure S6", "section": "Supplementary", "core_conclusion": "The single held-out confusion matrix and multiclass bridge importance provide context only.", "evidence": "row-normalized confusion + within-class mean-|SHAP| shares", "claim_boundary": "single held-out split is not the gating evidence layer"},
    ]
    table = pd.DataFrame(rows)
    table.to_csv(paths.logs / "figure_contract_manifest.csv", index=False, encoding="utf-8-sig")
    return table


def figure_artifact_quality_audit(paths: OutputPaths, figure_files: Sequence[str]) -> pd.DataFrame:
    """Fail fast on missing/tiny files and non-editable SVG text after a formal run."""
    rows: list[dict[str, Any]] = []
    for value in figure_files:
        path = Path(value)
        exists = path.is_file()
        size = path.stat().st_size if exists else 0
        svg_has_text = None
        if exists and path.suffix.lower() == ".svg":
            svg_has_text = "<text" in path.read_text(encoding="utf-8", errors="ignore")
        minimum = 2000 if path.suffix.lower() in {".svg", ".pdf"} else 5000
        passed = bool(exists and size >= minimum and (svg_has_text is not False))
        rows.append({"file": str(path), "suffix": path.suffix.lower(), "exists": exists,
                     "bytes": size, "minimum_bytes": minimum, "svg_contains_editable_text": svg_has_text,
                     "quality_gate_pass": passed})
    table = pd.DataFrame(rows)
    table.to_csv(paths.logs / "figure_artifact_quality_audit.csv", index=False, encoding="utf-8-sig")
    if len(table) and not table["quality_gate_pass"].all():
        failed = table.loc[~table["quality_gate_pass"], "file"].tolist()
        raise RuntimeError(f"Figure artifact quality gate failed: {failed}")
    return table


def write_run_report(
    paths: OutputPaths,
    cohort_flow: pd.DataFrame,
    feature_summary: Mapping[str, Any],
    associations: pd.DataFrame,
    decision: Mapping[str, Any],
) -> None:
    final = cohort_flow.iloc[-1]
    raw = associations.loc[associations["analysis"].eq("S support (raw)")].iloc[0]
    text = f"""# Part 4 run summary

## Cohort

- Matched valid OOF records: {int(final['records'])}
- Geological groups: {int(final['geological_groups'])}
- Connected dependency blocks: {int(final['dependency_blocks'])}

## Cross-task attribution

- Locked shared features: {feature_summary['shared_features']}
- Direction agreements: {feature_summary['direction_agreements']}/{feature_summary['shared_features']}
- Signed contribution-share cosine: {feature_summary['observed_signed_contribution_cosine']:.3f}
- Exact feature-label permutation p: {feature_summary['exact_feature_label_permutation_p']:.4f}

## Conditional S-support association

- Block-level Spearman rho: {raw['estimate']:.3f}
- 95% bootstrap CI: {raw['ci_low']:.3f} to {raw['ci_high']:.3f}
- Structure-preserving permutation p: {raw['permutation_p']:.4f}
- FDR q: {raw['fdr_q']:.4f}

## Gate decision

- Full I/A/S formal coupling eligible: {decision['gate_checks']['formal_full_three_class_coupling_eligible']}
- Restricted S diagnostic eligible: {decision['gate_checks']['restricted_s_diagnostic_eligible']}
- Figure 8 placement: {decision['figure8_placement']}
- Permitted primary claim: {decision['permitted_primary_claim']}

SHAP is interpreted as model attribution, not geological causality.
"""
    (paths.logs / "RUN_SUMMARY.md").write_text(text, encoding="utf-8")


def runtime_manifest(
    config_path: Path,
    files: Mapping[str, Path],
    paths: OutputPaths,
    decision: Mapping[str, Any],
    figure_files: Sequence[str],
) -> dict[str, Any]:
    manifest = {
        "pipeline": "JGE Part 4 guarded coupling",
        "python": sys.version,
        "platform": platform.platform(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "input_hashes": {name: sha256_file(path) for name, path in files.items()},
        "decision": decision,
        "figure_files": list(figure_files),
        "output_root": str(paths.root),
    }
    write_json(manifest, paths.logs / "run_manifest.json")
    return manifest


def run_pipeline(config_path: str | Path, make_figures: bool = True) -> dict[str, Any]:
    """Run the complete Part 4 analysis from saved OOF outputs."""
    config, config_path = load_config(config_path)
    project_root = config_path.parent.parent
    files = resolve_input_paths(config, project_root)
    output_root = resolve_configured_path(
        project_root, config["paths"]["output_root"]
    )
    paths = make_output_paths(output_root)
    audit_inputs(files, paths)

    contract_a = read_json(files["task_a_contract"])
    contract_b = read_json(files["task_b_contract"])
    readiness_b = read_json(files["task_b_readiness"])
    locked_a = pd.read_csv(files["task_a_locked_features"])
    rank_stability = numeric_frame(
        pd.read_csv(files["task_b_rank_stability"]),
        [
            "median_rank_spearman",
            "rank_spearman_q025",
            "rank_spearman_q975",
            "mean_top10_jaccard",
            "minimum_top10_jaccard",
        ],
    )
    bridge_features = determine_bridge_features(
        config, contract_a, contract_b, locked_a
    )

    matched, cohort_flow = build_matched_record_cohort(
        files, bridge_features, paths
    )
    groups = aggregate_to_geological_groups(matched, bridge_features, paths)
    concordance, feature_null, feature_summary = feature_concordance(
        files, groups, bridge_features, paths
    )
    dependence = dependence_source_data(
        groups,
        config["analysis"]["dependence_features"],
        int(config["analysis"]["dependence_bins"]),
        int(config["analysis"]["bootstrap_replicates"]),
        int(config["analysis"]["random_seed"]),
        paths,
    )
    associations, distributions, quartiles = association_analysis(
        groups, config, paths
    )
    readiness_table = model_readiness_table(
        contract_a,
        readiness_b,
        cohort_flow,
        feature_summary,
        paths,
    )
    decision = decision_gate(
        contract_a,
        readiness_b,
        rank_stability,
        feature_summary,
        associations,
        groups,
        config,
    )
    write_json(decision, paths.audit / "coupling_readiness_decision.json")

    figure_files: list[str] = []
    if make_figures:
        figure_files.extend(plot_main_figure6_v2(files, paths, config))
        figure_files.extend(
            plot_main_figure7_v2(
                concordance,
                feature_null,
                feature_summary,
                paths,
            )
        )
        figure_files.extend(
            plot_main_figure8_v2(
                groups,
                associations,
                quartiles,
                decision,
                paths,
            )
        )
        figure_files.extend(plot_main_figure9_v2(groups, concordance, bridge_features, paths))
        figure_files.extend(
            plot_supplementary_figures_v2(
                files,
                paths,
                config,
                cohort_flow,
                concordance,
                dependence,
                associations,
                distributions,
            )
        )
        figure_artifact_quality_audit(paths, figure_files)
    reference_manifest = figure_reference_manifest(paths)
    figure_contracts = figure_contract_manifest(paths)
    write_run_report(
        paths, cohort_flow, feature_summary, associations, decision
    )
    manifest = runtime_manifest(
        config_path, files, paths, decision, figure_files
    )

    return {
        "bridge_features": bridge_features,
        "cohort_flow": cohort_flow,
        "group_level_coupling": groups,
        "feature_concordance": concordance,
        "association_results": associations,
        "model_readiness": readiness_table,
        "decision": decision,
        "reference_manifest": reference_manifest,
        "figure_contracts": figure_contracts,
        "figure_files": figure_files,
        "manifest": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to coupling_config.json")
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Run data/statistical outputs only and skip figure export.",
    )
    args = parser.parse_args()
    result = run_pipeline(args.config, make_figures=not args.no_figures)
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
