"""Frozen, model-independent cohort screening for Task B.

The rules reproduce the original 1,864-record v5 analysis cohort before any
hold-out construction, tuning, model fitting or SHAP calculation.  Screening
uses only reported granite type, geological-group metadata and source-defined
group type; it never uses model predictions or Task A prospectivity labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_HARD_EXCLUDE_GROUP_TYPES = {
    "Regional multi-pluton assemblage",
    "Regional migmatite–granite assemblage",
    "Multi-pluton assemblage",
    "Composite pluton",
    "Composite pluton pair",
    "Composite batholith",
    "Composite granite pair",
    "Regional igneous assemblage",
    "Regional igneous suite",
    "Regional magmatic assemblage",
    "Regional metallogenic assemblage",
    "Regional metallogenic granitoid assemblage",
    "Regional metallogenic granitoid suite",
    "Intrusive assemblage",
    "Granite–migmatite complex",
    "Deposit-scale intrusive suite",
    "Deposit-scale granitoid suite",
    "Deposit-scale granitic-vein suite",
    "Deposit-scale intrusion",
    "Deposit-scale granite-porphyry suite",
    "Granite-porphyry suite",
    "Granite-porphyry body",
    "Granodiorite-porphyry body",
    "Batholith and evolved granite suite",
}


def _normalize_group_type(value: object) -> str:
    """Normalize punctuation corruption observed in the frozen S2 workbook."""
    text = str(value).strip()
    for token in ("\ufffdC", "â€“", "â€–", "--"):
        text = text.replace(token, "–")
    return " ".join(text.split())


def _read_s1_s2(s1_path: Path, s2_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    s1 = pd.read_excel(s1_path, sheet_name="Dataset", header=1)
    s2 = pd.read_excel(s2_path, sheet_name="Geological Groups", header=1)
    s1["Granite type"] = s1["Granite type"].astype(str).str.strip().str.upper()
    s1 = s1[s1["Granite type"].isin(("I", "A", "S"))].copy()
    s1["Geological Group ID"] = s1["Geological Group ID"].astype(str).str.strip()
    s2["Geological Group ID"] = s2["Geological Group ID"].astype(str).str.strip()
    if "Group Type" in s2:
        s2["Group Type normalized"] = s2["Group Type"].map(_normalize_group_type)
    return s1, s2


def _group_type_matrix(s1: pd.DataFrame) -> pd.DataFrame:
    return s1.groupby("Geological Group ID")["Granite type"].value_counts().unstack(fill_value=0)


def excluded_group_ids(s1_path: Path, s2_path: Path, config: dict[str, Any]) -> set[str]:
    rules = config.get("data_filter", {})
    s1, s2 = _read_s1_s2(s1_path, s2_path)
    excluded: set[str] = set()

    hard_types = {
        _normalize_group_type(value)
        for value in rules.get("hard_exclude_group_types", DEFAULT_HARD_EXCLUDE_GROUP_TYPES)
    }
    if rules.get("exclude_out_of_domain", True) and hard_types:
        if "Group Type normalized" not in s2:
            raise ValueError("Supplementary Table S1 lacks Group Type required by the frozen Task B data filter.")
        excluded.update(
            s2.loc[s2["Group Type normalized"].isin(hard_types), "Geological Group ID"]
        )

    matrix = _group_type_matrix(s1)
    mixed = matrix[(matrix > 0).sum(axis=1) > 1]
    threshold = float(rules.get("mixed_minority_threshold", 0.20))
    if rules.get("exclude_mixed_type_groups", True):
        for group, row in mixed.iterrows():
            total = int(row.sum())
            minority = total - int(row.max())
            if minority / total >= threshold:
                excluded.add(str(group))

    minimum_group_size = int(rules.get("minimum_group_size", 5))
    if minimum_group_size > 0:
        sizes = s1.groupby("Geological Group ID").size()
        excluded.update(sizes[sizes < minimum_group_size].index.astype(str))

    excluded.update(str(group).strip() for group in rules.get("exclude_groups", []))
    return {group for group in excluded if group and group.lower() != "nan"}


def reconcile_group_type_map(
    s1_path: Path, s2_path: Path, config: dict[str, Any]
) -> dict[str, str]:
    """Return the majority class for weakly mixed groups retained by the protocol."""
    rules = config.get("data_filter", {})
    if not rules.get("exclude_mixed_type_groups", True):
        return {}
    s1, _ = _read_s1_s2(s1_path, s2_path)
    matrix = _group_type_matrix(s1)
    mixed = matrix[(matrix > 0).sum(axis=1) > 1]
    threshold = float(rules.get("mixed_minority_threshold", 0.20))
    reconciliation: dict[str, str] = {}
    for group, row in mixed.iterrows():
        total = int(row.sum())
        minority = total - int(row.max())
        if 0 < minority / total < threshold:
            reconciliation[str(group)] = str(row.idxmax())
    return reconciliation


def cohort_attrition_audit(
    s1_path: Path, s2_path: Path, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a record-level audit and a deterministic cohort-flow summary."""
    raw = pd.read_excel(s1_path, sheet_name="Dataset", header=1)
    frame = raw[["Record ID", "Granite type", "Geological Group ID"]].copy()
    frame["reported_type_normalized"] = frame["Granite type"].astype(str).str.strip().str.upper()
    frame["Geological Group ID"] = frame["Geological Group ID"].astype(str).str.strip()
    frame["filter_status"] = "retained"
    frame.loc[~frame["reported_type_normalized"].isin(("I", "A", "S")), "filter_status"] = "excluded_non_IAS_label"

    excluded = excluded_group_ids(s1_path, s2_path, config)
    reconciled = reconcile_group_type_map(s1_path, s2_path, config)
    explicit = frame["reported_type_normalized"].isin(("I", "A", "S"))
    group_excluded = explicit & frame["Geological Group ID"].isin(excluded)
    frame.loc[group_excluded, "filter_status"] = "excluded_group_protocol"
    for group, majority in reconciled.items():
        minority = (
            explicit
            & frame["Geological Group ID"].eq(group)
            & ~frame["reported_type_normalized"].eq(majority)
            & frame["filter_status"].eq("retained")
        )
        frame.loc[minority, "filter_status"] = "excluded_weak_mixed_group_minority"

    retained = frame["filter_status"].eq("retained")
    summary = {
        "raw_S1_records": int(len(frame)),
        "explicit_IAS_records": int(explicit.sum()),
        "excluded_non_IAS_records": int((frame["filter_status"] == "excluded_non_IAS_label").sum()),
        "excluded_group_protocol_records": int((frame["filter_status"] == "excluded_group_protocol").sum()),
        "excluded_weak_mixed_group_minority_records": int((frame["filter_status"] == "excluded_weak_mixed_group_minority").sum()),
        "retained_analysis_records": int(retained.sum()),
        "excluded_geological_groups": int(len(excluded)),
        "reconciled_weak_mixed_groups": int(len(reconciled)),
    }
    rules = config.get("data_filter", {})
    if bool(rules.get("enforce_expected_counts", False)):
        expected_explicit = int(rules["expected_explicit_IAS_records"])
        expected_retained = int(rules["expected_retained_records"])
        if summary["explicit_IAS_records"] != expected_explicit or summary["retained_analysis_records"] != expected_retained:
            raise RuntimeError(
                "Frozen Task B cohort mismatch: expected "
                f"{expected_explicit} explicit I/A/S and {expected_retained} retained records, "
                f"observed {summary['explicit_IAS_records']} and {summary['retained_analysis_records']}. "
                "Do not run modelling until the input workbooks or filter contract are reconciled."
            )
    return frame, summary
