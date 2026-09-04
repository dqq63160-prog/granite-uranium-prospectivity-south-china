"""Reference-connected blocking and leakage audits for Task B.

Geological groups that share a Reference ID are joined into the same connected
component.  The resulting block is the indivisible resampling unit used by the
fixed holdout, nested cross-validation, and SHAP cross-fitting.
"""

from __future__ import annotations

import re
import hashlib
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd


REFERENCE_PATTERN = re.compile(r"REF\d{4}", flags=re.IGNORECASE)
REFERENCE_NORMALIZATION_RULE = "uppercase unique tokens matching REF\\d{4}; lexical order"


def normalize_reference_ids(value: object) -> list[str]:
    """Return unique upper-case REF identifiers in deterministic order."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return sorted(set(match.upper() for match in REFERENCE_PATTERN.findall(str(value))))


def reference_block_registry_hash(registry: pd.DataFrame) -> str:
    """Hash a canonical one-row-per-group block registry."""
    required = [
        "Geological Group ID", "Reference-connected block", "Reference ID normalized",
        "n_samples", "n_I", "n_A", "n_S",
    ]
    missing = set(required) - set(registry.columns)
    if missing:
        raise ValueError(f"Block registry hash fields are missing: {sorted(missing)}")
    canonical = registry[required].copy().sort_values("Geological Group ID").reset_index(drop=True)
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            # Lexical attachment makes the graph result independent of row order.
            low, high = sorted((left_root, right_root))
            self.parent[high] = low


def _reference_map_from_s2(group_reference_frame: pd.DataFrame | None) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    if group_reference_frame is None or group_reference_frame.empty:
        return mapping
    group_col = "Geological Group ID"
    reference_col = "Reference ID(s)"
    if group_col not in group_reference_frame or reference_col not in group_reference_frame:
        return mapping
    for group, value in group_reference_frame[[group_col, reference_col]].itertuples(index=False):
        group = str(group).strip()
        if group and group.lower() != "nan":
            mapping[group].update(normalize_reference_ids(value))
    return mapping


def build_reference_connected_blocks(
    metadata: pd.DataFrame,
    group_reference_frame: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Build fixed connected components from all explicit I/A/S records.

    Returns a record-aligned block Series, a record-aligned normalized source
    string, and a one-row-per-geological-group registry.
    """
    required = {"Geological Group ID", "Reference ID", "Granite type"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata lacks source-block fields: {sorted(missing)}")

    frame = metadata.reset_index(drop=True).copy()
    groups = frame["Geological Group ID"].astype(str).str.strip()
    if groups.isin(["", "nan", "None"]).any():
        raise ValueError("Every Task B record must have a Geological Group ID.")

    group_references = _reference_map_from_s2(group_reference_frame)
    for group, value in zip(groups, frame["Reference ID"]):
        group_references[group].update(normalize_reference_ids(value))

    union_find = _UnionFind()
    reference_owner: dict[str, str] = {}
    for group in sorted(groups.unique()):
        union_find.add(group)
        for reference in sorted(group_references[group]):
            if reference in reference_owner:
                union_find.union(group, reference_owner[reference])
            else:
                reference_owner[reference] = group

    components: dict[str, list[str]] = defaultdict(list)
    for group in sorted(groups.unique()):
        components[union_find.find(group)].append(group)
    ordered_components = sorted(
        (sorted(members) for members in components.values()), key=lambda members: members[0]
    )
    group_to_block = {
        group: f"RCB{block_number:04d}"
        for block_number, members in enumerate(ordered_components, 1)
        for group in members
    }
    block_series = groups.map(group_to_block).rename("Reference-connected block")
    reference_series = pd.Series(
        ["; ".join(sorted(group_references[group])) for group in groups],
        name="Reference ID normalized",
    )

    registry_rows: list[dict[str, object]] = []
    for group in sorted(groups.unique()):
        mask = groups.eq(group)
        counts = frame.loc[mask, "Granite type"].value_counts()
        registry_rows.append({
            "Geological Group ID": group,
            "Reference-connected block": group_to_block[group],
            "Reference ID normalized": "; ".join(sorted(group_references[group])),
            "n_samples": int(mask.sum()),
            "n_I": int(counts.get("I", 0)),
            "n_A": int(counts.get("A", 0)),
            "n_S": int(counts.get("S", 0)),
        })
    registry = pd.DataFrame(registry_rows)
    audit_reference_block_graph(frame, block_series, reference_series)
    return block_series, reference_series, registry


def audit_reference_block_graph(
    metadata: pd.DataFrame,
    block_series: pd.Series,
    reference_series: pd.Series | None = None,
) -> pd.DataFrame:
    """Validate the graph and return one audit row per normalized reference."""
    if len(metadata) != len(block_series):
        raise ValueError("Metadata and block Series have different lengths.")
    frame = metadata.reset_index(drop=True).copy()
    frame["Reference-connected block"] = pd.Series(block_series).reset_index(drop=True)
    group_blocks = frame.groupby("Geological Group ID")["Reference-connected block"].nunique()
    if (group_blocks != 1).any():
        raise RuntimeError("A geological group was assigned to more than one source block.")

    if reference_series is None:
        reference_series = frame["Reference ID"].map(
            lambda value: "; ".join(normalize_reference_ids(value))
        )
    rows: list[dict[str, object]] = []
    expanded: list[tuple[str, str, str]] = []
    for group, block, references in zip(
        frame["Geological Group ID"].astype(str),
        frame["Reference-connected block"].astype(str),
        reference_series,
    ):
        for reference in normalize_reference_ids(references):
            expanded.append((reference, group, block))
    if expanded:
        expanded_frame = pd.DataFrame(
            expanded, columns=["Reference ID", "Geological Group ID", "Reference-connected block"]
        ).drop_duplicates()
        for reference, subset in expanded_frame.groupby("Reference ID"):
            blocks = sorted(subset["Reference-connected block"].unique())
            if len(blocks) != 1:
                raise RuntimeError(f"Reference {reference} spans multiple connected blocks: {blocks}")
            rows.append({
                "Reference ID": reference,
                "Reference-connected block": blocks[0],
                "n_geological_groups": int(subset["Geological Group ID"].nunique()),
                "Geological Group IDs": "; ".join(sorted(subset["Geological Group ID"].unique())),
                "shared_reference": bool(subset["Geological Group ID"].nunique() > 1),
            })
    return pd.DataFrame(rows)


def assert_no_block_overlap(
    train_positions: Iterable[int],
    valid_positions: Iterable[int],
    metadata: pd.DataFrame,
) -> None:
    """Stop if Record IDs, geological groups, or source blocks cross a split."""
    train = metadata.iloc[np.asarray(list(train_positions), dtype=int)]
    valid = metadata.iloc[np.asarray(list(valid_positions), dtype=int)]
    levels = ("Record ID", "Geological Group ID", "Reference-connected block")
    for column in levels:
        overlap = set(train[column].astype(str)) & set(valid[column].astype(str))
        if overlap:
            preview = sorted(overlap)[:5]
            raise RuntimeError(f"Leakage at {column}: {preview} (n={len(overlap)})")
