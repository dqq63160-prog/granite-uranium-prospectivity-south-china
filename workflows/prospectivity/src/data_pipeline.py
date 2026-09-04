"""Leakage-safe data audit and cohort construction for uranium prospectivity modelling."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler

from export_utils import (
    IntegrityLedger,
    assert_columns,
    create_run_manifest,
    load_config,
    read_lines,
    resolve_package_path,
    save_json,
    save_table,
    sha256_file,
    sha256_payload,
    stage_complete,
)


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def normalize_reference_ids(value: object) -> list[str]:
    if pd.isna(value):
        return []
    tokens = re.split(r"[;,|/\s]+", str(value).strip())
    return sorted({token for token in tokens if token and token.lower() not in {"nan", "none"}})


def read_source_tables(config: dict[str, Any], root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = config["input"]
    s1_path = resolve_package_path(root, source["supplementary_table_s1"])
    s2_path = resolve_package_path(root, source["supplementary_table_s2"])
    if not s1_path.exists() or not s2_path.exists():
        raise FileNotFoundError(f"Input files not found: {s1_path}; {s2_path}")
    s1 = pd.read_excel(
        s1_path, sheet_name=source.get("s1_sheet", "Dataset"), header=int(source.get("header_row", 1))
    )
    s2 = pd.read_excel(
        s2_path, sheet_name=source.get("s2_sheet", "Geological Groups"), header=int(source.get("header_row", 1))
    )
    return s1, s2


def validate_source_tables(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    config: dict[str, Any],
    ledger: IntegrityLedger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = config["columns"]
    required_s1 = [
        columns["record_id"], columns["label"], columns["group_id"],
        columns["group_name"], columns["reference_id"],
    ]
    required_s2 = [columns["group_id"], columns["group_name"]]
    assert_columns(s1, required_s1, "Supplementary Table S1")
    assert_columns(s2, required_s2, "Supplementary Table S2")
    s1 = s1.copy()
    s2 = s2.copy()
    for column in [columns["record_id"], columns["group_id"], columns["reference_id"]]:
        s1[column] = s1[column].astype("string").str.strip()
    s2[columns["group_id"]] = s2[columns["group_id"]].astype("string").str.strip()
    record_ok = s1[columns["record_id"]].notna().all() and not s1[columns["record_id"]].duplicated().any()
    ledger.check("record_id_unique_and_complete", record_ok, {
        "rows": len(s1), "duplicates": int(s1[columns["record_id"]].duplicated().sum())
    })
    raw_labels = pd.to_numeric(s1[columns["label"]], errors="coerce")
    label_ok = raw_labels.notna().all() and set(raw_labels.unique()).issubset({0, 1})
    ledger.check("labels_are_binary_and_complete", label_ok, {
        "value_counts": raw_labels.value_counts(dropna=False).to_dict()
    })
    s1[columns["label"]] = raw_labels.astype(int)
    s2_ids = set(s2[columns["group_id"]].dropna().astype(str))
    missing_groups = sorted(set(s1[columns["group_id"]].dropna().astype(str)).difference(s2_ids))
    ledger.check("every_s1_group_traceable_to_s2", not missing_groups, {"missing_group_ids": missing_groups})
    reference_ok = s1[columns["reference_id"]].notna() & s1[columns["reference_id"]].str.len().gt(0)
    ledger.check("every_record_has_source_reference", bool(reference_ok.all()), {
        "records_without_reference": int((~reference_ok).sum())
    })
    s2_group_ok = s2[columns["group_id"]].notna().all() and not s2[columns["group_id"]].duplicated().any()
    ledger.check("s2_group_registry_unique", s2_group_ok, {
        "groups": int(s2[columns["group_id"]].nunique()),
        "duplicates": int(s2[columns["group_id"]].duplicated().sum()),
    })
    return s1, s2


def build_reference_connected_blocks_for_all_records(
    s1: pd.DataFrame, config: dict[str, Any], ledger: IntegrityLedger
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_column = config["columns"]["group_id"]
    reference_column = config["columns"]["reference_id"]
    block_column = config["columns"]["cv_block_id"]
    graph = UnionFind()
    for row in s1[[group_column, reference_column]].itertuples(index=False, name=None):
        group_id = str(row[0])
        references = normalize_reference_ids(row[1])
        graph.add(f"G::{group_id}")
        for reference in references:
            graph.union(f"G::{group_id}", f"R::{reference}")
    groups = sorted(s1[group_column].astype(str).unique())
    roots = {group: graph.find(f"G::{group}") for group in groups}
    ordered_roots = {root: f"CVB{index:04d}" for index, root in enumerate(sorted(set(roots.values())), 1)}
    group_to_block = {group: ordered_roots[root] for group, root in roots.items()}
    result = s1.copy()
    result[block_column] = result[group_column].astype(str).map(group_to_block)
    registry = result.groupby([block_column, group_column], as_index=False).agg(
        record_count=(config["columns"]["record_id"], "size"),
        reference_ids=(reference_column, lambda values: ";".join(sorted({r for v in values for r in normalize_reference_ids(v)}))),
    )
    mapping_count = registry.groupby(group_column)[block_column].nunique()
    ledger.check("each_group_maps_to_one_reference_connected_block", bool(mapping_count.eq(1).all()), {
        "maximum_blocks_per_group": int(mapping_count.max())
    })
    return result, registry


def classify_group_label_status(s1_with_blocks: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    columns = config["columns"]
    label_counts = s1_with_blocks.groupby(columns["group_id"])[columns["label"]].agg(
        group_record_count="size", label_min="min", label_max="max", label_nunique="nunique"
    ).reset_index()
    label_counts["group_label_status"] = np.select(
        [
            (label_counts["label_nunique"] == 1) & (label_counts["label_min"] == 0),
            (label_counts["label_nunique"] == 1) & (label_counts["label_min"] == 1),
            label_counts["label_nunique"] > 1,
        ],
        ["stable_negative", "stable_positive", "mixed_uncertain"],
        default="invalid_label",
    )
    block_map = s1_with_blocks.groupby(columns["group_id"])[columns["cv_block_id"]].first()
    label_counts[columns["cv_block_id"]] = label_counts[columns["group_id"]].map(block_map)
    return label_counts


def candidate_geochemical_columns(frame: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    excluded = set(config["data_rules"]["metadata_features_excluded"])
    candidates: list[str] = []
    for column in frame.columns:
        if column in excluded:
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        original_nonmissing = frame[column].notna().sum()
        numeric_fraction = converted.notna().sum() / max(int(original_nonmissing), 1)
        if numeric_fraction >= float(config["data_rules"]["minimum_numeric_fraction"]):
            candidates.append(column)
    return candidates


def coerce_geochemistry(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.copy()
    rows = []
    for feature in features:
        original_nonmissing = int(result[feature].notna().sum())
        numeric = pd.to_numeric(result[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        negative_count = int((numeric < 0).sum())
        numeric = numeric.mask(numeric < 0)
        result[feature] = numeric
        rows.append({
            "feature": feature,
            "original_nonmissing": original_nonmissing,
            "numeric_nonmissing": int(numeric.notna().sum()),
            "coercion_or_negative_to_missing": original_nonmissing - int(numeric.notna().sum()),
            "negative_values_removed": negative_count,
        })
    return result, pd.DataFrame(rows)


def build_primary_and_challenge_cohorts(
    all_records: pd.DataFrame,
    group_status: pd.DataFrame,
    chemistry_features: list[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = config["columns"]
    status_map = group_status.set_index(columns["group_id"])["group_label_status"]
    work = all_records.copy()
    work["group_label_status"] = work[columns["group_id"]].map(status_map)
    hard_excluded = set(config["data_rules"]["direct_ore_features_excluded"]) | set(
        config["data_rules"]["iron_features_excluded"]
    )
    preliminary_features = [
        feature for feature in chemistry_features
        if feature not in hard_excluded
        and all_records[feature].isna().mean() <= float(config["data_rules"]["maximum_feature_missing_fraction"])
    ]
    if not preliminary_features:
        raise ValueError("No preliminary predictors are available for record-level missingness screening.")
    work["record_missing_fraction_candidate_geochemistry"] = work[preliminary_features].isna().mean(axis=1)
    work["record_missingness_screen_feature_count"] = len(preliminary_features)
    high_missing = work["record_missing_fraction_candidate_geochemistry"] > float(
        config["data_rules"]["maximum_record_missing_fraction"]
    )
    excluded = work.loc[high_missing].copy()
    retained = work.loc[~high_missing].copy()
    primary = retained[retained["group_label_status"].isin(["stable_negative", "stable_positive"])].copy()
    challenge = retained[retained["group_label_status"].eq("mixed_uncertain")].copy()
    invalid = retained[retained["group_label_status"].eq("invalid_label")].copy()
    if not invalid.empty:
        raise ValueError(f"Invalid-label records remain after source validation: {len(invalid)}")
    flow = pd.DataFrame([
        {"stage": "source_records", "records": len(work), "groups": work[columns["group_id"]].nunique()},
        {"stage": "excluded_high_missingness", "records": len(excluded), "groups": excluded[columns["group_id"]].nunique()},
        {"stage": "primary_stable_groups", "records": len(primary), "groups": primary[columns["group_id"]].nunique()},
        {"stage": "mixed_group_challenge", "records": len(challenge), "groups": challenge[columns["group_id"]].nunique()},
    ])
    return primary, challenge, excluded, flow


def screen_predictor_panel(
    primary: pd.DataFrame, chemistry_features: list[str], config: dict[str, Any]
) -> tuple[list[str], pd.DataFrame]:
    rules = config["data_rules"]
    direct = set(rules["direct_ore_features_excluded"])
    iron_excluded = set(rules["iron_features_excluded"])
    threshold = float(rules["maximum_feature_missing_fraction"])
    rows = []
    retained = []
    for feature in chemistry_features:
        missing = float(primary[feature].isna().mean())
        reason = "retained"
        if feature in direct:
            reason = "excluded_direct_ore_feature"
        elif feature in iron_excluded:
            reason = "excluded_nonharmonized_total_iron"
        elif missing > threshold:
            reason = "excluded_feature_missingness"
        elif primary[feature].nunique(dropna=True) < 2:
            reason = "excluded_constant_or_empty"
        if reason == "retained":
            retained.append(feature)
        rows.append({"feature": feature, "primary_missing_fraction": missing, "decision": reason})
    if not retained:
        raise ValueError("No predictors remained after pre-specified screening.")
    return retained, pd.DataFrame(rows)


class GeochemicalKNNPreprocessor:
    """Fold-local nonnegative log1p, robust scaling, and KNN completion."""

    def __init__(self, features: list[str], n_neighbors: int = 5):
        self.features = list(features)
        self.n_neighbors = int(n_neighbors)
        self.log_scaler = RobustScaler(quantile_range=(25.0, 75.0))
        self.imputer = KNNImputer(n_neighbors=self.n_neighbors, weights="distance", keep_empty_features=False)

    def _numeric(self, frame: pd.DataFrame) -> pd.DataFrame:
        matrix = frame[self.features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        return matrix.mask(matrix < 0)

    def fit(self, frame: pd.DataFrame) -> "GeochemicalKNNPreprocessor":
        numeric = self._numeric(frame)
        empty = numeric.columns[numeric.notna().sum().eq(0)].tolist()
        if empty:
            raise ValueError(f"Training partition contains all-missing predictors: {empty}")
        logged = np.log1p(numeric)
        scaled = self.log_scaler.fit_transform(logged)
        self.imputer.fit(scaled)
        transformed = self.imputer.transform(scaled)
        if transformed.shape[1] != len(self.features) or not np.isfinite(transformed).all():
            raise RuntimeError("Fold-local KNN preprocessing produced an invalid training matrix.")
        self.fitted_ = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not getattr(self, "fitted_", False):
            raise RuntimeError("GeochemicalKNNPreprocessor must be fitted on training data before transform.")
        numeric = self._numeric(frame)
        scaled = self.log_scaler.transform(np.log1p(numeric))
        completed = self.imputer.transform(scaled)
        if completed.shape[1] != len(self.features) or not np.isfinite(completed).all():
            raise RuntimeError("Fold-local KNN preprocessing produced non-finite validation values.")
        return completed

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    def completed_raw(self, frame: pd.DataFrame) -> pd.DataFrame:
        completed = self.transform(frame)
        logged = self.log_scaler.inverse_transform(completed)
        raw = np.maximum(np.expm1(logged), 0.0)
        return pd.DataFrame(raw, columns=self.features, index=frame.index)


def build_missingness_and_quality_panel(
    primary: pd.DataFrame,
    features: list[str],
    config: dict[str, Any],
    paths: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Define the pre-specified data-quality sensitivity panel without model performance."""
    label = config["columns"]["label"]
    missing_rows = []
    for feature in features:
        by_label = primary.groupby(label)[feature].apply(lambda values: float(values.isna().mean()))
        negative = float(by_label.get(0, np.nan))
        positive = float(by_label.get(1, np.nan))
        missing_rows.append({
            "feature": feature,
            "missing_fraction": float(primary[feature].isna().mean()),
            "missing_fraction_label_0": negative,
            "missing_fraction_label_1": positive,
            "absolute_label_missingness_gap": abs(positive - negative),
        })
    missingness = pd.DataFrame(missing_rows)
    save_table(missingness, paths["audit"] / "missingness_by_label_audit.csv")

    rules = config["shap"]["quality_panel"]
    repeats = int(rules["mask_repeats"])
    fraction = float(rules["mask_fraction"])
    recovery_rows = []
    for repeat in range(1, repeats + 1):
        masked = primary.copy()
        held_out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        rng = np.random.default_rng(int(config["validation"]["base_seed"]) + 80000 + repeat)
        for feature in features:
            observed_indices = np.flatnonzero(masked[feature].notna().to_numpy())
            count = min(len(observed_indices), max(5, int(round(len(observed_indices) * fraction))))
            if len(observed_indices) < 10 or count < 5:
                held_out[feature] = (np.array([], dtype=int), np.array([], dtype=float))
                continue
            selected = rng.choice(observed_indices, size=count, replace=False)
            truth = masked.iloc[selected][feature].to_numpy(float)
            masked.loc[masked.index[selected], feature] = np.nan
            held_out[feature] = (selected, truth)
        processor = GeochemicalKNNPreprocessor(
            features, n_neighbors=int(config["data_rules"]["knn_neighbors"])
        )
        processor.fit(masked)
        completed = processor.completed_raw(masked)
        for feature in features:
            indices, truth = held_out[feature]
            if len(indices) < 5:
                pearson = np.nan
                normalized_mae = np.nan
            else:
                predicted = completed.iloc[indices][feature].to_numpy(float)
                pearson = float(np.corrcoef(truth, predicted)[0, 1]) if np.std(truth) > 0 and np.std(predicted) > 0 else np.nan
                scale = float(np.subtract(*np.quantile(truth, [0.75, 0.25])))
                normalized_mae = float(np.mean(np.abs(truth - predicted)) / max(scale, 1e-12))
            recovery_rows.append({
                "repeat": repeat, "feature": feature, "masked_count": len(indices),
                "masked_pearson_r": pearson, "normalized_mae": normalized_mae,
            })
    recovery_detail = pd.DataFrame(recovery_rows)
    recovery = recovery_detail.groupby("feature", as_index=False).agg(
        masked_count_total=("masked_count", "sum"),
        masked_pearson_r_mean=("masked_pearson_r", "mean"),
        masked_pearson_r_min=("masked_pearson_r", "min"),
        normalized_mae_mean=("normalized_mae", "mean"),
    )
    save_table(recovery_detail, paths["audit"] / "masking_recovery_detail.csv")
    save_table(recovery, paths["audit"] / "masking_recovery_summary.csv")
    audit = missingness.merge(recovery, on="feature", how="left", validate="one_to_one")
    audit["passes_missingness"] = audit["missing_fraction"].le(float(rules["maximum_missing_fraction"]))
    audit["passes_label_gap"] = audit["absolute_label_missingness_gap"].le(float(rules["maximum_label_missingness_gap"]))
    audit["passes_masked_recovery"] = audit["masked_pearson_r_mean"].ge(float(rules["minimum_masked_pearson_r"]))
    audit["retained_in_quality_panel"] = audit[[
        "passes_missingness", "passes_label_gap", "passes_masked_recovery"
    ]].all(axis=1)
    save_table(audit, paths["audit"] / "quality_controlled_panel_audit.csv")
    quality_features = audit.loc[audit["retained_in_quality_panel"], "feature"].tolist()
    (paths["processed"] / "quality_controlled_feature_list.txt").write_text(
        "\n".join(quality_features), encoding="utf-8"
    )
    return audit, recovery


def run_data_pipeline(config_path: str | Path) -> dict[str, Any]:
    config, root, paths = load_config(config_path)
    run_id = f"{config['analysis_revision']}::data_contract"
    ledger = IntegrityLedger(paths["audit"] / "preflight_and_integrity_checks.json", run_id=run_id)
    label_note = root / "config" / "label_definition_note.json"
    locked_file = root / config["shap"]["locked_bridge_feature_file"]
    input_files = [
        resolve_package_path(root, config["input"]["supplementary_table_s1"]),
        resolve_package_path(root, config["input"]["supplementary_table_s2"]), label_note, locked_file, Path(config_path),
    ]
    for path in input_files:
        ledger.check(f"input_exists::{path.name}", path.exists(), str(path))
    manifest = create_run_manifest(root, input_files)
    save_json(paths["audit"] / "input_hashes.json", manifest)
    s1, s2 = read_source_tables(config, root)
    s1, s2 = validate_source_tables(s1, s2, config, ledger)
    all_records, block_registry = build_reference_connected_blocks_for_all_records(s1, config, ledger)
    group_status = classify_group_label_status(all_records, config)
    chemistry = candidate_geochemical_columns(all_records, config)
    all_records, value_audit = coerce_geochemistry(all_records, chemistry)
    primary, challenge, excluded, flow = build_primary_and_challenge_cohorts(
        all_records, group_status, chemistry, config
    )
    features, feature_manifest = screen_predictor_panel(primary, chemistry, config)
    quality_panel_audit, _ = build_missingness_and_quality_panel(primary, features, config, paths)
    locked = read_lines(locked_file)
    ledger.check("locked_bridge_feature_file_has_exactly_nine_unique_features", len(locked) == 9 and len(set(locked)) == 9, locked)
    ledger.check("locked_bridge_features_exist_in_source_schema", set(locked).issubset(all_records.columns), {
        "missing": sorted(set(locked).difference(all_records.columns))
    })
    stable_blocks = set(primary[config["columns"]["cv_block_id"]])
    mixed_blocks = set(challenge[config["columns"]["cv_block_id"]])
    save_table(block_registry, paths["audit"] / "reference_connected_block_registry.csv")
    save_table(group_status, paths["audit"] / "group_label_status.csv")
    save_table(flow, paths["audit"] / "cohort_flow.csv")
    save_table(value_audit, paths["audit"] / "geochemical_value_audit.csv")
    save_table(feature_manifest, paths["processed"] / "feature_manifest.csv")
    save_table(primary, paths["processed"] / "primary_model_cohort_with_nan.csv")
    save_table(challenge, paths["processed"] / "mixed_group_challenge_cohort_with_nan.csv")
    save_table(excluded, paths["processed"] / "excluded_high_missingness_records.csv")
    (paths["processed"] / "primary_feature_list.txt").write_text("\n".join(features), encoding="utf-8")
    cohort_hash = sha256_payload({
        "records": primary[config["columns"]["record_id"]].tolist(),
        "features": features,
        "labels": primary[config["columns"]["label"]].tolist(),
        "blocks": primary[config["columns"]["cv_block_id"]].tolist(),
    })
    summary = {
        "schema_version": config["schema_version"],
        "source_records": len(all_records),
        "source_groups": int(all_records[config["columns"]["group_id"]].nunique()),
        "reference_connected_blocks": int(all_records[config["columns"]["cv_block_id"]].nunique()),
        "primary_records": len(primary),
        "primary_groups": int(primary[config["columns"]["group_id"]].nunique()),
        "primary_blocks": int(primary[config["columns"]["cv_block_id"]].nunique()),
        "challenge_records": len(challenge),
        "challenge_groups": int(challenge[config["columns"]["group_id"]].nunique()),
        "challenge_blocks_overlapping_primary_blocks": len(stable_blocks.intersection(mixed_blocks)),
        "excluded_records": len(excluded),
        "retained_predictors": len(features),
        "quality_controlled_predictors": int(quality_panel_audit["retained_in_quality_panel"].sum()),
        "cohort_hash": cohort_hash,
        "label_note_sha256": sha256_file(label_note),
        "locked_bridge_sha256": sha256_file(locked_file),
    }
    save_json(paths["processed"] / "data_pipeline_summary.json", summary)
    stage_complete(
        paths["logs"] / "stage_01_data_pipeline.complete.json",
        "data_pipeline", {"config": sha256_file(config_path)},
        [
            paths["processed"] / "primary_model_cohort_with_nan.csv",
            paths["processed"] / "mixed_group_challenge_cohort_with_nan.csv",
            paths["processed"] / "primary_feature_list.txt",
            paths["processed"] / "data_pipeline_summary.json",
        ],
        analysis_revision=config["analysis_revision"],
        analysis_protocol_hash=None,
    )
    return summary
