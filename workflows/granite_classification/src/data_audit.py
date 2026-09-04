"""Data-quality audit, primary-domain metrics and SHAP violin figures for Task B v5.

This module is additive.  It does not alter the source-connected, class-neutral
modelling pipeline.  It reads the same the Dataset and Geological Groups worksheets in Supplementary Table S1 inputs and the formal pipeline
outputs, then exports a traceable audit table, a primary-domain metric table and
class-specific SHAP violin figures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402


CLASSES = ("I", "A", "S")
CLASS_COLORS = {"I": "#0072B2", "A": "#E69F00", "S": "#009E73"}

DEFAULT_PRIMARY_GROUP_TYPES = {
    "Named pluton or intrusive unit",
    "Pluton",
    "Granite body",
    "Granodiorite body",
    "Granitoid body",
    "Batholith",
    "Intrusion",
    "Porphyritic granodiorite body",
}

DEFAULT_OUT_OF_DOMAIN_GROUP_TYPES = {
    "Composite or parent geological unit",
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
    "Granite complex",
    "Intrusive complex",
    "Batholith and evolved granite suite",
    "Granite suite",
    "Deposit-scale intrusive suite",
    "Deposit-scale granitoid suite",
    "Deposit-scale granitic-vein suite",
    "Deposit-scale intrusion",
    "Deposit-scale granite-porphyry suite",
    "Granite-porphyry suite",
    "Granite-porphyry body",
    "Granodiorite-porphyry body",
}


def _resolve(project_root: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (project_root / candidate).resolve()


def _read_s1_s2(project_root: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    s1 = pd.read_excel(
        _resolve(project_root, config["paths"]["s1"]), sheet_name="Dataset", header=1
    )
    s2 = pd.read_excel(
        _resolve(project_root, config["paths"]["s2"]), sheet_name="Geological Groups", header=1
    )
    s1["Granite type"] = s1["Granite type"].astype(str).str.strip().str.upper()
    return s1, s2


def _domain_map(s2: pd.DataFrame, config: dict[str, Any]) -> dict[str, str]:
    primary = set(config.get("domain", {}).get("primary_group_types", DEFAULT_PRIMARY_GROUP_TYPES))
    out = set(config.get("domain", {}).get("out_of_domain_group_types", DEFAULT_OUT_OF_DOMAIN_GROUP_TYPES))
    mapping: dict[str, str] = {}
    for _, row in s2.iterrows():
        group_type = str(row.get("Group Type", "")).strip()
        if group_type in primary:
            mapping[str(row["Geological Group ID"]).strip()] = "primary"
        elif group_type in out:
            mapping[str(row["Geological Group ID"]).strip()] = "out_of_domain"
        else:
            mapping[str(row["Geological Group ID"]).strip()] = "unclassified"
    return mapping


def run_data_audit(project_root: Path, config: dict[str, Any]) -> pd.DataFrame:
    """Export one audit row per flagged geological group."""
    s1, s2 = _read_s1_s2(project_root, config)
    output_root = _resolve(project_root, config["paths"]["output_root"])
    audit_dir = output_root / "00_Audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    domain = _domain_map(s2, config)
    group_name = dict(zip(
        s2["Geological Group ID"].astype(str),
        s2.get("Geological Group Name", s2["Geological Group ID"]).astype(str),
    ))

    # Raw group composition.
    raw = s1.copy()
    raw["Geological Group ID"] = raw["Geological Group ID"].astype(str).str.strip()
    group_size = raw.groupby("Geological Group ID").size()
    group_types = raw.groupby("Geological Group ID")["Granite type"].agg(
        lambda values: sorted({v for v in values.dropna() if v != "NAN"})
    )
    group_n_types = group_types.map(len)
    missing_label = raw["Granite type"].eq("NAN") | raw["Granite type"].isna()

    rows: list[dict[str, Any]] = []
    for group in group_size.index:
        flags: list[str] = []
        detail: list[str] = []

        if group_n_types.get(group, 0) > 1:
            flags.append("mixed_type_group")
            detail.append("types=" + ",".join(group_types.get(group, [])))
        if domain.get(group) == "out_of_domain":
            flags.append("out_of_domain_group")
            detail.append("group_type=" + str(
                s2.loc[s2["Geological Group ID"].astype(str).eq(group), "Group Type"].iloc[0]
            ))
        if group_size.get(group, 0) < 5:
            flags.append("low_sample_group")
            detail.append(f"n={int(group_size[group])}")
        if int(missing_label[raw["Geological Group ID"].eq(group)].sum()) > 0:
            flags.append("missing_label")
            detail.append(f"n_missing={int(missing_label[raw['Geological Group ID'].eq(group)].sum())}")

        if flags:
            rows.append({
                "Geological Group ID": group,
                "Geological Group Name": group_name.get(group, ""),
                "flag": "; ".join(flags),
                "detail": "; ".join(detail),
                "n_records": int(group_size[group]),
                "n_types": int(group_n_types.get(group, 0)),
                "domain": domain.get(group, "unclassified"),
                "recommended_action": "manual_geological_review_before_final_run",
                "evidence": "Supplementary Table the Dataset and Geological Groups worksheets in Supplementary Table S1",
            })

    # Missingness-based abstention groups.
    predictions_path = output_root / "04_SHAP" / "granite_type_full_oof_predictions_with_ids.csv"
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path)
        predictions["Geological Group ID"] = predictions["Geological Group ID"].astype(str).str.strip()
        abstain = predictions[~predictions["valid_oof"].astype(bool)]
        if len(abstain):
            abstain_counts = abstain.groupby("Geological Group ID").size()
            for group, count in abstain_counts.items():
                rows.append({
                    "Geological Group ID": group,
                    "Geological Group Name": group_name.get(group, ""),
                    "flag": "high_missingness_abstain",
                    "detail": f"n_abstained={int(count)}",
                    "n_records": int(group_size.get(group, 0)),
                    "n_types": int(group_n_types.get(group, 0)),
                    "domain": domain.get(group, "unclassified"),
                    "recommended_action": "review_missingness_pattern",
                    "evidence": "granite_type_full_oof_predictions_with_ids.csv",
                })

    # Cross-model consistent S-type misclassification.
    s_error_path = output_root / "02_Model_Comparison" / "S_type_source_block_error_audit.csv"
    if s_error_path.exists():
        s_errors = pd.read_csv(s_error_path)
        s_errors["Geological Group ID"] = s_errors["Geological Group ID"].astype(str).str.strip()
        summary = s_errors.groupby("Geological Group ID").agg(
            n_models=("model", "nunique"),
            median_S_recall=("S_recall_within_group", "median"),
            n_S=("n_S", "max"),
        ).reset_index()
        for _, row in summary.iterrows():
            if row["n_models"] >= 4 and row["median_S_recall"] < 0.5:
                rows.append({
                    "Geological Group ID": row["Geological Group ID"],
                    "Geological Group Name": group_name.get(row["Geological Group ID"], ""),
                    "flag": "consistent_cross_model_S_misclassification",
                    "detail": (
                        f"n_models={int(row['n_models'])}; "
                        f"median_S_recall={row['median_S_recall']:.3f}; n_S={int(row['n_S'])}"
                    ),
                    "n_records": int(group_size.get(row["Geological Group ID"], 0)),
                    "n_types": int(group_n_types.get(row["Geological Group ID"], 0)),
                    "domain": domain.get(row["Geological Group ID"], "unclassified"),
                    "recommended_action": "check_label_against_original_literature",
                    "evidence": "S_type_source_block_error_audit.csv",
                })

    frame = pd.DataFrame(rows).drop_duplicates(
        subset=["Geological Group ID", "flag", "detail"], keep="first"
    ).sort_values(["flag", "Geological Group ID"]).reset_index(drop=True)
    frame.to_csv(audit_dir / "data_audit_log_v5.csv", index=False)
    return frame


def compute_primary_domain_metrics(project_root: Path, config: dict[str, Any]) -> pd.DataFrame:
    """Recompute group-stratum macro-F1 restricted to the primary domain."""
    _, s2 = _read_s1_s2(project_root, config)
    domain = _domain_map(s2, config)
    output_root = _resolve(project_root, config["paths"]["output_root"])
    comparison_dir = output_root / "02_Model_Comparison"
    source = comparison_dir / "all_models_nested_source_block_group_strata_oof.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing group-stratum OOF file: {source}")
    frame = pd.read_csv(source)
    frame["Geological Group ID"] = frame["Geological Group ID"].astype(str).str.strip()
    frame["domain"] = frame["Geological Group ID"].map(domain).fillna("unclassified")
    primary = frame[frame["domain"].eq("primary")].copy()

    def macro_f1(subset: pd.DataFrame) -> float:
        labels = [CLASS_TO_INT_ORDER[x] for x in ["I", "A", "S"]]
        per_class = []
        for code in labels:
            tp = int(((subset["true_code"] == code) & (subset["predicted_code"] == code)).sum())
            fp = int(((subset["true_code"] != code) & (subset["predicted_code"] == code)).sum())
            fn = int(((subset["true_code"] == code) & (subset["predicted_code"] != code)).sum())
            denom = 2 * tp + fp + fn
            per_class.append(2 * tp / denom if denom > 0 else 0.0)
        return float(np.mean(per_class))

    CLASS_TO_INT_ORDER = {"I": 0, "A": 1, "S": 2}
    metrics = primary.groupby("model").apply(
        lambda subset: pd.Series({
            "primary_domain_macro_f1": macro_f1(subset),
            "n_strata": int(len(subset)),
            "n_groups": int(subset["Geological Group ID"].nunique()),
        })
    ).reset_index()
    metrics.to_csv(
        comparison_dir / "primary_domain_group_stratum_metrics_v5.csv", index=False
    )
    return metrics


def compute_oof_probability_audit(
    project_root: Path, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Audit canonical OOF probabilities without recalibrating or selecting a model.

    The same predictions are summarized at record level and at the primary
    geological-group × reported-type stratum level.  These diagnostics are
    descriptive evidence about probability reliability; they do not alter the
    frozen v5 model, thresholds, held-out set or SHAP values.
    """
    output_root = _resolve(project_root, config["paths"]["output_root"])
    shap_dir = output_root / "04_SHAP"
    audit_dir = output_root / "00_Audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    source = shap_dir / "granite_type_full_oof_predictions_with_ids.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing canonical OOF prediction file: {source}")

    frame = pd.read_csv(source)
    required = {
        "Record ID", "Geological Group ID", "Reference-connected block",
        "reported_granite_type", "P_I", "P_A", "P_S", "valid_oof",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"OOF prediction file lacks required columns: {missing}")
    valid = frame.loc[frame["valid_oof"].astype(bool)].copy()
    if valid["Record ID"].duplicated().any():
        raise ValueError("Canonical OOF probability audit requires one row per Record ID.")

    class_to_int = {name: index for index, name in enumerate(CLASSES)}
    valid["true_code"] = valid["reported_granite_type"].map(class_to_int)
    if valid["true_code"].isna().any():
        raise ValueError("Unexpected granite label in canonical OOF predictions.")
    probability_columns = [f"P_{name}" for name in CLASSES]
    row_sums = valid[probability_columns].sum(axis=1).to_numpy(float)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("OOF class probabilities do not sum to one within tolerance.")

    group = (
        valid.groupby(
            ["Geological Group ID", "Reference-connected block", "reported_granite_type"],
            as_index=False,
        )
        .agg(**{
            **{column: (column, "mean") for column in probability_columns},
            "n_records": ("Record ID", "size"),
        })
    )
    group["true_code"] = group["reported_granite_type"].map(class_to_int).astype(int)

    def summary_row(data: pd.DataFrame, unit: str) -> dict[str, Any]:
        probabilities = data[probability_columns].to_numpy(float)
        truth = data["true_code"].to_numpy(int)
        one_hot = np.eye(len(CLASSES))[truth]
        clipped = np.clip(probabilities, 1e-15, 1.0)
        confidence = probabilities.max(axis=1)
        correct = (probabilities.argmax(axis=1) == truth).astype(float)
        edges = np.linspace(0.0, 1.0, int(config.get("probability_audit", {}).get("n_bins", 10)) + 1)
        ece = 0.0
        for bin_index in range(len(edges) - 1):
            low, high = edges[bin_index], edges[bin_index + 1]
            keep = (confidence >= low) & (
                (confidence < high) if bin_index < len(edges) - 2 else (confidence <= high)
            )
            if keep.any():
                ece += keep.mean() * abs(confidence[keep].mean() - correct[keep].mean())
        return {
            "evaluation_unit": unit,
            "probability_scale": "uncalibrated_oof",
            "n_units": int(len(data)),
            "n_geological_groups": int(data["Geological Group ID"].nunique()),
            "n_source_blocks": int(data["Reference-connected block"].nunique()),
            "multiclass_log_loss": float(-np.log(clipped[np.arange(len(truth)), truth]).mean()),
            "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
            "top_label_ece": float(ece),
            **{
                f"support_{name}": int((data["reported_granite_type"] == name).sum())
                for name in CLASSES
            },
        }

    summary = pd.DataFrame([
        summary_row(valid, "record"),
        summary_row(group, "geological_group_by_reported_type_stratum"),
    ])

    n_bins = int(config.get("probability_audit", {}).get("n_bins", 10))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    reliability_rows: list[dict[str, Any]] = []
    for unit, data in [("record", valid), ("geological_group_by_reported_type_stratum", group)]:
        truth = data["true_code"].to_numpy(int)
        for class_index, class_name in enumerate(CLASSES):
            scores = data[f"P_{class_name}"].to_numpy(float)
            observed = (truth == class_index).astype(float)
            for bin_index in range(n_bins):
                low, high = edges[bin_index], edges[bin_index + 1]
                keep = (scores >= low) & (
                    (scores < high) if bin_index < n_bins - 1 else (scores <= high)
                )
                if keep.any():
                    reliability_rows.append({
                        "evaluation_unit": unit,
                        "class": class_name,
                        "bin": bin_index + 1,
                        "lower_probability": low,
                        "upper_probability": high,
                        "n_units": int(keep.sum()),
                        "mean_predicted_probability": float(scores[keep].mean()),
                        "observed_class_fraction": float(observed[keep].mean()),
                    })
    reliability = pd.DataFrame(reliability_rows)
    support = (
        valid.groupby("reported_granite_type", as_index=False)
        .agg(
            records=("Record ID", "size"),
            geological_groups=("Geological Group ID", "nunique"),
            source_blocks=("Reference-connected block", "nunique"),
        )
        .rename(columns={"reported_granite_type": "class"})
    )
    summary.to_csv(audit_dir / "canonical_oof_probability_diagnostics_v5_1.csv", index=False)
    reliability.to_csv(audit_dir / "canonical_oof_reliability_bins_v5_1.csv", index=False)
    support.to_csv(audit_dir / "canonical_oof_class_support_v5_1.csv", index=False)
    return summary, reliability, support


def _display_feature(name: str) -> str:
    return {
        "SiO2": r"SiO$_2$",
        "TiO2": r"TiO$_2$",
        "Al2O3": r"Al$_2$O$_3$",
        "Fe2O3": r"Fe$_2$O$_3$",
        "P2O5": r"P$_2$O$_5$",
        "Na2O": r"Na$_2$O",
        "K2O": r"K$_2$O",
    }.get(name, name)


def _top_shap_features(shap_frame: pd.DataFrame, top_n: int) -> list[str]:
    shap_columns = [column for column in shap_frame.columns if column.startswith("SHAP::")]
    importance = shap_frame[shap_columns].abs().mean().sort_values(ascending=False)
    return [column.replace("SHAP::", "") for column in importance.head(top_n).index]


def plot_shap_violin(project_root: Path, config: dict[str, Any], top_n: int | None = None) -> None:
    """Generate class-specific SHAP violin figures for the final tree model."""
    output_root = _resolve(project_root, config["paths"]["output_root"])
    shap_dir = output_root / "04_SHAP"
    figure_dir = output_root / "06_Figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    top_n = top_n or int(config["shap"]["top_features_per_class"])

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })

    figure, axes = plt.subplots(1, 3, figsize=(9.2, 4.4), constrained_layout=True)
    for panel, (axis, class_name) in enumerate(zip(axes, CLASSES)):
        path = shap_dir / f"granite_type_{class_name}_oof_shap_with_ids.csv"
        if not path.exists():
            axis.text(0.5, 0.5, "SHAP output not available", ha="center", va="center",
                      transform=axis.transAxes)
            axis.axis("off")
            continue
        shap_frame = pd.read_csv(path)
        features = _top_shap_features(shap_frame, top_n)
        melted = []
        for feature in features:
            values = pd.to_numeric(shap_frame[f"SHAP::{feature}"], errors="coerce").dropna()
            melted.append(pd.DataFrame({
                "feature": feature,
                "SHAP value": values.to_numpy(),
            }))
        if not melted:
            axis.axis("off")
            continue
        plot_frame = pd.concat(melted, ignore_index=True)
        order = features[::-1]
        sns.violinplot(
            data=plot_frame, y="feature", x="SHAP value", order=order,
            inner=None, linewidth=0.45, color=CLASS_COLORS[class_name], alpha=0.72, ax=axis,
        )
        axis.axvline(0, color="#888888", linewidth=0.75, zorder=0)
        axis.set_yticks(range(len(order)))
        axis.set_yticklabels([_display_feature(feature) for feature in order])
        axis.set_xlabel("SHAP value")
        axis.set_title(f"{class_name}-type", fontsize=8.8, pad=5)
        axis.grid(axis="x", linestyle=":", color="#D9D9D9", linewidth=0.6, alpha=0.85)
        axis.text(-0.14, 1.04, f"({chr(97 + panel)})", transform=axis.transAxes,
                  fontsize=8.8, fontweight="bold")

    stem = figure_dir / "Fig_granite_class_specific_SHAP_violin"
    figure.savefig(f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(f"{stem}.svg", bbox_inches="tight")
    figure.savefig(f"{stem}.png", bbox_inches="tight", dpi=600)
    temporary_tiff = stem.parent / f".{stem.name}.tmp.tiff"
    figure.savefig(temporary_tiff, bbox_inches="tight", dpi=600)
    temporary_tiff.replace(stem.with_suffix(".tiff"))
    plt.close(figure)


def run_all_audit(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    audit = run_data_audit(project_root, config)
    primary_metrics = compute_primary_domain_metrics(project_root, config)
    probability_summary, _, probability_support = compute_oof_probability_audit(
        project_root, config
    )
    plot_shap_violin(project_root, config)
    return {
        "data_audit_rows": int(len(audit)),
        "data_audit_flags": audit["flag"].value_counts().to_dict() if len(audit) else {},
        "primary_domain_metrics": primary_metrics.to_dict(orient="records"),
        "oof_probability_diagnostics": probability_summary.to_dict(orient="records"),
        "oof_class_support": probability_support.to_dict(orient="records"),
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    config_path = root / "config" / "granite_classification_config.json"
    with config_path.open("r", encoding="utf-8") as stream:
        configuration = json.load(stream)
    print(json.dumps(run_all_audit(root, configuration), ensure_ascii=False, indent=2, default=str))
