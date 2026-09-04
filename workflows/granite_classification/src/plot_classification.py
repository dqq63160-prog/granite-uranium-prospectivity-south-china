from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde


CLASSES = ("I", "A", "S")
BLUE_RED = LinearSegmentedColormap.from_list(
    "feature_value_blue_white_red",
    ["#2F56C6", "#87A5EA", "#F7F7F7", "#F08A72", "#B91F1F"],
    N=256,
)
MODEL_COLORS = {
    "RF": "#78A6C8", "SVM": "#8FC1A9", "MLP": "#C6A0C6", "XGBoost": "#D97A5B"
}


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.4,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def display_feature(name: str) -> str:
    return {
        "SiO2": r"SiO$_2$", "TiO2": r"TiO$_2$", "Al2O3": r"Al$_2$O$_3$",
        "Fe2O3": r"Fe$_2$O$_3$", "P2O5": r"P$_2$O$_5$",
        "Na2O": r"Na$_2$O", "K2O": r"K$_2$O",
    }.get(name, name)


def save_vector_and_png(figure: plt.Figure, folder: Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png", "tiff"):
        destination = folder / f"{name}.{extension}"
        if extension == "tiff":
            # Pillow's LZW TIFF writer crashes under the locked Python 3.13
            # Windows runtime. Write an uncompressed 600-dpi TIFF atomically;
            # PDF/SVG remain the preferred editable publication formats.
            temporary = folder / f".{name}.tmp.tiff"
            figure.savefig(
                temporary, dpi=600, bbox_inches="tight", facecolor="white"
            )
            temporary.replace(destination)
        else:
            figure.savefig(
                destination,
                dpi=600 if extension == "png" else None,
                bbox_inches="tight", facecolor="white",
            )
    plt.close(figure)


def vector_colorbar(axis: plt.Axes, cmap: Any, label: str) -> None:
    bar = axis.inset_axes([1.025, 0.12, 0.023, 0.76])
    bar.set_xlim(0, 1)
    bar.set_ylim(0, 1)
    for index in range(80):
        y0 = index / 80
        bar.add_patch(Rectangle(
            (0, y0), 1, 1 / 80 + 0.001,
            facecolor=cmap((index + 0.5) / 80), edgecolor="none",
        ))
    bar.set_xticks([])
    bar.set_yticks([0, 1])
    bar.set_yticklabels(["Low", "High"])
    bar.yaxis.tick_right()
    bar.tick_params(axis="y", length=2, pad=2, width=0.6)
    bar.set_ylabel(label, rotation=90, labelpad=9, fontsize=7.2)
    bar.yaxis.set_label_position("right")
    for spine in bar.spines.values():
        spine.set_linewidth(0.45)


def _local_feature_colour(x: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    value_low, value_high = np.nanquantile(values, [0.05, 0.95])
    scale = max(value_high - value_low, 1e-12)
    normalized = np.clip((values - value_low) / scale, 0, 1)
    span = max(np.nanquantile(x, 0.95) - np.nanquantile(x, 0.05), 1e-9)
    bandwidth = max(span * 0.10, np.std(x) * 0.08, 1e-9)
    colours = np.empty(len(grid), dtype=float)
    for index, location in enumerate(grid):
        weight = np.exp(-0.5 * ((x - location) / bandwidth) ** 2)
        colours[index] = np.average(normalized, weights=weight) if weight.sum() > 1e-10 else 0.5
    return colours


def ridge_panel(
    axis: plt.Axes,
    shap_frame: pd.DataFrame,
    value_frame: pd.DataFrame,
    features: list[str],
    class_name: str,
) -> None:
    ordered = list(reversed(features))
    for row, feature in enumerate(ordered):
        shap_column = f"SHAP::{feature}"
        value_column = f"imputed::{feature}"
        pair = shap_frame[["Record ID", shap_column]].merge(
            value_frame[["Record ID", value_column]], on="Record ID", how="inner"
        ).dropna()
        pair.columns = ["Record ID", "shap", "value"]
        pair = pair[np.isfinite(pair["shap"]) & np.isfinite(pair["value"])]
        if len(pair) < 25 or pair["shap"].nunique() < 4:
            continue
        x = pair["shap"].to_numpy(float)
        values = pair["value"].to_numpy(float)
        low, high = np.nanquantile(x, [0.003, 0.997])
        if not np.isfinite(low) or not np.isfinite(high) or low == high:
            continue
        grid = np.linspace(low, high, 260)
        try:
            density = gaussian_kde(x)(grid)
        except Exception:
            continue
        density = density / max(density.max(), 1e-12) * 0.225
        local_colour = _local_feature_colour(x, values, grid)
        # Expand both ends of the feature-value scale so high values remain visibly red
        # and low values visibly blue without changing the sign or position of SHAP values.
        local_colour = np.clip(0.5 + 1.20 * (local_colour - 0.5), 0.0, 1.0)

        # Each coloured strip is bounded by the same KDE outline. No fill can cross the black boundary.
        for segment in range(len(grid) - 1):
            xs = grid[segment:segment + 2]
            ds = density[segment:segment + 2] * 0.985
            colour = BLUE_RED(float(np.mean(local_colour[segment:segment + 2])))
            axis.fill_between(
                xs, row - ds, row + ds, facecolor=colour,
                edgecolor="none", alpha=0.98, zorder=2,
            )
        axis.plot(grid, row + density, color="#222222", linewidth=0.55, zorder=4)
        axis.plot(grid, row - density, color="#222222", linewidth=0.55, zorder=4)
        axis.axhline(row, color="#222222", linewidth=0.34, zorder=1)

    axis.axvline(0, color="#888888", linewidth=0.75, zorder=0)
    axis.set_yticks(range(len(ordered)))
    axis.set_yticklabels([display_feature(feature) for feature in ordered])
    axis.set_ylim(-0.75, len(ordered) - 0.25)
    axis.grid(axis="x", linestyle=":", color="#D9D9D9", linewidth=0.6, alpha=0.85)
    axis.set_xlabel(f"SHAP value for {class_name}-type output", fontsize=7.8)
    axis.set_title(f"{class_name}-type", fontsize=8.8, pad=5)
    axis.tick_params(axis="both", length=2.4, width=0.65, pad=2)
    vector_colorbar(axis, BLUE_RED, "Feature value")


def plot_shap_ridgeline(shap_dir: Path, figure_dir: Path, top_n: int) -> None:
    source = pd.read_csv(figure_dir / "Fig_granite_SHAP_source_data.csv")
    figure, axes = plt.subplots(1, 3, figsize=(9.2, 3.6), constrained_layout=True)
    for panel, (axis, class_name) in enumerate(zip(axes, CLASSES)):
        subset = source[source["class"].eq(class_name)].copy()
        features = (
            subset[["feature", "feature_rank"]].drop_duplicates()
            .sort_values("feature_rank").head(top_n)["feature"].tolist()
        )
        shap_values = subset.pivot(index="Record ID", columns="feature", values="SHAP value").reset_index()
        shap_values = shap_values.rename(columns=lambda value: value if value == "Record ID" else f"SHAP::{value}")
        values = subset.pivot(
            index="Record ID", columns="feature", values="imputed feature value"
        ).reset_index()
        values = values.rename(columns=lambda value: value if value == "Record ID" else f"imputed::{value}")
        ridge_panel(axis, shap_values, values, features, class_name)
        axis.text(-0.14, 1.04, f"({chr(97 + panel)})", transform=axis.transAxes,
                  fontsize=8.8, fontweight="bold")
    save_vector_and_png(figure, figure_dir, "Fig_granite_class_specific_SHAP_ridgeline")


def plot_model_comparison(comparison_dir: Path, figure_dir: Path) -> None:
    metrics = pd.read_csv(comparison_dir / "model_comparison_pooled_metrics.csv")
    ranking_path = comparison_dir / "model_comprehensive_ranking.csv"
    if ranking_path.exists():
        ranking = pd.read_csv(ranking_path)[[
            "model", "selection_score", "minimum_class_recall"
        ]]
        metrics = metrics.merge(ranking, on="model", how="left", validate="one_to_one")
    else:
        metrics["selection_score"] = metrics["macro_f1"]
        metrics["minimum_class_recall"] = metrics[[
            "recall_I", "recall_A", "recall_S"
        ]].min(axis=1)
    metric_names = [
        "macro_f1", "balanced_accuracy", "macro_ovr_auc",
        "f1_I", "f1_A", "f1_S", "minimum_class_recall",
    ]
    labels = [
        "Macro-F1", "Balanced\naccuracy", "Macro OVR-AUC",
        "I-type F1", "A-type F1", "S-type F1", "Minimum\nclass recall",
    ]
    figure, axis = plt.subplots(figsize=(9.0, 3.2))
    x = np.arange(len(metric_names))
    width = 0.19
    for index, model_name in enumerate(MODEL_COLORS):
        row = metrics.set_index("model").loc[model_name]
        axis.bar(
            x + (index - 1.5) * width,
            [row[metric] for metric in metric_names], width,
            color=MODEL_COLORS[model_name], edgecolor="#3F3F3F", linewidth=0.45,
            label=model_name,
        )
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Group-stratum nested OOF score")
    axis.grid(axis="y", linestyle=":", alpha=0.35)
    axis.legend(ncol=2, loc="upper left")
    save_vector_and_png(figure, figure_dir, "Fig_granite_model_fair_comparison")


def plot_heldout_confusion(final_dir: Path, figure_dir: Path) -> None:
    matrix = pd.read_csv(final_dir / "heldout_confusion_matrix.csv", index_col=0).to_numpy(float)
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    cmap = LinearSegmentedColormap.from_list("soft_orange", ["#FFFDF8", "#F5CBA7", "#D97A5B"])
    figure, axis = plt.subplots(figsize=(3.4, 3.1))
    sns.heatmap(
        normalized, annot=True, fmt=".2f", cmap=cmap, vmin=0, vmax=1, cbar=False,
        square=True, linewidths=0.7, linecolor="white",
        xticklabels=CLASSES, yticklabels=CLASSES, ax=axis,
    )
    axis.set_xlabel("Predicted type")
    axis.set_ylabel("Reported type")
    save_vector_and_png(figure, figure_dir, "Fig_granite_selected_model_heldout_confusion")


def generate_all_figures(project_root: Path, config: dict[str, Any]) -> None:
    set_style()
    output_root = (project_root / config["paths"]["output_root"]).resolve()
    comparison_dir = output_root / "02_Model_Comparison"
    final_dir = output_root / "03_Final_Model"
    shap_dir = output_root / "04_SHAP"
    figure_dir = output_root / "06_Figures"
    plot_model_comparison(comparison_dir, figure_dir)
    plot_heldout_confusion(final_dir, figure_dir)
    plot_shap_ridgeline(shap_dir, figure_dir, int(config["shap"]["top_features_per_class"]))


def generate_performance_figures(project_root: Path, config: dict[str, Any]) -> None:
    """Draw model-selection figures independently of the SHAP eligibility gate."""
    set_style()
    output_root = (project_root / config["paths"]["output_root"]).resolve()
    figure_dir = output_root / "06_Figures"
    plot_model_comparison(output_root / "02_Model_Comparison", figure_dir)
    plot_heldout_confusion(output_root / "03_Final_Model", figure_dir)


if __name__ == "__main__":
    import json
    root = Path(__file__).resolve().parents[1]
    with (root / "config" / "granite_classification_config.json").open("r", encoding="utf-8") as stream:
        configuration = json.load(stream)
    generate_all_figures(root, configuration)
