"""Shared configuration, hashing, audit, and export utilities for uranium prospectivity modelling."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


OUTPUT_SUBDIRECTORIES = {
    "audit": "../../results/prospectivity/01_Data_Audit",
    "processed": "../../results/prospectivity/02_Processed_Data",
    "optuna": "../../results/prospectivity/03_Optuna",
    "oof": "../../results/prospectivity/04_OOF",
    "results": "../../results/prospectivity/05_Model_Results",
    "shap": "../../results/prospectivity/06_SHAP",
    "challenge": "../../results/prospectivity/07_Challenge_Set",
    "bridge": "../../results/prospectivity/08_Coupling_Bridge",
    "figures": "../../results/prospectivity/09_Figures",
    "logs": "../../results/prospectivity/10_Logs",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(config_path: str | Path) -> tuple[dict[str, Any], Path, dict[str, Path]]:
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    root = path.parent.parent
    output_paths = {key: root / relative for key, relative in OUTPUT_SUBDIRECTORIES.items()}
    for directory in [root, *output_paths.values()]:
        directory.mkdir(parents=True, exist_ok=True)
    return config, root, output_paths


def resolve_package_path(root: Path, configured_path: str | Path) -> Path:
    """Resolve input paths against the package root, never the kernel working directory."""
    path = Path(configured_path).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=json_default).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=json_default)
    return target


def save_table(frame: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, encoding="utf-8-sig")
    return target


def read_lines(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def package_environment() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ["numpy", "pandas", "scipy", "sklearn", "optuna", "xgboost", "shap", "matplotlib"]:
        try:
            module = __import__(name)
            packages[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:
            packages[name] = f"unavailable: {exc.__class__.__name__}"
    return {
        "created_utc": utc_now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }


def validate_runtime_environment(config_path: str | Path) -> dict[str, Any]:
    """Fail before data processing when the active kernel is not the locked submission environment."""
    config, root, paths = load_config(config_path)
    runtime = config["runtime"]
    lock_path = resolve_package_path(root, runtime["requirements_lock_file"])
    if not lock_path.exists():
        raise FileNotFoundError(f"Locked requirements file not found: {lock_path}")
    expected: dict[str, str] = {}
    for raw_line in lock_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-r "):
            continue
        if "==" not in line:
            raise ValueError(f"Every submission dependency must use an exact version: {line}")
        name, version = line.split("==", 1)
        expected[name.strip()] = version.strip()
    installed: dict[str, str | None] = {}
    mismatches: dict[str, Any] = {}
    for distribution, expected_version in expected.items():
        try:
            actual_version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            actual_version = None
        installed[distribution] = actual_version
        if actual_version != expected_version:
            mismatches[distribution] = {"expected": expected_version, "actual": actual_version}
    expected_python = tuple(int(value) for value in runtime["expected_python_major_minor"])
    actual_python = (sys.version_info.major, sys.version_info.minor)
    if actual_python != expected_python:
        mismatches["python_major_minor"] = {
            "expected": ".".join(map(str, expected_python)),
            "actual": ".".join(map(str, actual_python)),
        }
    module_names = {
        "numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "scikit-learn": "sklearn",
        "optuna": "optuna", "xgboost": "xgboost", "shap": "shap", "matplotlib": "matplotlib",
        "openpyxl": "openpyxl", "joblib": "joblib",
    }
    prefix = Path(sys.prefix).resolve()
    module_origins: dict[str, str | None] = {}
    outside_prefix: dict[str, str] = {}
    for distribution, module_name in module_names.items():
        if distribution not in expected or installed.get(distribution) is None:
            continue
        module = importlib.import_module(module_name)
        origin_value = getattr(module, "__file__", None)
        origin = Path(origin_value).resolve() if origin_value else None
        module_origins[distribution] = str(origin) if origin else None
        if origin and runtime.get("require_packages_inside_sys_prefix", True):
            try:
                origin.relative_to(prefix)
            except ValueError:
                outside_prefix[distribution] = str(origin)
    if outside_prefix:
        mismatches["packages_outside_active_environment"] = outside_prefix
    report = {
        "analysis_revision": config["analysis_revision"],
        "passed": not mismatches,
        "strict_environment_check": bool(runtime.get("strict_environment_check", True)),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "sys_prefix": str(prefix),
        "requirements_lock_sha256": sha256_file(lock_path),
        "expected_versions": expected,
        "installed_versions": installed,
        "module_origins": module_origins,
        "mismatches": mismatches,
    }
    save_json(paths["audit"] / "runtime_environment_validation.json", report)
    if report["strict_environment_check"] and not report["passed"]:
        raise RuntimeError(
            "The active Jupyter kernel does not match requirements-lock.txt. "
            "Create the environment from environment.yml, select the 'JGE Prospectivity v7.3' kernel, "
            f"restart the kernel, and rerun. Details: {mismatches}"
        )
    return report


class IntegrityLedger:
    """Fail-fast integrity ledger with machine-readable evidence."""

    def __init__(self, output_path: str | Path, run_id: str = "unassigned"):
        self.output_path = Path(output_path)
        self.run_id = str(run_id)
        self.rows: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
        if self.output_path.exists():
            try:
                payload = json.loads(self.output_path.read_text(encoding="utf-8"))
                if "runs" in payload:
                    run = payload["runs"].get(self.run_id, {})
                    self.rows = list(run.get("checks", []))
                    self.history = list(run.get("history", []))
                    self.all_runs = dict(payload["runs"])
                else:
                    self.all_runs = {"legacy_pre_v7_1": {
                        "checks": list(payload.get("checks", [])),
                        "history": [],
                        "all_fatal_checks_passed": payload.get("all_fatal_checks_passed", False),
                    }}
            except (OSError, json.JSONDecodeError, TypeError):
                self.all_runs = {}
        else:
            self.all_runs = {}

    def check(self, name: str, passed: bool, evidence: Any = None, fatal: bool = True) -> None:
        row = {
            "check": name,
            "passed": bool(passed),
            "fatal": bool(fatal),
            "evidence": evidence,
            "checked_utc": utc_now(),
        }
        prior = [existing for existing in self.rows if existing.get("check") == name]
        if prior:
            self.history.extend(prior)
            self.rows = [existing for existing in self.rows if existing.get("check") != name]
        self.rows.append(row)
        self.all_runs[self.run_id] = {
            "checks": self.rows,
            "history": self.history,
            "all_fatal_checks_passed": self.passed,
            "updated_utc": utc_now(),
        }
        save_json(self.output_path, {
            "active_run_id": self.run_id,
            "runs": self.all_runs,
            "active_run_all_fatal_checks_passed": self.passed,
        })
        if fatal and not passed:
            raise RuntimeError(f"Integrity check failed: {name}. Evidence: {evidence}")

    @property
    def passed(self) -> bool:
        return all(row["passed"] for row in self.rows if row["fatal"])


def assert_columns(frame: pd.DataFrame, required: Iterable[str], table_name: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise KeyError(f"{table_name} is missing required columns: {missing}")


def create_run_manifest(root: Path, input_files: Iterable[Path]) -> dict[str, Any]:
    entries = []
    for path in input_files:
        entries.append({
            "path": str(path),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else None,
            "sha256": sha256_file(path) if path.exists() else None,
        })
    return {"created_utc": utc_now(), "package_root": str(root), "inputs": entries}


def build_analysis_protocol_contract(
    config_path: str | Path,
    frame: pd.DataFrame,
    features: list[str],
    split_registry: pd.DataFrame,
) -> dict[str, Any]:
    config, root, paths = load_config(config_path)
    input_hashes = json.loads((paths["audit"] / "input_hashes.json").read_text(encoding="utf-8"))
    columns = config["columns"]
    ordered_frame = frame.sort_values(columns["record_id"])
    ordered_splits = split_registry.sort_values(
        ["repeat", "outer_fold", "partition", "Geological Group ID"]
    ).reset_index(drop=True)
    feature_list_hash = sha256_payload(features)
    split_registry_hash = sha256_payload(ordered_splits.to_dict("records"))
    cohort_identity_hash = sha256_payload({
        "record_id": ordered_frame[columns["record_id"]].astype(str).tolist(),
        "label": ordered_frame[columns["label"]].astype(int).tolist(),
        "group_id": ordered_frame[columns["group_id"]].astype(str).tolist(),
        "cv_block_id": ordered_frame[columns["cv_block_id"]].astype(str).tolist(),
    })
    protocol_objects = {
        "analysis_revision": config["analysis_revision"],
        "protocol_status": config["protocol_status"],
        "input_hashes": input_hashes,
        "label_definition_hash": sha256_file(root / "config" / "label_definition_note.json"),
        "locked_bridge_features_hash": sha256_file(root / config["shap"]["locked_bridge_feature_file"]),
        "feature_list_hash": feature_list_hash,
        "cohort_identity_hash": cohort_identity_hash,
        "split_registry_hash": split_registry_hash,
        "validation": config["validation"],
        "data_rules": config["data_rules"],
        "search_spaces": config["optuna"]["search_spaces"],
        "shap_gate_and_candidate_rules": config["shap"],
        "runtime_contract": config["runtime"],
        "requirements_lock_hash": sha256_file(
            resolve_package_path(root, config["runtime"]["requirements_lock_file"])
        ),
    }
    protocol_hash = sha256_payload(protocol_objects)
    contract = {
        **protocol_objects,
        "analysis_protocol_hash": protocol_hash,
        "configuration_file_hash": sha256_file(config_path),
        "created_utc": utc_now(),
    }
    save_json(paths["audit"] / "analysis_protocol_contract.json", contract)
    save_json(paths["audit"] / "aggregation_protocol.json", {
        "analysis_revision": config["analysis_revision"],
        "analysis_protocol_hash": protocol_hash,
        "primary_group_aggregation": config["validation"]["geological_group_score_aggregation"],
        "sensitivity_group_aggregation": config["validation"]["geological_group_score_aggregation_sensitivity"],
        "applies_to": ["inner objective", "outer OOF", "consensus OOF", "SHAP", "mixed-group comparison", "subsequent granite-type coupling analysis"],
    })
    return contract


def stage_complete(
    marker_path: str | Path,
    stage: str,
    inputs: dict[str, str],
    outputs: Iterable[Path],
    analysis_revision: str | None = None,
    analysis_protocol_hash: str | None = None,
) -> None:
    output_rows = []
    for path in outputs:
        output_rows.append({
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
        })
    save_json(marker_path, {
        "stage": stage,
        "analysis_revision": analysis_revision,
        "analysis_protocol_hash": analysis_protocol_hash,
        "completed_utc": utc_now(),
        "inputs": inputs,
        "outputs": output_rows,
    })


def validate_stage_marker(marker_path: str | Path) -> bool:
    marker = Path(marker_path)
    if not marker.exists():
        return False
    payload = json.loads(marker.read_text(encoding="utf-8"))
    for row in payload.get("outputs", []):
        path = Path(row["path"])
        if not path.exists() or (row.get("sha256") and sha256_file(path) != row["sha256"]):
            return False
    return True


def write_output_manifest(config_path: str | Path) -> Path:
    config, root, paths = load_config(config_path)
    protocol_path = paths["audit"] / "analysis_protocol_contract.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.exists() else None
    rows = []
    for path in sorted((root / "../../results/prospectivity").rglob("*")):
        if path.is_file():
            rows.append({
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    manifest = {
        "analysis_revision": config["analysis_revision"],
        "created_utc": utc_now(),
        "environment": package_environment(),
        "configuration_hash": sha256_file(config_path),
        "analysis_protocol_hash": protocol.get("analysis_protocol_hash") if protocol else None,
        "input_hashes": protocol.get("input_hashes") if protocol else None,
        "outputs": rows,
    }
    return save_json(paths["logs"] / "run_output_manifest.json", manifest)


# Colourblind-safe qualitative palette (Okabe-Ito) shared by all model panels.
SCI_MODEL_COLORS = {
    "RF": "#0072B2",
    "SVM": "#D55E00",
    "MLP": "#009E73",
    "XGBoost": "#CC79A7",
}


def _configure_matplotlib(config: dict[str, Any]) -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": config["runtime"].get("matplotlib_font", "Arial"),
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": int(config["runtime"].get("figure_dpi", 600)),
    })


def _save_figure(fig: Any, directory: Path, stem: str, config: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{stem}.svg", bbox_inches="tight")


def _feature_label(feature: str) -> str:
    base = feature.replace(" (wt.%)", "").replace(" (ppm)", "")
    return "".join(f"$_{{{character}}}$" if character.isdigit() else character for character in base)


def _transform_geochemical_values(values: pd.Series, feature: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").mask(lambda series: series < 0)
    return numeric if "wt.%" in feature else np.log10(numeric + 1.0)


def _load_oof_completion_tables(
    config: dict[str, Any], paths: dict[str, Path]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    raw_path = paths["processed"] / "primary_model_cohort_with_nan.csv"
    completed_path = paths["processed"] / "fold_local_oof_completed_geochemistry.csv"
    feature_path = paths["processed"] / "primary_feature_list.txt"
    if not raw_path.exists() or not completed_path.exists() or not feature_path.exists():
        raise FileNotFoundError("Raw and fold-local OOF-completed geochemistry are required for data-quality figures.")
    features = read_lines(feature_path)
    record_id = config["columns"]["record_id"]
    observed = pd.read_csv(raw_path, low_memory=False).set_index(record_id)
    completed_long = pd.read_csv(completed_path, low_memory=False)
    completed = completed_long.groupby("Record ID", sort=False)[features].median(numeric_only=True)
    common = observed.index.intersection(completed.index)
    if common.empty:
        raise RuntimeError("No Record ID overlap between observed and OOF-completed geochemistry.")
    return observed.loc[common], completed.loc[common], features


def export_data_quality_figures(config_path: str | Path) -> list[str]:
    """Export OOF-completion KDE diagnostics and observed/completed correlations."""
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    config, _, paths = load_config(config_path)
    _configure_matplotlib(config)
    observed, completed, features = _load_oof_completion_tables(config, paths)
    configured_kde = config["runtime"].get("data_quality_figure_features", [])
    kde_features = [feature for feature in configured_kde if feature in features]
    if len(kde_features) < min(12, len(features)):
        remaining = sorted(
            [feature for feature in features if feature not in kde_features],
            key=lambda feature: float(observed[feature].isna().mean()), reverse=True,
        )
        kde_features.extend(remaining[: max(0, min(12, len(features)) - len(kde_features))])
    kde_features = kde_features[:12]

    rows = int(np.ceil(len(kde_features) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(10.8, 2.55 * rows), squeeze=False)
    palette = {"Observed": "#C83E4D", "Completed": "#356AA0", "Imputed only": "#E49322"}
    panel_letters = "abcdefghijklmnopqrstuvwxyz"

    def draw_density(axis: Any, values: np.ndarray, grid: np.ndarray, color: str, label: str, linestyle: str) -> None:
        values = values[np.isfinite(values)]
        if values.size < 3 or np.unique(values).size < 2:
            return
        try:
            density = gaussian_kde(values)(grid)
        except (np.linalg.LinAlgError, ValueError):
            return
        axis.plot(grid, density, color=color, lw=1.5, ls=linestyle, label=label)
        if label != "Imputed only":
            axis.fill_between(grid, 0, density, color=color, alpha=0.08)

    for index, feature in enumerate(kde_features):
        axis = axes.flat[index]
        observed_values = _transform_geochemical_values(observed[feature], feature).to_numpy(float)
        completed_values = _transform_geochemical_values(completed[feature], feature).to_numpy(float)
        imputed_values = completed_values[observed[feature].isna().to_numpy()]
        finite = np.concatenate([
            observed_values[np.isfinite(observed_values)], completed_values[np.isfinite(completed_values)]
        ])
        upper = max(float(np.nanquantile(finite, 0.995)) * 1.08, float(np.nanmax(finite)) * 0.25, 1e-6)
        grid = np.linspace(0.0, upper, 300)
        draw_density(axis, observed_values, grid, palette["Observed"], "Observed", "-")
        draw_density(axis, completed_values, grid, palette["Completed"], "Completed", "-")
        draw_density(axis, imputed_values, grid, palette["Imputed only"], "Imputed only", "--")
        missing_fraction = float(observed[feature].isna().mean())
        axis.set_title(f"({panel_letters[index]}) {_feature_label(feature)} ({missing_fraction:.1%})", fontsize=9)
        axis.set_xlabel("Content (wt.%)" if "wt.%" in feature else r"$\log_{10}$[Content (ppm) + 1]")
        axis.set_ylabel("Density")
        axis.set_xlim(left=0.0)
        axis.set_ylim(bottom=0.0)
        axis.grid(axis="y", color="#E6E6E6", lw=0.55, ls="--")
    for axis in axes.flat[len(kde_features):]:
        axis.set_visible(False)
    handles = [
        plt.Line2D([0], [0], color=palette["Observed"], lw=1.6, label="Observed"),
        plt.Line2D([0], [0], color=palette["Completed"], lw=1.6, label="Completed"),
        plt.Line2D([0], [0], color=palette["Imputed only"], lw=1.6, ls="--", label="Imputed only"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    quality_directory = paths["figures"] / "Data_Quality"
    kde_stem = "Fig_KDE_imputation_diagnostics_grouped_OOF"
    _save_figure(fig, quality_directory, kde_stem, config)
    plt.close(fig)

    configured_pearson = config["runtime"].get("pearson_figure_features", [])
    pearson_features = [feature for feature in configured_pearson if feature in features][:10]
    if len(pearson_features) < min(10, len(features)):
        pearson_features.extend(
            feature for feature in kde_features if feature not in pearson_features
        )
    pearson_features = pearson_features[:10]
    observed_transformed = pd.DataFrame({
        feature: _transform_geochemical_values(observed[feature], feature)
        for feature in pearson_features
    })
    completed_transformed = pd.DataFrame({
        feature: _transform_geochemical_values(completed[feature], feature)
        for feature in pearson_features
    })
    observed_correlation = observed_transformed.corr(method="pearson", min_periods=20)
    completed_correlation = completed_transformed.corr(method="pearson", min_periods=20)
    difference = (completed_correlation - observed_correlation).stack(future_stack=True).reset_index()
    difference.columns = ["feature_1", "feature_2", "completed_minus_observed_pearson_r"]
    save_table(difference, paths["audit"] / "pearson_observed_completed_difference.csv")

    from matplotlib.colors import LinearSegmentedColormap
    correlation_cmap = LinearSegmentedColormap.from_list(
        "jge_correlation", ["#2166AC", "#F7FBFF", "#D7263D"], N=256
    )
    fig = plt.figure(figsize=(11.0, 4.8))
    grid_spec = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045], wspace=0.25)
    axes = [fig.add_subplot(grid_spec[0, 0]), fig.add_subplot(grid_spec[0, 1])]
    colorbar_axis = fig.add_subplot(grid_spec[0, 2])
    matrices = [(observed_correlation, "(a) Observed data"), (completed_correlation, "(b) Completed data")]
    image = None
    labels = [_feature_label(feature) for feature in pearson_features]
    for axis, (matrix, title) in zip(axes, matrices):
        image = axis.imshow(matrix.to_numpy(), cmap=correlation_cmap, vmin=-1, vmax=1, interpolation="nearest")
        axis.set_xticks(np.arange(len(labels)), labels=labels, rotation=90)
        axis.set_yticks(np.arange(len(labels)), labels=labels)
        axis.set_title(title, fontsize=10)
        axis.tick_params(length=0, labelsize=8)
        axis.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.1)
        axis.tick_params(which="minor", length=0)
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Pearson correlation coefficient")
    colorbar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.08, right=0.94, bottom=0.18, top=0.90)
    pearson_stem = "Fig_Pearson_correlation_matrices_grouped_OOF"
    _save_figure(fig, quality_directory, pearson_stem, config)
    plt.close(fig)
    return [
        f"Data_Quality/{kde_stem}",
        f"Data_Quality/{pearson_stem}",
    ]


def _rule_selected_model(metrics: pd.DataFrame, config: dict[str, Any]) -> str:
    """Re-apply the frozen consensus selection rule when the decision JSON is absent."""
    maximum = float(metrics["roc_auc"].max())
    tolerance = float(config["validation"]["trial_auc_tolerance"])
    eligible = metrics[metrics["roc_auc"] >= maximum - tolerance].copy()
    ordered = eligible.sort_values(
        ["average_precision", "roc_auc_sd", "f1", "model"],
        ascending=[False, True, False, True],
    )
    return str(ordered.iloc[0]["model"])


def _selected_model_name(config: dict[str, Any], paths: dict[str, Path], metrics: pd.DataFrame) -> str:
    """Read the audited model-selection decision, falling back to the frozen rule."""
    decision_path = paths["results"] / "model_selection_decision.json"
    if decision_path.exists():
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if decision.get("selected_model") in set(metrics.index):
            return str(decision["selected_model"])
    return _rule_selected_model(metrics, config)


def _style_confusion_axis(axis: Any) -> None:
    """Restore a full frame and add cell separators for imshow-based confusion matrices."""
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.8)
    axis.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.5)
    axis.tick_params(which="minor", length=0)


def _export_merged_performance_figure(
    config: dict[str, Any], paths: dict[str, Path], data: pd.DataFrame,
    metrics: pd.DataFrame, models: list[str], palette: dict[str, str],
    performance_directory: Path,
) -> str:
    """Metrics bars + ROC/PR curves merged into the main performance-comparison figure."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig = plt.figure(figsize=(11.4, 6.5))
    grid = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.22], hspace=0.45, wspace=0.38)
    metric_axes = {
        "roc_auc": fig.add_subplot(grid[0, 0:2]),
        "average_precision": fig.add_subplot(grid[0, 2:4]),
        "f1": fig.add_subplot(grid[0, 4:6]),
    }
    metric_titles = {
        "roc_auc": "(a) ROC-AUC",
        "average_precision": "(b) Average precision",
        "f1": "(c) F1 (positive class)",
    }
    for column, axis in metric_axes.items():
        values = np.array([float(metrics.loc[model, column]) for model in models])
        bars = axis.barh(
            np.arange(len(models)), values,
            color=[palette[model] for model in models], edgecolor="#333333", linewidth=0.6,
        )
        axis.set_yticks(np.arange(len(models)), labels=models)
        axis.set_xlim(0, 1.0)
        axis.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axis.set_xlabel("Consensus grouped-OOF value")
        axis.set_title(metric_titles[column])
        axis.grid(axis="x", color="#E3E3E3", lw=0.6, ls="--")
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values):
            axis.text(min(value + 0.018, 0.99), bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")
    metric_axes["roc_auc"].invert_yaxis()

    roc_axis = fig.add_subplot(grid[1, 0:3])
    pr_axis = fig.add_subplot(grid[1, 3:6])
    for model in models:
        subset = data[data["model"].eq(model)]
        fpr, tpr, _ = roc_curve(subset["target"], subset["model_score"])
        precision, recall, _ = precision_recall_curve(subset["target"], subset["model_score"])
        roc_axis.plot(fpr, tpr, lw=1.8, color=palette[model], label=f"{model} (AUC = {metrics.loc[model, 'roc_auc']:.3f})")
        pr_axis.plot(recall, precision, lw=1.8, color=palette[model], label=f"{model} (AP = {metrics.loc[model, 'average_precision']:.3f})")
    roc_axis.plot([0, 1], [0, 1], "--", color="#777777", lw=0.9)
    prevalence = data.drop_duplicates("Geological Group ID")["target"].mean()
    pr_axis.axhline(prevalence, ls="--", color="#777777", lw=0.9)
    roc_axis.set(xlabel="False-positive rate", ylabel="True-positive rate", title="(d) ROC curves")
    pr_axis.set(xlabel="Recall", ylabel="Precision", title="(e) Precision-recall curves")
    for axis in (roc_axis, pr_axis):
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(color="#E8E8E8", lw=0.6)
        axis.legend(frameon=False, fontsize=7.6, loc="lower right")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.94, bottom=0.10, hspace=0.55, wspace=0.45)
    merged_stem = "Fig_3_consensus_grouped_OOF_performance_comparison"
    _save_figure(fig, performance_directory, merged_stem, config)
    plt.close(fig)
    return merged_stem


def _export_all_model_confusion_figure(
    config: dict[str, Any], paths: dict[str, Path], data: pd.DataFrame,
    metrics: pd.DataFrame, models: list[str], performance_directory: Path,
) -> str:
    """Group-level OOF confusion matrices for all candidate models with counts, row percentages,
    and per-class precision/recall/F1, using a full frame and cell separators."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    n_columns = 2
    n_rows = int(np.ceil(len(models) / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(7.8, 6.9), squeeze=False)
    confusion_by_model: dict[str, np.ndarray] = {}
    class_metrics: dict[str, list[tuple[float, float, float]]] = {}
    for model in models:
        subset = data[data["model"].eq(model)]
        model_matrix = confusion_matrix(subset["target"], subset["prediction"], labels=[0, 1])
        confusion_by_model[model] = model_matrix
        tn, fp, fn, tp = model_matrix.ravel()
        rows = []
        for label, true_positive, false_positive, false_negative in [(0, tn, fp, fn), (1, tp, fp, fn)]:
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            rows.append((precision, recall, f1))
        class_metrics[model] = rows
    common_vmax = max(1, max(int(value.max()) for value in confusion_by_model.values()))
    for index, (axis, model) in enumerate(zip(axes.flat, models)):
        model_matrix = confusion_by_model[model]
        row_totals = model_matrix.sum(axis=1, keepdims=True).astype(float)
        row_totals[row_totals == 0] = 1.0
        row_fractions = model_matrix / row_totals
        image = axis.imshow(model_matrix, cmap="Blues", vmin=0, vmax=common_vmax)
        for row in range(2):
            for column in range(2):
                count = int(model_matrix[row, column])
                axis.text(column, row, f"{count}\n({row_fractions[row, column] * 100:.1f}%)", ha="center", va="center",
                          fontsize=8.5,
                          color="white" if model_matrix[row, column] > model_matrix.max() * 0.55 else "black")
        axis.set_xticks([0, 1], labels=["Predicted 0", "Predicted 1"])
        axis.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
        axis.tick_params(labelsize=8)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        panel = "abcdefghijklmnopqrstuvwxyz"[index]
        m0, m1 = class_metrics[model]
        axis.set_title(
            f"({panel}) {model}\n"
            f"AUC = {metrics.loc[model, 'roc_auc']:.3f}; F1 = {metrics.loc[model, 'f1']:.3f}\n"
            f"P/R/F1 0: {m0[0]:.3f}/{m0[1]:.3f}/{m0[2]:.3f}\n"
            f"P/R/F1 1: {m1[0]:.3f}/{m1[1]:.3f}/{m1[2]:.3f}",
            fontsize=8,
        )
        _style_confusion_axis(axis)
    for axis in axes.flat[len(models):]:
        axis.set_visible(False)
    colorbar = fig.colorbar(image, ax=list(axes.flat[:len(models)]), fraction=0.035, pad=0.02)
    colorbar.set_label("Number of geological groups", fontsize=8)
    colorbar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.09, right=0.93, top=0.93, bottom=0.09, hspace=0.60, wspace=0.32)
    confusion_all_stem = "Fig_4_consensus_confusion_matrices_all_models"
    _save_figure(fig, performance_directory, confusion_all_stem, config)
    plt.close(fig)
    return confusion_all_stem


def export_model_performance_figures(config_path: str | Path, include_merged: bool = True) -> list[str]:
    """Export the merged performance-comparison figure and the all-model confusion matrices."""
    config, _, paths = load_config(config_path)
    _configure_matplotlib(config)
    source = paths["results"] / "geological_group_consensus_oof.csv"
    metrics_path = paths["results"] / "model_consensus_metrics.csv"
    if not source.exists() or not metrics_path.exists():
        raise FileNotFoundError("Consensus OOF outputs are required before figure export.")
    data = pd.read_csv(source)
    metrics = pd.read_csv(metrics_path).set_index("model")
    palette = SCI_MODEL_COLORS
    models = [model for model in config["validation"]["models"] if model in set(data["model"])]
    performance_directory = paths["figures"] / "Model_Performance"
    stems: list[str] = []
    if include_merged:
        stems.append(_export_merged_performance_figure(config, paths, data, metrics, models, palette, performance_directory))
    stems.append(_export_all_model_confusion_figure(config, paths, data, metrics, models, performance_directory))
    return [f"Model_Performance/{stem}" for stem in stems]


def export_shap_figure(config_path: str | Path) -> str | None:
    import matplotlib.pyplot as plt

    config, _, paths = load_config(config_path)
    _configure_matplotlib(config)
    source = paths["shap"] / "global_shap_importance_and_stability.csv"
    if not source.exists():
        return None
    data = pd.read_csv(source).sort_values("mean_abs_oof_shap", ascending=False).head(
        int(config["shap"]["top_features_for_figure"])
    )
    if data.empty:
        return None

    def category(feature: str) -> str:
        if "wt.%" in feature:
            return "Major oxides"
        if feature.split(" ")[0] in {"La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}:
            return "REEs"
        return "Trace elements"

    SHAP_CATEGORY_CMAPS = {"Major oxides": "Blues", "Trace elements": "Oranges", "REEs": "Purples"}
    data["category"] = data["feature"].map(category)
    data["cat_rank"] = data.groupby("category")["mean_abs_oof_shap"].rank(ascending=False).astype(int)
    data["cat_count"] = data.groupby("category")["mean_abs_oof_shap"].transform("size").astype(int)
    colors = []
    for _, row in data.iterrows():
        cmap = plt.get_cmap(SHAP_CATEGORY_CMAPS[row["category"]])
        frac = (row["cat_rank"] - 1) / max(row["cat_count"] - 1, 1)
        colors.append(cmap(0.95 - 0.60 * frac))
    plot_data = data.iloc[::-1]
    plot_colors = colors[::-1]
    fig, axis = plt.subplots(figsize=(8.2, max(6.0, 0.25 * len(plot_data) + 1.6)))
    y = np.arange(len(plot_data))
    axis.barh(y, plot_data["mean_abs_oof_shap"], color=plot_colors, edgecolor="#333333", linewidth=0.35)
    axis.set_yticks(y, labels=[_feature_label(feature) for feature in plot_data["feature"]])
    axis.set_xlabel("Mean absolute OOF SHAP value")
    axis.grid(axis="x", color="#E8E8E8", lw=0.6)
    axis.set_axisbelow(True)
    maximum = float(data["mean_abs_oof_shap"].max())
    axis.set_xlim(0, maximum * 1.62)
    for row_index, value in enumerate(plot_data["mean_abs_oof_shap"]):
        axis.text(float(value) + maximum * 0.012, row_index, f"{float(value):.3f}", va="center", fontsize=7)

    category_order = ["Major oxides", "Trace elements", "REEs"]
    feature_colors = dict(zip(data["feature"], colors))
    donut_data = data.copy()
    donut_data["category"] = pd.Categorical(donut_data["category"], categories=category_order, ordered=True)
    donut_data = donut_data.sort_values(["category", "cat_rank"])
    outer_sizes = donut_data.groupby("category", observed=False)["mean_abs_oof_shap"].sum().reindex(category_order)
    inner_sizes = donut_data["mean_abs_oof_shap"].to_numpy(float)
    outer_colors = [plt.get_cmap(SHAP_CATEGORY_CMAPS[name])(0.65) for name in category_order]
    inner_colors = [feature_colors[feature] for feature in donut_data["feature"]]
    inset = axis.inset_axes([0.64, 0.05, 0.32, 0.36])
    wedges, _ = inset.pie(
        outer_sizes.to_numpy(float), radius=1.0,
        colors=outer_colors,
        wedgeprops={"width": 0.30, "edgecolor": "white", "linewidth": 1.0},
    )
    inset.pie(
        inner_sizes, radius=0.70, colors=inner_colors,
        wedgeprops={"width": 0.34, "edgecolor": "white", "linewidth": 0.7},
    )
    inset.text(0, 0, "Top 30\ncontribution", ha="center", va="center", fontsize=8)
    for wedge, category_name in zip(wedges, category_order):
        angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        x, y_text = np.cos(angle), np.sin(angle)
        inset.annotate(
            category_name, xy=(0.84 * x, 0.84 * y_text), xytext=(1.32 * x, 1.32 * y_text),
            ha="left" if x >= 0 else "right", va="center", fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "#444444", "lw": 0.8},
        )
    inset.set_aspect("equal")
    fig.tight_layout()
    shap_directory = paths["figures"] / "SHAP"
    stem = "Fig_selected_model_OOF_TreeSHAP_top30_descriptive"
    _save_figure(fig, shap_directory, stem, config)
    plt.close(fig)
    return f"SHAP/{stem}"


def write_final_workflow_report(config_path: str | Path) -> Path:
    config, _, paths = load_config(config_path)
    decision_path = paths["results"] / "model_selection_decision.json"
    gate_path = paths["shap"] / "global_attribution_reliability_gate.json"
    challenge_path = paths["challenge"] / "challenge_set_interpretation.json"
    data_path = paths["processed"] / "data_pipeline_summary.json"
    integrity_path = paths["audit"] / "preflight_and_integrity_checks.json"
    payloads = {}
    for name, path in {
        "data": data_path, "decision": decision_path, "gate": gate_path,
        "challenge": challenge_path, "integrity": integrity_path,
    }.items():
        payloads[name] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    metrics_path = paths["results"] / "model_performance_mean_ci.csv"
    metrics_markdown = (
        "```text\n" + pd.read_csv(metrics_path).to_string(index=False) + "\n```"
        if metrics_path.exists() else "尚未生成。"
    )
    selected = payloads["decision"].get("selected_model") if payloads["decision"] else "尚未选择"
    gate = payloads["gate"].get("global_gate_passed") if payloads["gate"] else "尚未执行"
    integrity_passed = (
        payloads["integrity"].get("active_run_all_fatal_checks_passed")
        if payloads["integrity"] else "尚未执行"
    )
    lines = [
        "# 铀成矿有利性模型运行结果汇总",
        "",
        "## 1. 运行结论",
        "",
        f"- 最终模型：{selected}",
        f"- OOF SHAP 全局可靠性门禁：{gate}",
        f"- 当前运行完整性检查：{integrity_passed}",
        "- 性能术语：重复嵌套、来源连接块分组的 OOF 评估；不是外部验证。",
        "",
        "## 2. 数据与队列",
        "",
        "```json", json.dumps(payloads["data"], ensure_ascii=False, indent=2) if payloads["data"] else "尚未生成。", "```",
        "",
        "## 3. 四模型性能",
        "",
        metrics_markdown,
        "",
        "## 4. 模型选择",
        "",
        "```json", json.dumps(payloads["decision"], ensure_ascii=False, indent=2) if payloads["decision"] else "尚未生成。", "```",
        "",
        "## 5. SHAP 与耦合门禁",
        "",
        "```json", json.dumps(payloads["gate"], ensure_ascii=False, indent=2) if payloads["gate"] else "尚未生成或所选模型非树模型。", "```",
        "",
        "SHAP 只表示冻结模型中的变量归因。后续与花岗岩分类模型比较时，应使用标准化排名、贡献占比和方向一致性，不得直接比较两个模型的原始 SHAP 数值，也不得将统计对应解释为成矿因果证据。",
        "",
        "## 6. 混合标签挑战集",
        "",
        "```json", json.dumps(payloads["challenge"], ensure_ascii=False, indent=2) if payloads["challenge"] else "尚未生成。", "```",
        "",
        "## 7. 后续接口",
        "",
        "后续花岗岩分类耦合分析只应读取 `../../results/prospectivity/08_Coupling_Bridge/taskA_oof_bridge_one_row_per_Record_ID.csv`、锁定九变量表和 bridge contract。若全局归因门禁未通过，桥表保留用于审计，但不得作为正式耦合证据。",
    ]
    report = paths["logs"] / "uranium_prospectivity_model_report_CN.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def run_read_only_exports(config_path: str | Path) -> dict[str, Any]:
    figures = export_data_quality_figures(config_path)
    figures.extend(export_model_performance_figures(config_path))
    shap_figure = export_shap_figure(config_path)
    if shap_figure:
        figures.append(shap_figure)
    report = write_final_workflow_report(config_path)
    manifest = write_output_manifest(config_path)
    return {"figures": figures, "report": str(report), "output_manifest": str(manifest)}
