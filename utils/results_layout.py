"""
Helpers to align outputs with the Complete Results Directory Organization.

The manager creates the target structure and mirrors key artifacts from the
existing pipeline outputs into the standardized layout without breaking the
legacy locations.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

from utils.file_io import save_dataframe, read_dataframe
from utils import constants


class ResultsLayoutManager:
    """
    Best-effort layout helper to mirror existing outputs into the
    Parquet-first results structure.
    """

    def __init__(self, base_dir: Path, excel_copy: bool = False, logger=None):
        self.base_dir = Path(base_dir)
        self.excel_copy = excel_copy
        self.logger = logger

    # ------------------------------------------------------------------ #
    # Structure creation                                                 #
    # ------------------------------------------------------------------ #
    def ensure_base_structure(self) -> None:
        """Create base directory structure with sequential numbering."""
        for folder in constants.TOP_LEVEL_RESULT_DIRS:
            (self.base_dir / folder).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Top-level mirrors                                                  #
    # ------------------------------------------------------------------ #
    def mirror_config_artifacts(self) -> None:
        cfg_dir = self.base_dir / constants.CONFIG_DIR
        self._copy_if_exists(self.base_dir / constants.CONFIG_USED_FILE, cfg_dir / constants.CONFIG_USED_FILE)
        self._copy_if_exists(self.base_dir / constants.CONFIG_HASH_FILE, cfg_dir / constants.CONFIG_HASH_FILE)
        self._copy_if_exists(self.base_dir / constants.RUN_METADATA_FILE, cfg_dir / constants.RUN_METADATA_FILE)

    def mirror_splits(self) -> None:
        """Mirror smart split outputs into MasterSplits."""
        dest_dir = self.base_dir / constants.MASTER_SPLITS_DIR
        # Direct path only - no legacy aliasing
        for name in ["train.parquet", "val.parquet", "test.parquet", "split_balance_report.parquet", "split_summary.parquet"]:
            self._copy_if_exists(dest_dir / name, dest_dir / name)

    def mirror_global_tracking(self) -> None:
        """Mirror evolution tracker artifacts into RFEsummary."""
        src_dir = self.base_dir / constants.GLOBAL_ERROR_TRACKING_DIR
        dest_dir = self.base_dir / constants.RFE_SUMMARY_DIR
        if not src_dir.exists():
            return

        metrics_src = src_dir / constants.ROUND_METRICS_DIR / "metrics_all_rounds.parquet"
        features_src = src_dir / constants.ROUND_FEATURES_DIR / "features_eliminated_timeline.parquet"
        plots_src = src_dir / constants.ROUND_EVOLUTION_PLOTS_DIR

        self._copy_if_exists(metrics_src, dest_dir / "all_rounds_metrics.parquet")
        self._copy_if_exists(features_src, dest_dir / "feature_elimination_history.parquet")
        if plots_src.exists():
            self._copy_tree(plots_src, dest_dir / "plots")

        self._copy_if_exists(self.base_dir / "safety_threshold_summary_all_rounds.parquet", dest_dir / "safety_threshold_summary_all_rounds.parquet")
        self._copy_if_exists(self.base_dir / "significance_baseline_vs_dropped_over_rounds.parquet", dest_dir / "statistical_tests_round_comparisons.parquet")
        self._copy_if_exists(self.base_dir / "safety_gate_status_all_rounds.parquet", dest_dir / "safety_gate_status_all_rounds.parquet")

    def mirror_ensembling(self) -> None:
        """Mirror ensembling artifacts into Ensembling."""
        dest_dir = self.base_dir / constants.ENSEMBLING_DIR
        # Direct path only
        if dest_dir.exists():
            self._copy_tree(dest_dir, dest_dir)

    def mirror_reconstruction_mapping(self) -> None:
        """Mirror reconstruction mapping into ReconstructionMapping."""
        dest_dir = self.base_dir / constants.RECONSTRUCTION_MAPPING_DIR
        dest_dir.mkdir(parents=True, exist_ok=True)
        # Prefer explicit file names if present at root or reporting dir
        candidates = list(self.base_dir.rglob("model_reconstruction_mapping.parquet"))
        if candidates:
            self._copy_if_exists(candidates[0], dest_dir / "model_reconstruction_mapping.parquet")
        # Optional sidecars
        for name in [
            "model_reconstruction_summary.parquet",
            "model_reconstruction_hyperparameters.parquet",
            "model_reconstruction_features.parquet",
            "model_reconstruction_data_files.parquet",
            "model_reconstruction_code.parquet",
        ]:
            for cand in self.base_dir.rglob(name):
                self._copy_if_exists(cand, dest_dir / name)

    def mirror_reproducibility_package(self) -> None:
        """Mirror reproducibility bundle to ReproducibilityPackage."""
        dest_dir = self.base_dir / constants.REPRODUCIBILITY_PACKAGE_DIR
        # Direct path only
        if dest_dir.exists():
            self._copy_tree(dest_dir, dest_dir)

    def mirror_run(self) -> None:
        """
        Mirroring is disabled to avoid duplicate tree creation and self-copy errors.
        Intentional no-op.
        """
        if self.logger:
            self.logger.info("Results mirroring is disabled (no-op).")

    def ensure_round_structure(self, round_dir: Path) -> None:
        """Create round subdirectories with sequential numbering. Single source of truth."""
        round_dir = Path(round_dir)
        for folder in constants.ROUND_STRUCTURE_DIRS:
            (round_dir / folder).mkdir(parents=True, exist_ok=True)
        return list(constants.ROUND_STRUCTURE_DIRS)

    # ------------------------------------------------------------------ #
    # Mirroring helpers                                                  #
    # ------------------------------------------------------------------ #
    def mirror_baseline_outputs(self, round_dir: Path) -> None:
        """
        Copy core baseline artifacts into the standardized round layout.
        Non-destructive: legacy files remain in place.
        """
        round_dir = Path(round_dir)
        self.ensure_round_structure(round_dir)
        legacy_base = round_dir / constants.ROUND_BASE_MODEL_RESULTS_DIR
        if not legacy_base.exists():
            return

        # Predictions -> ROUND_PREDICTIONS_DIR
        preds_dest = round_dir / constants.ROUND_PREDICTIONS_DIR
        self._copy_if_exists(legacy_base / "baseline_predictions_val.parquet", preds_dest / "predictions_val.parquet")
        self._copy_if_exists(legacy_base / "baseline_predictions_test.parquet", preds_dest / "predictions_test.parquet")

        # Metrics -> ROUND_EVALUATION_DIR
        eval_dest = round_dir / constants.ROUND_EVALUATION_DIR
        val_metrics_path = legacy_base / "baseline_metrics_val.parquet"
        test_metrics_path = legacy_base / "baseline_metrics_test.parquet"
        self._copy_if_exists(val_metrics_path, eval_dest / "metrics_val.parquet")
        self._copy_if_exists(test_metrics_path, eval_dest / "metrics_test.parquet")
        self._write_combined_metrics(eval_dest, val_metrics_path, test_metrics_path)

        # Safety / error analysis
        self._copy_if_exists(legacy_base / "safety_threshold_summary.parquet", round_dir / constants.ROUND_ERROR_ANALYSIS_DIR / "safety_threshold_summary.parquet")
        error_dir_candidates = [
            legacy_base / constants.ERROR_ANALYSIS_ENGINE_DIR,
            legacy_base / "09_ERROR_ANALYSIS",
        ]
        for error_dir in error_dir_candidates:
            if error_dir.exists():
                self._copy_tree(error_dir, round_dir / constants.ROUND_ERROR_ANALYSIS_DIR)
                break

        # Diagnostics
        diag_dir_candidates = [
            legacy_base / constants.DIAGNOSTICS_ENGINE_DIR,
            legacy_base / "08_DIAGNOSTICS",
        ]
        for diag_dir in diag_dir_candidates:
            if diag_dir.exists():
                self._copy_tree(diag_dir, round_dir / constants.ROUND_DIAGNOSTICS_DIR)
                break

        # Advanced visualizations
        adv_dir_candidates = [
            legacy_base / constants.ROUND_ADVANCED_VISUALIZATIONS_DIR,
            legacy_base / "08_ADVANCED_VISUALIZATIONS",
        ]
        for adv_dir in adv_dir_candidates:
            if adv_dir.exists():
                self._copy_tree(adv_dir, round_dir / constants.ROUND_ADVANCED_VISUALIZATIONS_DIR)
                break

        # Comparison already aligned at ROUND_COMPARISON_DIR in legacy layout
        if (round_dir / constants.ROUND_COMPARISON_DIR).exists():
            pass  # exists from legacy flow

        # Round summary (metadata + pointers)
        self._write_round_summary(round_dir)

    # ------------------------------------------------------------------ #
    # Internal utilities                                                 #
    # ------------------------------------------------------------------ #
    def _copy_if_exists(self, src: Path, dest: Path) -> None:
        if not src.exists():
            return
        try:
            # Avoid self-copy (can happen if caller passes identical paths)
            if src.resolve() == dest.resolve():
                return
        except Exception:
            # Best-effort; if resolve fails, continue to copy attempt
            pass
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dest)
        except Exception as exc:  # noqa: BLE001
            if self.logger:
                self.logger.warning(f"Failed to mirror {src} -> {dest}: {exc}")

    def _copy_tree(self, src: Path, dest: Path) -> None:
        src = Path(src)
        dest = Path(dest)
        if not src.exists():
            return
        dest.mkdir(parents=True, exist_ok=True)
        for item in src.rglob("*"):
            if item.is_dir():
                continue
            rel = item.relative_to(src)
            self._copy_if_exists(item, dest / rel)

    def _write_combined_metrics(self, dest_dir: Path, val_path: Path, test_path: Path) -> None:
        if not val_path.exists() and not test_path.exists():
            return
        try:
            frames: List[pd.DataFrame] = []
            if val_path.exists():
                df_val = read_dataframe(val_path)
                df_val["split"] = "val"
                frames.append(df_val)
            if test_path.exists():
                df_test = read_dataframe(test_path)
                df_test["split"] = "test"
                frames.append(df_test)
            if frames:
                combined = pd.concat(frames, ignore_index=True)
                save_dataframe(combined, dest_dir / "combined_metrics.parquet", excel_copy=self.excel_copy, index=False)
        except Exception as exc:  # noqa: BLE001
            if self.logger:
                self.logger.warning(f"Failed to build combined metrics: {exc}")

    def _write_round_summary(self, round_dir: Path) -> None:
        summary_path = round_dir / "round_summary.json"
        payload: Dict[str, Any] = {
            "round_dir": str(round_dir),
            "mirrored": True,
            "artifacts": {
                "predictions_val": (round_dir / constants.ROUND_PREDICTIONS_DIR / "predictions_val.parquet").exists(),
                "predictions_test": (round_dir / constants.ROUND_PREDICTIONS_DIR / "predictions_test.parquet").exists(),
                "metrics_val": (round_dir / constants.ROUND_EVALUATION_DIR / "metrics_val.parquet").exists(),
                "metrics_test": (round_dir / constants.ROUND_EVALUATION_DIR / "metrics_test.parquet").exists(),
                "error_analysis": (round_dir / constants.ROUND_ERROR_ANALYSIS_DIR).exists(),
                "diagnostics": (round_dir / constants.ROUND_DIAGNOSTICS_DIR).exists(),
                "advanced_viz": (round_dir / constants.ROUND_ADVANCED_VISUALIZATIONS_DIR).exists(),
            },
        }
        summary_path.write_text(json.dumps(payload, indent=2))

    # Legacy aliasing methods removed - using direct sequential paths only
