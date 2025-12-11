import hashlib
import json
import logging
import platform
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from importlib import metadata
from scipy import stats

from modules.base.base_engine import BaseEngine
from utils.file_io import save_dataframe, read_dataframe
from utils.exceptions import RiserMLException
from utils.resource_monitor import write_resource_dashboard, capture_resource_snapshot
from utils import constants

class DataIntegrityTracker(BaseEngine):
    """
    Tracks data point consistency across all rounds of the RFE.
    """

    def __init__(self, config: dict, logger: logging.Logger):
        super().__init__(config, logger)
        self.excel_copy = self.config.get("outputs", {}).get("save_excel_copy", False)
        self.tracking_file = self.output_dir / "data_integrity_tracking_all_rounds.parquet"
        self.tracked_data = self._load_tracking_file()
        self.feature_stats_records: List[Dict[str, Any]] = []
        self.gate_records: List[Dict[str, Any]] = []
        self.split_similarity_records: List[Dict[str, Any]] = []
        self.lineage_records: List[Dict[str, Any]] = []
        gates_cfg = self.config.get("data_quality_gates", {})
        self.min_split_rows = gates_cfg.get("min_split_rows", 1)
        self.max_missing_pct = gates_cfg.get("max_missing_pct", 100.0)
        self.fail_on_violation = gates_cfg.get("fail_on_violation", False)

    def _get_engine_directory_name(self) -> str:
        return constants.DATA_INTEGRITY_DIR

    def _load_tracking_file(self) -> pd.DataFrame:
        if self.tracking_file.exists():
            df = read_dataframe(self.tracking_file)
            # Handle both old and new schema - check for 'original_index' or 'index' column
            if "original_index" in df.columns:
                return df.set_index("original_index")
            elif "index" in df.columns:
                return df.set_index("index")
            else:
                self.logger.warning(f"Tracking file exists but has unexpected schema. Creating fresh tracking data.")
                return pd.DataFrame()
        else:
            return pd.DataFrame()

    def track_round(self, round_num: int, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Tracks the data splits for a given round.
        """
        self.logger.info(f"Tracking data integrity for round {round_num}...")

        all_dfs = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }

        round_col_name = f"round_{round_num:03d}_split"

        # Ensure column for current round exists FIRST (before trying to add rows)
        # Use dtype='object' to avoid FutureWarning when assigning string values
        if round_col_name not in self.tracked_data.columns:
            self.tracked_data[round_col_name] = pd.Series(dtype='object')

        for split_name, df in all_dfs.items():
            for original_index in df.index:
                # Ensure index exists in tracked_data
                if original_index not in self.tracked_data.index:
                    # Create a new row with pd.NA (compatible with object dtype)
                    new_row = pd.Series({round_col_name: pd.NA}, name=original_index, dtype='object')
                    self.tracked_data = pd.concat([self.tracked_data, new_row.to_frame().T])

                # Now safely check and assign
                if pd.notna(self.tracked_data.loc[original_index, round_col_name]):
                    # Already tracked for this round, something is wrong
                    self.logger.warning(f"Index {original_index} already tracked in round {round_num}. Duplicates detected.")
                self.tracked_data.loc[original_index, round_col_name] = split_name

            # Collect feature distribution stats per split for this round
            self._collect_feature_stats(df, split_name, round_num)

            # Collect quality gate info
            missing_pct = df.isna().values.mean() * 100 if not df.empty else 0.0
            status = "PASS"
            if len(df) < self.min_split_rows or missing_pct > self.max_missing_pct:
                status = "FAIL" if self.fail_on_violation else "WARN"
            self.gate_records.append({
                "round": round_num,
                "split": split_name,
                "rows": int(len(df)),
                "missing_pct": float(missing_pct),
                "min_split_rows": self.min_split_rows,
                "max_missing_pct": self.max_missing_pct,
                "status": status,
            })
            # Data lineage hash per split/round
            self.lineage_records.append({
                "round": round_num,
                "split": split_name,
                "rows": int(len(df)),
                "cols": int(df.shape[1]),
                "sha256": self._hash_dataframe(df),
            })
        # Split similarity (KS) per numeric column
        self._collect_split_similarity(train_df, val_df, test_df, round_num)

    def finalize(self):
        """
        Analyzes the tracked data and saves the final report.
        """
        self.logger.info("Finalizing data integrity tracking...")

        # Add consistency checks
        round_cols = sorted([c for c in self.tracked_data.columns if c.startswith("round_")])
        
        # Check for leakage
        def check_leakage(row):
            seen_in = set(row[round_cols].dropna().unique())
            if len(seen_in) > 1:
                return f"LEAKAGE: {', '.join(sorted(list(seen_in)))}"
            elif len(seen_in) == 1:
                return seen_in.pop()
            else:
                return "Not Seen"

        self.tracked_data['consistency'] = self.tracked_data.apply(check_leakage, axis=1)

        # Reorder columns for readability
        final_cols = ['consistency'] + round_cols
        self.tracked_data = self.tracked_data[final_cols]

        # Save with explicit column name for the index
        save_dataframe(self.tracked_data.reset_index(names='original_index'), self.tracking_file, excel_copy=self.excel_copy, index=False)
        self.logger.info(f"Data integrity report saved to {self.tracking_file}")

        # Feature distribution stability across rounds
        feature_stats = self._compute_feature_distribution_stats()
        if feature_stats is not None:
            save_dataframe(
                feature_stats,
                self.output_dir / "feature_distribution_evolution_all_rounds.parquet",
                excel_copy=self.excel_copy,
                index=False,
            )

        # Quality checks summary per round
        checks_df = self._build_quality_checks()
        save_dataframe(
            checks_df,
            self.output_dir / "quality_checks_status_all_rounds.parquet",
            excel_copy=self.excel_copy,
            index=False,
        )

        # Data quality gates per round/split
        if self.gate_records:
            gates_df = pd.DataFrame(self.gate_records)
            save_dataframe(
                gates_df,
                self.output_dir / "data_quality_gates_results.parquet",
                excel_copy=self.excel_copy,
                index=False,
            )
            if self.fail_on_violation and (gates_df['status'] == "FAIL").any():
                raise RiserMLException(
                    "Data quality gates failed: one or more splits did not meet minimum rows or maximum missing percentage."
                )

        # Split similarity KS across splits
        if self.split_similarity_records:
            save_dataframe(
                pd.DataFrame(self.split_similarity_records),
                self.output_dir / "split_similarity_analysis.parquet",
                excel_copy=self.excel_copy,
                index=False,
            )

        # Data quality issues log (nan/inf/outliers per feature/split/round)
        if self.feature_stats_records:
            issues_cols = [
                "round",
                "split",
                "feature",
                "nan_count",
                "inf_count",
                "outlier_count",
                "count",
            ]
            issues_df = pd.DataFrame(self.feature_stats_records)[issues_cols]
            save_dataframe(
                issues_df,
                self.output_dir / "data_quality_issues_log_all_rounds.parquet",
                excel_copy=self.excel_copy,
                index=False,
            )

        # Data lineage checksums
        if self.lineage_records:
            save_dataframe(
                pd.DataFrame(self.lineage_records),
                self.output_dir / "data_lineage_validation_checksums.parquet",
                excel_copy=self.excel_copy,
                index=False,
            )

        # Seed and environment logs for reproducibility
        self._write_seed_log()
        self._write_environment_log()
        self._write_resource_snapshot()

    def execute(self, *args, **kwargs):
        """Not used for this engine."""
        pass

    def _compute_feature_distribution_stats(self) -> pd.DataFrame:
        """Compute per-feature stats per round to track drift."""
        if not self.feature_stats_records:
            return pd.DataFrame()

        stats_df = pd.DataFrame(self.feature_stats_records)
        # Compute deltas vs baseline (round 0) where possible
        baseline = stats_df[stats_df['round'] == 0].set_index(['split', 'feature'])
        def add_deltas(row):
            key = (row['split'], row['feature'])
            if key in baseline.index:
                base_row = baseline.loc[key]
                row['delta_mean'] = row['mean'] - base_row['mean']
                row['delta_std'] = row['std'] - base_row['std']
            else:
                row['delta_mean'] = np.nan
                row['delta_std'] = np.nan
            return row

        stats_df = stats_df.apply(add_deltas, axis=1)
        return stats_df

    def _collect_feature_stats(self, df: pd.DataFrame, split_name: str, round_num: int) -> None:
        """Collect per-feature numeric stats for drift tracking."""
        if df.empty:
            return
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            return
        # Use population std, skew, kurtosis to flag drift
        stds = numeric.std()
        means = numeric.mean()
        mins = numeric.min()
        maxs = numeric.max()
        medians = numeric.median()
        skews = numeric.skew()
        kurts = numeric.kurtosis()
        nans = numeric.isna().sum()
        infs = np.isinf(numeric).sum()
        for col in numeric.columns:
            col_series = numeric[col]
            outlier_count = (np.abs(col_series - means[col]) > 3 * (stds[col] if stds[col] != 0 else 1)).sum()
            self.feature_stats_records.append({
                "round": round_num,
                "split": split_name,
                "feature": col,
                "count": int(len(col_series)),
                "mean": float(means[col]),
                "std": float(stds[col]),
                "min": float(mins[col]),
                "max": float(maxs[col]),
                "median": float(medians[col]),
                "skew": float(skews[col]),
                "kurtosis": float(kurts[col]),
                "nan_count": int(nans[col]),
                "inf_count": int(infs[col]),
                "outlier_count": int(outlier_count),
            })

    def _collect_split_similarity(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, round_num: int) -> None:
        """Compute KS statistics between splits for numeric features."""
        splits = {
            "train": train_df,
            "val": val_df,
            "test": test_df,
        }
        numeric_cols = set(train_df.select_dtypes(include=[np.number]).columns)
        numeric_cols |= set(val_df.select_dtypes(include=[np.number]).columns)
        numeric_cols |= set(test_df.select_dtypes(include=[np.number]).columns)
        pairs = [("train", "val"), ("train", "test"), ("val", "test")]

        for col in numeric_cols:
            for a, b in pairs:
                a_vals = splits[a][col].dropna() if col in splits[a].columns else pd.Series(dtype=float)
                b_vals = splits[b][col].dropna() if col in splits[b].columns else pd.Series(dtype=float)
                if len(a_vals) == 0 or len(b_vals) == 0:
                    continue
                ks_stat, p_val = stats.ks_2samp(a_vals, b_vals)
                self.split_similarity_records.append({
                    "round": round_num,
                    "feature": col,
                    "split_a": a,
                    "split_b": b,
                    "ks_stat": float(ks_stat),
                    "p_value": float(p_val),
                    "n_a": int(len(a_vals)),
                    "n_b": int(len(b_vals)),
                })

    def _build_quality_checks(self) -> pd.DataFrame:
        """Summarize split counts and leakage flags per round."""
        rows = []
        round_cols = sorted([c for c in self.tracked_data.columns if c.startswith("round_")])
        for col in round_cols:
            col_vals = self.tracked_data[col].dropna()
            counts = col_vals.value_counts()
            leakage_count = (self.tracked_data['consistency'] == 'LEAKAGE').sum() if 'consistency' in self.tracked_data.columns else 0
            rows.append({
                "round": col,
                "train_count": int(counts.get('train', 0)),
                "val_count": int(counts.get('val', 0)),
                "test_count": int(counts.get('test', 0)),
                "leakage_rows": int(leakage_count),
                "timestamp": datetime.now().isoformat()
            })
        return pd.DataFrame(rows)

    def _write_seed_log(self) -> None:
        seeds = self.config.get("_internal_seeds", {})
        payload = {
            "timestamp": datetime.now().isoformat(),
            "master_seed": self.config.get("splitting", {}).get("seed"),
            "internal_seeds": seeds
        }
        seed_path = self.output_dir / "random_seed_provenance.json"
        seed_path.write_text(json.dumps(payload, indent=2))

    def _write_environment_log(self) -> None:
        libs = {}
        for pkg in ["pandas", "numpy", "scikit-learn", "matplotlib", "scipy", "pyarrow"]:
            try:
                libs[pkg] = metadata.version(pkg)
            except metadata.PackageNotFoundError:
                libs[pkg] = "not-installed"

        env_payload = {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "libraries": libs
        }
        env_path = self.output_dir / "environment_versioning_log.json"
        env_path.write_text(json.dumps(env_payload, indent=2))

    def _hash_dataframe(self, df: pd.DataFrame) -> str:
        """Compute a SHA256 hash of dataframe values for lineage checks."""
        # Use stable ordering: columns sorted, index sorted
        df_sorted = df.sort_index().sort_index(axis=1)
        # Convert to bytes via parquet buffer
        buffer = df_sorted.to_parquet(index=False)
        return hashlib.sha256(buffer).hexdigest()

    def _write_resource_snapshot(self) -> None:
        """Best-effort snapshot of system resources (CPU/memory)."""
        snapshot = capture_resource_snapshot()
        resource_path = self.output_dir / "resource_utilization_snapshot.json"
        resource_path.write_text(json.dumps(snapshot, indent=2))
        # Also persist a richer dashboard (00_DATA_INTEGRITY/resource_utilization_dashboard.*)
        try:
            write_resource_dashboard(self.base_dir, excel_copy=self.excel_copy)
        except Exception:
            # Keep snapshot even if dashboard fails
            pass
