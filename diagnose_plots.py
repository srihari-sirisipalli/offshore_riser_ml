#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagnostic script to check plot issues
"""
import pandas as pd
from pathlib import Path
import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')

def check_plot_data():
    """Check if prediction data looks correct"""

    results_dir = Path("results/ROUND_000")

    if not results_dir.exists():
        print("[ERROR] results/ROUND_000 directory not found")
        return

    # Check baseline predictions
    print("\n" + "="*60)
    print("CHECKING BASELINE MODEL DATA")
    print("="*60)

    base_dir = results_dir / "04_BaseModel_WithAllFeatures"

    # Check val predictions
    val_path = base_dir / "baseline_predictions_val.parquet"
    if val_path.exists():
        df_val = pd.read_parquet(val_path)
        print(f"\n[OK] Val predictions: {len(df_val)} rows")
        print(f"  Columns: {list(df_val.columns)}")
        print(f"  abs_error stats: min={df_val['abs_error'].min():.2f}, "
              f"max={df_val['abs_error'].max():.2f}, mean={df_val['abs_error'].mean():.2f}")
        print(f"  true_angle range: [{df_val['true_angle'].min():.1f}, {df_val['true_angle'].max():.1f}]")

        # Check for Hs column
        hs_cols = [c for c in df_val.columns if 'Hs' in c or 'hs' in c]
        if hs_cols:
            print(f"  Hs columns found: {hs_cols}")
            for col in hs_cols:
                print(f"    {col} range: [{df_val[col].min():.2f}, {df_val[col].max():.2f}]")
        else:
            print(f"  [WARNING] No Hs column found!")
    else:
        print(f"[ERROR] Val predictions not found at {val_path}")

    # Check test predictions
    test_path = base_dir / "baseline_predictions_test.parquet"
    if test_path.exists():
        df_test = pd.read_parquet(test_path)
        print(f"\n[OK] Test predictions: {len(df_test)} rows")
        print(f"  Columns: {list(df_test.columns)}")
        print(f"  abs_error stats: min={df_test['abs_error'].min():.2f}, "
              f"max={df_test['abs_error'].max():.2f}, mean={df_test['abs_error'].mean():.2f}")
    else:
        print(f"[ERROR] Test predictions not found at {test_path}")

    # Check LOFO results
    print("\n" + "="*60)
    print("CHECKING LOFO DATA")
    print("="*60)

    lofo_path = results_dir / "05_Feature_ImportanceRanking" / "lofo_summary.parquet"
    if lofo_path.exists():
        df_lofo = pd.read_parquet(lofo_path)
        print(f"\n[OK] LOFO summary: {len(df_lofo)} features")
        print(f"  Columns: {list(df_lofo.columns)}")
        if 'val_cmae' in df_lofo.columns:
            print(f"  val_cmae range: [{df_lofo['val_cmae'].min():.4f}, {df_lofo['val_cmae'].max():.4f}]")
        if 'delta_val_cmae' in df_lofo.columns:
            print(f"  delta_val_cmae range: [{df_lofo['delta_val_cmae'].min():.4f}, {df_lofo['delta_val_cmae'].max():.4f}]")
        print(f"\n  Top 5 features (lowest val_cmae when removed):")
        if 'val_cmae' in df_lofo.columns:
            top5 = df_lofo.nsmallest(5, 'val_cmae')[['feature', 'val_cmae', 'delta_val_cmae']]
            print(top5.to_string(index=False))
    else:
        print(f"[ERROR] LOFO summary not found at {lofo_path}")

    # Check comparison data
    print("\n" + "="*60)
    print("CHECKING COMPARISON DATA")
    print("="*60)

    comp_dir = results_dir / "07_ModelComparison_BaseVsReduced"

    delta_path = comp_dir / "delta_metrics.parquet"
    if delta_path.exists():
        df_delta = pd.read_parquet(delta_path)
        print(f"\n[OK] Delta metrics: {len(df_delta)} metrics")
        print(df_delta[['metric', 'baseline', 'dropped', 'delta', 'status']].to_string(index=False))
    else:
        print(f"[ERROR] Delta metrics not found at {delta_path}")

    # Check for dropped model predictions
    dropped_val_path = results_dir / "06_ReducedModel_FeatureDropped" / "dropped_predictions_val.parquet"
    if dropped_val_path.exists():
        df_dropped = pd.read_parquet(dropped_val_path)
        print(f"\n[OK] Dropped model val predictions: {len(df_dropped)} rows")
    else:
        print(f"[ERROR] Dropped model predictions not found")

    # Check plot directories
    print("\n" + "="*60)
    print("CHECKING PLOT FILES")
    print("="*60)

    plot_dirs = [
        ("Feature Evaluation Plots", results_dir / "05_Feature_ImportanceRanking" / "feature_evaluation_plots"),
        ("Comparison Plots", results_dir / "07_ModelComparison_BaseVsReduced" / "comparison_plots"),
        ("Advanced Viz", results_dir / "04_BaseModel_WithAllFeatures" / "10_DiagnosticPlots_Advanced"),
        ("Standard Diagnostics", results_dir / "04_BaseModel_WithAllFeatures" / "DiagnosticPlots"),
    ]

    for name, plot_dir in plot_dirs:
        if plot_dir.exists():
            plots = list(plot_dir.glob("*.png"))
            if plots:
                print(f"\n[OK] {name}: {len(plots)} plots")
                # Check file sizes
                small_plots = [p for p in plots if p.stat().st_size < 10000]  # < 10KB
                if small_plots:
                    print(f"  [WARNING] {len(small_plots)} potentially blank plots (< 10KB):")
                    for p in small_plots[:5]:  # Show first 5
                        print(f"    - {p.name} ({p.stat().st_size} bytes)")
            else:
                print(f"[ERROR] {name}: No plots found")
        else:
            print(f"[ERROR] {name}: Directory not found")

    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    check_plot_data()
