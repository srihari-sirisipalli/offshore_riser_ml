"""
Lightweight Plotly HTML dashboard for interactive exploration of prediction results.

Inputs:
- predictions DataFrame with at least: true_angle, pred_angle, abs_error, error
- optional: Hs_ft (or custom hs_col)

Outputs:
- Single self-contained HTML file with interactive plots (hover/zoom/filter via built-in controls).
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_dashboard(
    predictions: pd.DataFrame,
    output_path: Path,
    *,
    hs_col: str = "Hs_ft",
    title: str = "Interactive Diagnostics Dashboard",
) -> Path:
    """
    Create a static HTML dashboard (no server required) for quick interactive review.

    Sections:
    - True vs Predicted (colored by abs_error)
    - Hs vs Abs Error (colored by true_angle)
    - Error Histogram
    - Index vs Error line plot
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = predictions.copy()
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for visualization",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color="red"),
        )
        fig.update_layout(
            title=title,
            template="plotly_white",
            height=400,
            showlegend=False,
        )
        fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)
        return output_path

    if "abs_error" not in df.columns or "true_angle" not in df.columns or "pred_angle" not in df.columns:
        raise ValueError("Predictions must include columns: true_angle, pred_angle, abs_error.")

    has_hs = hs_col in df.columns

    # Subplots canvas
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "True vs Predicted (colored by abs_error)",
            f"{hs_col} vs Abs Error" if has_hs else "Abs Error vs Index",
            "Error Histogram",
            "Error vs Index",
        ),
    )

    # True vs Predicted
    scatter_tp = px.scatter(
        df,
        x="true_angle",
        y="pred_angle",
        color="abs_error",
        color_continuous_scale="Viridis",
        labels={"true_angle": "True Angle (deg)", "pred_angle": "Pred Angle (deg)", "abs_error": "Abs Error (deg)"},
    )
    for trace in scatter_tp.data:
        fig.add_trace(trace, row=1, col=1)

    # Hs vs Abs Error (fallback to index if no Hs)
    if has_hs:
        scatter_hs = px.scatter(
            df,
            x=hs_col,
            y="abs_error",
            color="true_angle",
            color_continuous_scale="Plasma",
            labels={hs_col: "Hs (ft)", "abs_error": "Abs Error (deg)", "true_angle": "True Angle (deg)"},
        )
        for trace in scatter_hs.data:
            fig.add_trace(trace, row=1, col=2)
    else:
        line_abs = go.Scatter(
            x=df.index,
            y=df["abs_error"],
            mode="lines+markers",
            line=dict(color="firebrick"),
            name="Abs Error",
        )
        fig.add_trace(line_abs, row=1, col=2)

    # Error histogram
    hist_err = go.Histogram(
        x=df["abs_error"],
        nbinsx=50,
        marker=dict(color="#1f77b4"),
        name="Abs Error Distribution",
        opacity=0.8,
    )
    fig.add_trace(hist_err, row=2, col=1)

    # Error vs Index
    line_err = go.Scatter(
        x=df.index,
        y=df["error"] if "error" in df.columns else df["abs_error"],
        mode="lines",
        line=dict(color="#ff7f0e"),
        name="Error",
    )
    fig.add_trace(line_err, row=2, col=2)

    fig.update_layout(
        title=title,
        height=900,
        coloraxis=dict(colorbar_title="Abs Error"),
        template="plotly_white",
        showlegend=False,
    )

    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)
    return output_path
