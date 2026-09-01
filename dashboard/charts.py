"""
SAFEBAND AI - Dashboard Charts Module

Provides reusable Plotly chart functions for:
- Heart-rate monitoring
- SpO2 monitoring
- Temperature monitoring
- Risk-score trends
- Sensor activity
"""

from typing import List, Dict, Any

import plotly.graph_objects as go


# ============================================================
# COMMON CHART CONFIGURATION
# ============================================================

CHART_HEIGHT = 300

COMMON_LAYOUT = {
    "height": CHART_HEIGHT,
    "margin": {
        "l": 40,
        "r": 20,
        "t": 45,
        "b": 40,
    },
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {
        "family": "Arial",
        "size": 12,
    },
    "hovermode": "x unified",
}


# ============================================================
# GENERIC LINE CHART
# ============================================================

def create_line_chart(
    x_values: List[Any],
    y_values: List[float],
    title: str,
    y_axis_title: str,
    color: str = "#00C896",
) -> go.Figure:
    """
    Create a reusable line chart.

    Parameters
    ----------
    x_values : list
        X-axis values.

    y_values : list
        Y-axis values.

    title : str
        Chart title.

    y_axis_title : str
        Y-axis label.

    color : str
        Line color.
    """

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines+markers",
            line={
                "color": color,
                "width": 3,
            },
            marker={
                "size": 6,
            },
            hovertemplate=(
                f"{y_axis_title}: %{{y}}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        **COMMON_LAYOUT,
        title={
            "text": title,
            "x": 0.02,
        },
        xaxis={
            "showgrid": False,
            "title": "Time",
        },
        yaxis={
            "title": y_axis_title,
            "showgrid": True,
            "gridcolor": "rgba(128,128,128,0.15)",
        },
    )

    return fig


# ============================================================
# HEART RATE CHART
# ============================================================

def create_heart_rate_chart(
    timestamps: List[Any],
    heart_rates: List[float],
) -> go.Figure:
    """Create heart-rate monitoring chart."""

    return create_line_chart(
        x_values=timestamps,
        y_values=heart_rates,
        title="Heart Rate",
        y_axis_title="BPM",
        color="#E63946",
    )


# ============================================================
# SPO2 CHART
# ============================================================

def create_spo2_chart(
    timestamps: List[Any],
    spo2_values: List[float],
) -> go.Figure:
    """Create SpO2 monitoring chart."""

    fig = create_line_chart(
        x_values=timestamps,
        y_values=spo2_values,
        title="Blood Oxygen Saturation",
        y_axis_title="SpO₂ (%)",
        color="#3B82F6",
    )

    fig.update_yaxes(
        range=[85, 100]
    )

    return fig


# ============================================================
# TEMPERATURE CHART
# ============================================================

def create_temperature_chart(
    timestamps: List[Any],
    temperatures: List[float],
) -> go.Figure:
    """Create environmental temperature chart."""

    return create_line_chart(
        x_values=timestamps,
        y_values=temperatures,
        title="Environmental Temperature",
        y_axis_title="Temperature (°C)",
        color="#F59E0B",
    )


# ============================================================
# RISK SCORE CHART
# ============================================================

def create_risk_chart(
    timestamps: List[Any],
    risk_scores: List[float],
) -> go.Figure:
    """
    Create risk-score monitoring chart.

    Risk score:
        0-29   Low
        30-59  Moderate
        60-79  High
        80-100 Critical
    """

    fig = create_line_chart(
        x_values=timestamps,
        y_values=risk_scores,
        title="Safety Risk Score",
        y_axis_title="Risk Score",
        color="#8B5CF6",
    )

    fig.update_yaxes(
        range=[0, 100]
    )

    # Moderate-risk boundary
    fig.add_hline(
        y=30,
        line_dash="dash",
        line_width=1,
        annotation_text="Moderate",
        annotation_position="top left",
    )

    # High-risk boundary
    fig.add_hline(
        y=60,
        line_dash="dash",
        line_width=1,
        annotation_text="High",
        annotation_position="top left",
    )

    # Critical-risk boundary
    fig.add_hline(
        y=80,
        line_dash="dash",
        line_width=1,
        annotation_text="Critical",
        annotation_position="top left",
    )

    return fig


# ============================================================
# MOTION CHART
# ============================================================

def create_motion_chart(
    timestamps: List[Any],
    motion_values: List[float],
) -> go.Figure:
    """Create motion-intensity monitoring chart."""

    return create_line_chart(
        x_values=timestamps,
        y_values=motion_values,
        title="Motion Intensity",
        y_axis_title="Intensity",
        color="#10B981",
    )


# ============================================================
# MULTI-SENSOR CHART
# ============================================================

def create_sensor_overview_chart(
    timestamps: List[Any],
    heart_rates: List[float],
    risk_scores: List[float],
) -> go.Figure:
    """
    Create combined heart-rate and risk-score visualization.

    Uses two Y axes because BPM and risk score have different units.
    """

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=heart_rates,
            name="Heart Rate",
            mode="lines+markers",
            line={
                "color": "#E63946",
                "width": 2,
            },
            marker={
                "size": 5,
            },
        )
    )

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=risk_scores,
            name="Risk Score",
            mode="lines+markers",
            yaxis="y2",
            line={
                "color": "#8B5CF6",
                "width": 2,
            },
            marker={
                "size": 5,
            },
        )
    )

    fig.update_layout(
        **COMMON_LAYOUT,
        title={
            "text": "Safety Monitoring Overview",
            "x": 0.02,
        },
        xaxis={
            "title": "Time",
            "showgrid": False,
        },
        yaxis={
            "title": "Heart Rate (BPM)",
            "showgrid": True,
            "gridcolor": "rgba(128,128,128,0.15)",
        },
        yaxis2={
            "title": "Risk Score",
            "overlaying": "y",
            "side": "right",
            "range": [0, 100],
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )

    return fig


# ============================================================
# ACTIVITY DISTRIBUTION
# ============================================================

def create_activity_distribution_chart(
    activities: List[str],
) -> go.Figure:
    """
    Create activity-distribution bar chart.

    Parameters
    ----------
    activities : list
        Activity labels collected during monitoring.
    """

    counts: Dict[str, int] = {}

    for activity in activities:
        activity = str(activity).upper()
        counts[activity] = counts.get(activity, 0) + 1

    labels = list(counts.keys())
    values = list(counts.values())

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                text=values,
                textposition="auto",
                marker={
                    "color": "#00C896",
                },
            )
        ]
    )

    fig.update_layout(
        **COMMON_LAYOUT,
        title={
            "text": "Activity Distribution",
            "x": 0.02,
        },
        xaxis={
            "title": "Activity",
            "showgrid": False,
        },
        yaxis={
            "title": "Occurrences",
            "showgrid": True,
            "gridcolor": "rgba(128,128,128,0.15)",
        },
    )

    return fig


# ============================================================
# SENSOR HISTORY HELPERS
# ============================================================

def extract_sensor_history(
    history: List[Dict[str, Any]],
    key: str,
) -> tuple:
    """
    Extract timestamps and values from sensor history.

    Expected history format:

    [
        {
            "timestamp": "...",
            "heart_rate": 80
        },
        ...
    ]
    """

    timestamps = []
    values = []

    for record in history:
        if key not in record:
            continue

        timestamps.append(
            record.get("timestamp", "")
        )

        try:
            values.append(
                float(record[key])
            )
        except (TypeError, ValueError):
            continue

    return timestamps, values


# ============================================================
# CHART EXPORT
# ============================================================

def chart_to_dict(fig: go.Figure) -> Dict[str, Any]:
    """
    Convert a Plotly figure into a dictionary.

    Useful if the dashboard needs to inspect or serialize
    chart configuration.
    """

    return fig.to_dict()