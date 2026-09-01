"""
SAFEBAND AI - Dashboard Charts Module

Reusable Plotly visualizations for the SAFEBAND AI dashboard.

Supported visualizations:
    - Heart-rate monitoring
    - SpO2 monitoring
    - Environmental temperature
    - Body temperature
    - Risk-score trends
    - Motion intensity
    - Combined safety overview
    - Activity distribution

Design goals:
    - Consistent dashboard appearance
    - Dark/light theme compatibility
    - Predictable chart dimensions
    - Safe handling of empty or invalid data
    - No Streamlit dependency

The chart layer is presentation-only. It does not perform
sensor processing, AI inference, or risk calculation.
"""


from typing import Any, Dict, List, Optional, Sequence, Tuple

import plotly.graph_objects as go


# ============================================================
# CHART CONFIGURATION
# ============================================================

CHART_HEIGHT = 300

CHART_MARGIN = {
    "l": 45,
    "r": 25,
    "t": 50,
    "b": 45,
}

TRANSPARENT = "rgba(0,0,0,0)"

GRID_COLOR = "rgba(128,128,128,0.15)"

DEFAULT_FONT = "Arial"

DEFAULT_FONT_SIZE = 12


# ============================================================
# COMMON LAYOUT
# ============================================================

COMMON_LAYOUT: Dict[str, Any] = {
    "height": CHART_HEIGHT,
    "margin": CHART_MARGIN,
    "paper_bgcolor": TRANSPARENT,
    "plot_bgcolor": TRANSPARENT,
    "font": {
        "family": DEFAULT_FONT,
        "size": DEFAULT_FONT_SIZE,
    },
    "hovermode": "x unified",
    "autosize": True,
}


# ============================================================
# DATA HELPERS
# ============================================================

def _safe_float(
    value: Any,
) -> Optional[float]:
    """Convert a value to float when possible."""

    try:
        return float(value)

    except (TypeError, ValueError):
        return None


def _clean_series(
    x_values: Sequence[Any],
    y_values: Sequence[Any],
) -> Tuple[List[Any], List[float]]:
    """
    Remove invalid Y values while keeping X/Y alignment.

    This prevents malformed sensor readings from breaking
    dashboard charts.
    """

    timestamps: List[Any] = []
    values: List[float] = []

    for x_value, y_value in zip(
        x_values,
        y_values,
    ):

        numeric_value = _safe_float(
            y_value
        )

        if numeric_value is None:
            continue

        timestamps.append(
            x_value
        )

        values.append(
            numeric_value
        )

    return timestamps, values


def _base_figure() -> go.Figure:
    """Create an empty Plotly figure with common configuration."""

    fig = go.Figure()

    fig.update_layout(
        **COMMON_LAYOUT
    )

    return fig


# ============================================================
# GENERIC LINE CHART
# ============================================================

def create_line_chart(
    x_values: Sequence[Any],
    y_values: Sequence[Any],
    title: str,
    y_axis_title: str,
    color: str = "#00C896",
) -> go.Figure:
    """
    Create a reusable sensor line chart.

    Invalid numeric readings are ignored safely.
    """

    x_clean, y_clean = _clean_series(
        x_values,
        y_values,
    )

    fig = _base_figure()

    fig.add_trace(
        go.Scatter(
            x=x_clean,
            y=y_clean,
            mode="lines+markers",
            line={
                "color": color,
                "width": 3,
            },
            marker={
                "size": 6,
            },
            hovertemplate=(
                f"{y_axis_title}: "
                "%{y}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title={
            "text": title,
            "x": 0.02,
        },
        xaxis={
            "title": "Time",
            "showgrid": False,
            "automargin": True,
        },
        yaxis={
            "title": y_axis_title,
            "showgrid": True,
            "gridcolor": GRID_COLOR,
            "automargin": True,
        },
    )

    return fig


# ============================================================
# HEART RATE
# ============================================================

def create_heart_rate_chart(
    timestamps: Sequence[Any],
    heart_rates: Sequence[Any],
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
# SPO2
# ============================================================

def create_spo2_chart(
    timestamps: Sequence[Any],
    spo2_values: Sequence[Any],
) -> go.Figure:
    """Create blood-oxygen monitoring chart."""

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
# ENVIRONMENTAL TEMPERATURE
# ============================================================

def create_temperature_chart(
    timestamps: Sequence[Any],
    temperatures: Sequence[Any],
) -> go.Figure:
    """
    Create environmental-temperature chart.

    This represents the BME680 environmental sensor and should
    not be confused with MAX30208 body temperature.
    """

    return create_line_chart(
        x_values=timestamps,
        y_values=temperatures,
        title="Environmental Temperature",
        y_axis_title="Temperature (°C)",
        color="#F59E0B",
    )


# ============================================================
# BODY TEMPERATURE
# ============================================================

def create_body_temperature_chart(
    timestamps: Sequence[Any],
    body_temperatures: Sequence[Any],
) -> go.Figure:
    """
    Create body-temperature monitoring chart.

    Source:
        MAX30208

    Body temperature is intentionally kept separate from
    environmental temperature.
    """

    return create_line_chart(
        x_values=timestamps,
        y_values=body_temperatures,
        title="Body Temperature",
        y_axis_title="Body Temperature (°C)",
        color="#EF4444",
    )


# ============================================================
# RISK SCORE
# ============================================================

def create_risk_chart(
    timestamps: Sequence[Any],
    risk_scores: Sequence[Any],
) -> go.Figure:
    """
    Create safety risk-score monitoring chart.

    Risk levels:

        0-29    LOW
        30-59   MODERATE
        60-79   HIGH
        80-100  CRITICAL
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

    # --------------------------------------------------------
    # RISK BOUNDARIES
    # --------------------------------------------------------

    fig.add_hline(
        y=30,
        line_dash="dash",
        line_width=1,
        annotation_text="Moderate",
        annotation_position="top left",
    )

    fig.add_hline(
        y=60,
        line_dash="dash",
        line_width=1,
        annotation_text="High",
        annotation_position="top left",
    )

    fig.add_hline(
        y=80,
        line_dash="dash",
        line_width=1,
        annotation_text="Critical",
        annotation_position="top left",
    )

    return fig


# ============================================================
# MOTION
# ============================================================

def create_motion_chart(
    timestamps: Sequence[Any],
    motion_values: Sequence[Any],
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
# MULTI-SENSOR OVERVIEW
# ============================================================

def create_sensor_overview_chart(
    timestamps: Sequence[Any],
    heart_rates: Sequence[Any],
    risk_scores: Sequence[Any],
) -> go.Figure:
    """
    Create combined heart-rate and risk-score visualization.

    Two Y axes are used because BPM and risk score represent
    different units and ranges.
    """

    hr_x, hr_y = _clean_series(
        timestamps,
        heart_rates,
    )

    risk_x, risk_y = _clean_series(
        timestamps,
        risk_scores,
    )

    fig = _base_figure()

    # --------------------------------------------------------
    # HEART RATE
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=hr_x,
            y=hr_y,
            name="Heart Rate",
            mode="lines+markers",
            line={
                "color": "#E63946",
                "width": 2,
            },
            marker={
                "size": 5,
            },
            hovertemplate=(
                "Heart Rate: %{y} BPM"
                "<extra></extra>"
            ),
        )
    )

    # --------------------------------------------------------
    # RISK SCORE
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=risk_x,
            y=risk_y,
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
            hovertemplate=(
                "Risk Score: %{y}"
                "<extra></extra>"
            ),
        )
    )

    # --------------------------------------------------------
    # LAYOUT
    # --------------------------------------------------------

    fig.update_layout(
        title={
            "text": "Safety Monitoring Overview",
            "x": 0.02,
        },

        xaxis={
            "title": "Time",
            "showgrid": False,
            "automargin": True,
        },

        yaxis={
            "title": "Heart Rate (BPM)",
            "showgrid": True,
            "gridcolor": GRID_COLOR,
            "automargin": True,
        },

        yaxis2={
            "title": "Risk Score",
            "overlaying": "y",
            "side": "right",
            "range": [0, 100],
            "automargin": True,
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
    activities: Sequence[str],
) -> go.Figure:
    """
    Create activity-distribution bar chart.

    Activities are counted exactly as provided after
    normalization to uppercase.
    """

    counts: Dict[str, int] = {}

    for activity in activities:

        normalized = str(
            activity
        ).strip().upper()

        if not normalized:
            continue

        counts[normalized] = (
            counts.get(
                normalized,
                0,
            )
            + 1
        )

    labels = list(
        counts.keys()
    )

    values = list(
        counts.values()
    )

    fig = _base_figure()

    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            text=values,
            textposition="auto",
            marker={
                "color": "#00C896",
            },
            hovertemplate=(
                "Activity: %{x}"
                "<br>Occurrences: %{y}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title={
            "text": "Activity Distribution",
            "x": 0.02,
        },

        xaxis={
            "title": "Activity",
            "showgrid": False,
            "automargin": True,
        },

        yaxis={
            "title": "Occurrences",
            "showgrid": True,
            "gridcolor": GRID_COLOR,
            "automargin": True,
        },
    )

    return fig


# ============================================================
# SENSOR HISTORY HELPERS
# ============================================================

def extract_sensor_history(
    history: Sequence[Dict[str, Any]],
    key: str,
) -> Tuple[List[Any], List[float]]:
    """
    Extract timestamps and numeric values from sensor history.

    Expected record format:

        {
            "timestamp": "...",
            "heart_rate": 80
        }

    Invalid or missing values are ignored.
    """

    timestamps: List[Any] = []
    values: List[float] = []

    for record in history:

        if not isinstance(
            record,
            dict,
        ):
            continue

        if key not in record:
            continue

        numeric_value = _safe_float(
            record.get(key)
        )

        if numeric_value is None:
            continue

        timestamps.append(
            record.get(
                "timestamp",
                "",
            )
        )

        values.append(
            numeric_value
        )

    return timestamps, values


# ============================================================
# CHART EXPORT
# ============================================================

def chart_to_dict(
    fig: go.Figure,
) -> Dict[str, Any]:
    """
    Convert a Plotly figure into a dictionary.

    Useful for debugging, testing, or future serialization.
    """

    if not isinstance(
        fig,
        go.Figure,
    ):
        raise TypeError(
            "Expected a Plotly Figure."
        )

    return fig.to_dict()


# ============================================================
# CHART VALIDATION
# ============================================================

def validate_chart(
    fig: go.Figure,
) -> bool:
    """
    Perform a lightweight validation of a Plotly figure.

    Returns True when the supplied object is a valid Plotly
    Figure instance.
    """

    if not isinstance(
        fig,
        go.Figure,
    ):
        return False

    try:
        fig.to_dict()
        return True

    except Exception:
        return False