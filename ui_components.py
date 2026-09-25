"""
UI Components: Interactive Plotly visualizers for Temperature, Rain, Cloudiness, and Spread.
"""

import pandas as pd
import plotly.graph_objects as go

MODEL_PALETTE = [
    "#2563EB",
    "#059669",
    "#7C3AED",
    "#EA580C",
    "#D97706",
    "#0891B2",
    "#DC2626",
]


def plot_master_consensus(
    df_max: pd.DataFrame,
    df_min: pd.DataFrame,
    df_summary: pd.DataFrame,
    unit: str,
    city: str,
) -> go.Figure:
    """Master chart showing ensemble temperature envelope, consensus means, and individual model traces."""
    dates = df_summary.index.tolist()
    fig = go.Figure()

    # 1. Shaded Uncertainty Ribbon (Between Lowest Min and Highest Max)
    upper_bounds = df_summary["Highest Max"].tolist()
    lower_bounds = df_summary["Lowest Min"].tolist()
    if not df_summary["Highest Max"].isna().all():
        fig.add_trace(
            go.Scatter(
                x=dates + dates[::-1],
                y=upper_bounds + lower_bounds[::-1],
                fill="toself",
                fillcolor="rgba(59, 130, 246, 0.12)",
                line={"color": "rgba(255,255,255,0)"},
                hoverinfo="skip",
                name="Ensemble Uncertainty Envelope",
            )
        )

    # 2. Ensemble Averages
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df_summary["Daily Avg Max"],
            name="Ensemble Avg Max",
            mode="lines+markers",
            line={"color": "#DC2626", "width": 3.5},
            marker={"size": 7},
            hovertemplate="<b>%{x}</b><br>Ensemble Avg Max: %{y:.1f} "
            + unit
            + "<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df_summary["Daily Avg Min"],
            name="Ensemble Avg Min",
            mode="lines+markers",
            line={"color": "#1D4ED8", "width": 3.5},
            marker={"size": 7},
            hovertemplate="<b>%{x}</b><br>Ensemble Avg Min: %{y:.1f} "
            + unit
            + "<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df_summary["Daily Mean Temp"],
            name="Daily Mean Temp",
            mode="lines",
            line={"color": "#4B5563", "width": 2, "dash": "dot"},
            hovertemplate="<b>%{x}</b><br>Daily Mean: %{y:.1f} "
            + unit
            + "<extra></extra>",
        )
    )

    # 3. Individual Models
    for idx, col in enumerate(df_max.columns):
        col_color = MODEL_PALETTE[idx % len(MODEL_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df_max[col],
                name=f"{col} (Max)",
                mode="lines",
                line={"color": col_color, "width": 1.5, "dash": "dash"},
                opacity=0.65,
                hovertemplate=f"<b>%{{x}}</b><br>{col} Max: %{{y:.1f}} {unit}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df_min[col],
                name=f"{col} (Min)",
                mode="lines",
                line={"color": col_color, "width": 1.2, "dash": "dot"},
                opacity=0.55,
                hovertemplate=f"<b>%{{x}}</b><br>{col} Min: %{{y:.1f}} {unit}<extra></extra>",
            )
        )

    fig.update_layout(
        title={
            "text": f"Temperature Consensus & Multi-Model Forecast \u2014 {city}",
            "font": {"size": 18},
        },
        xaxis={
            "title": "Date",
            "gridcolor": "#F3F4F6",
            "tickformat": "%a, %b %d",
            "showgrid": True,
        },
        yaxis={
            "title": f"Temperature ({unit})",
            "gridcolor": "#F3F4F6",
            "zeroline": True,
            "zerolinecolor": "#CBD5E1",
        },
        hovermode="x unified",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.38,
            "xanchor": "center",
            "x": 0.5,
        },
        margin={"l": 40, "r": 40, "t": 50, "b": 100},
        height=520,
    )
    return fig


def plot_rain_forecast(
    df_rain: pd.DataFrame, df_summary: pd.DataFrame, city: str
) -> go.Figure:
    """Grouped bar & trend chart comparing daily rainfall predictions (mm) across models."""
    dates = df_summary.index.tolist()
    fig = go.Figure()

    # Per-model bars
    for idx, col in enumerate(df_rain.columns):
        color = MODEL_PALETTE[idx % len(MODEL_PALETTE)]
        fig.add_trace(
            go.Bar(
                x=dates,
                y=df_rain[col],
                name=f"{col}",
                marker={"color": color, "opacity": 0.8},
                hovertemplate=f"<b>%{{x}}</b><br>{col}: %{{y:.1f}} mm<extra></extra>",
            )
        )

    # Ensemble Average Rain line overlay
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df_summary["Daily Avg Rain (mm)"],
            name="Ensemble Avg Rain",
            mode="lines+markers",
            line={"color": "#0F172A", "width": 3.5},
            marker={"size": 8, "symbol": "diamond"},
            hovertemplate="<b>%{x}</b><br>Ensemble Avg Rain: %{y:.1f} mm<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": f"Daily Precipitation (Rainfall) Forecast by Model \u2014 {city}",
            "font": {"size": 18},
        },
        xaxis={
            "title": "Date",
            "gridcolor": "#F3F4F6",
            "tickformat": "%a, %b %d",
            "showgrid": True,
        },
        yaxis={
            "title": "Daily Precipitation (mm)",
            "gridcolor": "#F3F4F6",
            "rangemode": "tozero",
        },
        barmode="group",
        hovermode="x unified",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.35,
            "xanchor": "center",
            "x": 0.5,
        },
        margin={"l": 40, "r": 40, "t": 50, "b": 90},
        height=480,
    )
    return fig


def plot_cloud_forecast(
    df_cloud: pd.DataFrame, df_summary: pd.DataFrame, city: str
) -> go.Figure:
    """Line chart tracking cloud cover percentage (0-100%) with qualitative sky cover bands."""
    dates = df_summary.index.tolist()
    fig = go.Figure()

    # Qualitative background bands for sky condition
    fig.add_hrect(
        y0=0,
        y1=20,
        fillcolor="#FEF9C3",
        opacity=0.35,
        layer="below",
        line_width=0,
        annotation_text="\u2600\ufe0f Sunny / Clear (<20%)",
        annotation_position="top left",
    )
    fig.add_hrect(
        y0=20,
        y1=60,
        fillcolor="#F1F5F9",
        opacity=0.35,
        layer="below",
        line_width=0,
        annotation_text="\u26c5 Partly Cloudy (20\u201360%)",
        annotation_position="top left",
    )
    fig.add_hrect(
        y0=60,
        y1=85,
        fillcolor="#E2E8F0",
        opacity=0.35,
        layer="below",
        line_width=0,
        annotation_text="\U0001f325\ufe0f Mostly Cloudy (60\u201385%)",
        annotation_position="top left",
    )
    fig.add_hrect(
        y0=85,
        y1=100,
        fillcolor="#CBD5E1",
        opacity=0.35,
        layer="below",
        line_width=0,
        annotation_text="\u2601\ufe0f Overcast (>85%)",
        annotation_position="top left",
    )

    # Individual model traces
    for idx, col in enumerate(df_cloud.columns):
        color = MODEL_PALETTE[idx % len(MODEL_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df_cloud[col],
                name=f"{col}",
                mode="lines+markers",
                line={"color": color, "width": 2},
                marker={"size": 5},
                hovertemplate=f"<b>%{{x}}</b><br>{col}: %{{y:.0f}}%<extra></extra>",
            )
        )

    # Ensemble Average Cloudiness
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df_summary["Daily Avg Cloud (%)"],
            name="Ensemble Avg Cloud Cover",
            mode="lines+markers",
            line={"color": "#1E293B", "width": 4},
            marker={"size": 8, "symbol": "circle"},
            hovertemplate="<b>%{x}</b><br>Ensemble Avg Cloud Cover: %{y:.0f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": f"Daily Cloud Cover / Cloudiness Forecast (%) \u2014 {city}",
            "font": {"size": 18},
        },
        xaxis={
            "title": "Date",
            "gridcolor": "#F3F4F6",
            "tickformat": "%a, %b %d",
            "showgrid": True,
        },
        yaxis={
            "title": "Cloud Cover (%)",
            "range": [0, 105],
            "gridcolor": "#F3F4F6",
            "dtick": 20,
        },
        hovermode="x unified",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.35,
            "xanchor": "center",
            "x": 0.5,
        },
        margin={"l": 40, "r": 40, "t": 50, "b": 90},
        height=480,
    )
    return fig


def plot_temp_comparison(
    df: pd.DataFrame, ensemble_series: pd.Series, title: str, unit: str
) -> go.Figure:
    """Dedicated comparative chart for either Daily Max or Daily Min across models."""
    dates = df.index.tolist()
    fig = go.Figure()

    for idx, col in enumerate(df.columns):
        color = MODEL_PALETTE[idx % len(MODEL_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df[col],
                name=col,
                mode="lines+markers",
                line={"color": color, "width": 2},
                marker={"size": 6},
                hovertemplate=f"<b>%{{x}}</b><br>{col}: %{{y:.1f}} {unit}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ensemble_series,
            name="Ensemble Average",
            mode="lines",
            line={"color": "#111827", "width": 3.5, "dash": "longdash"},
            hovertemplate=f"<b>%{{x}}</b><br>Ensemble Average: %{{y:.1f}} {unit}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis={"title": "Date", "gridcolor": "#F3F4F6", "tickformat": "%a, %b %d"},
        yaxis={"title": f"Temperature ({unit})", "gridcolor": "#F3F4F6"},
        hovermode="x unified",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.3,
            "xanchor": "center",
            "x": 0.5,
        },
        margin={"l": 40, "r": 40, "t": 50, "b": 80},
        height=450,
    )
    return fig


def plot_spread(df_summary: pd.DataFrame, unit: str, city: str) -> go.Figure:
    """Bar chart indicating model disagreement / spread in temperature."""
    dates = df_summary.index.tolist()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=dates,
            y=df_summary["Model Spread"],
            name="Inter-Model Spread",
            marker={
                "color": df_summary["Model Spread"],
                "colorscale": "Blues",
                "showscale": True,
                "colorbar": {"title": f"Spread ({unit})"},
            },
            hovertemplate="<b>%{x}</b><br>Spread: %{y:.1f} " + unit + "<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Daily Model Temperature Spread (Highest Max \u2212 Lowest Min) \u2014 {city}",
        xaxis={"title": "Date", "gridcolor": "#F3F4F6", "tickformat": "%a, %b %d"},
        yaxis={"title": f"Spread ({unit})", "gridcolor": "#F3F4F6"},
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin={"l": 40, "r": 40, "t": 50, "b": 60},
        height=400,
    )
    return fig
