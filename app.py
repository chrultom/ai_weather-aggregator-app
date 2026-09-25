"""
Weather Forecast Aggregator (7–14 Days)
Multi-model weather forecasting dashboard using Open-Meteo API & 7Timer! API.
"""

import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="Multi-Model Weather Forecast Aggregator",
    page_icon="⛅",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .stMetric {
        background: #F8FAFC;
        padding: 10px 14px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Supported Models Metadata
AVAILABLE_MODELS = {
    "best_match": {
        "label": "Open-Meteo Best Match",
        "provider": "Open-Meteo Blended",
        "horizon": "14 days",
        "color": "#2563EB",
    },
    "metno_nordic": {
        "label": "MET Norway Nordic (yr.no)",
        "provider": "Norwegian Meteorological Institute",
        "horizon": "3 days (Europe)",
        "color": "#059669",
    },
    "ecmwf_ifs025": {
        "label": "ECMWF IFS (0.25°)",
        "provider": "European Centre for Medium-Range Forecasts",
        "horizon": "14 days",
        "color": "#7C3AED",
    },
    "gfs_seamless": {
        "label": "NOAA GFS Seamless",
        "provider": "US National Weather Service",
        "horizon": "14 days",
        "color": "#EA580C",
    },
    "icon_seamless": {
        "label": "DWD ICON Seamless",
        "provider": "German Weather Service",
        "horizon": "8 days",
        "color": "#D97706",
    },
}

API_TIMEOUT = 10  # seconds


# ==============================================================================
# Geocoding Service (Open-Meteo Geocoding API with Nominatim Fallback)
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def geocode_city(city_query: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    Converts a city name to geographical coordinates using Open-Meteo Geocoding API.
    Falls back to OpenStreetMap Nominatim if needed.
    """
    city = city_query.strip()
    if not city:
        return None, "Please provide a non-empty city name."

    # 1. Try Open-Meteo Geocoding API
    try:
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {"name": city, "count": 5, "language": "en", "format": "json"}
        res = requests.get(url, params=params, timeout=API_TIMEOUT)
        res.raise_for_status()
        data = res.json()
        results = data.get("results", [])
        if results:
            standardized = []
            for r in results:
                standardized.append({
                    "name": r.get("name", city),
                    "country": r.get("country", ""),
                    "admin1": r.get("admin1", ""),
                    "latitude": float(r["latitude"]),
                    "longitude": float(r["longitude"]),
                    "elevation": r.get("elevation", "N/A"),
                    "timezone": r.get("timezone", "UTC"),
                })
            return standardized, None
    except requests.exceptions.RequestException:
        pass

    # 2. Fallback to OpenStreetMap Nominatim
    try:
        nom_url = "https://nominatim.openstreetmap.org/search"
        headers = {"User-Agent": "WeatherForecastAggregator/1.0"}
        nom_params = {"q": city, "format": "json", "limit": 5, "addressdetails": 1}
        res = requests.get(nom_url, params=nom_params, headers=headers, timeout=API_TIMEOUT)
        res.raise_for_status()
        nom_data = res.json()
        if nom_data:
            standardized = []
            for item in nom_data:
                addr = item.get("address", {})
                country = addr.get("country", "")
                state = addr.get("state", addr.get("county", ""))
                standardized.append({
                    "name": item.get("display_name", "").split(",")[0],
                    "country": country,
                    "admin1": state,
                    "latitude": float(item["lat"]),
                    "longitude": float(item["lon"]),
                    "elevation": "N/A",
                    "timezone": "UTC",
                })
            return standardized, None
    except Exception as e:
        return None, f"Geocoding network error: {e}"

    return None, f"Could not find location coordinates for '{city}'. Please verify spelling."


# ==============================================================================
# Weather Data Retrieval: Open-Meteo & 7Timer!
# ==============================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_open_meteo(
    lat: float,
    lon: float,
    models: List[str],
    forecast_days: int = 14,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Fetches 7-14 day forecast across selected numerical models via Open-Meteo."""
    if not models:
        return None, "No Open-Meteo models selected."

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": round(lat, 5),
        "longitude": round(lon, 5),
        "daily": ["temperature_2m_max", "temperature_2m_min"],
        "forecast_days": int(forecast_days),
        "timezone": "auto",
        "models": ",".join(models),
    }

    try:
        resp = requests.get(url, params=params, timeout=API_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            return None, f"Open-Meteo API Error: {data.get('reason', 'Unknown')}"
        return data, None
    except requests.exceptions.Timeout:
        return None, "Open-Meteo request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return None, f"Open-Meteo connection error: {e}"
    except Exception as e:
        return None, f"Error parsing Open-Meteo data: {e}"


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_7timer(lat: float, lon: float) -> Tuple[Optional[Dict[str, Dict[str, float]]], Optional[str]]:
    """
    Fetches 7-day civil weather forecast from 7Timer! Keyless Meteorological API.
    Returns { 'YYYY-MM-DD': {'max': temp, 'min': temp} }.
    """
    url = "http://www.7timer.info/bin/api.pl"
    params = {"lon": round(lon, 4), "lat": round(lat, 4), "product": "civillight", "output": "json"}
    try:
        resp = requests.get(url, params=params, timeout=API_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        series = data.get("dataseries", [])
        if not series:
            return None, "7Timer returned empty dataseries."

        forecast_map = {}
        for entry in series:
            d_str = str(entry.get("date", ""))
            t2m = entry.get("temp2m", {})
            if len(d_str) == 8 and "max" in t2m and "min" in t2m:
                formatted_d = f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}"
                forecast_map[formatted_d] = {"max": float(t2m["max"]), "min": float(t2m["min"])}
        return forecast_map, None
    except requests.exceptions.Timeout:
        return None, "7Timer! request timed out."
    except requests.exceptions.RequestException as e:
        return None, f"7Timer! API error: {e}"
    except Exception as e:
        return None, f"7Timer! data error: {e}"


# ==============================================================================
# Aggregation & Ensemble Logic
# ==============================================================================
def process_forecast_data(
    open_meteo_raw: Optional[Dict[str, Any]],
    selected_models: List[str],
    include_7timer: bool,
    seven_timer_data: Optional[Dict[str, Dict[str, float]]],
    temp_unit: str = "°C",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Merges data from all models into standardized DataFrames:
      - df_max: Daily maximum temperatures by source
      - df_min: Daily minimum temperatures by source
      - df_summary: Daily aggregated ensemble averages, spreads, and extrema
    """
    if not open_meteo_raw or "daily" not in open_meteo_raw:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    daily = open_meteo_raw["daily"]
    dates = daily.get("time", [])
    if not dates:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_max = pd.DataFrame(index=dates)
    df_min = pd.DataFrame(index=dates)

    single_model = len(selected_models) == 1

    for model_key in selected_models:
        label = AVAILABLE_MODELS.get(model_key, {}).get("label", model_key)
        if single_model and "temperature_2m_max" in daily:
            max_key = "temperature_2m_max"
            min_key = "temperature_2m_min"
        else:
            max_key = f"temperature_2m_max_{model_key}"
            min_key = f"temperature_2m_min_{model_key}"

        max_vals = daily.get(max_key, [None] * len(dates))
        min_vals = daily.get(min_key, [None] * len(dates))

        df_max[label] = [float(v) if v is not None else np.nan for v in max_vals]
        df_min[label] = [float(v) if v is not None else np.nan for v in min_vals]

    # Additional source: 7Timer! API
    if include_7timer and seven_timer_data:
        col_name = "7Timer! Civil API"
        t7_max = [seven_timer_data.get(d, {}).get("max", np.nan) for d in dates]
        t7_min = [seven_timer_data.get(d, {}).get("min", np.nan) for d in dates]
        df_max[col_name] = t7_max
        df_min[col_name] = t7_min

    # Fahrenheit conversion if needed
    if temp_unit == "°F":
        df_max = df_max * 9.0 / 5.0 + 32.0
        df_min = df_min * 9.0 / 5.0 + 32.0

    # Summary Ensemble Statistics
    summary_rows = []
    for d in dates:
        row_max = df_max.loc[d].dropna()
        row_min = df_min.loc[d].dropna()
        count = len(row_max)

        if count > 0:
            avg_max = row_max.mean()
            avg_min = row_min.mean()
            overall_mean = (avg_max + avg_min) / 2.0
            peak_max = row_max.max()
            low_min = row_min.min()
            spread = peak_max - low_min
        else:
            avg_max, avg_min, overall_mean, peak_max, low_min, spread = (
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            )

        summary_rows.append({
            "Date": d,
            "Daily Avg Max": avg_max,
            "Daily Avg Min": avg_min,
            "Daily Mean Temp": overall_mean,
            "Highest Max": peak_max,
            "Lowest Min": low_min,
            "Model Spread": spread,
            "Sources Available": count,
        })

    df_summary = pd.DataFrame(summary_rows).set_index("Date")
    return df_max, df_min, df_summary


# ==============================================================================
# Plotly Chart Creators
# ==============================================================================
def plot_master_consensus(
    df_max: pd.DataFrame,
    df_min: pd.DataFrame,
    df_summary: pd.DataFrame,
    unit: str,
    city: str,
) -> go.Figure:
    dates = df_summary.index.tolist()
    fig = go.Figure()

    # 1. Shaded Corridor (Ensemble Uncertainty)
    fig.add_trace(
        go.Scatter(
            x=dates + dates[::-1],
            y=df_summary["Highest Max"].tolist() + df_summary["Lowest Min"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(59, 130, 246, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Model Uncertainty Envelope",
        )
    )

    # 2. Ensemble Mean Lines
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df_summary["Daily Avg Max"],
            name="Ensemble Avg Max",
            mode="lines+markers",
            line=dict(color="#DC2626", width=3.5),
            marker=dict(size=7),
            hovertemplate="<b>%{x}</b><br>Ensemble Avg Max: %{y:.1f} " + unit + "<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df_summary["Daily Avg Min"],
            name="Ensemble Avg Min",
            mode="lines+markers",
            line=dict(color="#1D4ED8", width=3.5),
            marker=dict(size=7),
            hovertemplate="<b>%{x}</b><br>Ensemble Avg Min: %{y:.1f} " + unit + "<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df_summary["Daily Mean Temp"],
            name="Daily Mean Temp",
            mode="lines",
            line=dict(color="#4B5563", width=2, dash="dot"),
            hovertemplate="<b>%{x}</b><br>Daily Mean: %{y:.1f} " + unit + "<extra></extra>",
        )
    )

    # 3. Individual Models
    palette = ["#9333EA", "#059669", "#D97706", "#EA580C", "#0891B2", "#4F46E5"]
    for idx, col in enumerate(df_max.columns):
        col_color = palette[idx % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df_max[col],
                name=f"{col} (Max)",
                mode="lines",
                line=dict(color=col_color, width=1.5, dash="dash"),
                opacity=0.6,
                hovertemplate=f"<b>%{{x}}</b><br>{col} Max: %{{y:.1f}} {unit}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df_min[col],
                name=f"{col} (Min)",
                mode="lines",
                line=dict(color=col_color, width=1.2, dash="dot"),
                opacity=0.5,
                hovertemplate=f"<b>%{{x}}</b><br>{col} Min: %{{y:.1f}} {unit}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Multi-Model Consensus & Daily Temperature Forecast — {city}",
        xaxis=dict(title="Date", gridcolor="#F3F4F6", tickformat="%a, %b %d"),
        yaxis=dict(title=f"Temperature ({unit})", gridcolor="#F3F4F6", zeroline=True, zerolinecolor="#CBD5E1"),
        hovermode="x unified",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=50, b=90),
        height=520,
    )
    return fig


def plot_comparison(df: pd.DataFrame, ensemble_series: pd.Series, title: str, unit: str) -> go.Figure:
    dates = df.index.tolist()
    fig = go.Figure()
    palette = ["#2563EB", "#059669", "#7C3AED", "#EA580C", "#D97706", "#0891B2"]

    for idx, col in enumerate(df.columns):
        color = palette[idx % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df[col],
                name=col,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=6),
                hovertemplate=f"<b>%{{x}}</b><br>{col}: %{{y:.1f}} {unit}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ensemble_series,
            name="Ensemble Average",
            mode="lines",
            line=dict(color="#111827", width=3.5, dash="longdash"),
            hovertemplate=f"<b>%{{x}}</b><br>Ensemble Average: %{{y:.1f}} {unit}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(title="Date", gridcolor="#F3F4F6", tickformat="%a, %b %d"),
        yaxis=dict(title=f"Temperature ({unit})", gridcolor="#F3F4F6"),
        hovermode="x unified",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=50, b=80),
        height=450,
    )
    return fig


def plot_spread(df_summary: pd.DataFrame, unit: str, city: str) -> go.Figure:
    dates = df_summary.index.tolist()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=dates,
            y=df_summary["Model Spread"],
            name="Inter-Model Spread",
            marker=dict(
                color=df_summary["Model Spread"],
                colorscale="Blues",
                showscale=True,
                colorbar=dict(title=f"Spread ({unit})"),
            ),
            hovertemplate="<b>%{x}</b><br>Spread: %{y:.1f} " + unit + "<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Daily Model Disagreement / Spread (Highest Max − Lowest Min) — {city}",
        xaxis=dict(title="Date", gridcolor="#F3F4F6", tickformat="%a, %b %d"),
        yaxis=dict(title=f"Temperature Spread ({unit})", gridcolor="#F3F4F6"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=40, r=40, t=50, b=60),
        height=400,
    )
    return fig


# ==============================================================================
# UI Sidebar Controls
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Forecast Configuration")

    city_input = st.text_input(
        "City Name:",
        value="Szczecin",
        help="Type any city in the world (e.g. Szczecin, Oslo, Berlin, Warsaw, New York).",
    )

    quick_options = ["(Custom)", "Szczecin", "Warsaw", "Berlin", "Oslo", "London", "New York", "Tokyo"]
    chosen_quick = st.selectbox("Quick City Presets:", quick_options)
    if chosen_quick != "(Custom)":
        city_input = chosen_quick

    st.markdown("---")

    days_horizon = st.slider(
        "Forecast Horizon (Days):",
        min_value=7,
        max_value=14,
        value=14,
        step=1,
    )

    unit_selection = st.radio(
        "Temperature Unit:",
        options=["°C", "°F"],
        index=0,
        horizontal=True,
    )

    st.markdown("---")
    st.subheader("Weather Models")

    checked_open_meteo = []
    for m_id, meta in AVAILABLE_MODELS.items():
        is_default = m_id in ["best_match", "metno_nordic", "ecmwf_ifs025", "gfs_seamless"]
        if st.checkbox(f"{meta['label']} ({meta['horizon']})", value=is_default, key=f"m_{m_id}"):
            checked_open_meteo.append(m_id)

    use_7timer = st.checkbox(
        "7Timer! Civil API (7 days)",
        value=True,
        help="Independent, keyless meteorological service based on NOAA GFS.",
        key="m_7timer",
    )

    st.markdown("---")
    submit_clicked = st.button("🔄 Update Forecast", type="primary", use_container_width=True)

    st.caption("Data Sources: Open-Meteo & 7Timer! (Free & Keyless APIs)")


# ==============================================================================
# Application Main Interface
# ==============================================================================
st.markdown('<div class="main-header">⛅ Long-Term Weather Forecast Aggregator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Comparative 7–14 day forecast synthesizing Open-Meteo models (Best Match, MET Norway / yr.no, ECMWF, GFS) & 7Timer!</div>',
    unsafe_allow_html=True,
)

if not city_input.strip():
    st.warning("👈 Please enter a city name in the sidebar to begin.")
    st.stop()

if not checked_open_meteo and not use_7timer:
    st.error("Please select at least one weather model from the sidebar.")
    st.stop()

# 1. Geocoding
with st.spinner(f"Geocoding '{city_input}'..."):
    locations, geo_err = geocode_city(city_input)

if geo_err:
    st.error(f"❌ Geocoding Failed: {geo_err}")
    st.stop()

if not locations:
    st.error(f"❌ No matching coordinates found for '{city_input}'. Please check spelling.")
    st.stop()

# Multi-result disambiguation
selected_location = locations[0]
if len(locations) > 1:
    options_fmt = [
        f"{loc['name']}{', ' + loc['admin1'] if loc['admin1'] else ''}, {loc['country']} (Lat: {loc['latitude']:.2f}, Lon: {loc['longitude']:.2f})"
        for loc in locations
    ]
    with st.expander("📍 Disambiguate Location", expanded=False):
        pick = st.selectbox("Select your target location:", range(len(options_fmt)), format_func=lambda i: options_fmt[i])
        selected_location = locations[pick]

lat = selected_location["latitude"]
lon = selected_location["longitude"]
city_clean_name = selected_location["name"]
country = selected_location["country"]
admin = selected_location["admin1"]
tz = selected_location["timezone"]
elev = selected_location["elevation"]

# Header metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("📍 City / Region", f"{city_clean_name}", f"{admin}, {country}" if admin else country)
with c2:
    st.metric("🌐 Coordinates", f"{lat:.4f}°, {lon:.4f}°", f"Elev: {elev} m" if elev != "N/A" else None)
with c3:
    st.metric("🕒 Timezone", f"{tz}")
with c4:
    total_sources = len(checked_open_meteo) + (1 if use_7timer else 0)
    st.metric("📡 Active Sources", f"{total_sources} Models", f"{days_horizon}-Day Horizon")

st.markdown("---")

# 2. Fetch Data
with st.spinner("Fetching multi-model forecast data..."):
    om_data, om_err = None, None
    if checked_open_meteo:
        om_data, om_err = fetch_open_meteo(lat, lon, checked_open_meteo, days_horizon)

    t7_data, t7_err = None, None
    if use_7timer:
        t7_data, t7_err = fetch_7timer(lat, lon)

if om_err:
    st.error(f"⚠️ Open-Meteo Notice: {om_err}")
if t7_err and use_7timer:
    st.warning(f"ℹ️ 7Timer! Notice: {t7_err} (Continuing with available models)")

if not om_data and not t7_data:
    st.error("❌ Unable to retrieve data from any weather source. Please check connection.")
    st.stop()


# 3. Assemble and Aggregate
df_max, df_min, df_summary = process_forecast_data(
    open_meteo_raw=om_data,
    selected_models=checked_open_meteo,
    include_7timer=use_7timer,
    seven_timer_data=t7_data,
    temp_unit=unit_selection,
)

if df_summary.empty:
    st.error("Could not construct forecast dataframe.")
    st.stop()

# 4. Summary Key Indicators
st.subheader("📊 Aggregated Forecast Summary")
k1, k2, k3, k4, k5 = st.columns(5)

today_idx = df_summary.index[0]
today_row = df_summary.iloc[0]

with k1:
    st.metric(
        label=f"Today ({today_idx}) Mean",
        value=f"{today_row['Daily Mean Temp']:.1f} {unit_selection}",
        delta=f"Min: {today_row['Daily Avg Min']:.1f} / Max: {today_row['Daily Avg Max']:.1f}",
        delta_color="off",
    )
with k2:
    mean_val = df_summary["Daily Mean Temp"].mean()
    st.metric(
        label=f"{len(df_summary)}-Day Ensemble Mean",
        value=f"{mean_val:.1f} {unit_selection}",
    )
with k3:
    peak_date = df_summary["Highest Max"].idxmax()
    peak_val = df_summary["Highest Max"].max()
    st.metric(
        label="Forecast Peak Max",
        value=f"{peak_val:.1f} {unit_selection}",
        delta=f"On {peak_date}",
        delta_color="inverse",
    )
with k4:
    lowest_date = df_summary["Lowest Min"].idxmin()
    lowest_val = df_summary["Lowest Min"].min()
    st.metric(
        label="Forecast Lowest Min",
        value=f"{lowest_val:.1f} {unit_selection}",
        delta=f"On {lowest_date}",
        delta_color="normal",
    )
with k5:
    spread_val = df_summary["Model Spread"].mean()
    st.metric(
        label="Average Spread / Uncertainty",
        value=f"{spread_val:.1f} {unit_selection}",
        help="Average difference between the warmest and coldest model per day.",
    )

st.markdown("<br>", unsafe_allow_html=True)

# 5. Visualizations & Tabs
tab_all, tab_max_view, tab_min_view, tab_spread_view, tab_table_view, tab_docs = st.tabs([
    "📈 Consensus Overview",
    "☀️ Max Temperatures",
    "🌙 Min Temperatures",
    "📊 Model Spread & Uncertainty",
    "📋 Comparative Table & CSV",
    "ℹ️ Model Reference",
])

with tab_all:
    st.plotly_chart(
        plot_master_consensus(df_max, df_min, df_summary, unit_selection, city_clean_name),
        use_container_width=True,
    )
    st.caption("💡 **Tip:** Click on any legend item to toggle that model on or off in the chart.")

with tab_max_view:
    st.plotly_chart(
        plot_comparison(
            df_max,
            df_summary["Daily Avg Max"],
            f"Daily Maximum Temperatures Across Models — {city_clean_name}",
            unit_selection,
        ),
        use_container_width=True,
    )

with tab_min_view:
    st.plotly_chart(
        plot_comparison(
            df_min,
            df_summary["Daily Avg Min"],
            f"Daily Minimum Temperatures Across Models — {city_clean_name}",
            unit_selection,
        ),
        use_container_width=True,
    )

with tab_spread_view:
    st.plotly_chart(
        plot_spread(df_summary, unit_selection, city_clean_name),
        use_container_width=True,
    )
    st.info(
        "Inter-model spread highlights where numerical forecast models diverge. "
        "Forecast confidence is highest when the spread is narrow."
    )


with tab_table_view:
    st.markdown("### Daily Aggregated Forecast Table")

    export_df = pd.DataFrame(index=df_summary.index)
    export_df[f"Ensemble Mean ({unit_selection})"] = df_summary["Daily Mean Temp"].round(1)
    export_df[f"Avg Max ({unit_selection})"] = df_summary["Daily Avg Max"].round(1)
    export_df[f"Avg Min ({unit_selection})"] = df_summary["Daily Avg Min"].round(1)
    export_df[f"Spread ({unit_selection})"] = df_summary["Model Spread"].round(1)
    export_df["Available Sources"] = df_summary["Sources Available"]

    for col in df_max.columns:
        export_df[f"{col} Max ({unit_selection})"] = df_max[col].round(1)
        export_df[f"{col} Min ({unit_selection})"] = df_min[col].round(1)

    st.dataframe(
        export_df.style.format(precision=1, na_rep="—"),
        use_container_width=True,
        height=450,
    )

    csv_data = export_df.to_csv().encode("utf-8")
    st.download_button(
        label="📥 Download Full Forecast as CSV",
        data=csv_data,
        file_name=f"forecast_{city_clean_name}_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

with tab_docs:
    st.markdown("### Integrated Weather Models & Data Sources")
    st.markdown(
        """
        | Model / Source | Provider | Resolution | Horizon | Details |
        |---|---|---|---|---|
        | **Open-Meteo Best Match** | Open-Meteo | Blended (~1–11 km) | 16 days | Blends high-resolution regional models (DWD ICON, Met Norway) with global ECMWF & GFS. |
        | **MET Norway Nordic (yr.no)** | Norwegian Met Institute | ~2.5 km | 3 days | MEPS numerical prediction model behind yr.no, focusing on Scandinavia & Central/Northern Europe. |
        | **ECMWF IFS (0.25°)** | ECMWF | ~25 km | 14 days | Gold-standard European Centre medium-range forecasting model. |
        | **NOAA GFS Seamless** | NOAA NWS (USA) | ~13–27 km | 14 days | US flagship Global Forecast System. |
        | **DWD ICON Seamless** | Deutscher Wetterdienst | ~7–13 km | 8 days | Non-hydrostatic model developed in Germany. |
        | **7Timer! Civil API** | 7Timer! / NOAA | ~0.25° | 7 days | Keyless global meteorological API providing civil and astronomical forecasts. |
        """
    )
    st.markdown(
        """
        **Aggregation Methodology:**
        1. **Daily Avg Max & Min:** Arithmetic mean of all models available on that day.
        2. **Daily Mean Temperature:** Calculated as `(Daily Avg Max + Daily Avg Min) / 2`.
        3. **Spread / Uncertainty:** Calculated as `Highest Max - Lowest Min` across all reporting sources.
        4. **Handling Short Horizons:** Models with shorter horizons (e.g. MET Norway Nordic for 3 days or 7Timer for 7 days) contribute for the days they are active; subsequent days dynamically compute the ensemble across the active medium-range models without skew or missing values.
        """
    )

st.markdown("---")
st.markdown(
    "<center><small style='color: #6B7280;'>Weather Forecast Aggregator • Powered by Open-Meteo & 7Timer! APIs • Built with Streamlit</small></center>",
    unsafe_allow_html=True,
)

