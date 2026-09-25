"""
Multi-Model Long-Term Weather Forecast Aggregator (7\u201314 Days)
Forecasting Temperature, Rain (Precipitation), and Cloudiness.
Integrates Open-Meteo (Best Match, MET Norway, ECMWF, GFS, ICON) & 7Timer!
"""

import datetime
import pandas as pd
import streamlit as st

from forecast_service import (
    AVAILABLE_MODELS,
    geocode_city,
    fetch_open_meteo,
    fetch_7timer,
    process_forecast_data,
)
from ui_components import (
    plot_master_consensus,
    plot_rain_forecast,
    plot_cloud_forecast,
    plot_temp_comparison,
    plot_spread,
)
from table_view import render_table_subpage

# ==============================================================================
# 1. Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="Multi-Model Weather Forecast Aggregator",
    page_icon="\u26c5",
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

# ==============================================================================
# 2. Sidebar Controls
# ==============================================================================
with st.sidebar:
    st.header("\u2699\ufe0f Forecast Settings")

    app_view = st.radio(
        "\U0001f5fa\ufe0f View Mode:",
        options=[
            "\U0001f4ca Visual Dashboard (Charts & KPIs)",
            "\U0001f4cb Table View",
        ],
        index=0,
    )

    st.markdown("---")

    city_input = st.text_input(
        "City Name:",
        value="Szczecin",
        help="Type any city worldwide (e.g. Szczecin, Oslo, Berlin, Warsaw, New York).",
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
        options=["\u00b0C", "\u00b0F"],
        index=0,
        horizontal=True,
    )

    st.markdown("---")
    st.subheader("Weather Models")

    checked_open_meteo = []
    for m_id, meta in AVAILABLE_MODELS.items():
        # Default checked models: best_match, metno_seamless, ecmwf_ifs025, gfs_seamless
        is_default = m_id in ["best_match", "metno_seamless", "ecmwf_ifs025", "gfs_seamless"]
        if st.checkbox(f"{meta['label']} ({meta['horizon']})", value=is_default, key=f"m_{m_id}"):
            checked_open_meteo.append(m_id)

    use_7timer = st.checkbox(
        "7Timer! Civil API (7 days)",
        value=True,
        help="Independent, keyless meteorological service providing civil temperature & weather.",
        key="m_7timer",
    )

    st.markdown("---")
    submit_clicked = st.button("\U0001f504 Update Forecast", type="primary", use_container_width=True)

    st.caption("Data Sources: Open-Meteo & 7Timer! APIs (Keyless & Free)")


# ==============================================================================
# 3. Main Dashboard Body
# ==============================================================================
st.markdown('<div class="main-header">\u26c5 Long-Term Weather Forecast Aggregator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multi-model 7\u201314 day forecast synthesizing Open-Meteo models (Best Match, MET Norway / yr.no, ECMWF, GFS, ICON) & 7Timer! for <b>Temperature</b>, <b>Rain</b>, and <b>Cloudiness</b></div>',
    unsafe_allow_html=True,
)

if not city_input.strip():
    st.warning("\U0001f448 Please enter a city name in the sidebar to begin.")
    st.stop()

if not checked_open_meteo and not use_7timer:
    st.error("Please select at least one weather model from the sidebar.")
    st.stop()

# --- Step A: Geocoding ---
with st.spinner(f"Geocoding '{city_input}'..."):
    locations, geo_err = geocode_city(city_input)

if geo_err:
    st.error(f"\u274c Geocoding Failed: {geo_err}")
    st.stop()

if not locations:
    st.error(f"\u274c No matching coordinates found for '{city_input}'. Please check spelling.")
    st.stop()

# Multi-result disambiguation if user entered ambiguous city name
selected_location = locations[0]
if len(locations) > 1:
    options_fmt = [
        f"{loc['name']}{', ' + loc['admin1'] if loc['admin1'] else ''}, {loc['country']} (Lat: {loc['latitude']:.2f}, Lon: {loc['longitude']:.2f})"
        for loc in locations
    ]
    with st.expander("\U0001f4cd Disambiguate Location", expanded=False):
        pick = st.selectbox("Select your target location:", range(len(options_fmt)), format_func=lambda i: options_fmt[i])
        selected_location = locations[pick]

lat = selected_location["latitude"]
lon = selected_location["longitude"]
city_clean_name = selected_location["name"]
country = selected_location["country"]
admin = selected_location["admin1"]
elev = selected_location["elevation"]

# Location Info Banner (Timezone removed as requested)
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("\U0001f4cd City / Region", f"{city_clean_name}", f"{admin}, {country}" if admin else country)
with c2:
    st.metric("\U0001f310 Coordinates & Elevation", f"{lat:.4f}\u00b0, {lon:.4f}\u00b0", f"Elevation: {elev} m" if elev != "N/A" else None)
with c3:
    total_sources = len(checked_open_meteo) + (1 if use_7timer else 0)
    met_badge = "MET Norway active" if "metno_seamless" in checked_open_meteo else "MET Norway off"
    st.metric("\U0001f4e1 Active Models", f"{total_sources} Sources", f"{days_horizon}-Day Horizon \u2022 {met_badge}")

st.markdown("---")

# --- Step B: Weather Data Fetching ---
with st.spinner("Fetching multi-variable forecast data across weather models..."):
    om_data, om_err = None, None
    if checked_open_meteo:
        om_data, om_err = fetch_open_meteo(lat, lon, checked_open_meteo, days_horizon)

    t7_data, t7_err = None, None
    if use_7timer:
        t7_data, t7_err = fetch_7timer(lat, lon)

if om_err:
    st.warning(f"\u26a0\ufe0f Open-Meteo Notice: {om_err}")
if t7_err and use_7timer:
    st.warning(f"\u2139\ufe0f 7Timer! Notice: {t7_err} (Continuing with available models)")

if not om_data and not t7_data:
    st.warning("\u274c Unable to retrieve data from any weather source. Please check connection.")
    st.stop()

# --- Step C: Data Processing & Aggregation ---
df_max, df_min, df_rain, df_cloud, df_weather, df_summary = process_forecast_data(
    open_meteo_raw=om_data,
    selected_models=checked_open_meteo,
    include_7timer=use_7timer,
    seven_timer_data=t7_data,
    temp_unit=unit_selection,
)

if df_summary.empty:
    st.error("Could not construct forecast dataframe.")
    st.stop()

# --- Subpage Routing: Table View ---
if app_view == "\U0001f4cb Table View":
    render_table_subpage(
        df_max=df_max,
        df_min=df_min,
        df_rain=df_rain,
        df_cloud=df_cloud,
        df_weather=df_weather,
        df_summary=df_summary,
        temp_unit=unit_selection,
        city_name=city_clean_name,
        horizon_days=days_horizon,
    )
    st.markdown("---")
    st.markdown(
        "<center><small style='color: #6B7280;'>Weather Forecast Aggregator \u2022 Open-Meteo & 7Timer! Keyless APIs \u2022 Built with Streamlit</small></center>",
        unsafe_allow_html=True,
    )
    st.stop()

# --- Step D: Comprehensive Summary KPIs ---
st.subheader("\U0001f4ca Forecast Highlights (Temperature \u2022 Rain \u2022 Cloudiness)")
k1, k2, k3, k4, k5 = st.columns(5)

today_idx = df_summary.index[0]
today_row = df_summary.iloc[0]

with k1:
    st.metric(
        label=f"Today ({today_idx})",
        value=f"{today_row['Daily Mean Temp']:.1f} {unit_selection}",
        delta=f"{today_row['Sky Condition']}",
        delta_color="off",
        help=f"Consensus Min: {today_row['Daily Avg Min']:.1f} {unit_selection} | Max: {today_row['Daily Avg Max']:.1f} {unit_selection}",
    )

with k2:
    total_rain = df_summary["Daily Avg Rain (mm)"].sum()
    st.metric(
        label=f"{len(df_summary)}-Day Total Rain",
        value=f"{total_rain:.1f} mm",
        delta="Precipitation Sum",
        delta_color="off",
        help="Cumulative expected rainfall across the forecast horizon.",
    )

with k3:
    wettest_day = df_summary["Daily Avg Rain (mm)"].idxmax()
    wettest_val = df_summary["Daily Avg Rain (mm)"].max()
    st.metric(
        label="Wettest Day",
        value=f"{wettest_val:.1f} mm",
        delta=f"On {wettest_day}" if wettest_val > 0 else "Dry Period Ahead",
        delta_color="inverse" if wettest_val > 5.0 else "off",
    )

with k4:
    period_avg_cloud = df_summary["Daily Avg Cloud (%)"].mean()
    cloud_label = (
        "Sunny / Clear" if period_avg_cloud < 25
        else "Partly Cloudy" if period_avg_cloud < 60
        else "Mostly Cloudy" if period_avg_cloud < 80
        else "Overcast"
    )
    st.metric(
        label="Avg Cloud Cover",
        value=f"{period_avg_cloud:.0f}%",
        delta=cloud_label,
        delta_color="off",
    )

with k5:
    peak_val = df_summary["Highest Max"].max()
    low_val = df_summary["Lowest Min"].min()
    st.metric(
        label=f"Temp Range ({len(df_summary)}d)",
        value=f"{peak_val:.1f} {unit_selection}",
        delta=f"Lowest: {low_val:.1f} {unit_selection}",
        delta_color="normal",
    )

st.markdown("<br>", unsafe_allow_html=True)

# --- Step E: Visualizations & Detailed Tabs ---
tab_temp, tab_rain, tab_cloud, tab_detailed_temp, tab_table, tab_models = st.tabs([
    "\U0001f321\ufe0f Temperature Consensus",
    "\U0001f327\ufe0f Rain / Precipitation",
    "\u2601\ufe0f Cloudiness & Sky Cover",
    "\u2600\ufe0f vs \U0001f319 Temp Extrema",
    "\U0001f4cb Comprehensive Table & CSV",
    "\u2139\ufe0f Weather Models Info",
])

with tab_temp:
    st.plotly_chart(
        plot_master_consensus(df_max, df_min, df_summary, unit_selection, city_clean_name),
        use_container_width=True,
    )
    st.caption("\U0001f4a1 **Tip:** Click on any legend item to toggle that model on or off. The light blue shaded corridor indicates the ensemble uncertainty envelope.")

with tab_rain:
    st.plotly_chart(
        plot_rain_forecast(df_rain, df_summary, city_clean_name),
        use_container_width=True,
    )
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        rainy_days = (df_summary["Daily Avg Rain (mm)"] >= 1.0).sum()
        st.info(f"\U0001f327\ufe0f **Rain Outlook:** {rainy_days} out of {len(df_summary)} days are projected to have noticeable rain (\u2265 1.0 mm). Total expected accumulation: **{total_rain:.1f} mm**.")
    with col_r2:
        max_single_model_rain = df_summary["Max Model Rain (mm)"].max()
        st.info(f"\u26a0\ufe0f **Peak Intensity:** The highest single-model rainfall forecast is **{max_single_model_rain:.1f} mm** on {df_summary['Max Model Rain (mm)'].idxmax()}.")

with tab_cloud:
    st.plotly_chart(
        plot_cloud_forecast(df_cloud, df_summary, city_clean_name),
        use_container_width=True,
    )
    sunniest_day = df_summary["Daily Avg Cloud (%)"].idxmin()
    sunniest_val = df_summary["Daily Avg Cloud (%)"].min()
    cloudiest_day = df_summary["Daily Avg Cloud (%)"].idxmax()
    cloudiest_val = df_summary["Daily Avg Cloud (%)"].max()
    st.info(
        f"\u2600\ufe0f **Sky Clearness Summary:** Sunniest projected day is **{sunniest_day}** (only {sunniest_val:.0f}% cloud cover). "
        f"Cloudiest day is **{cloudiest_day}** ({cloudiest_val:.0f}% cloud cover)."
    )

with tab_detailed_temp:
    st.plotly_chart(
        plot_temp_comparison(
            df_max,
            df_summary["Daily Avg Max"],
            f"Daily Maximum (Peak Daytime) Temperature Across Models \u2014 {city_clean_name}",
            unit_selection,
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        plot_temp_comparison(
            df_min,
            df_summary["Daily Avg Min"],
            f"Daily Minimum (Nighttime Low) Temperature Across Models \u2014 {city_clean_name}",
            unit_selection,
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        plot_spread(df_summary, unit_selection, city_clean_name),
        use_container_width=True,
    )

with tab_table:
    st.markdown("### Unified Multi-Variable Forecast Table")

    export_df = pd.DataFrame(index=df_summary.index)
    export_df[f"Ensemble Mean Temp ({unit_selection})"] = df_summary["Daily Mean Temp"].round(1)
    export_df[f"Avg Max Temp ({unit_selection})"] = df_summary["Daily Avg Max"].round(1)
    export_df[f"Avg Min Temp ({unit_selection})"] = df_summary["Daily Avg Min"].round(1)
    export_df["Avg Rain (mm)"] = df_summary["Daily Avg Rain (mm)"].round(1)
    export_df["Avg Cloud Cover (%)"] = df_summary["Daily Avg Cloud (%)"].round(0)
    export_df["Sky Condition"] = df_summary["Sky Condition"]
    export_df[f"Temp Spread ({unit_selection})"] = df_summary["Model Spread"].round(1)

    # Individual model temperatures and rain
    for col in df_max.columns:
        export_df[f"{col} Max ({unit_selection})"] = df_max[col].round(1)
    for col in df_rain.columns:
        export_df[f"{col} Rain (mm)"] = df_rain[col].round(1)

    st.dataframe(
        export_df.style.format(precision=1, na_rep="\u2014"),
        use_container_width=True,
        height=480,
    )

    csv_data = export_df.to_csv().encode("utf-8")
    st.download_button(
        label="\U0001f4e5 Download Full Weather Forecast (CSV)",
        data=csv_data,
        file_name=f"weather_forecast_{city_clean_name}_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

with tab_models:
    st.markdown("### Integrated Weather Forecast Models & Data Sources")
    st.markdown(
        """
        | Model / Source | Provider | Resolution | Horizon | Variables Forecasted | Description |
        |---|---|---|---|---|---|
        | **Open-Meteo Best Match** | Open-Meteo | Blended (~1\u201311 km) | 16 days | Temp, Rain, Clouds | Blends regional models (DWD ICON, Met Norway) with global ECMWF & GFS. |
        | **MET Norway / yr.no** | Norwegian Met Institute (yr.no) | Seamless (~2.5\u201313 km) | 14 days | Temp, Rain, Clouds | Official yr.no forecast model providing complete 14-day forecasts for Northern Europe and worldwide. |
        | **ECMWF IFS (0.25\u00b0)** | ECMWF | ~25 km | 14 days | Temp, Rain, Clouds | Renowned European Centre medium-range forecasting model. |
        | **NOAA GFS Seamless** | NOAA NWS (USA) | ~13\u201327 km | 14 days | Temp, Rain, Clouds | Flagship US Global Forecast System, updated 4x daily. |
        | **DWD ICON Seamless** | Deutscher Wetterdienst | ~7\u201313 km | 8 days | Temp, Rain, Clouds | Advanced non-hydrostatic model developed in Germany. |
        | **7Timer! Civil API** | 7Timer! / NOAA | ~0.25\u00b0 | 7 days | Temp, Weather, Clouds | Keyless meteorological service oriented around astronomical and civil forecasting. |
        """
    )
    st.markdown(
        """
        **Forecast Aggregation & Variables Guide:**
        1. **Temperature (\u00b0C or \u00b0F):** Daily Max, Min, and Mean. Mean is computed as $(Avg Max + Avg Min) / 2$.
        2. **Rain / Precipitation Sum (mm):** Total daily expected liquid accumulation. Computed across all numerical models.
        3. **Cloudiness / Cloud Cover (%):** Daily mean sky cloud cover from 0% (clear sky) to 100% (completely overcast).
        4. **yr.no Provider:** Uses the full `metno_seamless` engine from MET Norway, providing complete coverage across the entire 14-day horizon.
        """
    )

# Footer
st.markdown("---")
st.markdown(
    "<center><small style='color: #6B7280;'>Weather Forecast Aggregator \u2022 Open-Meteo & 7Timer! Keyless APIs \u2022 Built with Streamlit</small></center>",
    unsafe_allow_html=True,
)
