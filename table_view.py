"""
Multi-Model Weather Table View.
Provides complete tabular forecasts comparing all available models
(MET Norway, ECMWF, GFS, Best Match, ICON, 7Timer) on one page.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st


def format_table_date(date_str: str) -> str:
    """Formats '2026-09-25' into clean 'Friday 25 Sep'."""
    try:
        dt = pd.to_datetime(date_str)
        return dt.strftime("%A %d %b")
    except Exception:
        return date_str


def build_ensemble_table_df(df_summary: pd.DataFrame, temp_unit: str) -> pd.DataFrame:
    """Builds the primary consensus table matching standard daily layout."""
    rows = []
    for d in df_summary.index:
        row = df_summary.loc[d]
        date_label = format_table_date(d)
        sky = row.get("Sky Condition", "N/A")

        max_t = row.get("Daily Avg Max", np.nan)
        min_t = row.get("Daily Avg Min", np.nan)
        temp_str = f"{max_t:.1f}{temp_unit} / {min_t:.1f}{temp_unit}" if not np.isnan(max_t) and not np.isnan(min_t) else "\u2014"

        rain = row.get("Daily Avg Rain (mm)", 0.0)
        rain_str = f"{rain:.1f} mm" if not np.isnan(rain) else "\u2014"

        cloud = row.get("Daily Avg Cloud (%)", np.nan)
        cloud_str = f"{cloud:.0f}%" if not np.isnan(cloud) else "\u2014"

        spread = row.get("Model Spread", np.nan)
        spread_str = f"\u00b1{spread/2:.1f}{temp_unit}" if not np.isnan(spread) else "\u2014"

        sources = int(row.get("Sources Available", 0))

        rows.append({
            "Date": date_label,
            "Raw Date": d,
            "Weather Condition": sky,
            f"Max / Min Temp ({temp_unit})": temp_str,
            "Precipitation (Rain)": rain_str,
            "Cloud Cover": cloud_str,
            "Model Spread": spread_str,
            "Models Reporting": f"{sources} models",
        })

    return pd.DataFrame(rows).set_index("Date")


def build_single_model_table_df(
    df_max: pd.DataFrame,
    df_min: pd.DataFrame,
    df_rain: pd.DataFrame,
    df_cloud: pd.DataFrame,
    df_weather: pd.DataFrame,
    model_col: str,
    temp_unit: str,
) -> pd.DataFrame:
    """Builds a daily table for a specific model."""
    dates = df_max.index.tolist()
    rows = []

    for d in dates:
        date_label = format_table_date(d)

        # Weather string
        weather_val = df_weather.loc[d, model_col] if model_col in df_weather.columns and d in df_weather.index else "\u2014"
        if pd.isna(weather_val):
            weather_val = "\u2014"

        # Max / Min
        t_max = df_max.loc[d, model_col] if model_col in df_max.columns and d in df_max.index else np.nan
        t_min = df_min.loc[d, model_col] if model_col in df_min.columns and d in df_min.index else np.nan
        if not np.isnan(t_max) and not np.isnan(t_min):
            temp_str = f"{t_max:.1f}{temp_unit} / {t_min:.1f}{temp_unit}"
        elif not np.isnan(t_max):
            temp_str = f"{t_max:.1f}{temp_unit} / \u2014"
        else:
            temp_str = "\u2014"

        # Rain
        r_val = df_rain.loc[d, model_col] if model_col in df_rain.columns and d in df_rain.index else np.nan
        rain_str = f"{r_val:.1f} mm" if not np.isnan(r_val) else "\u2014"

        # Cloud
        c_val = df_cloud.loc[d, model_col] if model_col in df_cloud.columns and d in df_cloud.index else np.nan
        cloud_str = f"{c_val:.0f}%" if not np.isnan(c_val) else "\u2014"

        rows.append({
            "Date": date_label,
            "Weather Condition": weather_val,
            f"Max / Min Temp ({temp_unit})": temp_str,
            "Precipitation (Rain)": rain_str,
            "Cloud Cover": cloud_str,
        })

    return pd.DataFrame(rows).set_index("Date")



def build_comparison_matrix(
    df_max: pd.DataFrame,
    df_min: pd.DataFrame,
    df_rain: pd.DataFrame,
    df_weather: pd.DataFrame,
    df_summary: pd.DataFrame,
    temp_unit: str,
    view_metric: str,
) -> pd.DataFrame:
    """
    Constructs a multi-model matrix where rows are Dates and columns are Models.
    view_metric can be:
      - 'Overview (Weather • Temp • Rain)'
      - 'Max / Min Temperature'
      - 'Precipitation (Rain mm)'
      - 'Weather & Sky Condition'
    """
    dates = df_summary.index.tolist()
    matrix_rows = []

    models = df_max.columns.tolist()

    for d in dates:
        row_dict = {"Date": format_table_date(d)}

        # Add Ensemble Consensus first
        sum_row = df_summary.loc[d]
        e_max = sum_row.get("Daily Avg Max", np.nan)
        e_min = sum_row.get("Daily Avg Min", np.nan)
        e_rain = sum_row.get("Daily Avg Rain (mm)", 0.0)
        e_sky = sum_row.get("Sky Condition", "")

        if view_metric == "Overview (Weather \u2022 Temp \u2022 Rain)":
            row_dict["\U0001f3c6 Ensemble Consensus"] = f"{e_sky.split(' ')[0]} {e_max:.1f}\u00b0/{e_min:.1f}\u00b0 \u2022 {e_rain:.1f}mm"
        elif view_metric == "Max / Min Temperature":
            row_dict["\U0001f3c6 Ensemble Consensus"] = f"{e_max:.1f}{temp_unit} / {e_min:.1f}{temp_unit}"
        elif view_metric == "Precipitation (Rain mm)":
            row_dict["\U0001f3c6 Ensemble Consensus"] = f"{e_rain:.1f} mm"
        elif view_metric == "Weather & Sky Condition":
            row_dict["\U0001f3c6 Ensemble Consensus"] = e_sky

        # Add each model
        for m in models:
            m_max = df_max.loc[d, m] if m in df_max.columns and d in df_max.index else np.nan
            m_min = df_min.loc[d, m] if m in df_min.columns and d in df_min.index else np.nan
            m_rain = df_rain.loc[d, m] if m in df_rain.columns and d in df_rain.index else np.nan
            m_w = df_weather.loc[d, m] if m in df_weather.columns and d in df_weather.index else "\u2014"
            if pd.isna(m_w):
                m_w = "\u2014"

            if view_metric == "Overview (Weather \u2022 Temp \u2022 Rain)":
                w_symbol = m_w.split(" ")[0] if isinstance(m_w, str) and m_w != "\u2014" else "\u26c5"
                t_str = f"{m_max:.1f}\u00b0/{m_min:.1f}\u00b0" if not np.isnan(m_max) and not np.isnan(m_min) else "\u2014"
                r_str = f"{m_rain:.1f}mm" if not np.isnan(m_rain) else "\u2014"
                row_dict[m] = f"{w_symbol} {t_str} \u2022 {r_str}"
            elif view_metric == "Max / Min Temperature":
                row_dict[m] = f"{m_max:.1f}{temp_unit} / {m_min:.1f}{temp_unit}" if not np.isnan(m_max) and not np.isnan(m_min) else "\u2014"
            elif view_metric == "Precipitation (Rain mm)":
                row_dict[m] = f"{m_rain:.1f} mm" if not np.isnan(m_rain) else "\u2014"
            elif view_metric == "Weather & Sky Condition":
                row_dict[m] = m_w

        matrix_rows.append(row_dict)

    return pd.DataFrame(matrix_rows).set_index("Date")




def render_table_subpage(
    df_max: pd.DataFrame,
    df_min: pd.DataFrame,
    df_rain: pd.DataFrame,
    df_cloud: pd.DataFrame,
    df_weather: pd.DataFrame,
    df_summary: pd.DataFrame,
    temp_unit: str,
    city_name: str,
    horizon_days: int,
):
    """Renders the comprehensive multi-model table page."""
    st.markdown(
        '<div class="main-header">\U0001f4cb Multi-Model Forecast Table View</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="sub-header">Comprehensive daily forecast matrix comparing all available weather models on one page for <b>{city_name}</b> ({horizon_days}-day horizon)</div>',
        unsafe_allow_html=True,
    )

    # Subpage navigation tabs for table views
    table_tab_consensus, table_tab_matrix, table_tab_inspector, table_tab_models = st.tabs([
        "\U0001f3c6 Master Ensemble Table",
        "\U0001f4ca All Models Comparison Matrix (All on One Screen)",
        "\U0001f50d Daily Cross-Model Inspector",
        "\U0001f4d1 Individual Model Tables",
    ])

    with table_tab_consensus:
        st.subheader(f"\U0001f3c6 Master Ensemble Forecast Table \u2014 {city_name}")
        st.caption("Consensus forecast aggregated from all reporting weather models in a clean tabular view.")

        ensemble_df = build_ensemble_table_df(df_summary, temp_unit)
        st.dataframe(ensemble_df, use_container_width=True, height=520)

        csv_bytes = ensemble_df.to_csv().encode("utf-8")
        st.download_button(
            label="\U0001f4e5 Download Ensemble Table (CSV)",
            data=csv_bytes,
            file_name=f"ensemble_table_{city_name}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    with table_tab_matrix:
        st.subheader(f"\U0001f4ca All Weather Models Comparison Matrix \u2014 {city_name}")
        st.caption("Side-by-side comparative table showing all weather models simultaneously across each calendar day.")

        view_metric = st.selectbox(
            "Matrix Display Mode:",
            options=[
                "Overview (Weather \u2022 Temp \u2022 Rain)",
                "Max / Min Temperature",
                "Precipitation (Rain mm)",
                "Weather & Sky Condition",
            ],
            key="matrix_metric_selector",
        )

        matrix_df = build_comparison_matrix(
            df_max=df_max,
            df_min=df_min,
            df_rain=df_rain,
            df_weather=df_weather,
            df_summary=df_summary,
            temp_unit=temp_unit,
            view_metric=view_metric,
        )

        st.dataframe(matrix_df, use_container_width=True, height=520)

        csv_bytes_m = matrix_df.to_csv().encode("utf-8")
        st.download_button(
            label="\U0001f4e5 Download All-Models Matrix (CSV)",
            data=csv_bytes_m,
            file_name=f"all_models_matrix_{city_name}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="dl_matrix_csv",
        )



    with table_tab_inspector:
        st.subheader(f"\U0001f50d Daily Cross-Model Inspector \u2014 {city_name}")
        st.caption("Inspect any individual day in detail to see the exact predictions from every active model.")

        date_options = df_summary.index.tolist()
        date_labels = [f"{format_table_date(d)} ({d})" for d in date_options]

        chosen_idx = st.selectbox(
            "Select Date to Inspect:",
            range(len(date_options)),
            format_func=lambda i: date_labels[i],
            key="inspector_date_pick",
        )
        selected_d = date_options[chosen_idx]
        selected_label = format_table_date(selected_d)

        day_rows = []

        # Ensemble row first
        sum_row = df_summary.loc[selected_d]
        e_max = sum_row.get("Daily Avg Max", np.nan)
        e_min = sum_row.get("Daily Avg Min", np.nan)
        day_rows.append({
            "Forecast Source / Model": "\U0001f3c6 Ensemble Consensus",
            "Weather Condition": sum_row.get("Sky Condition", "N/A"),
            f"Max Temp ({temp_unit})": f"{e_max:.1f}" if not np.isnan(e_max) else "\u2014",
            f"Min Temp ({temp_unit})": f"{e_min:.1f}" if not np.isnan(e_min) else "\u2014",
            "Precipitation (Rain)": f"{sum_row.get('Daily Avg Rain (mm)', 0.0):.1f} mm",
            "Cloud Cover": f"{sum_row.get('Daily Avg Cloud (%)', 0):.0f}%",
            "Model Spread / Note": "Consensus ensemble average",
        })

        for m in df_max.columns:
            m_max = df_max.loc[selected_d, m] if m in df_max.columns else np.nan
            m_min = df_min.loc[selected_d, m] if m in df_min.columns else np.nan
            m_rain = df_rain.loc[selected_d, m] if m in df_rain.columns else np.nan
            m_cloud = df_cloud.loc[selected_d, m] if m in df_cloud.columns else np.nan
            m_w = df_weather.loc[selected_d, m] if m in df_weather.columns else "\u2014"
            if pd.isna(m_w):
                m_w = "\u2014"

            diff_str = f"{m_max - e_max:+.1f}{temp_unit}" if not np.isnan(m_max) and not np.isnan(e_max) else "\u2014"

            day_rows.append({
                "Forecast Source / Model": m,
                "Weather Condition": m_w,
                f"Max Temp ({temp_unit})": f"{m_max:.1f}" if not np.isnan(m_max) else "\u2014",
                f"Min Temp ({temp_unit})": f"{m_min:.1f}" if not np.isnan(m_min) else "\u2014",
                "Precipitation (Rain)": f"{m_rain:.1f} mm" if not np.isnan(m_rain) else "\u2014",
                "Cloud Cover": f"{m_cloud:.0f}%" if not np.isnan(m_cloud) else "\u2014",
                "Model Spread / Note": f"Diff from consensus: {diff_str}",
            })

        day_df = pd.DataFrame(day_rows).set_index("Forecast Source / Model")
        st.markdown(f"#### Forecast for **{selected_label}** across all active models:")
        st.dataframe(day_df, use_container_width=True)

    with table_tab_models:
        st.subheader(f"\U0001f4d1 Individual Model Tables \u2014 {city_name}")
        st.caption("View the full forecast table for each individual weather model.")

        model_tabs = st.tabs(df_max.columns.tolist())
        for idx, m_col in enumerate(df_max.columns):
            with model_tabs[idx]:
                st.markdown(f"#### Model: **{m_col}**")
                single_df = build_single_model_table_df(
                    df_max=df_max,
                    df_min=df_min,
                    df_rain=df_rain,
                    df_cloud=df_cloud,
                    df_weather=df_weather,
                    model_col=m_col,
                    temp_unit=temp_unit,
                )
                st.dataframe(single_df, use_container_width=True, height=450)
                csv_bytes_single = single_df.to_csv().encode("utf-8")
                st.download_button(
                    label=f"\U0001f4e5 Download {m_col} Table (CSV)",
                    data=csv_bytes_single,
                    file_name=f"table_{m_col}_{city_name}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key=f"dl_single_{m_col}",
                )

