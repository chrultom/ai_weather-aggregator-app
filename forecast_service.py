"""
Forecast Service: Geocoding, multi-model weather fetching, and ensemble calculations.
Supported sources: Open-Meteo (Best Match, MET Norway / yr.no, ECMWF, GFS, ICON) and 7Timer!
Variables: Temperature (Max, Min, Mean), Rain / Precipitation Sum (mm), Cloud Cover (%).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import requests
import streamlit as st

API_TIMEOUT = 10  # seconds

AVAILABLE_MODELS = {
    "best_match": {
        "label": "Open-Meteo Best Match",
        "provider": "Open-Meteo Blended Ensemble",
        "horizon": "14 days",
        "color": "#2563EB",
    },
    "metno_seamless": {
        "label": "MET Norway / yr.no",
        "provider": "Norwegian Meteorological Institute (yr.no official)",
        "horizon": "14 days",
        "color": "#059669",
    },
    "ecmwf_ifs025": {
        "label": "ECMWF IFS (0.25\u00b0)",
        "provider": "European Centre for Medium-Range Forecasts",
        "horizon": "14 days",
        "color": "#7C3AED",
    },
    "gfs_seamless": {
        "label": "NOAA GFS Seamless",
        "provider": "US National Weather Service (NOAA)",
        "horizon": "14 days",
        "color": "#EA580C",
    },
    "icon_seamless": {
        "label": "DWD ICON Seamless",
        "provider": "German Weather Service (DWD)",
        "horizon": "8 days",
        "color": "#D97706",
    },
}


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
                })
            return standardized, None
    except requests.exceptions.RequestException:
        pass

    # 2. Fallback to OpenStreetMap Nominatim
    try:
        nom_url = "https://nominatim.openstreetmap.org/search"
        headers = {"User-Agent": "WeatherForecastAggregator/2.0"}
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
                })
            return standardized, None
    except Exception as e:
        return None, f"Geocoding network error: {e}"

    return None, f"Could not find location coordinates for '{city}'. Please verify spelling."


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_open_meteo(
    lat: float,
    lon: float,
    models: List[str],
    forecast_days: int = 14,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Fetches temperature, rain (precipitation sum), and cloud cover forecasts
    across selected numerical models via Open-Meteo.
    """
    if not models:
        return None, "No Open-Meteo models selected."

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": round(lat, 5),
        "longitude": round(lon, 5),
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "cloudcover_mean",
        ],
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
def fetch_7timer(lat: float, lon: float) -> Tuple[Optional[Dict[str, Dict[str, Any]]], Optional[str]]:
    """
    Fetches 7-day civil weather forecast from 7Timer! Keyless Meteorological API.
    Returns { 'YYYY-MM-DD': {'max': temp, 'min': temp, 'weather': str, 'cloud_est': float} }.
    """
    url = "http://www.7timer.info/bin/api.pl"
    params = {"lon": round(lon, 4), "lat": round(lat, 4), "product": "civillight", "output": "json"}

    weather_cloud_map = {
        "clear": 10.0,
        "pcloudy": 35.0,
        "mcloudy": 65.0,
        "cloudy": 90.0,
        "humid": 50.0,
        "lightrain": 85.0,
        "rain": 95.0,
        "snow": 90.0,
        "ts": 95.0,
        "tsrain": 100.0,
    }

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
            w_code = str(entry.get("weather", "clear")).lower()
            cloud_pct = weather_cloud_map.get(w_code, 50.0)

            if len(d_str) == 8 and "max" in t2m and "min" in t2m:
                formatted_d = f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}"
                forecast_map[formatted_d] = {
                    "max": float(t2m["max"]),
                    "min": float(t2m["min"]),
                    "weather": w_code,
                    "cloud_est": cloud_pct,
                }
        return forecast_map, None
    except requests.exceptions.Timeout:
        return None, "7Timer! request timed out."
    except requests.exceptions.RequestException as e:
        return None, f"7Timer! API error: {e}"
    except Exception as e:
        return None, f"7Timer! data error: {e}"


def get_sky_condition(cloud_pct: float, rain_mm: float) -> str:
    """Helper to convert cloud cover percentage and rain mm into an intuitive emoji & label."""
    if np.isnan(cloud_pct):
        return "N/A"
    if not np.isnan(rain_mm) and rain_mm >= 1.0:
        return f"\U0001f327\ufe0f Rain ({rain_mm:.1f} mm)"
    if not np.isnan(rain_mm) and rain_mm > 0.1:
        return f"\U0001f326\ufe0f Light Rain ({rain_mm:.1f} mm)"
    if cloud_pct < 20.0:
        return "\u2600\ufe0f Sunny / Clear"
    elif cloud_pct < 60.0:
        return "\u26c5 Partly Cloudy"
    elif cloud_pct < 85.0:
        return "\U0001f325\ufe0f Mostly Cloudy"
    else:
        return "\u2601\ufe0f Overcast"


def process_forecast_data(
    open_meteo_raw: Optional[Dict[str, Any]],
    selected_models: List[str],
    include_7timer: bool,
    seven_timer_data: Optional[Dict[str, Dict[str, Any]]],
    temp_unit: str = "\u00b0C",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Merges multi-variable data from all models into standardized DataFrames:
      - df_max: Daily maximum temperatures by source
      - df_min: Daily minimum temperatures by source
      - df_rain: Daily precipitation sum (mm) by source
      - df_cloud: Daily mean cloud cover (%) by source
      - df_summary: Daily aggregated ensemble averages, spreads, and weather conditions
    """
    if not open_meteo_raw or "daily" not in open_meteo_raw:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    daily = open_meteo_raw["daily"]
    dates = daily.get("time", [])
    if not dates:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_max = pd.DataFrame(index=dates)
    df_min = pd.DataFrame(index=dates)
    df_rain = pd.DataFrame(index=dates)
    df_cloud = pd.DataFrame(index=dates)

    single_model = len(selected_models) == 1

    for model_key in selected_models:
        label = AVAILABLE_MODELS.get(model_key, {}).get("label", model_key)
        if single_model and "temperature_2m_max" in daily:
            max_key = "temperature_2m_max"
            min_key = "temperature_2m_min"
            rain_key = "precipitation_sum"
            cloud_key = "cloudcover_mean"
        else:
            max_key = f"temperature_2m_max_{model_key}"
            min_key = f"temperature_2m_min_{model_key}"
            rain_key = f"precipitation_sum_{model_key}"
            cloud_key = f"cloudcover_mean_{model_key}"

        max_vals = daily.get(max_key, [None] * len(dates))
        min_vals = daily.get(min_key, [None] * len(dates))
        rain_vals = daily.get(rain_key, [None] * len(dates))
        cloud_vals = daily.get(cloud_key, [None] * len(dates))

        df_max[label] = [float(v) if v is not None else np.nan for v in max_vals]
        df_min[label] = [float(v) if v is not None else np.nan for v in min_vals]
        df_rain[label] = [float(v) if v is not None else np.nan for v in rain_vals]
        df_cloud[label] = [float(v) if v is not None else np.nan for v in cloud_vals]

    # Map 7Timer! if selected
    if include_7timer and seven_timer_data:
        col_name = "7Timer! Civil API"
        t7_max = [seven_timer_data.get(d, {}).get("max", np.nan) for d in dates]
        t7_min = [seven_timer_data.get(d, {}).get("min", np.nan) for d in dates]
        t7_cloud = [seven_timer_data.get(d, {}).get("cloud_est", np.nan) for d in dates]

        df_max[col_name] = t7_max
        df_min[col_name] = t7_min
        df_cloud[col_name] = t7_cloud

    # Fahrenheit conversion for temperature if requested
    if temp_unit == "\u00b0F":
        df_max = df_max * 9.0 / 5.0 + 32.0
        df_min = df_min * 9.0 / 5.0 + 32.0

    # Summary Ensemble Statistics
    summary_rows = []
    for d in dates:
        row_max = df_max.loc[d].dropna()
        row_min = df_min.loc[d].dropna()
        row_rain = df_rain.loc[d].dropna() if not df_rain.empty and d in df_rain.index else pd.Series(dtype=float)
        row_cloud = df_cloud.loc[d].dropna() if not df_cloud.empty and d in df_cloud.index else pd.Series(dtype=float)

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

        avg_rain = row_rain.mean() if len(row_rain) > 0 else 0.0
        max_rain = row_rain.max() if len(row_rain) > 0 else 0.0
        avg_cloud = row_cloud.mean() if len(row_cloud) > 0 else np.nan

        condition = get_sky_condition(avg_cloud, avg_rain)

        summary_rows.append({
            "Date": d,
            "Daily Avg Max": avg_max,
            "Daily Avg Min": avg_min,
            "Daily Mean Temp": overall_mean,
            "Highest Max": peak_max,
            "Lowest Min": low_min,
            "Model Spread": spread,
            "Daily Avg Rain (mm)": avg_rain,
            "Max Model Rain (mm)": max_rain,
            "Daily Avg Cloud (%)": avg_cloud,
            "Sky Condition": condition,
            "Sources Available": count,
        })

    df_summary = pd.DataFrame(summary_rows).set_index("Date")
    return df_max, df_min, df_rain, df_cloud, df_summary
