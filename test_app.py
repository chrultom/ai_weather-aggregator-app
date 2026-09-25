"""
Unit and integration test suite for Multi-Model Weather Forecast Aggregator.
Tests Temperature, Rain, Cloudiness, Geocoding, and MET Norway (metno_seamless) integration.
"""

import unittest
import numpy as np
import pandas as pd

from forecast_service import (
    AVAILABLE_MODELS,
    geocode_city,
    fetch_open_meteo,
    fetch_7timer,
    get_sky_condition,
    process_forecast_data,
)


class TestForecastService(unittest.TestCase):

    def test_geocoding_szczecin(self):
        results, err = geocode_city("Szczecin")
        self.assertIsNone(err)
        self.assertIsNotNone(results)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["name"], "Szczecin")
        self.assertAlmostEqual(results[0]["latitude"], 53.428, places=1)
        self.assertAlmostEqual(results[0]["longitude"], 14.553, places=1)
        # Verify timezone is not in the standardized location dict
        self.assertNotIn("timezone", results[0])

    def test_geocoding_invalid(self):
        results, err = geocode_city("   ")
        self.assertIsNone(results)
        self.assertIsNotNone(err)

    def test_metno_provider_live(self):
        """Verifies that the MET Norway provider (metno_seamless) works for the full 14 days."""
        models = ["metno_seamless"]
        data, err = fetch_open_meteo(53.4289, 14.5530, models=models, forecast_days=14)
        self.assertIsNone(err)
        self.assertIsNotNone(data)
        self.assertIn("daily", data)

        daily = data["daily"]
        self.assertEqual(len(daily["time"]), 14)
        self.assertIn("temperature_2m_max", daily)
        self.assertIn("precipitation_sum", daily)
        self.assertIn("cloudcover_mean", daily)

        # Check that MET Norway has non-null data
        self.assertIsNotNone(daily["temperature_2m_max"][0])
        self.assertIsNotNone(daily["precipitation_sum"][0])
        self.assertIsNotNone(daily["cloudcover_mean"][0])

    def test_multi_model_with_rain_and_clouds(self):
        """Verifies multi-model query with Best Match, MET Norway, ECMWF, and GFS."""
        models = ["best_match", "metno_seamless", "ecmwf_ifs025", "gfs_seamless"]
        data, err = fetch_open_meteo(53.4289, 14.5530, models=models, forecast_days=14)
        self.assertIsNone(err)
        daily = data["daily"]

        for m in models:
            self.assertIn(f"temperature_2m_max_{m}", daily)
            self.assertIn(f"precipitation_sum_{m}", daily)
            self.assertIn(f"cloudcover_mean_{m}", daily)

    def test_7timer_live(self):
        data, err = fetch_7timer(53.4289, 14.5530)
        self.assertIsNone(err)
        self.assertIsNotNone(data)
        self.assertTrue(len(data) >= 7)
        first_date = next(iter(data))
        self.assertIn("max", data[first_date])
        self.assertIn("min", data[first_date])
        self.assertIn("cloud_est", data[first_date])

    def test_get_sky_condition(self):
        self.assertIn("Sunny", get_sky_condition(10.0, 0.0))
        self.assertIn("Partly Cloudy", get_sky_condition(40.0, 0.0))
        self.assertIn("Mostly Cloudy", get_sky_condition(75.0, 0.0))
        self.assertIn("Overcast", get_sky_condition(95.0, 0.0))
        self.assertIn("Rain", get_sky_condition(95.0, 4.5))

    def test_process_forecast_data_multi_variable(self):
        mock_om = {
            "daily": {
                "time": ["2026-09-25", "2026-09-26"],
                "temperature_2m_max_best_match": [20.0, 22.0],
                "temperature_2m_min_best_match": [10.0, 12.0],
                "precipitation_sum_best_match": [0.0, 5.0],
                "cloudcover_mean_best_match": [25.0, 80.0],
                "temperature_2m_max_metno_seamless": [18.0, 20.0],
                "temperature_2m_min_metno_seamless": [8.0, 10.0],
                "precipitation_sum_metno_seamless": [0.2, 7.0],
                "cloudcover_mean_metno_seamless": [35.0, 90.0],
            }
        }
        df_max, df_min, df_rain, df_cloud, df_weather, df_summary = process_forecast_data(
            open_meteo_raw=mock_om,
            selected_models=["best_match", "metno_seamless"],
            include_7timer=False,
            seven_timer_data=None,
            temp_unit="\u00b0C",
        )

        # Day 1: Max Avg = 19.0, Min Avg = 9.0, Mean = 14.0
        # Rain Avg = (0.0 + 0.2) / 2 = 0.1 mm
        # Cloud Avg = (25 + 35) / 2 = 30.0%
        self.assertAlmostEqual(df_summary.loc["2026-09-25", "Daily Avg Max"], 19.0)
        self.assertAlmostEqual(df_summary.loc["2026-09-25", "Daily Avg Min"], 9.0)
        self.assertAlmostEqual(df_summary.loc["2026-09-25", "Daily Mean Temp"], 14.0)
        self.assertAlmostEqual(df_summary.loc["2026-09-25", "Daily Avg Rain (mm)"], 0.1)
        self.assertAlmostEqual(df_summary.loc["2026-09-25", "Daily Avg Cloud (%)"], 30.0)
        self.assertEqual(len(df_weather.columns), 2)

        # Day 2: Rain Avg = (5.0 + 7.0) / 2 = 6.0 mm, Max Model Rain = 7.0
        self.assertAlmostEqual(df_summary.loc["2026-09-26", "Daily Avg Rain (mm)"], 6.0)
        self.assertEqual(df_summary.loc["2026-09-26", "Max Model Rain (mm)"], 7.0)
        self.assertIn("Rain", df_summary.loc["2026-09-26", "Sky Condition"])

    def test_table_components(self):
        from table_view import format_table_date, build_ensemble_table_df, build_comparison_matrix
        formatted = format_table_date("2026-09-25")
        self.assertEqual(formatted, "Friday 25 Sep")

        mock_summary = pd.DataFrame([{
            "Date": "2026-09-25",
            "Daily Avg Max": 20.0,
            "Daily Avg Min": 10.0,
            "Daily Mean Temp": 15.0,
            "Daily Avg Rain (mm)": 0.5,
            "Daily Avg Cloud (%)": 50.0,
            "Model Spread": 2.0,
            "Sources Available": 3,
            "Sky Condition": "\u26c5 Partly Cloudy",
        }]).set_index("Date")

        ensemble_df = build_ensemble_table_df(mock_summary, "\u00b0C")
        self.assertEqual(len(ensemble_df), 1)
        self.assertIn("Friday 25 Sep", ensemble_df.index)
        self.assertEqual(ensemble_df.loc["Friday 25 Sep", "Weather Condition"], "\u26c5 Partly Cloudy")


if __name__ == "__main__":
    unittest.main()
