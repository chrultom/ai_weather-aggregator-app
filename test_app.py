"""
Unit & integration tests for Weather Forecast Aggregator.
"""
import unittest
import numpy as np
import pandas as pd
from app import (
    geocode_city,
    fetch_open_meteo,
    fetch_7timer,
    process_forecast_data,
    AVAILABLE_MODELS,
)

class TestWeatherForecastAggregator(unittest.TestCase):

    def test_geocoding_valid_city(self):
        results, err = geocode_city("Szczecin")
        self.assertIsNone(err)
        self.assertIsNotNone(results)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["name"], "Szczecin")
        self.assertAlmostEqual(results[0]["latitude"], 53.428, places=1)
        self.assertAlmostEqual(results[0]["longitude"], 14.553, places=1)

    def test_geocoding_empty_input(self):
        results, err = geocode_city("   ")
        self.assertIsNone(results)
        self.assertIn("non-empty", err)

    def test_geocoding_nonexistent_city(self):
        results, err = geocode_city("xyz_fake_city_nonexistent_99999")
        self.assertIsNone(results)
        self.assertIsNotNone(err)

    def test_fetch_open_meteo_live(self):
        models = ["best_match", "metno_nordic", "ecmwf_ifs025", "gfs_seamless"]
        data, err = fetch_open_meteo(53.4289, 14.5530, models=models, forecast_days=14)
        self.assertIsNone(err)
        self.assertIsNotNone(data)
        self.assertIn("daily", data)
        self.assertIn("time", data["daily"])
        self.assertEqual(len(data["daily"]["time"]), 14)
        self.assertIn("temperature_2m_max_best_match", data["daily"])
        self.assertIn("temperature_2m_max_metno_nordic", data["daily"])
        self.assertIn("temperature_2m_max_ecmwf_ifs025", data["daily"])
        self.assertIn("temperature_2m_max_gfs_seamless", data["daily"])

    def test_fetch_7timer_live(self):
        data, err = fetch_7timer(53.4289, 14.5530)
        self.assertIsNone(err)
        self.assertIsNotNone(data)
        self.assertTrue(len(data) >= 7)
        first_date = next(iter(data))
        self.assertIn("max", data[first_date])
        self.assertIn("min", data[first_date])

    def test_process_forecast_data_ensemble_aggregation(self):
        mock_open_meteo = {
            "daily": {
                "time": ["2026-09-25", "2026-09-26"],
                "temperature_2m_max_best_match": [20.0, 22.0],
                "temperature_2m_min_best_match": [10.0, 12.0],
                "temperature_2m_max_metno_nordic": [18.0, None],  # missing 2nd day
                "temperature_2m_min_metno_nordic": [8.0, None],
            }
        }
        mock_7timer = {
            "2026-09-25": {"max": 22.0, "min": 12.0},
            "2026-09-26": {"max": 24.0, "min": 14.0},
        }

        df_max, df_min, df_summary = process_forecast_data(
            open_meteo_raw=mock_open_meteo,
            selected_models=["best_match", "metno_nordic"],
            include_7timer=True,
            seven_timer_data=mock_7timer,
            temp_unit="°C",
        )

        # Day 1: Best Match (20), Metno (18), 7Timer (22)
        # Avg Max = (20 + 18 + 22) / 3 = 20.0
        # Avg Min = (10 + 8 + 12) / 3 = 10.0
        # Overall Mean = (20.0 + 10.0) / 2 = 15.0
        # Highest Max = 22, Lowest Min = 8, Spread = 14
        self.assertAlmostEqual(df_summary.loc["2026-09-25", "Daily Avg Max"], 20.0)
        self.assertAlmostEqual(df_summary.loc["2026-09-25", "Daily Avg Min"], 10.0)
        self.assertAlmostEqual(df_summary.loc["2026-09-25", "Daily Mean Temp"], 15.0)
        self.assertEqual(df_summary.loc["2026-09-25", "Highest Max"], 22.0)
        self.assertEqual(df_summary.loc["2026-09-25", "Lowest Min"], 8.0)
        self.assertEqual(df_summary.loc["2026-09-25", "Model Spread"], 14.0)
        self.assertEqual(df_summary.loc["2026-09-25", "Sources Available"], 3)

        # Day 2: Metno is None (NaN) -> Avg over Best Match (22) and 7Timer (24)
        # Avg Max = (22 + 24) / 2 = 23.0
        # Avg Min = (12 + 14) / 2 = 13.0
        # Overall Mean = (23 + 13) / 2 = 18.0
        self.assertAlmostEqual(df_summary.loc["2026-09-26", "Daily Avg Max"], 23.0)
        self.assertAlmostEqual(df_summary.loc["2026-09-26", "Daily Avg Min"], 13.0)
        self.assertAlmostEqual(df_summary.loc["2026-09-26", "Daily Mean Temp"], 18.0)
        self.assertEqual(df_summary.loc["2026-09-26", "Sources Available"], 2)

    def test_fahrenheit_conversion(self):
        mock_open_meteo = {
            "daily": {
                "time": ["2026-09-25"],
                "temperature_2m_max_best_match": [0.0],
                "temperature_2m_min_best_match": [100.0],
            }
        }
        df_max, df_min, _ = process_forecast_data(
            open_meteo_raw=mock_open_meteo,
            selected_models=["best_match"],
            include_7timer=False,
            seven_timer_data=None,
            temp_unit="°F",
        )
        # 0 °C -> 32 °F, 100 °C -> 212 °F
        col = AVAILABLE_MODELS["best_match"]["label"]
        self.assertAlmostEqual(df_max.loc["2026-09-25", col], 32.0)
        self.assertAlmostEqual(df_min.loc["2026-09-25", col], 212.0)


if __name__ == "__main__":
    unittest.main()
