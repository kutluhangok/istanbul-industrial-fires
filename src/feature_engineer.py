from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
import numpy as np


WEATHER_COLUMNS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "windspeed_10m_max",
]


def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 41.0082,
        "longitude": 28.9784,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(WEATHER_COLUMNS),
        "timezone": "Europe/Istanbul",
    }
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    daily = response.json()["daily"]
    df_weather = pd.DataFrame(daily)
    df_weather["date"] = pd.to_datetime(df_weather["time"])
    df_weather = df_weather.drop(columns=["time"])
    df_weather["extreme_heat"] = df_weather["temperature_2m_max"] > 30
    df_weather["low_humidity"] = df_weather["relative_humidity_2m_mean"] < 40
    df_weather["high_wind"] = df_weather["windspeed_10m_max"] > 40
    df_weather["has_precipitation"] = df_weather["precipitation_sum"] > 0.1
    df_weather["temp_lag1"] = df_weather["temperature_2m_mean"].shift(1)
    df_weather["humidity_lag1"] = df_weather["relative_humidity_2m_mean"].shift(1)
    return df_weather


def enrich_istanbul(clean: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    df_istanbul = clean[clean["is_istanbul"]].copy()
    df_istanbul["date"] = pd.to_datetime(df_istanbul["tarih_parsed"]).dt.normalize()
    weather = weather.copy()
    weather["date"] = pd.to_datetime(weather["date"]).dt.normalize()
    return df_istanbul.merge(weather, on="date", how="left")


def build_enrichment(
    clean_path: Path = Path("data/processed/kmo_incidents_clean.xlsx"),
    weather_path: Path = Path("data/enrichment/weather_daily.xlsx"),
    istanbul_path: Path = Path("data/processed/istanbul_enriched.xlsx"),
) -> pd.DataFrame:
    clean = pd.read_excel(clean_path)
    min_date = pd.to_datetime(clean["tarih_parsed"]).min().strftime("%Y-%m-%d")
    max_date = pd.to_datetime(clean["tarih_parsed"]).max().strftime("%Y-%m-%d")
    try:
        weather = fetch_weather(min_date, max_date)
    except Exception as exc:
        print(f"Weather API failed: {exc}. Continuing with missing weather values.")
        dates = pd.date_range(min_date, max_date, freq="D")
        weather = pd.DataFrame({"date": dates})
        for col in WEATHER_COLUMNS:
            weather[col] = np.nan
        weather["extreme_heat"] = False
        weather["low_humidity"] = False
        weather["high_wind"] = False
        weather["has_precipitation"] = False
        weather["temp_lag1"] = np.nan
        weather["humidity_lag1"] = np.nan

    weather_path.parent.mkdir(parents=True, exist_ok=True)
    istanbul_path.parent.mkdir(parents=True, exist_ok=True)
    weather.to_excel(weather_path, index=False)
    enriched = enrich_istanbul(clean, weather)
    enriched.to_excel(istanbul_path, index=False)
    return enriched


if __name__ == "__main__":
    result = build_enrichment()
    print(f"Istanbul enriched rows: {len(result)}")
