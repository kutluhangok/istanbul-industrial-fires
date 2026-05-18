from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.data_cleaner import OFFICIAL_PROVINCES


WEATHER_COLUMNS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "windspeed_10m_max",
]


def fetch_province_coordinates(provinces: list[str], cache_path: Path = Path("data/enrichment/province_coordinates.xlsx")) -> pd.DataFrame:
    if cache_path.exists():
        cached = pd.read_excel(cache_path)
        if set(provinces).issubset(set(cached["il"])):
            return cached[cached["il"].isin(provinces)].copy()

    rows = []
    for province in sorted(provinces):
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {"name": province, "count": 10, "language": "tr", "format": "json"}
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        results = response.json().get("results", [])
        tr_results = [r for r in results if r.get("country_code") == "TR"]
        result = tr_results[0] if tr_results else (results[0] if results else None)
        if result is None:
            raise RuntimeError(f"No coordinate found for province: {province}")
        rows.append(
            {
                "il": province,
                "latitude": result["latitude"],
                "longitude": result["longitude"],
                "geocode_name": result.get("name", province),
                "geocode_admin1": result.get("admin1", ""),
            }
        )

    df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(cache_path, index=False)
    return df


def fetch_weather(latitude: float, longitude: float, start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
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
    return df_weather


def fetch_weather_batch(coords: pd.DataFrame, start_date: str, end_date: str, batch_size: int = 25) -> pd.DataFrame:
    frames = []
    url = "https://archive-api.open-meteo.com/v1/archive"
    for start in range(0, len(coords), batch_size):
        batch = coords.iloc[start : start + batch_size].reset_index(drop=True)
        params = {
            "latitude": ",".join(batch["latitude"].astype(str)),
            "longitude": ",".join(batch["longitude"].astype(str)),
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(WEATHER_COLUMNS),
            "timezone": "Europe/Istanbul",
        }
        response = requests.get(url, params=params, timeout=120)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            payload = [payload]
        for location, row in zip(payload, batch.itertuples(index=False)):
            daily = pd.DataFrame(location["daily"])
            daily["date"] = pd.to_datetime(daily["time"])
            daily = daily.drop(columns=["time"])
            daily.insert(0, "il", row.il)
            daily.insert(1, "latitude", row.latitude)
            daily.insert(2, "longitude", row.longitude)
            frames.append(daily)
    return pd.concat(frames, ignore_index=True)


def fetch_nasa_power_weather(coords: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    frames = []
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    start_key = pd.Timestamp(start_date).strftime("%Y%m%d")
    end_key = pd.Timestamp(end_date).strftime("%Y%m%d")
    nasa_to_project = {
        "T2M_MAX": "temperature_2m_max",
        "T2M_MIN": "temperature_2m_min",
        "T2M": "temperature_2m_mean",
        "RH2M": "relative_humidity_2m_mean",
        "PRECTOTCORR": "precipitation_sum",
        "WS10M_MAX": "windspeed_10m_max",
    }
    for row in coords.itertuples(index=False):
        params = {
            "parameters": ",".join(nasa_to_project),
            "community": "RE",
            "longitude": row.longitude,
            "latitude": row.latitude,
            "start": start_key,
            "end": end_key,
            "format": "JSON",
        }
        response = requests.get(url, params=params, timeout=90)
        response.raise_for_status()
        parameters = response.json()["properties"]["parameter"]
        daily = pd.DataFrame({project_col: pd.Series(parameters[nasa_col]) for nasa_col, project_col in nasa_to_project.items()})
        daily["date"] = pd.to_datetime(daily.index)
        daily = daily.reset_index(drop=True)
        daily.insert(0, "il", row.il)
        daily.insert(1, "latitude", row.latitude)
        daily.insert(2, "longitude", row.longitude)
        frames.append(daily)
    weather = pd.concat(frames, ignore_index=True)
    weather["weather_source"] = "NASA POWER Daily API"
    return weather


def add_weather_flags(df_weather: pd.DataFrame) -> pd.DataFrame:
    df_weather = df_weather.copy()
    df_weather["extreme_heat"] = df_weather["temperature_2m_max"] > 30
    df_weather["low_humidity"] = df_weather["relative_humidity_2m_mean"] < 40
    df_weather["high_wind"] = df_weather["windspeed_10m_max"] > 40
    df_weather["has_precipitation"] = df_weather["precipitation_sum"] > 0.1
    df_weather["temp_lag1"] = df_weather.groupby("il")["temperature_2m_mean"].shift(1)
    df_weather["humidity_lag1"] = df_weather.groupby("il")["relative_humidity_2m_mean"].shift(1)
    return df_weather


def fetch_weather_by_province(clean: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    valid_provinces = set(OFFICIAL_PROVINCES)
    provinces = sorted(
        province
        for province in clean["il"].dropna().astype(str).loc[lambda s: s.str.len() > 0].unique()
        if province in valid_provinces
    )
    coords = fetch_province_coordinates(provinces)
    try:
        weather = fetch_weather_batch(coords, start_date, end_date)
        weather["weather_source"] = "Open-Meteo Archive API"
    except requests.HTTPError as exc:
        print(f"Open-Meteo failed ({exc}); falling back to NASA POWER Daily API.")
        weather = fetch_nasa_power_weather(coords, start_date, end_date)
    return add_weather_flags(weather)


def fallback_missing_weather(clean: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    dates = pd.date_range(start_date, end_date, freq="D")
    valid_provinces = set(OFFICIAL_PROVINCES)
    provinces = sorted(
        province
        for province in clean["il"].dropna().astype(str).loc[lambda s: s.str.len() > 0].unique()
        if province in valid_provinces
    )
    base = pd.MultiIndex.from_product([provinces, dates], names=["il", "date"]).to_frame(index=False)
    base["latitude"] = np.nan
    base["longitude"] = np.nan
    for col in WEATHER_COLUMNS:
        base[col] = np.nan
    base["extreme_heat"] = False
    base["low_humidity"] = False
    base["high_wind"] = False
    base["has_precipitation"] = False
    base["temp_lag1"] = np.nan
    base["humidity_lag1"] = np.nan
    return base


def enrich_incidents_with_weather(clean: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    enriched = clean.copy()
    existing_weather_cols = [col for col in [*WEATHER_COLUMNS, "extreme_heat", "low_humidity", "high_wind", "has_precipitation", "temp_lag1", "humidity_lag1", "weather_source"] if col in enriched.columns]
    enriched = enriched.drop(columns=existing_weather_cols)
    enriched["date"] = pd.to_datetime(enriched["tarih_parsed"]).dt.normalize()
    weather = weather.copy()
    weather["date"] = pd.to_datetime(weather["date"]).dt.normalize()
    weather_cols = ["il", "date", *WEATHER_COLUMNS, "extreme_heat", "low_humidity", "high_wind", "has_precipitation", "temp_lag1", "humidity_lag1", "weather_source"]
    enriched = enriched.merge(weather[weather_cols], on=["il", "date"], how="left")
    return enriched


def build_enrichment(
    clean: pd.DataFrame | None = None,
    clean_path: Path = Path("data/processed/kmo_incidents_clean.xlsx"),
    weather_path: Path = Path("data/enrichment/weather_daily_by_province.xlsx"),
    legacy_istanbul_weather_path: Path = Path("data/enrichment/weather_daily.xlsx"),
    istanbul_path: Path = Path("data/processed/istanbul_enriched.xlsx"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if clean is None:
        clean = pd.read_excel(clean_path)
    min_date = pd.to_datetime(clean["tarih_parsed"]).min().strftime("%Y-%m-%d")
    max_date = pd.to_datetime(clean["tarih_parsed"]).max().strftime("%Y-%m-%d")
    if weather_path.exists():
        cached_weather = pd.read_excel(weather_path)
        has_observed_weather = cached_weather[WEATHER_COLUMNS].notna().any().any()
        cached_min = pd.to_datetime(cached_weather["date"]).min().strftime("%Y-%m-%d") if not cached_weather.empty else ""
        cached_max = pd.to_datetime(cached_weather["date"]).max().strftime("%Y-%m-%d") if not cached_weather.empty else ""
        if has_observed_weather and cached_min <= min_date and cached_max >= max_date:
            weather = cached_weather
        else:
            weather = fetch_weather_by_province(clean, min_date, max_date)
    else:
        weather = fetch_weather_by_province(clean, min_date, max_date)

    weather_path.parent.mkdir(parents=True, exist_ok=True)
    istanbul_path.parent.mkdir(parents=True, exist_ok=True)
    weather.to_excel(weather_path, index=False)
    weather[weather["il"].eq("İstanbul")].to_excel(legacy_istanbul_weather_path, index=False)
    enriched = enrich_incidents_with_weather(clean, weather)
    enriched.to_excel(clean_path, index=False)
    enriched[enriched["is_istanbul"]].to_excel(istanbul_path, index=False)
    return enriched, weather


if __name__ == "__main__":
    result, weather_df = build_enrichment()
    print(f"Incidents enriched rows: {len(result)}")
    print(f"Province weather rows: {len(weather_df)}")
