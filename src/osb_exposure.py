from __future__ import annotations

import re
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


OSBUK_URL = "https://osbuk.org.tr/view/sayilarlaosb/osbliste.php"


def normalize_city(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def fetch_osbuk_table() -> pd.DataFrame:
    response = requests.get(OSBUK_URL, timeout=60, verify=False)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    if not tables:
        raise RuntimeError("No OSBÜK table found.")
    df = tables[0].copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def clean_osbuk_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(
        columns={
            "OSB İli": "il",
            "OSB Ünvanı": "osb_unvani",
            "OSB Türü": "osb_turu",
            "Fiili Durumu": "fiili_durumu",
            "Kuruluş Onayı Veren Bakanlık": "bakanlik",
            "OSB Alanı (Hektar)": "osb_alani_hektar",
            "Toplam Parsel Sayısı": "toplam_parsel_sayisi",
        }
    )
    keep = ["il", "osb_unvani", "osb_turu", "fiili_durumu", "bakanlik", "osb_alani_hektar", "toplam_parsel_sayisi"]
    out = out[[c for c in keep if c in out.columns]]
    out["il"] = out["il"].map(normalize_city)
    out["osb_alani_hektar"] = pd.to_numeric(out["osb_alani_hektar"], errors="coerce").fillna(0)
    out["toplam_parsel_sayisi"] = pd.to_numeric(out["toplam_parsel_sayisi"], errors="coerce").fillna(0)
    out["is_operational"] = out["fiili_durumu"].astype(str).str.contains("İŞLETMEDE|FAAL", case=False, na=False)
    return out


def aggregate_city_exposure(osb: pd.DataFrame) -> pd.DataFrame:
    grouped = osb.groupby("il", dropna=False).agg(
        osb_count=("osb_unvani", "count"),
        osb_operational_count=("is_operational", "sum"),
        osb_area_hectare=("osb_alani_hektar", "sum"),
        osb_parcels=("toplam_parsel_sayisi", "sum"),
        osb_operational_area_hectare=("osb_alani_hektar", lambda s: s[osb.loc[s.index, "is_operational"]].sum()),
        osb_operational_parcels=("toplam_parsel_sayisi", lambda s: s[osb.loc[s.index, "is_operational"]].sum()),
    ).reset_index()
    return grouped


def add_exposure_to_incidents(clean: pd.DataFrame, city_exposure: pd.DataFrame) -> pd.DataFrame:
    merged = clean.copy()
    merged["il"] = merged["il"].map(normalize_city)
    exposure = city_exposure.copy()
    exposure["il"] = exposure["il"].map(normalize_city)
    merged = merged.merge(exposure, on="il", how="left")
    exposure_cols = [
        "osb_count",
        "osb_operational_count",
        "osb_area_hectare",
        "osb_parcels",
        "osb_operational_area_hectare",
        "osb_operational_parcels",
    ]
    for col in exposure_cols:
        merged[col] = merged[col].fillna(0)
    merged["has_city_osb_exposure"] = merged["osb_count"] > 0
    return merged


def build_city_year_panel(clean: pd.DataFrame, city_exposure: pd.DataFrame) -> pd.DataFrame:
    panel = clean.groupby(["il", "year"], dropna=False).agg(
        incident_count=("Tarih", "count"),
        deaths=("olum", "sum"),
        injuries=("yaralanma", "sum"),
        high_severity_count=("severity", lambda s: (s == "high").sum()),
        explosion_count=("olay_turu", lambda s: (s == "explosion").sum()),
    ).reset_index()
    panel = panel.merge(city_exposure, on="il", how="left")
    exposure_cols = ["osb_count", "osb_operational_count", "osb_area_hectare", "osb_parcels", "osb_operational_area_hectare", "osb_operational_parcels"]
    for col in exposure_cols:
        panel[col] = panel[col].fillna(0)
    panel["incidents_per_1000_parcels"] = panel["incident_count"] / panel["osb_parcels"].replace(0, pd.NA) * 1000
    panel["incidents_per_1000_operational_parcels"] = panel["incident_count"] / panel["osb_operational_parcels"].replace(0, pd.NA) * 1000
    panel["incidents_per_100_hectare"] = panel["incident_count"] / panel["osb_area_hectare"].replace(0, pd.NA) * 100
    return panel


def build_osb_outputs(
    clean: pd.DataFrame,
    osb_raw_path: Path = Path("data/enrichment/osbuk_osb_list.xlsx"),
    city_path: Path = Path("data/enrichment/osb_exposure_by_city.xlsx"),
    panel_path: Path = Path("data/processed/city_year_osb_panel.xlsx"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = fetch_osbuk_table()
    osb = clean_osbuk_table(raw)
    city = aggregate_city_exposure(osb)
    panel = build_city_year_panel(clean, city)
    osb_raw_path.parent.mkdir(parents=True, exist_ok=True)
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    osb.to_excel(osb_raw_path, index=False)
    city.to_excel(city_path, index=False)
    panel.to_excel(panel_path, index=False)
    return osb, city, panel


if __name__ == "__main__":
    clean_df = pd.read_excel("data/processed/kmo_incidents_clean.xlsx")
    osb_df, city_df, panel_df = build_osb_outputs(clean_df)
    enriched = add_exposure_to_incidents(clean_df, city_df)
    enriched.to_excel("data/processed/kmo_incidents_clean.xlsx", index=False)
    print(f"OSBs: {len(osb_df)}, cities: {len(city_df)}, panel rows: {len(panel_df)}")
