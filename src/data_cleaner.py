from __future__ import annotations

import re
from datetime import datetime

import numpy as np
import pandas as pd


TURKISH_HOLIDAYS = {
    "01-01",
    "04-23",
    "05-01",
    "05-19",
    "07-15",
    "08-30",
    "10-29",
}

ISTANBUL_OSB_DISTRICTS = [
    "tuzla",
    "ikitelli",
    "hadımköy",
    "hadimkoy",
    "başakşehir",
    "basaksehir",
    "dudullu",
    "ümraniye",
    "umraniye",
    "esenyurt",
    "arnavutköy",
    "arnavutkoy",
    "silivri",
    "gebze",
    "dilovası",
    "dilovasi",
    "çerkezköy",
    "cerkezkoy",
    "velimeşe",
    "velimese",
    "avcılar",
    "avcilar",
]

SEKTOR_MAP = {
    "ağaç": "Ağaç,Kağıt,Mobilya",
    "agac": "Ağaç,Kağıt,Mobilya",
    "kağıt": "Ağaç,Kağıt,Mobilya",
    "kagit": "Ağaç,Kağıt,Mobilya",
    "mobilya": "Ağaç,Kağıt,Mobilya",
    "metal": "Metal",
    "demir": "Metal",
    "çelik": "Metal",
    "celik": "Metal",
    "tekstil": "Tekstil",
    "kauçuk": "Kauçuk,Plastik",
    "kaucuk": "Kauçuk,Plastik",
    "plastik": "Kauçuk,Plastik",
    "gıda": "Gıda",
    "gida": "Gıda",
    "petrokimya": "Petrokimya,Yağ",
    "kimya": "Petrokimya,Yağ",
    "yağ": "Petrokimya,Yağ",
    "yag": "Petrokimya,Yağ",
    "çimento": "Çimento,Cam,Seramik",
    "cimento": "Çimento,Cam,Seramik",
    "cam": "Çimento,Cam,Seramik",
    "seramik": "Çimento,Cam,Seramik",
    "enerji": "Enerji",
    "kozmetik": "Kozmetik,Temizlik",
    "temizlik": "Kozmetik,Temizlik",
    "ilaç": "İlaç",
    "ilac": "İlaç",
    "boya": "Boya",
}


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("- ", "-")
    text = text.replace(" / ", "/").replace("/ ", "/").replace(" /", "/")
    return text


def repair_hyphenated_words(text: object) -> str:
    text = _clean_text(text)
    replacements = {
        "Arnavut- köy": "Arnavutköy",
        "Ar-navutköy": "Arnavutköy",
        "Küçükçek- mece": "Küçükçekmece",
        "Kü-çükçekmece": "Küçükçekmece",
        "Kahraman- maraş": "Kahramanmaraş",
        "Kahraman- kazan": "Kahramankazan",
        "Mene- men": "Menemen",
        "Kemalpa- şa": "Kemalpaşa",
        "Ga-ziemir": "Gaziemir",
        "Ke-malpaşa": "Kemalpaşa",
        "Bayram-paşa": "Bayrampaşa",
        "Os-mangazi": "Osmangazi",
        "Bey-şehir": "Beyşehir",
        "Hem-şin": "Hemşin",
        "Merkeze- fendi": "Merkezefendi",
        "Nilu- fer": "Nilüfer",
        "Nilü- fer": "Nilüfer",
        "Fabri- kası": "Fabrikası",
        "Atöl- yesi": "Atölyesi",
        "Petro- kimya": "Petrokimya",
        "Kau- çuk": "Kauçuk",
        "P- lastik": "Plastik",
        "Kağıt,- Mobilya": "Kağıt,Mobilya",
        "Ağaç,-": "Ağaç,",
        "Kağıt,-": "Kağıt,",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"(?<=\w)- (?=\w)", "", text)
    return text.strip()


def parse_turkish_date(date_str: object) -> pd.Timestamp | pd.NaT:
    value = _clean_text(date_str)
    if not value or value == "-":
        return pd.NaT
    match = re.search(r"(\d{1,2})[./](\d{1,2})[./](\d{4})", value)
    if not match:
        return pd.NaT
    day, month, year = map(int, match.groups())
    try:
        return pd.Timestamp(datetime(year, month, day))
    except ValueError:
        return pd.NaT


def parse_casualties(value: object) -> tuple[int, int]:
    text = _clean_text(value).lower()
    if not text or text in {"-", "nan"}:
        return 0, 0
    text = text.replace("ölü:", "ölü ").replace("yaralı:", "yaralı ")
    dead = 0
    injured = 0
    m_dead = re.search(r"(?:ölü\s*[:.]?\s*(\d+)|(\d+)\s*[öo]l[üu])", text)
    m_injured = re.search(r"(?:yara-?\s*lı\s*[:.]?\s*(\d+)|(\d+)\s*yaral)", text)
    if m_dead:
        dead = int(next(g for g in m_dead.groups() if g))
    if m_injured:
        injured = int(next(g for g in m_injured.groups() if g))
    return dead, injured


def severity_label(row: pd.Series) -> str:
    if row["olum"] > 0 or row["yaralanma"] > 5:
        return "high"
    if row["yaralanma"] >= 1:
        return "medium"
    return "low"


def parse_location(value: object) -> tuple[str, str]:
    text = repair_hyphenated_words(value)
    if not text or text == "-":
        return "", ""
    parts = [p.strip() for p in text.split("/", 1)]
    il = parts[0]
    ilce = parts[1] if len(parts) > 1 else ""
    return il, ilce


def has_osb(row: pd.Series) -> bool:
    yer = _clean_text(row.get("Yer", "")).lower()
    ilce = _clean_text(row.get("ilce", "")).lower()
    if "osb" in yer or "organize sanayi" in yer:
        return True
    return any(district in ilce for district in ISTANBUL_OSB_DISTRICTS)


def standardize_sektor(value: object) -> str:
    text = repair_hyphenated_words(value)
    if not text or text in {"-", "nan"}:
        return "Bilinmeyen"
    lowered = (
        text.lower()
        .replace(",", " ")
        .replace("-", " ")
        .replace("ı", "i")
        .replace("ğ", "g")
        .replace("ü", "u")
        .replace("ş", "s")
        .replace("ö", "o")
        .replace("ç", "c")
    )
    normalized_map = {k.replace("ı", "i").replace("ğ", "g").replace("ü", "u").replace("ş", "s").replace("ö", "o").replace("ç", "c"): v for k, v in SEKTOR_MAP.items()}
    for key, value in normalized_map.items():
        if key in lowered:
            return value
    return "Bilinmeyen"


def standardize_event_type(value: object) -> str:
    text = _clean_text(value).lower()
    if "yang" in text:
        return "fire"
    if "patlama" in text:
        return "explosion"
    return "unknown"


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_weekend"] = df["day_of_week"] >= 5
    df["is_holiday"] = df["tarih_parsed"].apply(
        lambda d: d.strftime("%m-%d") in TURKISH_HOLIDAYS if pd.notna(d) else False
    )
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def clean_incidents(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].map(repair_hyphenated_words)

    df["tarih_parsed"] = df["Tarih"].map(parse_turkish_date)
    df["year"] = df["tarih_parsed"].dt.year.fillna(df["source_year"]).astype("Int64")
    df["month"] = df["tarih_parsed"].dt.month
    df["day_of_week"] = df["tarih_parsed"].dt.dayofweek
    df["season"] = df["month"].map(
        {
            12: "winter",
            1: "winter",
            2: "winter",
            3: "spring",
            4: "spring",
            5: "spring",
            6: "summer",
            7: "summer",
            8: "summer",
            9: "autumn",
            10: "autumn",
            11: "autumn",
        }
    )

    casualties = df["Kayıp"].map(parse_casualties)
    df[["olum", "yaralanma"]] = pd.DataFrame(casualties.tolist(), index=df.index)
    df["severity"] = df.apply(severity_label, axis=1)
    df[["il", "ilce"]] = pd.DataFrame(df["İl/İlçe"].map(parse_location).tolist(), index=df.index)
    df["is_istanbul"] = df["il"].str.contains("istanbul|İstanbul", case=False, na=False)
    df["has_osb"] = df.apply(has_osb, axis=1)
    df["sektor_std"] = df["Sektör"].map(standardize_sektor)
    df["olay_turu"] = df["Olay Türü"].map(standardize_event_type)
    df["is_duplicate"] = df.duplicated(subset=["tarih_parsed", "Firma İsmi", "ilce"], keep=False)
    df = add_calendar_features(df)

    cols = [
        "source_file",
        "source_year",
        "extraction_method",
        "Tarih",
        "tarih_parsed",
        "year",
        "month",
        "day_of_week",
        "season",
        "olay_turu",
        "olum",
        "yaralanma",
        "severity",
        "Firma İsmi",
        "il",
        "ilce",
        "is_istanbul",
        "has_osb",
        "Yer",
        "Tesis Türü",
        "sektor_std",
        "Sektör",
        "Tutuşturma Kaynağı",
        "Oluş Biçimleri",
        "Bölüm",
        "Ekipman/Malzeme",
        "Diğer",
        "is_weekend",
        "is_holiday",
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
        "is_duplicate",
    ]
    return df[[c for c in cols if c in df.columns]]


if __name__ == "__main__":
    raw = pd.read_excel("data/raw/kmo_incidents_raw.xlsx")
    clean = clean_incidents(raw)
    clean.to_excel("data/processed/kmo_incidents_clean.xlsx", index=False)
    print(f"Cleaned {len(clean)} rows")
    print(clean.groupby("year").size().to_string())
