from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import pdfplumber


CANONICAL_COLUMNS = [
    "Tarih",
    "Olay Türü",
    "Kayıp",
    "Firma İsmi",
    "Tutuşturma Kaynağı",
    "Oluş Biçimleri",
    "İl/İlçe",
    "Yer",
    "Tesis Türü",
    "Sektör",
    "Bölüm",
    "Ekipman/Malzeme",
    "Diğer",
]

PDF_DIR = Path("/Users/kutluhangok/Desktop/DSA210 Analiz")
MANUAL_2024_PATH = Path("data/raw/kmo2024_manual.xlsx")


DATE_RE = re.compile(r"^\d{1,2}[./]\d{1,2}[./]\d{4}$")
DATE_TOKEN_RE = re.compile(r"^\d{1,2}[./]\d{1,2}[./]\d{4}$")


def _clean_cell(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("- ", "-")
    return text


def _looks_like_date(value: str) -> bool:
    return bool(DATE_RE.match(value.strip()))


def _extract_table_rows(pdf_path: Path) -> list[list[str]]:
    """Extract rows from PDFs where pdfplumber detects real table cells."""
    rows: list[list[str]] = []
    year = int(re.search(r"(\d{4})", pdf_path.stem).group(1))

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                if not table or not table[0]:
                    continue
                ncols = len(table[0])
                if ncols < 8:
                    continue
                for raw_row in table:
                    cells = [_clean_cell(c) for c in raw_row]
                    if not cells or any("Tarih" == c for c in cells[:2]):
                        continue

                    normalized: list[str] | None = None
                    if year == 2017:
                        normalized = _normalize_2017(cells)
                    elif ncols == 9:
                        normalized = _normalize_9_col(cells)
                    elif ncols == 11:
                        normalized = _normalize_11_col(cells)
                    elif ncols == 13 and _looks_like_date(cells[0]):
                        normalized = cells[:13]

                    if normalized and _looks_like_date(normalized[0]):
                        rows.append(normalized)
    return rows


def _normalize_2017(cells: list[str]) -> list[str] | None:
    if len(cells) < 8:
        return None
    cells = cells + [""] * (11 - len(cells))
    date = cells[0]
    if re.match(r"^\d{1,2}[./]\d{1,2}[./]201$", date) and cells[1].strip() == "7":
        date = date + "7"
    if not _looks_like_date(date):
        return None
    equip = " ".join(c for c in [cells[8], cells[9], cells[10]] if c).strip()
    return [
        date,
        cells[2],
        cells[3],
        "",
        "",
        "",
        cells[4],
        cells[5],
        cells[6],
        cells[7],
        "",
        equip,
        "",
    ]


def _normalize_9_col(cells: list[str]) -> list[str] | None:
    cells = cells + [""] * (9 - len(cells))
    if not _looks_like_date(cells[0]):
        return None
    return [
        cells[0],
        cells[1],
        cells[2],
        cells[3],
        cells[4],
        "",
        cells[5],
        "",
        cells[6],
        cells[7],
        "",
        cells[8],
        "",
    ]


def _normalize_11_col(cells: list[str]) -> list[str] | None:
    cells = cells + [""] * (11 - len(cells))
    if not _looks_like_date(cells[0]):
        return None
    return [
        cells[0],
        cells[1],
        cells[2],
        cells[3],
        cells[4],
        cells[5],
        cells[6],
        "",
        cells[7],
        cells[8],
        cells[9],
        cells[10],
        "",
    ]


def _boundaries(starts: Iterable[float]) -> list[float]:
    starts = list(starts)
    mids = [(a + b) / 2 for a, b in zip(starts, starts[1:])]
    return [-1.0, *mids, 9999.0]


COORD_LAYOUTS = {
    2022: {
        "starts": [10, 54, 98, 142, 187, 231, 275, 320, 366, 411, 455, 499, 543],
        "page_start": 22,
        "page_end": 44,
        "data_top": 55,
        "data_bottom": 790,
    },
    2023: {
        "starts": [6, 37, 70, 105, 160, 217, 253, 316, 364, 409, 460, 498, 548],
        "page_start": 24,
        "page_end": 38,
        "data_top": 55,
        "data_bottom": 790,
    },
}


def _assign_col(x0: float, boundaries: list[float]) -> int:
    for i in range(len(boundaries) - 1):
        if boundaries[i] <= x0 < boundaries[i + 1]:
            return i
    return len(boundaries) - 2


def _extract_coordinate_rows(pdf_path: Path) -> list[list[str]]:
    year_match = re.search(r"(\d{4})", pdf_path.stem)
    if not year_match:
        return []
    year = int(year_match.group(1))
    layout = COORD_LAYOUTS.get(year)
    if not layout:
        return []

    rows: list[list[str]] = []
    bounds = _boundaries(layout["starts"])

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_no in range(layout["page_start"], min(layout["page_end"], len(pdf.pages)) + 1):
            page = pdf.pages[page_no - 1]
            words = [
                w
                for w in page.extract_words(x_tolerance=1, y_tolerance=3)
                if layout["data_top"] <= w["top"] <= layout["data_bottom"]
            ]
            date_words = [w for w in words if DATE_TOKEN_RE.match(w["text"]) and w["x0"] < 45]
            date_words = sorted(date_words, key=lambda w: w["top"])
            for idx, date_word in enumerate(date_words):
                top = date_word["top"] - 3
                bottom = date_words[idx + 1]["top"] - 3 if idx + 1 < len(date_words) else layout["data_bottom"]
                record_words = [w for w in words if top <= w["top"] < bottom]
                cols: list[list[tuple[float, float, str]]] = [[] for _ in CANONICAL_COLUMNS]
                for word in record_words:
                    col_idx = _assign_col(word["x0"], bounds)
                    if 0 <= col_idx < len(cols):
                        cols[col_idx].append((word["top"], word["x0"], word["text"]))
                row = [" ".join(t for _, _, t in sorted(col)).strip() for col in cols]
                if _looks_like_date(row[0]):
                    rows.append(row)
    return rows


def extract_pdf(pdf_path: Path) -> pd.DataFrame:
    year = int(re.search(r"(\d{4})", pdf_path.stem).group(1))
    if year in COORD_LAYOUTS:
        rows = _extract_coordinate_rows(pdf_path)
        method = "coordinate_words"
    else:
        rows = _extract_table_rows(pdf_path)
        method = "pdfplumber_tables"

    df = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    df.insert(0, "source_file", pdf_path.name)
    df.insert(1, "source_year", year)
    df.insert(2, "extraction_method", method)
    return df


def extract_all(pdf_dir: Path = PDF_DIR, output_path: Path | None = None) -> pd.DataFrame:
    frames = []
    logs = []
    for pdf_path in sorted(pdf_dir.glob("kmo*.pdf")):
        year = int(re.search(r"(\d{4})", pdf_path.stem).group(1))
        if year == 2024:
            if MANUAL_2024_PATH.exists():
                manual = pd.read_excel(MANUAL_2024_PATH)
                for col in CANONICAL_COLUMNS:
                    if col not in manual.columns:
                        manual[col] = ""
                manual = manual[CANONICAL_COLUMNS].copy()
                manual.insert(0, "source_file", pdf_path.name)
                manual.insert(1, "source_year", year)
                manual.insert(2, "extraction_method", "manual_2024_excel")
                frames.append(manual)
                logs.append({"source_file": pdf_path.name, "status": "ok_manual", "reason": f"Loaded {MANUAL_2024_PATH}", "rows": len(manual)})
            else:
                logs.append(
                    {
                        "source_file": pdf_path.name,
                        "status": "skipped",
                        "reason": "Appendix pages are image-based and OCR tests were not reliable. Provide data/raw/kmo2024_manual.xlsx with canonical columns to include 2024.",
                        "rows": 0,
                    }
                )
            continue
        df_year = extract_pdf(pdf_path)
        frames.append(df_year)
        logs.append(
            {
                "source_file": pdf_path.name,
                "status": "ok",
                "reason": "",
                "rows": len(df_year),
            }
        )

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["source_file", "source_year", "extraction_method", *CANONICAL_COLUMNS])
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(output_path, index=False)
        pd.DataFrame(logs).to_csv(output_path.parent / "extraction_log.csv", index=False)
    return df


if __name__ == "__main__":
    out = Path("data/raw/kmo_incidents_raw.xlsx")
    extracted = extract_all(output_path=out)
    print(f"Extracted {len(extracted)} rows to {out}")
    print(extracted.groupby("source_year").size().to_string())
