from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.analysis import build_figures, hypothesis_tests, run_ml
from src.data_cleaner import clean_incidents
from src.feature_engineer import build_enrichment
from src.pdf_extractor import extract_all


def main() -> None:
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/enrichment").mkdir(parents=True, exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    raw = extract_all(output_path=Path("data/raw/kmo_incidents_raw.xlsx"))
    clean = clean_incidents(raw)
    clean.to_excel("data/processed/kmo_incidents_clean.xlsx", index=False)
    istanbul = build_enrichment()
    build_figures(clean, istanbul)
    hypotheses = hypothesis_tests(clean, istanbul)
    model_comparison = run_ml(clean, istanbul)

    summary = {
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(clean)),
        "rows_by_year": {str(k): int(v) for k, v in clean.groupby("year").size().items()},
        "istanbul_rows": int(clean["is_istanbul"].sum()),
        "figures": len(list(Path("figures").glob("*.png"))),
        "best_model": model_comparison.iloc[0].to_dict() if not model_comparison.empty else {},
        "hypothesis_results": hypotheses,
    }
    Path("reports/pipeline_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
