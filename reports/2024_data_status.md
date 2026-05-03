# 2024 Data Status

The supplied `kmo2024.pdf` includes appendix pages as full-page table images rather than extractable table cells or text. `pdfplumber` and `PyMuPDF` can render the pages, but they cannot recover incident rows from the embedded table image.

OCR attempts:

- macOS Vision through `ocrmac`: returned no usable text for the table.
- EasyOCR with Turkish and English recognition: returned hundreds of low-confidence fragments that did not preserve dates, city names, or table cells reliably.

Decision: 2024 is not included in `data/processed/kmo_incidents_clean.xlsx` because adding noisy OCR rows would contaminate the analysis.

Low-friction way to include 2024:

1. Open the 2024 PDF in Adobe Acrobat, Microsoft Word, Google Drive OCR, or another table OCR tool.
2. Export the appendix table to Excel.
3. Save it as `data/raw/kmo2024_manual.xlsx`.
4. Use these canonical columns:
   `Tarih`, `Olay Türü`, `Kayıp`, `Firma İsmi`, `Tutuşturma Kaynağı`, `Oluş Biçimleri`, `İl/İlçe`, `Yer`, `Tesis Türü`, `Sektör`, `Bölüm`, `Ekipman/Malzeme`, `Diğer`.
5. Run `python3 run_pipeline.py`.

The extractor will automatically load `data/raw/kmo2024_manual.xlsx` if it exists.
