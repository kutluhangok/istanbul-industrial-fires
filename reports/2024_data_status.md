# 2024 Data Status

The supplied `kmo2024.pdf` includes appendix pages as full-page table images rather than extractable table cells or text. `pdfplumber` and `PyMuPDF` can render the pages, but they cannot recover incident rows from the embedded table image.

OCR attempts:

- macOS Vision through `ocrmac`: returned no usable text for the table.
- EasyOCR with Turkish and English recognition: returned hundreds of low-confidence fragments that did not preserve dates, city names, or table cells reliably.

Updated decision: a manually provided Excel file, `/Users/kutluhangok/Downloads/2024- YANGIN VE PATLAMA VERİLERİ - taslak.xlsx`, is now converted into `data/raw/kmo2024_manual.xlsx` and included in the pipeline.

Validation:

- 2024 rows: 720
- 2024 fires: 694
- 2024 explosions: 26
- Date range: 2024-01-02 to 2024-12-31

Column mapping:

- `İl` + `İlçe` -> `İl/İlçe`
- `Mahalle / OSB` -> `Yer`
- `Ek Bilgi` -> `Diğer`
- `Kaynak` is retained as a source/reference field.
