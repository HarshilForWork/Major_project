# Data

## Provenance

All data originate from the **Parkinson's Progression Markers Initiative (PPMI)**,
obtained through the LONI Image & Data Archive (<https://ida.loni.usc.edu>).
Extract date: **16 August 2026**, 19 source tables.

## Licence and redistribution

PPMI data are released under a **Data Use Agreement** and are *not* redistributable.
The extract is committed to this repository for the project team's convenience, so the
repository must remain **private** and shared only with people covered by the DUA.

## Layout

```
data/
├── raw/
│   ├── ppmi_csv/            23 source CSVs exactly as extracted from LONI
│   └── zips_as_downloaded/  the original archive downloads, unmodified
├── interim/                 scratch space for intermediate artifacts
└── processed/
    ├── ppmi_tdpigd_long.csv            6,922 x 75 - one row per patient-visit
    ├── ppmi_tdpigd_dictionary.csv      per-column role: feature / label / excluded
    └── ppmi_tdpigd_column_inventory.csv  column -> source table mapping
```

## Regenerating `processed/`

```bash
make preprocess
```

The preprocessing stage is deterministic — no random seeds — so the same raw
extract reproduces `ppmi_tdpigd_long.csv` byte for byte.
