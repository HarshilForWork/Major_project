"""Render the machine-readable data dictionary as docs/data_dictionary.md.

Joins three files in data/processed/:
  ppmi_tdpigd_dictionary.csv          bucket, dtype, completeness   (written by preprocess.py)
  ppmi_tdpigd_column_inventory.csv    role, source table
  ppmi_tdpigd_column_descriptions.csv human-readable description

The markdown is generated, never hand-edited: re-run `make docs` after
preprocessing changes so the document can't drift from the data.
"""
from pathlib import Path as _Path
import pandas as pd

_ROOT = _Path(__file__).resolve().parents[3]
P = _ROOT / "data" / "processed"
OUT = _ROOT / "docs" / "data_dictionary.md"

dd   = pd.read_csv(P / "ppmi_tdpigd_dictionary.csv")
inv  = pd.read_csv(P / "ppmi_tdpigd_column_inventory.csv")[["column", "role", "source_table"]]
desc = pd.read_csv(P / "ppmi_tdpigd_column_descriptions.csv")
m = dd.merge(inv, on="column", how="left").merge(desc, on="column", how="left")

# The dtype recorded at preprocessing time is pre-coercion (many columns show
# `object`). Report the type as the table is actually read back instead.
_long = pd.read_csv(P / "ppmi_tdpigd_long.csv", low_memory=False)
_kind = {"f": "numeric", "i": "integer", "b": "bool", "O": "text"}
m["dtype"] = m["column"].map(lambda c: _kind.get(_long[c].dtype.kind, str(_long[c].dtype)))

ORDER = [
    ("CHEAP_FEATURE",              "✅ Model input (X)"),
    ("RESOURCE_DEPENDENT_FEATURE", "⏸ Held out — imaging / genetics"),
    ("BANNED_LABEL_DEFINING",      "⛔ Banned — defines the label"),
    ("TARGET",                     "🎯 Targets (Y and derived)"),
    ("KEY",                        "🔑 Keys"),
    ("ADMIN",                      "🗂 Admin / provenance"),
]

lines = [
    "# Data Dictionary — `ppmi_tdpigd_long.csv`",
    "",
    "> **Generated** by `src/trace_pd/data/export_dictionary.py` (`make docs`). Do not edit by hand.",
    "",
    f"**{len(m)} columns** · 6,922 rows · one row per patient-visit · 439 patients.",
    "",
    "Every column's bucket is assigned **by rule** in `preprocess.py`, and training code selects",
    "features **by bucket only**. `tests/test_dataset_contract.py` fails if a label-defining",
    "column ever lands in a feature bucket.",
    "",
    "| Bucket | Columns | Used for training |",
    "|---|---|---|",
]
for b, title in ORDER:
    n = (m.bucket == b).sum()
    lines.append(f"| `{b}` | {n} | {'**yes**' if b == 'CHEAP_FEATURE' else 'no'} |")
lines.append("")

for b, title in ORDER:
    sub = m[m.bucket == b].sort_values("column")
    meaning = sub.bucket_meaning.iloc[0] if len(sub) else ""
    lines += ["---", "", f"## {title} — `{b}` ({len(sub)})", "", f"*{meaning}*", "",
              "| Column | Description | Source table | Type | % present | Unique |",
              "|---|---|---|---|---|---|"]
    for _, r in sub.iterrows():
        lines.append(
            f"| `{r.column}` | {r.description} | {r.source_table} | {r['dtype']} "
            f"| {r.pct_present:.1f} | {int(r.n_unique)} |")
    lines.append("")

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {OUT.relative_to(_ROOT)}  ({len(m)} columns)")
