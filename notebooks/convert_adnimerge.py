"""Convert the ADNIMERGE2 R package's source tables into plain CSVs.

ADNI's current download (ADNIMERGE2, the ATRI Biostatistics package) is NOT one
pre-merged table. It's an R package with ~235 separate raw tables, one .rda file
each, under <package_root>/data/. A single "ADNIMERGE.csv" no longer exists --
you build your own analysis table from these, the same way preprocess.py builds
ppmi_tdpigd_long.csv from 19 raw PPMI CSVs.

This script converts the TABLES WE ACTUALLY NEED for the longitudinal
explainability framework (TARGET_TABLES below) from .rda to .csv using pyreadr
(pure Python, no R install needed). One table currently fails to load (ADSL --
it uses a newer R storage format pyreadr can't parse); it isn't needed, since
PTDEMOG + DXSUM + REGISTRY cover the same ground.

Usage:
  python notebooks/convert_adnimerge.py                 # converts TARGET_TABLES only (fast)
  python notebooks/convert_adnimerge.py --all            # converts every .rda in the package (slow, ~235 files)
  python notebooks/convert_adnimerge.py TABLE1 TABLE2    # convert specific tables by name
"""
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # repo root
OUT_DIR = ROOT / "data" / "raw" / "adni_csv"

# The tables that map onto our PPMI structure (label source, demographics, visit
# registry, and the "cheap" repeat-visit clinical instruments). See the table
# above in the chat response / HANDOFF_AR1_SHAP.md for what each one is.
TARGET_TABLES = [
    "REGISTRY", "VISITS", "PTDEMOG", "DXSUM",
    "MMSE", "MOCA", "CDR", "GDSCALE", "FAQ",
    "NEUROBAT", "ECOGPT", "ECOGSP", "NPIQ", "VITALS", "APOERES",
]


def ensure_pyreadr():
    try:
        import pyreadr  # noqa: F401
    except ImportError:
        print("pyreadr not installed -- installing now (pip install pyreadr)...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyreadr"], check=True)
    import pyreadr
    return pyreadr


def find_package_root(search_root):
    """Find the extracted ADNIMERGE2 (or similarly named) package: any directory
    containing a data/ subfolder with multiple .rda files."""
    for data_dir in Path(search_root).rglob("data"):
        if data_dir.is_dir() and len(list(data_dir.glob("*.rda"))) > 5:
            return data_dir.parent
    return None


def main():
    pyreadr = ensure_pyreadr()

    args = sys.argv[1:]
    convert_all = "--all" in args
    explicit = [a for a in args if not a.startswith("--")]

    print(f"Searching for the extracted ADNIMERGE package under {ROOT.parent} ...")
    pkg_root = find_package_root(ROOT.parent) or find_package_root(ROOT) or find_package_root(Path.home() / "Downloads")
    if pkg_root is None:
        print("\nCould not find an extracted package (a folder with data/*.rda inside).")
        print("Extract the ADNIMERGE .tar.gz somewhere under the repo or its parent folder,")
        print("then re-run this script.")
        sys.exit(1)

    data_dir = pkg_root / "data"
    print(f"Found package: {pkg_root}  ({len(list(data_dir.glob('*.rda')))} tables available)")

    if explicit:
        targets = explicit
    elif convert_all:
        targets = sorted(p.stem for p in data_dir.glob("*.rda"))
    else:
        targets = TARGET_TABLES

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok, failed = [], []
    for name in targets:
        src = data_dir / f"{name}.rda"
        if not src.exists():
            failed.append((name, "file not found in package"))
            continue
        try:
            result = pyreadr.read_r(str(src))
            df = list(result.values())[0]
        except Exception as e:
            failed.append((name, str(e)[:80]))
            continue
        out_path = OUT_DIR / f"{name}.csv"
        df.to_csv(out_path, index=False)
        ok.append((name, df.shape))

    print(f"\nConverted {len(ok)}/{len(targets)} tables to {OUT_DIR}:")
    for name, shape in ok:
        print(f"  {name:<12} {shape[0]:>7,} rows x {shape[1]:>3} cols  ->  {name}.csv")
    if failed:
        print(f"\n{len(failed)} table(s) could not be converted:")
        for name, err in failed:
            print(f"  {name:<12} {err}")
        print("(ADSL failing is expected and not needed -- PTDEMOG + DXSUM + REGISTRY cover it.)")

    print(f"\nDone. Next: build data/preprocess_adni.py to join these on RID + VISCODE, the")
    print("same way preprocess.py joins the PPMI tables -- see HANDOFF_AR1_SHAP.md.")


if __name__ == "__main__":
    main()
