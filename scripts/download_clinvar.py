"""Download ClinVar variant_summary and emit a cleaned, labeled parquet.

Run:
    python scripts/download_clinvar.py
    python scripts/download_clinvar.py --out data/interim/clinvar_labeled.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

CLINVAR_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"

COLS_TO_KEEP = [
    "GeneSymbol",
    "Name",
    "Type",
    "Assembly",
    "Oncogenicity",
    "ReviewStatusOncogenicity",
    "ClinicalSignificance",
]

AA_3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Glu": "E", "Gln": "Q", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Ter": "*", "=": "=",
}


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip download] {dest} already exists ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"Downloading {url} -> {dest}")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with dest.open("wb") as f, tqdm(
        desc="ClinVar", total=total, unit="iB", unit_scale=True, unit_divisor=1024
    ) as bar:
        for chunk in resp.iter_content(chunk_size=1024):
            bar.update(f.write(chunk))


def translate_3_to_1(protein_str: str) -> str:
    if pd.isna(protein_str):
        return protein_str
    out = str(protein_str)
    for three, one in AA_3_TO_1.items():
        out = out.replace(three, one)
    return out


def assign_label(row: pd.Series) -> str | None:
    """Oncogenicity first, fall back to ClinicalSignificance. Drop conflicting."""
    onco = str(row["Oncogenicity"]).lower()
    if "oncogenic" in onco and "benign" not in onco:
        return "oncogenic"
    if "benign" in onco and "oncogenic" not in onco:
        return "benign"

    clin = str(row["ClinicalSignificance"]).lower()
    if "conflicting" in clin:
        return None
    if "pathogenic" in clin:
        return "oncogenic"
    if "benign" in clin:
        return "benign"
    return None


def clean(raw_path: Path) -> pd.DataFrame:
    print(f"Reading {raw_path}")
    df = pd.read_csv(
        raw_path, sep="\t", compression="gzip", usecols=COLS_TO_KEEP, low_memory=False
    )
    print(f"  {len(df):,} raw rows")

    df = df[df["Assembly"] == "GRCh38"]
    df = df[df["Type"] == "single nucleotide variant"].copy()
    print(f"  {len(df):,} rows after GRCh38 + SNV filter")

    df["protein_change"] = df["Name"].str.extract(r"\(p\.(.*?)\)")
    df["protein_change_clean"] = df["protein_change"].apply(translate_3_to_1)

    df["ML_Label"] = df.apply(assign_label, axis=1)
    df = df.dropna(subset=["ML_Label", "protein_change_clean"]).copy()
    print(f"  {len(df):,} labeled rows")
    print(df["ML_Label"].value_counts().to_string())

    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw", type=Path)
    ap.add_argument("--out", default="data/interim/clinvar_labeled.parquet", type=Path)
    args = ap.parse_args()

    raw_file = args.raw_dir / "variant_summary.txt.gz"
    download_file(CLINVAR_URL, raw_file)

    df = clean(raw_file)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Wrote {args.out} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
