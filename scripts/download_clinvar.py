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
ONCOKB_GENE_LIST_URL = "https://www.oncokb.org/api/v1/utils/cancerGeneList"
REPO_ROOT = Path(__file__).resolve().parents[1]
ONCOKB_GENE_CACHE = REPO_ROOT / "data/interim/oncokb_cancer_genes.tsv"

COLS_TO_KEEP = [
    "GeneSymbol",
    "Name",
    "Type",
    "Assembly",
    "Oncogenicity",
    "ReviewStatusOncogenicity",
    "ClinicalSignificance",
    # Genomic coordinates for joining variant-level features
    # (e.g. UCSC phyloP/phastCons bigWigs in build_conservation_cache.py).
    "Chromosome",
    "PositionVCF",
    "ReferenceAlleleVCF",
    "AlternateAlleleVCF",
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


def load_oncokb_token() -> str | None:
    """Read ONCOKB_TOKEN from .env. Returns None if missing -- the cancer
    gene list endpoint is reachable without auth in practice, but with a
    token we get higher rate limits and avoid surprise lockouts."""
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return None
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, val = line.partition("=")
        if key.strip() == "ONCOKB_TOKEN":
            val = val.strip().strip('"').strip("'")
            return val or None
    return None


def load_cancer_genes(cache_path: Path = ONCOKB_GENE_CACHE) -> set[str]:
    """Return the set of HUGO gene symbols on the OncoKB cancer gene list.
    Cached as TSV after the first fetch so we don't re-hit the API.
    """
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        token = load_oncokb_token()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        print(f"Fetching OncoKB cancer gene list -> {cache_path}")
        resp = requests.get(ONCOKB_GENE_LIST_URL, headers=headers, timeout=30)
        resp.raise_for_status()
        genes_df = pd.DataFrame(resp.json())
        genes_df.to_csv(cache_path, sep="\t", index=False)
        print(f"  cached {len(genes_df):,} genes")
    else:
        genes_df = pd.read_csv(cache_path, sep="\t")
    return set(genes_df["hugoSymbol"].dropna().astype(str))


def translate_3_to_1(protein_str: str) -> str:
    if pd.isna(protein_str):
        return protein_str
    out = str(protein_str)
    for three, one in AA_3_TO_1.items():
        out = out.replace(three, one)
    return out


def assign_label(row: pd.Series) -> tuple[str | None, str | None]:
    """Return (label, source). Source is 'oncogenicity' if the dedicated
    column resolved it, 'clinsig_fallback' if we fell back to
    ClinicalSignificance, or None if dropped."""
    onco = str(row["Oncogenicity"]).lower()
    if "oncogenic" in onco and "benign" not in onco:
        return "oncogenic", "oncogenicity"
    if "benign" in onco and "oncogenic" not in onco:
        return "benign", "oncogenicity"

    clin = str(row["ClinicalSignificance"]).lower()
    if "conflicting" in clin:
        return None, None
    if "pathogenic" in clin:
        return "oncogenic", "clinsig_fallback"
    if "benign" in clin:
        return "benign", "clinsig_fallback"
    return None, None


def clean(raw_path: Path) -> pd.DataFrame:
    print(f"Reading {raw_path}")
    df = pd.read_csv(
        raw_path, sep="\t", compression="gzip", usecols=COLS_TO_KEEP, low_memory=False
    )
    print(f"  {len(df):,} raw rows")

    n_raw = len(df)
    df = df[df["Assembly"] == "GRCh38"]
    n_after_assembly = len(df)
    df = df[df["Type"] == "single nucleotide variant"].copy()
    n_after_snv = len(df)
    print(f"  {n_after_assembly:,} after GRCh38 filter "
          f"(dropped {n_raw - n_after_assembly:,})")
    print(f"  {n_after_snv:,} after SNV filter "
          f"(dropped {n_after_assembly - n_after_snv:,})")

    df["protein_change"] = df["Name"].str.extract(r"\(p\.(.*?)\)")
    df["protein_change_clean"] = df["protein_change"].apply(translate_3_to_1)

    # Keep only canonical single-residue substitutions (missense). Excludes
    # synonymous (e.g. A123=), nonsense (R123*), start-loss (M1?), and any
    # non-substitution HGVS that slipped past the SNV type filter. AlphaFold
    # WT-structure features are only meaningful when one residue swaps for
    # another, so this is a scope filter, not just a cleanup.
    missense_pat = r"[ACDEFGHIKLMNPQRSTVWY]\d+[ACDEFGHIKLMNPQRSTVWY]"
    is_missense = df["protein_change_clean"].str.fullmatch(missense_pat, na=False)

    # Bucket the non-missense rows so we can see what the SNV filter was
    # actually carrying. Mutually exclusive: NaN -> synonymous -> nonsense ->
    # other, in that priority. Stop-codon-synonymous (`*123=`) hits both `=`
    # and `*`; we count those as synonymous since the substitution at the
    # protein level is the no-op, not the stop.
    non_missense = df.loc[~is_missense, "protein_change_clean"]
    n_no_protein = non_missense.isna().sum()
    rest = non_missense.dropna()
    is_syn = rest.str.contains("=", regex=False)
    n_synonymous = is_syn.sum()
    rest_after_syn = rest[~is_syn]
    is_non = rest_after_syn.str.contains("*", regex=False)
    n_nonsense = is_non.sum()
    n_other = (~is_non).sum()

    df = df[is_missense]
    print(f"  {len(df):,} after missense filter "
          f"(dropped {n_after_snv - len(df):,}: "
          f"no_protein_change={n_no_protein:,}, "
          f"synonymous={n_synonymous:,}, "
          f"nonsense={n_nonsense:,}, "
          f"other={n_other:,})")

    n_pre_label = len(df)
    label_pairs = df.apply(assign_label, axis=1, result_type="expand")
    df["ML_Label"] = label_pairs[0]
    df["label_source"] = label_pairs[1]
    df = df.dropna(subset=["ML_Label", "protein_change_clean"]).copy()
    print(f"  {len(df):,} labeled rows "
          f"(dropped {n_pre_label - len(df):,} unlabeled / conflicting)")
    print("  by source (pre cancer-gene filter):")
    print(df.groupby(["label_source", "ML_Label"]).size().to_string())

    # Cancer-only filter. The clinsig_fallback rows pull in hereditary-disease
    # variants (CFTR, HBB, etc.) that aren't cancer-related; gate them by
    # OncoKB's cancer gene list. Oncogenicity-sourced rows are already curated
    # by ClinVar as cancer-relevant so we leave them alone.
    cancer_genes = load_cancer_genes()
    print(f"  {len(cancer_genes):,} genes on OncoKB cancer list")
    is_fallback = df["label_source"] == "clinsig_fallback"
    in_cancer = df["GeneSymbol"].isin(cancer_genes)
    keep = ~is_fallback | in_cancer
    n_pre_cancer = len(df)
    df = df[keep].copy()
    n_dropped = n_pre_cancer - len(df)
    print(f"  {len(df):,} after cancer-gene filter "
          f"(dropped {n_dropped:,} clinsig_fallback rows in non-cancer genes)")
    print("  by source (post):")
    print(df.groupby(["label_source", "ML_Label"]).size().to_string())
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
