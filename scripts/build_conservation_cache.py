"""Build the per-variant evolutionary-conservation cache from UCSC bigWigs.

For each ClinVar missense variant, query phyloP100way and phastCons100way at
the variant's GRCh38 genomic position. Output:

    data/processed/conservation_cache.parquet
    columns: [Chromosome, PositionVCF, phylop100way, phastcons100way]

phyloP captures the per-position rate of evolution (positive = conserved,
negative = accelerated). phastCons is an HMM-smoothed posterior probability
of conservation across windows. Both are the canonical evolutionary signal
in clinical genomics (CADD, REVEL, dbNSFP all use them).

--------------------------------------------------------------------------
PIPELINE
--------------------------------------------------------------------------
    1. [download] Mirror UCSC bigWigs locally if --phylop / --phastcons point
                  at default paths and the files don't exist yet.
                  Sizes: phyloP100way ~9 GB, phastCons100way ~6 GB.
    2. [extract]  Open both bigWigs, dedupe variants by (Chromosome,
                  PositionVCF) so we don't re-query the same locus across
                  multi-allelic variants, and write the parquet.

Both stages are resumable: download supports HTTP Range resumption, and
already-extracted parquets are NOT overwritten unless --force is passed.

--------------------------------------------------------------------------
GOTCHAS
--------------------------------------------------------------------------
* Chromosome naming: ClinVar variant_summary uses bare names ("1", "X", "MT").
  UCSC bigWigs use "chr1", "chrX", "chrM" (note "chrM" not "chrMT"). We
  translate at query time.
* GRCh38 only. download_clinvar.py already filters to GRCh38; the bigWigs
  here are hg38 too. If you switch the ClinVar assembly, change the URLs.
* pyBigWig.values(chrom, start, end) is 0-based half-open. ClinVar's
  PositionVCF is VCF-style 1-based, so we query [pos-1, pos).
* Positions on alt contigs / decoys won't be in the bigWig and return NaN.
  Same with rare positions where multi-species alignment failed. The feature
  block drops NaN rows -- expected coverage is >99% of canonical-chrom SNVs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_clinvar_labeled  # noqa: E402

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
DEFAULT_CLINVAR = ROOT / "data/interim/clinvar_labeled.parquet"
UCSC_DIR = ROOT / "data/raw/ucsc"
DEFAULT_PHYLOP = UCSC_DIR / "hg38.phyloP100way.bw"
DEFAULT_PHASTCONS = UCSC_DIR / "hg38.phastCons100way.bw"
DEFAULT_OUT = ROOT / "data/processed/conservation_cache.parquet"

PHYLOP_URL = "https://hgdownload.soe.ucsc.edu/goldenpath/hg38/phyloP100way/hg38.phyloP100way.bw"
PHASTCONS_URL = "https://hgdownload.soe.ucsc.edu/goldenpath/hg38/phastCons100way/hg38.phastCons100way.bw"


# --------------------------------------------------------------------------
# STAGE 1: download bigWigs (resumable)
# --------------------------------------------------------------------------


def download_resumable(url: str, dest: Path) -> None:
    """HTTP GET with Range-based resume so a half-finished 9 GB transfer
    doesn't cost us another 9 GB."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    head = requests.head(url, allow_redirects=True, timeout=30)
    head.raise_for_status()
    total = int(head.headers.get("content-length", 0))

    have = dest.stat().st_size if dest.exists() else 0
    if have and have == total:
        print(f"[download] cache hit: {dest}  ({total / 1e9:.2f} GB)")
        return
    if have > total:
        print(f"[download] {dest} is larger than remote -- redownloading")
        have = 0
        dest.unlink()

    headers = {"Range": f"bytes={have}-"} if have else {}
    mode = "ab" if have else "wb"
    print(f"[download] {url}\n           -> {dest}  "
          f"(start={have / 1e9:.2f} GB / {total / 1e9:.2f} GB)")
    r = requests.get(url, headers=headers, stream=True, timeout=60)
    r.raise_for_status()
    with dest.open(mode) as fh, tqdm(
        total=total, initial=have, unit="iB", unit_scale=True,
        unit_divisor=1024, desc=dest.name,
    ) as bar:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            bar.update(fh.write(chunk))


def _is_url(spec: str) -> bool:
    return spec.startswith("http://") or spec.startswith("https://")


def stage_download(phylop_spec: str, phastcons_spec: str,
                   skip_phylop: bool, skip_phastcons: bool) -> None:
    """Mirror UCSC bigWigs locally only when --phylop/--phastcons point at
    the default local paths and the files are missing. URLs and custom
    paths skip download (caller is providing their own source)."""
    if not skip_phylop and phylop_spec == str(DEFAULT_PHYLOP):
        download_resumable(PHYLOP_URL, DEFAULT_PHYLOP)
    if not skip_phastcons and phastcons_spec == str(DEFAULT_PHASTCONS):
        download_resumable(PHASTCONS_URL, DEFAULT_PHASTCONS)


# --------------------------------------------------------------------------
# STAGE 2: extract per-variant conservation
# --------------------------------------------------------------------------


def _ucsc_chrom(clinvar_chrom: str) -> str:
    """Translate ClinVar chrom -> UCSC chrom. 'MT' -> 'chrM', else 'chr<X>'."""
    if clinvar_chrom == "MT":
        return "chrM"
    return f"chr{clinvar_chrom}"


def _query_position(bw, chrom: str, pos_vcf: int) -> float:
    """Read a single bigWig score at VCF-1-based position `pos_vcf`.

    pyBigWig.values uses 0-based half-open intervals, so the 1-bp window
    around VCF position N is [N-1, N).
    """
    import math
    try:
        vals = bw.values(chrom, pos_vcf - 1, pos_vcf)
    except RuntimeError:
        # pyBigWig raises RuntimeError on chroms not in the bigWig header
        # (e.g. alt contigs). Treat as missing data.
        return float("nan")
    if not vals:
        return float("nan")
    v = vals[0]
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return float("nan")
    return float(v)


def stage_extract(
    clinvar_path: Path,
    phylop_spec: str,
    phastcons_spec: str,
    out_path: Path,
    use_phylop: bool,
    use_phastcons: bool,
    limit: int | None,
) -> None:
    import pyBigWig

    df = load_clinvar_labeled(clinvar_path)
    print(f"[extract] {len(df):,} ClinVar rows in {clinvar_path.name}")

    # Score is position-only -- multi-allelic variants at the same locus get
    # the same phyloP / phastCons. Dedupe before query so we don't pay for
    # them twice.
    loci = (
        df[["Chromosome", "PositionVCF"]]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if limit is not None:
        loci = loci.head(limit)
        print(f"[extract] --limit {limit}: {len(loci):,} unique loci")
    else:
        print(f"[extract] {len(loci):,} unique (chrom, pos) loci to query")

    bw_phylop = pyBigWig.open(phylop_spec) if use_phylop else None
    bw_phastcons = pyBigWig.open(phastcons_spec) if use_phastcons else None
    if bw_phylop is None and bw_phastcons is None:
        sys.exit("[extract] both phyloP and phastCons disabled -- nothing to do.")

    phylop_scores: list[float] = []
    phastcons_scores: list[float] = []
    for chrom, pos in tqdm(
        zip(loci["Chromosome"].astype(str), loci["PositionVCF"].astype(int)),
        total=len(loci),
        desc="[extract]",
    ):
        ucsc = _ucsc_chrom(chrom)
        phylop_scores.append(
            _query_position(bw_phylop, ucsc, pos) if bw_phylop else float("nan")
        )
        phastcons_scores.append(
            _query_position(bw_phastcons, ucsc, pos) if bw_phastcons else float("nan")
        )

    if bw_phylop is not None:
        bw_phylop.close()
    if bw_phastcons is not None:
        bw_phastcons.close()

    out = loci.copy()
    if use_phylop:
        out["phylop100way"] = phylop_scores
    if use_phastcons:
        out["phastcons100way"] = phastcons_scores

    score_cols = [c for c in ("phylop100way", "phastcons100way") if c in out.columns]
    nonnull = out.dropna(subset=score_cols, how="all")
    print(
        f"[extract] coverage: {len(nonnull):,}/{len(out):,} "
        f"({len(nonnull) / max(1, len(out)):.1%}) loci have at least one score"
    )
    for col in score_cols:
        hit = out[col].notna().sum()
        print(f"           {col}: {hit:,} non-null  "
              f"(mean={out[col].mean():.3f}, "
              f"min={out[col].min():.3f}, max={out[col].max():.3f})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[extract] wrote {out_path}  rows={len(out):,}")


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clinvar", default=DEFAULT_CLINVAR, type=Path,
                    help="Labeled ClinVar parquet "
                         "(default: data/interim/clinvar_labeled.parquet).")
    ap.add_argument("--phylop", default=str(DEFAULT_PHYLOP), type=str,
                    help=f"phyloP bigWig path or URL "
                         f"(default downloads {PHYLOP_URL} on first use).")
    ap.add_argument("--phastcons", default=str(DEFAULT_PHASTCONS), type=str,
                    help=f"phastCons bigWig path or URL "
                         f"(default downloads {PHASTCONS_URL} on first use).")
    ap.add_argument("--no-phylop", action="store_true",
                    help="Skip phyloP entirely.")
    ap.add_argument("--no-phastcons", action="store_true",
                    help="Skip phastCons entirely.")
    ap.add_argument("--stage", choices=["all", "download", "extract"],
                    default="all",
                    help="'all' runs download then extract.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N unique loci. For smoke tests.")
    ap.add_argument("--out", default=DEFAULT_OUT, type=Path,
                    help="Output parquet path. Use a different path with --limit "
                         "so debug runs don't clobber the real cache.")
    args = ap.parse_args()

    if args.stage in ("all", "download"):
        stage_download(
            args.phylop, args.phastcons,
            skip_phylop=args.no_phylop, skip_phastcons=args.no_phastcons,
        )

    if args.stage in ("all", "extract"):
        stage_extract(
            clinvar_path=args.clinvar,
            phylop_spec=args.phylop,
            phastcons_spec=args.phastcons,
            out_path=args.out,
            use_phylop=not args.no_phylop,
            use_phastcons=not args.no_phastcons,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
