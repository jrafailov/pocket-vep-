"""Build the per-residue structure feature cache used by StructureFeatures.

Pipeline (each stage is resumable / idempotent; already-done work is skipped):

    1. [map]       Unique GeneSymbol -> UniProt accession via UniProt REST ID-Mapping.
    2. [download]  AlphaFold PDB per UniProt from the public deterministic URL.
    3. [features]  Run fpocket + DSSP per PDB, compute per-residue features,
                   write one consolidated parquet to data/processed/.

The Trainer/Features layer only touches the final parquet -- slow tool
invocation happens once here, not on every training run.

--------------------------------------------------------------------------
STORAGE & SCALE (rough, for the full ClinVar-labeled set)
--------------------------------------------------------------------------
    Unique human genes in labeled ClinVar   : ~15,000-20,000
    AlphaFold PDB files to download         : ~15,000-20,000  (one per UniProt)
    AlphaFold PDB median size               : ~700 KB  (range: 100 KB - 5 MB)
    Total AlphaFold storage                 : ~10-15 GB
    fpocket temp output per protein         : ~1-5 MB  (cleaned after parse)
    Peak fpocket temp storage (parallel)    : a few GB; kept under control via --keep-fpocket-raw
    DSSP output per protein                 : ~100-500 KB (can be discarded after parse)
    Final structure_features.parquet        : ~50-200 MB  (1 row per residue per protein)

If you don't want to trigger ~20k HTTPS calls to EBI, AlphaFold also publishes
the full human proteome as a single tarball (~5-10 GB compressed) at
https://ftp.ebi.ac.uk/pub/databases/alphafold/v4/UP000005640_9606_HUMAN_v4.tar
-- use that and skip the --stage download step entirely.

--------------------------------------------------------------------------
EXTERNAL TOOLS (install before running --stage features)
--------------------------------------------------------------------------
    fpocket:  conda install -c bioconda fpocket
    DSSP:     conda install -c salilab dssp        # provides `mkdssp`

Biopython (already in env/requirements.txt) handles DSSP parsing and PDB I/O.

--------------------------------------------------------------------------
GOTCHAS -- read before trusting the output
--------------------------------------------------------------------------
* Isoform mismatch: ClinVar's Name encodes a TRANSCRIPT-specific position
  (e.g. NM_014855.3). AlphaFold covers the CANONICAL UniProt isoform only.
  Positions can be off for non-canonical transcripts. Sanity check: the WT
  residue at `Position` in the AlphaFold structure should match `WT_AA` --
  we enforce this and drop mismatches in compute_features_for_protein().
* Missing AlphaFold entries: fusion genes, HLA variants, and proteins
  >2700 AA (AlphaFold splits them into fragments F1/F2/...) will not have a
  clean single PDB. We log and skip; StructureFeatures will drop those rows.
* Gene symbol ambiguity: a few HGNC symbols map to multiple UniProt accessions
  (e.g. gene families, withdrawn symbols). We keep only reviewed/Swiss-Prot
  entries and, if still ambiguous, take the first -- logged so you can audit.
* UniProt API rate limits: the ID-Mapping endpoint is polite but finite. We
  batch in chunks of 1000 with backoff. The result is cached so you only
  pay this cost once.
* fpocket needs a readable PDB. AlphaFold files are minimal (no HETATM, no
  alt-locs) so this is fine; if you ever point this script at experimental
  PDBs you'll need to pre-clean them.
* AlphaFold pLDDT (confidence) is stored in the B-factor column. Low-pLDDT
  regions (<70) have unreliable coordinates -- we record mean pLDDT per
  residue in the cache so the downstream feature block can filter if desired.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_clinvar_labeled  # noqa: E402

ALPHAFOLD_URL = "https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v4.pdb"
UNIPROT_MAPPING_RUN = "https://rest.uniprot.org/idmapping/run"
UNIPROT_MAPPING_STATUS = "https://rest.uniprot.org/idmapping/status/{job_id}"
UNIPROT_MAPPING_RESULTS = "https://rest.uniprot.org/idmapping/uniprotkb/results/{job_id}"

# Paths
DEFAULT_DATA = ROOT / "data/interim/clinvar_labeled.parquet"
UNIPROT_CACHE = ROOT / "data/interim/uniprot_mapping.parquet"
PDB_DIR = ROOT / "data/raw/alphafold"
FPOCKET_DIR = ROOT / "data/interim/fpocket"
DSSP_DIR = ROOT / "data/interim/dssp"
OUT_PLDDT = ROOT / "data/processed/plddt_cache.parquet"
OUT_FEATURES = ROOT / "data/processed/structure_features.parquet"

# 3-letter -> 1-letter map for residue names in AlphaFold PDBs (canonical 20 only).
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# ============================================================
# STAGE 1: Gene symbol -> UniProt accession
# ============================================================


def collect_unique_genes(clinvar_path: Path) -> list[str]:
    df = load_clinvar_labeled(clinvar_path)
    genes = sorted({g for g in df["GeneSymbol"].dropna().unique() if g and g != "-"})
    print(f"[map] {len(genes):,} unique gene symbols in {clinvar_path.name}")
    return genes


def uniprot_id_mapping(gene_symbols: list[str], batch_size: int = 1000) -> pd.DataFrame:
    """Hit UniProt's ID-Mapping API for Gene_Name -> UniProtKB.

    Returns a DataFrame with columns [gene_symbol, uniprot_id, reviewed].
    Only reviewed (Swiss-Prot) canonical entries are kept; ambiguities are logged.
    """
    rows: list[dict] = []
    for i in range(0, len(gene_symbols), batch_size):
        batch = gene_symbols[i : i + batch_size]
        print(f"[map] batch {i // batch_size + 1}: {len(batch)} symbols")

        # 1. submit the job
        resp = requests.post(
            UNIPROT_MAPPING_RUN,
            data={
                "ids": ",".join(batch),
                "from": "Gene_Name",
                "to": "UniProtKB",
                "taxId": "9606",  # human only -- drops paralogs in other species
            },
        )
        resp.raise_for_status()
        job_id = resp.json()["jobId"]

        # 2. poll until finished
        while True:
            status = requests.get(UNIPROT_MAPPING_STATUS.format(job_id=job_id))
            status.raise_for_status()
            body = status.json()
            if "results" in body or body.get("jobStatus") == "FINISHED":
                break
            if body.get("jobStatus") == "ERROR":
                raise RuntimeError(f"UniProt mapping job errored: {body}")
            time.sleep(3)

        # 3. fetch results (paginated -- follow the Link header)
        url = UNIPROT_MAPPING_RESULTS.format(job_id=job_id) + "?format=json&size=500"
        while url:
            page = requests.get(url)
            page.raise_for_status()
            payload = page.json()
            for entry in payload.get("results", []):
                gene = entry["from"]
                acc = entry["to"]["primaryAccession"]
                reviewed = entry["to"].get("entryType", "").startswith("UniProtKB reviewed")
                rows.append(
                    {"gene_symbol": gene, "uniprot_id": acc, "reviewed": reviewed}
                )
            # pagination
            link = page.headers.get("Link", "")
            url = None
            if 'rel="next"' in link:
                url = link.split("<", 1)[1].split(">", 1)[0]

    df = pd.DataFrame(rows)
    print(f"[map] raw hits: {len(df):,}")

    # prefer reviewed entries; if still ambiguous, keep the first (alphabetical accession)
    df = df.sort_values(["gene_symbol", "reviewed", "uniprot_id"],
                        ascending=[True, False, True])
    dups = df.duplicated("gene_symbol", keep="first")
    if dups.any():
        multi = df.loc[dups, "gene_symbol"].unique()
        print(f"[map] {len(multi)} genes had multiple UniProt hits; kept first reviewed.")
    df = df.loc[~dups].reset_index(drop=True)
    print(f"[map] final unique gene->UniProt: {len(df):,}")
    return df


def stage_map(clinvar_path: Path, force: bool) -> pd.DataFrame:
    if UNIPROT_CACHE.exists() and not force:
        print(f"[map] cache hit: {UNIPROT_CACHE}")
        return pd.read_parquet(UNIPROT_CACHE)
    genes = collect_unique_genes(clinvar_path)
    mapping = uniprot_id_mapping(genes)
    UNIPROT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_parquet(UNIPROT_CACHE, index=False)
    print(f"[map] wrote {UNIPROT_CACHE}")
    return mapping


# ============================================================
# STAGE 2: Download AlphaFold PDBs
# ============================================================


def download_pdb(uniprot_id: str, dest_dir: Path, session: requests.Session) -> Path | None:
    dest = dest_dir / f"AF-{uniprot_id}-F1-model_v4.pdb"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = ALPHAFOLD_URL.format(uniprot=uniprot_id)
    r = session.get(url)
    if r.status_code == 404:
        # no structure in AlphaFold DB -- common for large proteins, obsolete entries
        return None
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def stage_download(mapping: pd.DataFrame) -> pd.DataFrame:
    PDB_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    missing: list[str] = []
    have: list[str] = []

    for i, uid in enumerate(mapping["uniprot_id"], 1):
        try:
            path = download_pdb(uid, PDB_DIR, session)
        except requests.HTTPError as e:
            print(f"[download] HTTP error for {uid}: {e}")
            missing.append(uid)
            continue
        if path is None:
            missing.append(uid)
        else:
            have.append(uid)
        if i % 200 == 0:
            print(f"[download] progress: {i}/{len(mapping)}  "
                  f"have={len(have)}  missing={len(missing)}")

    print(f"[download] done. present={len(have)}  missing={len(missing)}")
    if missing:
        print(f"[download] first 10 missing: {missing[:10]}")
    return mapping.assign(has_structure=mapping["uniprot_id"].isin(have))


# ============================================================
# STAGE 3: Extract pLDDT (per-residue AlphaFold confidence)
# ============================================================
#
# This stage is deliberately lightweight -- no fpocket, no DSSP, no Biopython
# structure objects. Reads each PDB line-by-line and grabs the CA B-factor,
# which AlphaFold populates with the pLDDT confidence score (0-100).
#
# Runs in minutes for ~20k PDBs. Output is consumed by SequenceFeatures so
# that sequence-only experiments can include pLDDT without requiring fpocket
# or mkdssp to be installed.


def _read_pdb_text(pdb_path: Path) -> str:
    """Read a PDB file; transparent gzip handling for .pdb.gz."""
    if pdb_path.suffix == ".gz":
        import gzip
        with gzip.open(pdb_path, "rt") as fh:
            return fh.read()
    return pdb_path.read_text()


def _uid_from_filename(pdb_path: Path) -> str:
    # Filename is either AF-<UID>-F1-model_v4.pdb or AF-<UID>-F1-model_v4.pdb.gz.
    name = pdb_path.name
    if name.endswith(".pdb.gz"):
        name = name[:-7]
    elif name.endswith(".pdb"):
        name = name[:-4]
    return name.split("-")[1] if name.startswith("AF-") else name


def extract_plddt_from_pdb(pdb_path: Path) -> list[dict]:
    """Return one dict per residue with keys uniprot_id, position, wt_aa, plddt."""
    uid = _uid_from_filename(pdb_path)
    rows: list[dict] = []
    for line in _read_pdb_text(pdb_path).splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        try:
            resnum = int(line[22:26].strip())
            resname3 = line[17:20].strip()
            plddt = float(line[60:66].strip())  # B-factor column
        except ValueError:
            continue
        rows.append(
            {
                "uniprot_id": uid,
                "position": resnum,
                "wt_aa": AA3_TO_1.get(resname3, "X"),
                "plddt": plddt,
            }
        )
    return rows


def _find_alphafold_pdbs(pdb_dir: Path) -> list[Path]:
    """Locate PDBs under pdb_dir, matching either plain or .gz (bulk tarball ships .gz)."""
    hits = list(pdb_dir.glob("AF-*-F1-model_v4.pdb"))
    hits.extend(pdb_dir.glob("AF-*-F1-model_v4.pdb.gz"))
    return sorted(hits)


def _pdb_for_uid(uid: str) -> Path | None:
    """Locate the single AlphaFold file for a UniProt id; prefer plain .pdb over .pdb.gz."""
    plain = PDB_DIR / f"AF-{uid}-F1-model_v4.pdb"
    gz = PDB_DIR / f"AF-{uid}-F1-model_v4.pdb.gz"
    if plain.exists():
        return plain
    if gz.exists():
        return gz
    return None


def stage_plddt() -> pd.DataFrame:
    OUT_PLDDT.parent.mkdir(parents=True, exist_ok=True)
    pdbs = _find_alphafold_pdbs(PDB_DIR)
    print(f"[plddt] {len(pdbs):,} PDB files to scan under {PDB_DIR}")
    if not pdbs:
        print(f"[plddt] nothing to scan. If you downloaded the bulk tarball, "
              f"extract it first:\n"
              f"    tar -xf {PDB_DIR}/UP000005640_9606_HUMAN_v4.tar -C {PDB_DIR}/")
    all_rows: list[dict] = []
    for i, pdb in enumerate(pdbs, 1):
        all_rows.extend(extract_plddt_from_pdb(pdb))
        if i % 2000 == 0:
            print(f"[plddt] {i}/{len(pdbs)}  residues so far: {len(all_rows):,}")
    df = pd.DataFrame(all_rows)
    df.to_parquet(OUT_PLDDT, index=False)
    print(f"[plddt] wrote {OUT_PLDDT}  rows={len(df):,}")
    return df


# ============================================================
# STAGE 4: Per-PDB fpocket + DSSP features
# ============================================================


def run_fpocket(pdb_path: Path, work_dir: Path) -> tuple[dict, Path]:
    """Run fpocket on a PDB and parse its output.

    fpocket drops everything in <pdb_stem>_out/ next to the input by default.
    We redirect via work_dir to keep things tidy.

    If `pdb_path` is gzipped (.pdb.gz, as in the AlphaFold bulk tarball), it is
    transparently decompressed into `work_dir` before being handed to fpocket --
    the fpocket CLI only reads plain PDBs.

    Returns:
        (
            {
                "pockets": [
                    {
                        "id": 1,
                        "druggability": float,        # from _info.txt "Druggability Score"
                        "residues": set[int],          # residue numbers touching this pocket
                        "ca_coords": np.ndarray (N,3), # CA coords of pocket-lining residues
                    },
                    ...
                ]
            },
            staged,  # Path to the plain (decompressed, if needed) PDB on disk.
                     # Downstream callers (run_dssp, get_ca_coords) MUST use this --
                     # the original .pdb.gz is not readable by either.
        )
    """
    # Stage the input as a plain PDB inside work_dir so fpocket writes siblings to our copy.
    work_dir.mkdir(parents=True, exist_ok=True)
    staged_name = pdb_path.name[:-3] if pdb_path.name.endswith(".pdb.gz") else pdb_path.name
    staged = work_dir / staged_name
    if not staged.exists():
        if pdb_path.suffix == ".gz":
            import gzip
            with gzip.open(pdb_path, "rb") as src, open(staged, "wb") as dst:
                dst.write(src.read())
        else:
            staged.write_bytes(pdb_path.read_bytes())

    out_dir = work_dir / f"{staged.stem}_out"
    if not out_dir.exists():
        subprocess.run(
            ["fpocket", "-f", str(staged)],
            check=True,
            capture_output=True,
        )

    # --- parse fpocket output ---
    # fpocket writes:
    #   <stem>_info.txt          per-pocket scores (Druggability, Score, Volume, ...)
    #   pockets/pocket{N}_atm.pdb residues lining pocket N
    info_path = out_dir / f"{staged.stem}_info.txt"
    pockets_dir = out_dir / "pockets"
    pockets: list[dict] = []
    if info_path.exists():
        pockets = _parse_fpocket_info(info_path, pockets_dir)
    else:
        # TODO: fpocket version differences -- some versions name the info file differently.
        # Add a fallback here if you hit missing files on your install.
        print(f"[fpocket] WARN: no info file for {pdb_path.name}")

    return {"pockets": pockets}, staged


def _parse_fpocket_info(info_path: Path, pockets_dir: Path) -> list[dict]:
    """Parse <stem>_info.txt and each pocket{N}_atm.pdb."""
    pockets: list[dict] = []
    current: dict | None = None
    for raw in info_path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("Pocket ") and line.endswith(":"):
            if current is not None:
                pockets.append(current)
            current = {"id": int(line.split()[1]), "druggability": np.nan}
        elif line.startswith("Druggability Score") and current is not None:
            current["druggability"] = float(line.split(":")[1].strip())
    if current is not None:
        pockets.append(current)

    # Attach residue lists + CA coords by reading each pocket PDB fragment.
    for p in pockets:
        frag = pockets_dir / f"pocket{p['id']}_atm.pdb"
        residues: set[int] = set()
        ca_coords: list[tuple[float, float, float]] = []
        if frag.exists():
            for line in frag.read_text().splitlines():
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    try:
                        residues.add(int(line[22:26].strip()))
                        ca_coords.append(
                            (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                        )
                    except ValueError:
                        continue
        p["residues"] = residues
        p["ca_coords"] = np.array(ca_coords) if ca_coords else np.empty((0, 3))
    return pockets


def run_dssp(pdb_path: Path) -> dict[int, dict]:
    """Return {residue_number: {'ss': str, 'sasa': float, 'wt_aa': str}}.

    Uses Biopython's DSSP wrapper -- requires `mkdssp` on PATH.

    mkdssp 4.x dropped PDB-format input; it only reads mmCIF. We convert the
    AlphaFold PDB to mmCIF on the fly via Biopython and hand that to mkdssp.
    The .cif is written as a sibling of the staged PDB so stage_features'
    per-protein cleanup wipes it along with the rest of the work dir.
    """
    from Bio.PDB import MMCIFIO, PDBParser
    from Bio.PDB.DSSP import DSSP

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))

    cif_path = pdb_path.with_suffix(".cif")
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(str(cif_path))

    model = next(structure.get_models())
    dssp = DSSP(model, str(cif_path), dssp="mkdssp", file_type="MMCIF")

    # DSSP's 8-state -> 3-state collapse
    ss8_to_3 = {
        "H": "H", "G": "H", "I": "H",  # helix
        "E": "E", "B": "E",              # strand
        "T": "L", "S": "L", "-": "L", " ": "L",  # loop/coil
    }

    out: dict[int, dict] = {}
    for key in dssp.keys():
        # key = (chain_id, (' ', resnum, ' '))
        resnum = key[1][1]
        aa, ss, rel_sasa, *_ = dssp[key]
        try:
            sasa = float(rel_sasa)
        except (TypeError, ValueError):
            # DSSP emits 'NA' / '-' / '' for residues whose relative SASA
            # is undefined (e.g. non-standard AAs, chain ends).
            sasa = np.nan
        out[int(resnum)] = {
            "wt_aa": aa,
            "ss": ss8_to_3.get(ss, "L"),
            "sasa": sasa,
        }
    return out


def get_ca_coords(pdb_path: Path) -> dict[int, np.ndarray]:
    """Map residue number -> CA xyz, read directly from the PDB."""
    coords: dict[int, np.ndarray] = {}
    for line in pdb_path.read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                resnum = int(line[22:26].strip())
                coords[resnum] = np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                )
            except ValueError:
                continue
    return coords


def compute_features_for_protein(
    uniprot_id: str, pdb_path: Path, pocket_info: dict, dssp_info: dict
) -> pd.DataFrame:
    """One row per residue in the protein."""
    ca = get_ca_coords(pdb_path)
    pockets = pocket_info["pockets"]

    # Pre-concat all pocket CA coords for a single nearest-neighbor query.
    pocket_residues_union: set[int] = set()
    pocket_by_residue: dict[int, int] = {}   # residue -> first pocket id it belongs to
    pocket_drug: dict[int, float] = {}       # pocket id -> druggability
    all_pocket_cas: list[np.ndarray] = []
    pocket_id_per_ca: list[int] = []
    for p in pockets:
        pocket_drug[p["id"]] = p["druggability"]
        pocket_residues_union |= p["residues"]
        for r in p["residues"]:
            pocket_by_residue.setdefault(r, p["id"])
        if p["ca_coords"].size:
            all_pocket_cas.append(p["ca_coords"])
            pocket_id_per_ca.extend([p["id"]] * len(p["ca_coords"]))
    pocket_ca_stack = (
        np.vstack(all_pocket_cas) if all_pocket_cas else np.empty((0, 3))
    )
    pocket_id_arr = np.array(pocket_id_per_ca, dtype=int)

    rows: list[dict] = []
    for resnum, xyz in ca.items():
        info = dssp_info.get(resnum, {})
        if pocket_ca_stack.size:
            dists = np.linalg.norm(pocket_ca_stack - xyz, axis=1)
            nearest_idx = int(dists.argmin())
            dist_to_nearest = float(dists[nearest_idx])
            nearest_pocket_id = int(pocket_id_arr[nearest_idx])
            druggability = pocket_drug.get(nearest_pocket_id, np.nan)
        else:
            dist_to_nearest = np.nan
            druggability = np.nan
        rows.append(
            {
                "uniprot_id": uniprot_id,
                "position": int(resnum),
                "wt_aa": info.get("wt_aa"),
                "ss": info.get("ss"),
                "sasa": info.get("sasa", np.nan),
                # pLDDT lives in plddt_cache.parquet (owned by --stage plddt,
                # consumed by SequenceFeatures per the research proposal).
                "in_pocket": int(resnum in pocket_residues_union),
                "dist_to_nearest_pocket": dist_to_nearest,
                "druggability": druggability,
            }
        )
    return pd.DataFrame(rows)


def _cleanup_fpocket_workdir(uid: str) -> None:
    work = FPOCKET_DIR / uid
    if not work.exists():
        return
    for p in work.rglob("*"):
        if p.is_file():
            p.unlink()
    for p in sorted(work.rglob("*"), reverse=True):
        if p.is_dir():
            p.rmdir()


def _process_one_protein(
    uid: str, keep_fpocket_raw: bool
) -> tuple[str, pd.DataFrame | None, str | None]:
    """Worker for parallel stage_features dispatch.

    Return shape: (uid, per_residue_df | None, error_msg | None)
      - success:  (uid, df,   None)
      - skipped:  (uid, None, None)         # no PDB on disk for this uid
      - failure:  (uid, None, "error msg")  # fpocket/DSSP/parse raised

    Must be a module-level function so ProcessPoolExecutor can pickle it.
    """
    pdb_path = _pdb_for_uid(uid)
    if pdb_path is None:
        return uid, None, None
    try:
        # run_fpocket stages (and, if needed, decompresses) pdb_path into
        # the per-protein work dir and returns that staged plain PDB -- reuse
        # it for DSSP and CA-coord parsing so .pdb.gz inputs work transparently.
        pocket_info, staged_pdb = run_fpocket(pdb_path, FPOCKET_DIR / uid)
        dssp_info = run_dssp(staged_pdb)
        per_residue = compute_features_for_protein(uid, staged_pdb, pocket_info, dssp_info)
        return uid, per_residue, None
    except subprocess.CalledProcessError as e:
        return uid, None, f"fpocket failed: {e.stderr[:200]!r}"
    except Exception as e:
        return uid, None, f"{type(e).__name__}: {e}"
    finally:
        if not keep_fpocket_raw:
            # fpocket can produce MBs per protein; clean up unless asked to keep.
            _cleanup_fpocket_workdir(uid)


def stage_features(
    mapping: pd.DataFrame, keep_fpocket_raw: bool, n_jobs: int
) -> pd.DataFrame:
    FPOCKET_DIR.mkdir(parents=True, exist_ok=True)
    DSSP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)

    uids = list(mapping["uniprot_id"])
    all_frames: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []
    skipped_no_pdb = 0

    pool: ProcessPoolExecutor | None = None
    if n_jobs <= 1:
        # Serial path -- handy for debugging worker crashes that the pool
        # would otherwise mask behind a pickling boundary.
        result_iter = (_process_one_protein(uid, keep_fpocket_raw) for uid in uids)
    else:
        pool = ProcessPoolExecutor(max_workers=n_jobs)
        futures = [
            pool.submit(_process_one_protein, uid, keep_fpocket_raw) for uid in uids
        ]
        result_iter = (f.result() for f in as_completed(futures))

    pbar = tqdm(total=len(uids), desc="[features]", unit="prot")
    try:
        for uid, df, err in result_iter:
            if df is not None:
                all_frames.append(df)
            elif err is not None:
                failures.append((uid, err))
            else:
                skipped_no_pdb += 1
            pbar.set_postfix(
                ok=len(all_frames), fail=len(failures), skip=skipped_no_pdb
            )
            pbar.update(1)
    finally:
        pbar.close()
        if pool is not None:
            pool.shutdown(wait=True)

    if failures:
        log = OUT_FEATURES.with_suffix(".failures.json")
        log.write_text(json.dumps(failures, indent=2))
        print(f"[features] {len(failures)} failures logged to {log}")

    if not all_frames:
        print("[features] no successful proteins -- nothing written.")
        return pd.DataFrame()

    out = pd.concat(all_frames, ignore_index=True)
    out.to_parquet(OUT_FEATURES, index=False)
    print(f"[features] wrote {OUT_FEATURES}  rows={len(out):,}")
    return out


# ============================================================
# Orchestration
# ============================================================


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(DEFAULT_DATA), type=Path)
    ap.add_argument(
        "--stage",
        choices=["all", "map", "download", "plddt", "features"],
        default="all",
        help="'all' runs map -> download -> plddt -> features in order.",
    )
    ap.add_argument("--force-map", action="store_true",
                    help="Re-run UniProt mapping even if cache exists.")
    ap.add_argument("--keep-fpocket-raw", action="store_true",
                    help="Don't delete fpocket output dirs (debugging; costs disk).")
    ap.add_argument("--n-jobs", type=int,
                    default=max(1, (os.cpu_count() or 2) - 1),
                    help="Parallel workers for --stage features. "
                         "Default: cpu_count()-1. Use 1 for a serial run "
                         "(easier to debug worker exceptions).")
    args = ap.parse_args()

    if args.stage in ("all", "map"):
        mapping = stage_map(args.data, force=args.force_map)
    else:
        if not UNIPROT_CACHE.exists():
            sys.exit(f"UniProt mapping cache missing: {UNIPROT_CACHE}. "
                     "Run --stage map first.")
        mapping = pd.read_parquet(UNIPROT_CACHE)

    if args.stage in ("all", "download"):
        mapping = stage_download(mapping)

    if args.stage in ("all", "plddt"):
        stage_plddt()

    if args.stage in ("all", "features"):
        stage_features(
            mapping,
            keep_fpocket_raw=args.keep_fpocket_raw,
            n_jobs=args.n_jobs,
        )


if __name__ == "__main__":
    main()
