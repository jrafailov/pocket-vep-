# pocket-vep — End-to-end runbook

This document walks a fresh clone all the way from raw downloads to trained
models. Follow top-to-bottom; each section's commands are copy-pasteable.

If you only care about the **sequence-only** baseline (no fpocket / DSSP),
you can skip section 3 and step 5. Sections 1, 2, 4, 6, 7, 9 are sufficient
to train a sequence model with pLDDT.

---

## 1. Prerequisites

- macOS (Apple Silicon or Intel) or Linux
- Python 3.11+ available as `python3`
- ~20 GB free disk (mostly for the AlphaFold tarball + extracted PDBs)
- Internet access for the one-time downloads in steps 1, 2, and 4

---

## 2. Python environment

```bash
cd /path/to/pocket-vep-
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r env/requirements.txt
```

This covers everything for the data pull, sequence features, pLDDT cache,
and model training. The structure features stage (`--stage features`) needs
two extra **system binaries** installed via conda — see section 3.

---

## 3. System binaries for `--stage features` (skip if sequence-only)

`fpocket` (pocket prediction) and `mkdssp` (secondary structure + SASA) are
not pip-installable. Use Miniforge/conda for them; keep your Python in venv.

### 3a. Install Miniforge (skip if conda already on PATH)

macOS arm64:

```bash
curl -fL -o /tmp/miniforge.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash /tmp/miniforge.sh -b -p $HOME/miniforge3
$HOME/miniforge3/bin/conda init zsh
exec zsh   # reload shell so `conda` is on PATH
```

For Intel macOS or Linux, swap the installer URL — see
<https://github.com/conda-forge/miniforge/releases>.

### 3b. Create the `bio-tools` env (binaries only)

```bash
conda create -n bio-tools -c bioconda -c conda-forge fpocket dssp -y
```

This env is **separate** from your `.venv`. We only borrow its binaries.

### 3c. Put the binaries on PATH alongside the venv

Add to `~/.zshrc` so it persists:

```bash
export PATH="$HOME/miniforge3/envs/bio-tools/bin:$PATH"
```

Then `source ~/.zshrc` (or open a new terminal).

### 3d. libcifpp dictionary files (mkdssp 4.x needs these)

mkdssp 4.5+ only reads mmCIF, and the conda package on macOS doesn't ship
the reference dictionaries. Download them once:

```bash
CIFPP=$HOME/miniforge3/envs/bio-tools/var/cache/libcifpp
mkdir -p "$CIFPP"
curl -fL -o "$CIFPP/components.cif" \
  https://files.wwpdb.org/pub/pdb/data/monomers/components.cif
curl -fL -o "$CIFPP/mmcif_pdbx.dic" \
  https://raw.githubusercontent.com/PDB-REDO/libcifpp/trunk/rsrc/mmcif_pdbx.dic
curl -fL -o "$CIFPP/mmcif_ma.dic" \
  https://raw.githubusercontent.com/PDB-REDO/libcifpp/trunk/rsrc/mmcif_ma.dic
```

`components.cif` is ~500 MB; the two `.dic` files are small.

### 3e. Smoke-test the install

```bash
which fpocket
which mkdssp
mkdssp --help | head -1   # should print `mkdssp 4.5.x` or similar
```

---

## 4. Step 1 — Download ClinVar variants

```bash
python scripts/download_clinvar.py
```

What this does:

- Pulls `variant_summary.txt.gz` (~250 MB) from NCBI into `data/raw/`.
- Filters to GRCh38 SNVs, parses `protein_change`, applies the
  Oncogenicity → ClinicalSignificance fallback labeler.
- Writes `data/interim/clinvar_labeled.parquet` (~50 MB, ~960k rows,
  binary `ML_Label ∈ {oncogenic, benign}`).

Runtime: ~5 minutes on a normal connection.

---

## 5. Step 2 — Download the AlphaFold human proteome (one-time, ~8 GB)

```bash
mkdir -p data/raw/alphafold
curl -fL -o data/raw/alphafold/UP000005640_9606_HUMAN_v4.tar \
  https://ftp.ebi.ac.uk/pub/databases/alphafold/v4/UP000005640_9606_HUMAN_v4.tar
tar -xf data/raw/alphafold/UP000005640_9606_HUMAN_v4.tar \
  -C data/raw/alphafold/
```

After extraction you should see ~16,808 files named
`AF-<UNIPROT>-F1-model_v4.pdb.gz` directly inside `data/raw/alphafold/`.

The pipeline reads `.pdb.gz` directly — **do not** `gunzip` them. Disk:

| What | Size |
|---|---|
| Tarball (can delete after extracting) | ~8 GB |
| Extracted `.pdb.gz` files | ~10 GB |
| If you unzipped them (don't) | ~25 GB |

---

## 6. Step 3 — Map ClinVar gene symbols to UniProt accessions

```bash
python scripts/build_structure_cache.py --stage map
```

Hits the UniProt ID-Mapping REST API for the ~16k unique gene symbols in
ClinVar and writes `data/interim/uniprot_mapping.parquet`.

Runtime: ~10 minutes. Expect logs like `137k raw hits → 16k unique
mappings` — that's normal (UniProt returns multiple isoforms per gene; the
script keeps only the Swiss-Prot reviewed canonical entry).

---

## 7. Step 4 — Build the pLDDT cache (lightweight, no fpocket/DSSP)

```bash
python scripts/build_structure_cache.py --stage plddt
```

Scans every `.pdb.gz` under `data/raw/alphafold/`, extracts each CA atom's
B-factor (= pLDDT in AlphaFold files), and writes
`data/processed/plddt_cache.parquet` (~100–300 MB, ~10M rows).

Runtime: ~5–10 minutes single-threaded.

After this step, sequence-only experiments work end-to-end. You can skip
straight to section 9 and come back to section 8 later if you want.

---

## 8. Step 5 — Build structure features (fpocket + DSSP, parallel)

Requires section 3 to be complete (`fpocket` and `mkdssp` on PATH +
libcifpp dictionaries).

### 8a. Pick `--n-jobs` for your machine

```bash
sysctl -n hw.physicalcpu              # total physical cores
sysctl -n hw.perflevel0.physicalcpu   # performance cores (Apple Silicon)
```

Rule of thumb: `n_jobs = physical_cores - 1` (leave one core for the OS).

| Machine | Suggested `--n-jobs` |
|---|---|
| M1/M2/M3 Pro (8–10 p-cores) | 7 or 9 |
| M3/M4 Max (10–16 p-cores) | 9–15 |
| 8-core Intel laptop | 6 or 7 |

### 8b. Run

```bash
python scripts/build_structure_cache.py --stage features --n-jobs 8
```

Outputs:

- `data/processed/structure_features.parquet` (~50–200 MB)
- `data/processed/structure_features.failures.json` (per-protein errors,
  if any)

Runtime: **~2 hours** on Apple Silicon at `--n-jobs 8`, vs ~14 hours
serial. Memory: ~250–300 MB per worker (8 workers ≈ 2.5 GB peak).

For debugging worker-side errors, run with `--n-jobs 1` to get the real
traceback in the foreground (the pool otherwise hides them behind a
pickling boundary).

---

## 9. Step 6 — Train models

Run with command:
`python scripts/run_experiments.py`

`Trainer.run(feature_keys, model_name)` always evaluates on **both** val and
test in one call and returns:

```python
{
    "metrics": [val_metrics, test_metrics],   # two compute_metrics dicts
    "model": <fitted estimator>,
    "feature_names": [...],
    "X_train": ..., "X_val": ..., "X_test": ...,
    "y_val_enc": ..., "y_test_enc": ...,
    "label_names": [...],
}
```

The runner script writes a fixed layout under `--out-dir` (default
`results/`):

```
results/
    experiments.csv                     # one row per (feature_set, model, split)
    interpretations/
        {feature_set}_{model_name}.csv  # method, feature, importance, rank
```

Useful flags:

- `--models` / `--feature-sets` — subset what runs.
- `--interpret-methods native permutation shap` — pick interpretation
  methods (default: all three). MLP has no `native` and skips it with a
  warning.
- `--no-interpret` — skip the interpretation pass.
- `--shap-sample-size 500` — caps SHAP cost (KernelExplainer on MLP).
- `--out-dir results/run_a/` — redirect the whole output tree.

The headline metrics for this project are `macro_f1` and
`balanced_accuracy` (the dataset is ~2:1 oncogenic:benign — raw accuracy
would reward an "always-benign" classifier).

<!-- ---

## 10. macOS multiprocessing note

Only relevant if you write a **new** script that calls `stage_features()`
directly (e.g. for a 50-protein smoke test). macOS uses `spawn` for
`ProcessPoolExecutor`, which re-imports the entry module — so any such
script must guard the call:

```python
if __name__ == "__main__":
    stage_features(mapping, keep_fpocket_raw=False, n_jobs=8)
```

The shipping `scripts/build_structure_cache.py` already has this guard, so
the production command in step 8b works without any extra setup.

--- -->

## 11. Storage budget

| Item | Size | Path |
|---|---|---|
| ClinVar raw + parquet | ~300 MB | `data/raw/`, `data/interim/clinvar_labeled.parquet` |
| AlphaFold tarball (deletable post-extract) | ~8 GB | `data/raw/alphafold/UP000005640_9606_HUMAN_v4.tar` |
| AlphaFold extracted `.pdb.gz` | ~10 GB | `data/raw/alphafold/AF-*-F1-model_v4.pdb.gz` |
| libcifpp dictionaries | ~500 MB | `$HOME/miniforge3/envs/bio-tools/var/cache/libcifpp/` |
| pLDDT cache | ~100–300 MB | `data/processed/plddt_cache.parquet` |
| Structure features | ~50–200 MB | `data/processed/structure_features.parquet` |
| **Total** | **~20 GB** | |

You can delete the AlphaFold tarball after extracting (~8 GB freed).

---

## 12. Troubleshooting

**`--stage plddt` reports `0 PDB files to scan`**
The tarball wasn't extracted, or was extracted to the wrong directory.
Re-run the `tar -xf ...` command from section 5 and verify
`ls data/raw/alphafold/AF-*.pdb.gz | head` lists files.

**`fpocket failed: QH6047 qhull input error` on every protein**
You're hitting a broken `fpocket` build (often Homebrew's). Make sure
section 3c put the conda env's bin first on PATH:
`which fpocket` should print `…/miniforge3/envs/bio-tools/bin/fpocket`,
not `/opt/homebrew/bin/fpocket`. If brew's is shadowing it,
`brew uninstall fpocket` and reopen the terminal.

**`mkdssp ... Could not load dictionary mmcif_pdbx.dic`**
The libcifpp data files aren't where mkdssp expects them. Re-run section
3d and verify the three files exist under
`$HOME/miniforge3/envs/bio-tools/var/cache/libcifpp/`.

**`mkdssp ... This file does not seem to be an mmCIF file`**
mkdssp 4.5+ rejects PDB input. The script already converts PDB → mmCIF
on the fly via Biopython, so this only appears if `run_dssp` is being
called with a path that isn't going through the conversion. Make sure
you're on the latest `scripts/build_structure_cache.py`.

**`BrokenProcessPool` / `attempt has been made to start a new process
before the current process has finished its bootstrapping phase`**
Your custom script that calls `stage_features` is missing the
`if __name__ == "__main__":` guard. See section 10.
