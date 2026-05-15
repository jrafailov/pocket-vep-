# pocket-vep

Somatic missense variant effect predictor that mixes AlphaFold pocket
geometry with sequence and evolutionary conservation features. The question
we're testing is whether pocket geometry actually adds anything on top of
the sequence and evolution signals that existing VEPs already use.

The main experiment is a three-way ablation over `sequence`, `structure`, and
`evolution` feature blocks (plus the three pairwise combos and the full
union), trained with three classifiers across five seeds. Report at
[`report/main.pdf`](report/main.pdf), proposal at
[`proposal.pdf`](proposal.pdf).

## Layout

- `env/`, conda env spec and pip requirements
- `scripts/`, runnable entry points for data pulls, feature building,
  training, plotting
- `src/`, importable code
  - `data/`, loader, splits, AlphaFold cache wrapper
  - `features/`, sequence, structure, evolution feature blocks
  - `models/`, RF, MLP, XGBoost classifiers and the trainer
  - `eval/`, metrics and interpretation
- `notebooks/`, per-person exploration (one file per author)
- `report/`, NeurIPS-style writeup, LaTeX source and figures
- `data/`, raw downloads and processed parquets (gitignored)
- `results/`, experiment CSVs, interpretations, plots (gitignored except
  for the figures used in the paper)

## Setup

The main analysis env is pip-based for portability. The structure features
stage also needs two conda-only binaries (`fpocket` and `mkdssp`). Short
version below; full walkthrough with troubleshooting in
[`instructions.md`](instructions.md).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r env/requirements.txt
```

For the structure stage only,

```bash
conda create -n bio-tools -c bioconda -c conda-forge fpocket dssp -y
export PATH="$HOME/miniforge3/envs/bio-tools/bin:$PATH"
```

`mkdssp` 4.5+ also needs libcifpp dictionaries staged in
`$HOME/miniforge3/envs/bio-tools/var/cache/libcifpp/`, see section 3d of
`instructions.md`.

OncoKB annotation was an exploratory path that we did not end up using as
the label source. Its env lives separately at `env/oncokb-environment.yml`.

## Reproducing the study

Two options. One-shot script, or step-by-step.

### One-shot

```bash
# Point this at a directory containing AF-{uniprot}-F1-model_v4.pdb.gz files.
# Pulled once from https://ftp.ebi.ac.uk/pub/databases/alphafold/v4/UP000005640_9606_HUMAN_v4.tar
bash scripts/run_pipeline.sh \
    --alphafold-dir /path/to/alphafold/human_proteome \
    --n-jobs 8 \
    --seeds "42 43 44 45 46" \
    --output-dir results/full_run
```

Stage-skipping by default. Re-running on top of an existing
`results/full_run/` no-ops every stage whose sentinel output already
exists. Pass `--force` to redo everything from scratch.

### Step by step

What we actually ran, in order. All commands assume the `pocket-vep` venv
is activated and the repo root is the working directory.

1. ClinVar pull and label. Downloads `variant_summary.txt.gz`, filters to
   GRCh38 SNVs, keeps missense only, applies the Oncogenicity then
   ClinicalSignificance fallback labeler, dedupes codon-degenerate copies,
   and writes ~30k labeled rows.

   ```bash
   python scripts/download_clinvar.py \
       --raw-dir data/raw \
       --out data/interim/clinvar_labeled.parquet
   ```

2. Gene symbol to UniProt mapping. Hits the UniProt ID-mapping API for
   every gene in the labeled set and keeps the Swiss-Prot canonical entry.

   ```bash
   python scripts/build_structure_cache.py --stage map
   ```

3. AlphaFold proteome. One-time, ~8 GB tarball and ~10 GB extracted.
   Extract into `data/raw/alphafold/` so the files land as
   `AF-{uniprot}-F1-model_v4.pdb.gz` directly inside that directory. Do
   not `gunzip` them, the pipeline reads `.pdb.gz`.

   ```bash
   mkdir -p data/raw/alphafold
   curl -fL -o data/raw/alphafold/UP000005640_9606_HUMAN_v4.tar \
       https://ftp.ebi.ac.uk/pub/databases/alphafold/v4/UP000005640_9606_HUMAN_v4.tar
   tar -xf data/raw/alphafold/UP000005640_9606_HUMAN_v4.tar \
       -C data/raw/alphafold/
   ```

4. pLDDT cache. Per-residue AlphaFold confidence scores pulled from each
   PDB's CA B-factors.

   ```bash
   python scripts/build_structure_cache.py --stage plddt
   ```

5. Structure features (fpocket and DSSP, parallel). This is the long
   stage, around 2 hours at `--n-jobs 8`. Outputs the pocket, SASA, SS,
   and pLDDT feature parquet.

   ```bash
   python scripts/build_structure_cache.py --stage features --n-jobs 8
   ```

6. Conservation cache (phyloP100way and phastCons100way bigWigs, ~15 GB
   one-time download). Per-locus query keyed on `(chrom, PositionVCF)`.
   Resumable, safe to re-run.

   ```bash
   python scripts/build_conservation_cache.py
   ```

7. Materialize the feature matrix. Inner-joins all three feature blocks so
   every ablation arm trains on the same rows.

   ```bash
   python scripts/build_feature_matrix.py
   ```

8. Experiment grid. Three models, seven feature configurations, five
   seeds, so 105 fits total. Splits are gene-grouped so no gene's variants
   leak between train, val, and test. Writes one row per
   `(seed, fs, model, split)` to `experiments.csv`, plus per-`(fs, model)`
   interpretation CSVs (native importances, permutation, SHAP) on the
   first seed.

   ```bash
   python scripts/run_experiments.py \
       --feature-matrix data/processed/feature_matrix.parquet \
       --out-dir results/full_run \
       --seeds 42 43 44 45 46
   ```

9. EDA plots. Per-feature distributions used in the report's feature
   overview figure.

   ```bash
   python scripts/visualize_features.py \
       --matrix data/processed/feature_matrix.parquet \
       --out results/full_run/eda
   ```

10. Results plots. Headline macro-F1 bar plots used in the report.

    ```bash
    python scripts/visualize_results.py \
        --csv results/full_run/experiments.csv \
        --out results/full_run/analysis_graphs
    ```

11. ROC curves. Best classifier per arm, plus the 3-panel per-model view.
    Refits on seed 42.

    ```bash
    python scripts/plot_roc_curves.py
    ```

12. Pipeline schematic for the report.

    ```bash
    python scripts/visualize_pipeline.py --format pdf png
    ```

13. Case study. Finds variants where the full-feature model rescues a
    sequence-only miss, then renders the ABL1 p.A433T pocket figure with
    PyMOL. Requires `pymol` on the PATH (separate from the analysis env,
    install via `conda install -c conda-forge pymol-open-source` into its
    own env if needed).

    ```bash
    python scripts/case_study_finder.py
    python scripts/visualize_case_study.py
    python scripts/_patch_case_study_labels.py   # enlarges in-image (A)/(B) labels
    ```

14. Build the report.

    ```bash
    cd report
    make            # builds main.pdf
    ```

## Outputs

Everything that landed in the report comes from `results/full_run/`. The
shipped copy in this archive is the exact set used. Files worth knowing
about,

- `results/full_run/experiments.csv`, one row per
  `(seed, feature_set, model, split)`
- `results/full_run/experiments_summary.csv`, mean and std across seeds
- `results/full_run/interpretations/{fs}_{model}.csv`, native, permutation,
  and SHAP importances per `(feature_set, model)` on seed 42
- `results/full_run/analysis_graphs/`, bar plots, PDFs are copied into
  `report/figures/`
- `results/case_study_candidates.csv`, top flipped variants shortlist
- `report/main.pdf`, final writeup

## Notes

- Headline metric is macro-F1. The label set is roughly 60/40
  benign/oncogenic so raw accuracy rewards an always-benign classifier.
  Balanced accuracy, ROC-AUC, PR-AUC, Brier, and ECE are reported
  alongside.
- Splits are gene-grouped via `GroupShuffleSplit` on `GeneSymbol`, so no
  gene appears in more than one of train, val, or test. Closes the obvious
  gene-level leakage path that an unblocked random split would open.
- Variants are deduped at `(GeneSymbol, protein_change_clean)` to drop
  codon-degenerate copies, i.e. different alt alleles encoding the same AA
  substitution. Confirmed zero label conflicts at that key.
- The full per-stage runbook with troubleshooting, disk budget, debug
  subsets, and macOS multiprocessing notes is in
  [`instructions.md`](instructions.md).
