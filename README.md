# pocket-vep

Somatic missense variant effect predictor.

Report at [`report/main.pdf`](report/main.pdf), proposal at [`proposal.pdf`](proposal.pdf).

## Reproducing the study



```bash
# point at a directory containing AF-{uniprot}-F1-model_v4.pdb.gz files.
# pulled once from https://ftp.ebi.ac.uk/pub/databases/alphafold/v4/UP000005640_9606_HUMAN_v4.tar
bash scripts/run_pipeline.sh \
    --alphafold-dir /path/to/alphafold/human_proteome \
    --n-jobs 8 \
    --seeds "42 43 44 45 46" \
    --output-dir results/full_run
```


### Step by step

1. ClinVar pull and label. 
   ```bash
   python scripts/download_clinvar.py \
       --raw-dir data/raw \
       --out data/interim/clinvar_labeled.parquet
   ```

2. Gene symbol to UniProt mapping.
   ```bash
   python scripts/build_structure_cache.py --stage map
   ```

3. AlphaFold proteome. 
   ```bash
   mkdir -p data/raw/alphafold
   curl -fL -o data/raw/alphafold/UP000005640_9606_HUMAN_v4.tar \
       https://ftp.ebi.ac.uk/pub/databases/alphafold/v4/UP000005640_9606_HUMAN_v4.tar
   tar -xf data/raw/alphafold/UP000005640_9606_HUMAN_v4.tar \
       -C data/raw/alphafold/
   ```

4. pLDDT cache.  

   ```bash
   python scripts/build_structure_cache.py --stage plddt
   ```

5. Structure features (fpocket and DSSP, parallel). 

   ```bash
   python scripts/build_structure_cache.py --stage features --n-jobs 8
   ```

6. Conservation cache . 

   ```bash
   python scripts/build_conservation_cache.py
   ```

7. feature matrix. 

   ```bash
   python scripts/build_feature_matrix.py
   ```

8. Experiment grid.
   ```bash
   python scripts/run_experiments.py \
       --feature-matrix data/processed/feature_matrix.parquet \
       --out-dir results/full_run \
       --seeds 42 43 44 45 46
   ```

9. plots. 

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

11. ROC curves.
    
    ```bash
    python scripts/plot_roc_curves.py
    ```

12. Pipeline schematic for the report.

    ```bash
    python scripts/visualize_pipeline.py --format pdf png
    ```

13. Case study. 

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

