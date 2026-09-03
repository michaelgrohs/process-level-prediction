# System-level prediction 
This repository serves as the supplementary code base and results files for a conference submission. 

## Structure

| Path                                                                         | What it is                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `execution_helper_ts_models/`                                                | Pipelines for all statistical and ML time-series models as well as baselines (`pipeline_synthetic.ipynb`, `pipeline_real.ipynb`). Must run before `pipelines/robustness/robustness_ts_models.ipynb`, which reads its recorded hyperparameters.                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `pipelines/`                                                                 | Pipelines for all the trace-level PPM approaches. One notebook per model (`camargo/` for GLSTM, `bukhsh/` for PT, `amiri/` for PGT) covering the default trims (`ssd` real-life + `none` synthetic) -- full HPO through all three prediction regimes (first/half/plain-field). `other_trims/` subfolders under each cover `peak_*`/`magnitude_*` real-life trims. Plus `pmsd/`, `robustness/` (incl. `robustness_prophet.ipynb`), `intercase/` (incl. `intercase_summary_comparison.ipynb`), `compile_results/compile_results.ipynb` (results_auto.xlsx + aggregated-comparison charts). `run_experiments_reallife_ts.py`/`run_experiments_synthetic_ts.py` -- the scripts to use for running Chronos/TabPFN. |
| `analysis/`                                                                  | Files to obtain analytical insights into error directions and trace-level bias.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `camargo/`, `amiri/`, `bukhsh/`, `inter-case-camargo/`, `inter-case-bukhsh/` | The three PPM model wrappers + their inter-case variants.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `GenerativeLSTM/`, `pgtnet/`, `processtransformer/`                          | Vendored architectures camargo/amiri/bukhsh wrap.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `plain-field/`                                                               | Zero-knowledge (plain-field) prediction regime: arrival forecasting + SOS case simulation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `simulation_baselines/`                                                      | `PMSD-main/` -- PMSD training/prediction.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `steady_state_detection/`                                                    | `ssd_trim.py` -- the steady-state cutoff logic behind the `ssd` real-life trim.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `helpers/`                                                                   | `visualizations.ipynb` Produces paper visualizations. Reads straight from `data/real-life/` -- no dependency on any pipeline's output.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `data/real-life/`, `data/synthetic/`                                         | Input XES event logs. **Not contained on GitHub** has tp be downloaded from https://figshare.com/s/c1a3f2a952ffc2ad58f1.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Top-level `.py` files                                                        | Shared utilities imported bare (`sys.path`-based) throughout: `create_prefixes_from_windows.py`, `time_series_creation.py`, `time_series_preprocessing.py`, `time_series_prediction.py`, `setttings.py`, `prophet_baseline.py`, `hpo_val_scoring.py`, `run_experiments_real.py`, `run_predictions_real.py`.                                                                                                                                                                                                                                                                                                                                                                                                   |
| Top-level `appendix.pdf`                                                     | Details on hyperparameter selection and excerpt from standard deviations in MAE across runs with three different seeds, indicating models' robustness to random weight initialization.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |

Everything not listed above (`best_models/`, `results/`, `robustness/` output,
`GenerativeLSTM/GenerativeLSTM/output_files/`) is a **generated output**,
gitignored, and gets created fresh by running the pipelines.

## Additional Results

In ```/results```, we provide additional files the supplement the findings from the paper:
1. **results_auto.xlsx**: This file contains all the results, including additional metrics (MAE, MSE, RMSE) for all our experiements. It also contains results for the other applied truncation strategies (peak and magnitude).
2. **inter_case_summary_comparison.csv**: The results from investigating whether inter-case features improve GLSTM or PT_RT.
3. **robustness_by_dataset.xlsx**: We here report the average MAE ± standard deviation across three random seeds for all the models, indicating their robustness to random weight initialization. This is the extended version of the tables referenced in ```appendix.pdf```.


## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Before running anything, sanity-check the three PPM trainers import cleanly:

```bash
python3 -c "
import sys; from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
from camargo.trainer import CamargoTrainer
from amiri.trainer import AmiriTrainer
from bukhsh.trainer import BukhshTrainer
print('OK')
"
```

## Running everything, in order

Each pipeline notebook is resumable (skips work that's already been done), so
re-running after an interruption just picks up where it left off.

1. **PPM models** -- `pipelines/camargo/camargo_pipeline.ipynb`,
   `pipelines/bukhsh/bukhsh_pipeline.ipynb`,
   `pipelines/amiri/amiri_pipeline.ipynb`. Each covers `ssd` (real-life) +
   `none` (synthetic): HPO, then first/half/plain-field prediction, for every
   dataset. Add `pipelines/{camargo,bukhsh,amiri}/other_trims/*.ipynb` if you
   also want `peak_*`/`magnitude_*` real-life trims.
2. **PMSD** -- `pipelines/pmsd/ssd_ts_pmsd.ipynb` (real-life) and
   `pipelines/pmsd/pmsd_synthetic_pipeline.ipynb` (synthetic).
3. **TS model baselines** -- `execution_helper_ts_models/pipeline_synthetic.ipynb`
   and `pipeline_real.ipynb`. Needed before step 5's TS robustness sweep.
4. **Intercase** -- `pipelines/intercase/intercase_camargo_suffix.ipynb`,
   `pipelines/intercase/intercase_bukhsh_remaining_time.ipynb`.
5. **Robustness** -- `pipelines/robustness/robustness_pipeline.ipynb`
   (camargo + bukhsh + amiri, new seeds + half/plain-field), then
   `robustness_pmsd.ipynb`, `robustness_baseline.ipynb`,
   `robustness_ts_models.ipynb`, `robustness_prophet.ipynb`, then finally
   `pipelines/robustness/compile_robustness.ipynb` to aggregate everything
   (mean/std/CV% tables, best-of-3-runs tables, `robustness_by_dataset.xlsx`).
6. **`results_auto.xlsx` + the aggregated-comparison `_grid.pdf` charts** --
   `pipelines/compile_results/compile_results.ipynb`.
7. **`error_direction`** -- `analysis/error_direction.ipynb`.
8. **`prefix_length_analysis`** -- `analysis/prefix_length_analysis.ipynb`
   (and `_filtered`). Each has a coverage-check cell near the top that scans
   for missing raw predictions from steps 1-2 and tells you exactly which
   pipeline notebook to (re-)run if anything's missing, instead of silently
   under-reporting.


