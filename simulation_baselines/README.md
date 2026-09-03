# simulation_baselines/

Baseline approaches that live outside the main PPM/TS-model pipelines.

| Path | What it is |
|---|---|
| `PMSD-main/` | The PMSD simulation baseline (vendored). See its own `README.md`/`further_docs.md`. |

`run_experiments_reallife_ts.py`/`run_experiments_synthetic_ts.py` (Chronos/TabPFN
local execution) live in `pipelines/` now, not here.

## Chronos/TabPFN — read-only everywhere else, except in those two scripts

Everywhere else in this project, Chronos and TabPFN ("tabular foundation
models" / "Tab. FM" in `compile_results.ipynb`'s taxonomy) are **read-only**
— `analysis/ts_comparison.py`'s `load_foundation_predictions()` reads a
colleague's precomputed `predictions_<dataset>.csv` files; nothing else in
the project fits either model fresh. That's still true for the main results
tree (`results/<trim>/<dataset>/`, what `compile_results.ipynb`/
`results_auto.xlsx`/`analysis/error_direction.py` all read).

`pipelines/run_experiments_reallife_ts.py` / `run_experiments_synthetic_ts.py`
are the one exception: they call `time_series_prediction.py`'s
`forecast_tabpfn`/`forecast_chronos` directly, which really do fit/run
inference locally (`TabPFNRegressor`, `amazon/chronos-t5-{model_size}` via
`BaseChronosPipeline.from_pretrained`) — no precomputed file involved. Their
`DEFAULT_MODELS` is `["tabpfn", "chronos"]`

**Output goes to a separate tree**, not the main `results/` layout every
other approach uses:
- Real-life: `results/reallife_ts/<trim>/<dataset>/`
- Synthetic: `results/synthetic/none/split_<train>_<val>_<test>/<dataset>/`

## Usage (run from the repo root)

```bash
# Real-life, all datasets/trims, Chronos + TabPFN (default):
python pipelines/run_experiments_reallife_ts.py

# Synthetic, all datasets, Chronos + TabPFN (default):
python pipelines/run_experiments_synthetic_ts.py

# Single dataset / single trim / specific models — see each script's own
# module docstring (--help) for the full flag set.
```
