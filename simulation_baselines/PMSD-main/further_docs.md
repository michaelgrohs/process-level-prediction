# PMSD baseline — concurrent_cases / throughput_time via System Dynamics

Reimplementation of Pourbafrani & van der Aalst, *"Discovering System Dynamics
Simulation Models Using Process Mining"* (IEEE Access, 2022,
https://vdaalst.com/publications/p1260.pdf), adapted as a baseline for this
project's `concurrent_cases` / `throughput_time` prediction task.



## What "training" and "prediction" mean here

1. **Fit** — build an SD-Log (a per-day table of aggregate process variables)
   from the train+val portion of an event log, discover which variables
   correlate with which others at what lag, and fit a simple regression
   equation for each variable that needs to be forecast.
2. **Simulate** — walk a fixed recursive stock-flow formula forward over the
   test horizon, producing `concurrent_cases` and `throughput_time` for every
   test-period day. This *is* PMSD's own idea of "prediction": a coarse-grained
   simulation of the process, not a per-case/per-trace forecast.

