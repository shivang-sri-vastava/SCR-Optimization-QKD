# Simulation code for the shared transmitter CV-QKD link

This repository contains the Python model and numerical results for *System-Design Optimization of a Single-Transmitter, Single-Fiber QKD-based ITS Communication Link* (Srivastava, Ricard, Jaouën, Schiavon, and Alléaume). It covers the forward Alice-to-Bob link, its provisioned classical return service, the separated-link comparison, and the receiver and transmitter-noise sensitivity studies. The paper gives the equations and distinguishes measured inputs from design assumptions.

## Files

- `simulations/key_rate_model.py`: binomial 64-QAM asymptotic key-rate calculation with a trusted receiver.
- `simulations/system_model.py`: classical traffic, leakage, Raman noise, bandwidth constraints, and optimization.
- `simulations/run_studies.py`: platform point, spectral and distance sweeps, power sweep, and receiver study.
- `simulations/run_sensitivities.py`: transmitter-noise and transport-efficiency sweeps.
- `simulations/build_figures.py` and `build_frequency_layout.py`: generated plot and explanatory frequency layout.
- `simulations/verify_package.py`: numerical checks against the supplied result tables.
- `data/platform_snr_digitized.csv`: digitized relative-power/SNR markers used in the regression.
- `results/`: numerical tables corresponding to the paper's plots, tables, and sensitivities. The parameter ledger records the source or assumption for each main input.

The architecture and resource-sharing illustrations in the manuscript are separate artwork; this repository generates the numerical figures, not those illustrations.

## Run

Use Python 3.12. Install the packages in `requirements.txt`, preferably in a fresh environment. From the repository root:

```bash
python -m venv .venv
# Activate .venv using the command appropriate for your shell.
python -m pip install -r requirements.txt
python simulations/verify_package.py
```

To regenerate the tables and plots:

```bash
python simulations/run_studies.py
python simulations/run_sensitivities.py
python simulations/build_figures.py
python simulations/build_frequency_layout.py
python simulations/verify_package.py
```

The full sweeps involve repeated security calculations and numerical optimization, so they can take substantially longer than the initial verification. On a multi-core machine, setting `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` avoids unnecessary BLAS oversubscription. The scripts write CSV files in `results/`, figures in `figures/`, and a verification report in `validation/`.

## Interpretation

The rates are **asymptotic model projections**, not experimentally demonstrated OTP throughput or finite-size composable keys. The return service is assumed to provide 10 Gbit/s net authenticated capacity on a separate fiber. The forward classical transport efficiency is assumed to be 0.95. The receiver-noise values at 350, 500, and 750 MHz follow an adopted quadratic model; the 350 MHz value is an assumed two-sided complex-baseband budget.

Two calibration choices especially affect absolute predictions: the reported 4.3 mSNU platform excess noise is treated as channel-output-referred before trusted detection, and zero dB on the digitized relative classical-power curve is assumed to mean 1 mW. The latter reference is not established by the source curve. The Raman coefficient is a representative design value, not a measurement at the optimized GHz channel separations. Additional admitted transmitter noise is a sensitivity scenario, not a measured spectrum. These assumptions are recorded in `results/parameter_ledger.csv` and discussed in the paper.

The optimizers use fixed starting points and seeds. Their results are numerical optima over the stated bounds, without a global-optimality certificate. `verify_package.py` checks identities, feasibility, four direct key-rate evaluations, and independent optimizer runs; it is not a proof of the security model.

## Reuse and citation

Please cite the accompanying paper when referring to the model or results. The repository URL and any archival DOI should be added to the paper's code-availability statement after the repository is public. A software license should be selected with the code owners before public release; until one is added, public visibility alone does not grant reuse rights.
