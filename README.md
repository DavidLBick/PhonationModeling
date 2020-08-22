# Description

This repo contains source codes for phonation modeling and estimating vocal fold and vocal tract models from speech.

# Package structure
```
.
├── README.md
├── gnu-gpl-v3.0.md
└── src
    ├── PhonationModeling
    │   ├── __init__.py
    │   ├── data
    │   │   ├── data_preprocess.py
    │   │   └── extract_glottal_flow.py
    │   ├── external
    │   │   ├── __init__.py
    │   │   └── pypevoc
    │   ├── main_scripts
    │   │   ├── bifurcation_plot.py
    │   │   ├── configurations
    │   │   │   ├── run_e2e.configure.json
    │   │   │   └── vocal_tract_estimate.configure.json
    │   │   ├── process_output.py
    │   │   ├── run_e2e.py
    │   │   ├── vocal_fold_estimate.py
    │   │   └── vocal_tract_estimate.py
    │   ├── models
    │   │   ├── __init__.py
    │   │   └── vocal_fold
    │   │       ├── __init__.py
    │   │       ├── adjoint_model_displacement.py
    │   │       ├── adjoint_model_volume_velocity.py
    │   │       ├── vocal_fold_model_displacement.py
    │   │       └── vocal_fold_model_volume_velocity.py
    │   └── solvers
    │       ├── __init__.py
    │       ├── ode_solvers
    │       │   ├── __init__.py
    │       │   ├── dae_solver.py
    │       │   └── ode_solver.py
    │       ├── optimization.py
    │       └── pde_solvers
    │           ├── __init__.py
    │           ├── fem_solver.py
    │           └── test_fem_solver.py
    ├── PhonationModeling.egg-info
    ├── requirements.txt
    └── setup.py
```

## data
`data_preprocess.py`: data pre-processing

`extract_glottal_flow.py`: script for extracting glottal flow

## external
`pypevoc`: external signal & speech processing library

## main_scripts
`configurations`: configuration files containing experiment settings, input to main scripts

`vocal_fold_estimate.py`: script for estimating vocal fold model parameters

`vocal_tract_estimate.py`: script for estimating vocal fold + vocal tract models

`run_e2e.py`: script for running model parameter estimation end-to-end over list of files

`process_output.py`: process experiment outputs and collect results

`bifurcation_plot.py`: script for plotting bifurcation and entrainment diagrams

## models
`vocal_fold`: contains vocal fold models. 
    
    `vocal_fold_model_displacement.py`: asymmetric vocal fold model with displacement as variable
    `adjoint_model_displacement.py`: adjoint of vocal fold displacement model
    `vocal_fold_model_volume_velocity.py`: model of volume velocity flow through vocal fold
    `adjoint_model_volume_velocity.py`: adjoint of vocal fold velocity model

## solvers
`ode_solvers`: `ode_solver.py` is used for solving ODEs (primary model), `dae_solver.py` is used for solving DAEs (adjoint model).

`pde_solvers`: `fem_solver.py` is used for solving PDEs via FEM.

`optimization.py`: optimizers.

# Usage

## set up python virtual environment, install dependencies
Use `conda`, `requirements.txt`, and `setup.py`.

## run main scripts

```
# run one of the main scripts: run_e2e, vocal_fold_estimate, vocal_tract_estimate
python <main script> --configure_file <path to configuration file>
```

# Contact
Author: Wenbo Zhao

Email: waynero1954@gmail.com

# References & useful links

[Speech-Based Parameter Estimation of an Asymmetric Vocal Fold Oscillation Model and Its Application in Discriminating Vocal Fold Pathologies](https://arxiv.org/abs/1910.08886)

[Introduction to Numerical Methods for Variational Problems](https://hplgit.github.io/fem-book/doc/web/index.html)

[FEniCSx Project](https://fenicsproject.org)
