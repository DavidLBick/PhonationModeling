# Description

This repo contains source codes for phonation modeling and estimating vocal fold and vocal tract models from speech. It also contains a script to compute features over the outputs of the models, and perform statistical tests between groups such as those with and without vocal fold pathology like COVID-19. 

# Usage

## Setup and running the VFO model 
1. Create conda environment 
2. `conda install --file /path/to/requirements.txt`
3. Make sure to `cd` into the same directory as `setup.py`, and then run `python setup.py install`
4. Edit `PhonationModeling/main_scripts/configurations/run_e2e.configure.json` on lines 5, 6, 7, 8, and 25 to put your own values in for `wav_dir`, `list_dir`, `wav_list`, `results_save_dir`, `log_dir`, and `filename`. The meaning for each of these respectively is as follows: the path to the directory contained wav files of the audios, the directory of a .txt file that lists the audios that you want to process, the filename of a .txt file that lists all the audios within `wav_dir` that you want to process, the directory to save your outputs, and the directory to save log files.
5. Run `python -m PhonationModeling.main_scripts.run_e2e --configure_file src/PhonationModeling/main_scripts/configurations/run_e2e.configure.json`

## Running the feature computation and statistical tests
1. Edit `statistics_config.json` to put in your values for `data_dir`, `label_df_path`, `pos_definitions`, `pos_neg_values`, and `measures`. 
    * The meaning for these respectively is as follows: the directory of the outputs of the VFO model, the path to the .csv file that contains the labels for each audio, and the list of features that you want to compute
    * Note that `label_df_path` expects minimally expects the column `path`. `path` should be the path to the audio file _without_ the directory that it is in. This script assumes that all the paths are in `data_dir`. The script supports testing multiple columns as the positive-negative values, which is explained in the next bullet point.
    * Note that `pos_definitions` is a list of the columns in `label_df_path` that you want to use as labels for the statistical tests. For example, if you want to compare the outputs of the model for audios corresponding to positive and negative COVID-19 test results, as well as symptomatic and asymptomatic COVID-19 patients, then you would include `pos_definitions` as `["label", "symptomatic"]`. This would run the statistical tests for both the positive and negative COVID-19 test results, as well as the symptomatic and asymptomatic COVID-19 patients.
    * Note that `pos_neg_values` is a dictionary that maps the columns you listed in `pos_definitions` to the values for positive and negative within that column. To continue the example from the previous bullet point, if you want to compare the outputs of the model for audios corresponding to positive and negative COVID-19 test results, as well as symptomatic and asymptomatic COVID-19 patients, then you would include `pos_neg_values` as `{"label": ["positive", "negative"], "symptomatic": ["yes", "no"]}`. This would run the statistical tests for both the positive and negative COVID-19 test results, as well as the symptomatic and asymptomatic COVID-19 patients. It supports using integers such as 1 and 0 for labels as well. 
    * Note that the currently supported options for `measures` are `stats`, `info-theory`, and `all`. Currently `all` simply means both `stats` and `info-theory`. This allows flexibility for running different tests on different sets of data depending on one's desires.
2. Run `python statistics.py`

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

# Contact
Author: Wenbo Zhao
Email: waynero1954@gmail.com

Edits by David Bick
Email: davidlbick@gmail.com

# References & useful links

[Speech-Based Parameter Estimation of an Asymmetric Vocal Fold Oscillation Model and Its Application in Discriminating Vocal Fold Pathologies](https://arxiv.org/abs/1910.08886)

[Introduction to Numerical Methods for Variational Problems](https://hplgit.github.io/fem-book/doc/web/index.html)

[FEniCSx Project](https://fenicsproject.org)