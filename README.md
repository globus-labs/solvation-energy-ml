# Predicting Solvation Energy for Small Molecules with Machine Learning

This repository contains the code associated with a recent paper by Ward et al., 
[Datasets and Machine Learning Models for Accurate Estimates of Solvation Energy in Multiple Solvents](http://arxiv.org/)

## Using the Model

The best model from our study has been made available for easy use via [DLHub](https://www.dlhub.org/).
To run it, you only need install the [DLHub SDK](https://github.com/DLHub-Argonne/dlhub_sdk) for Python via Pip:

```bash
pip install dlhub_sdk
```

Once installed, run the model through the `run_model.py` script in the [dlhub](./dlhub) directory:

```bash
>>> python run_model.py C CC CC=O
{
  "smiles": [
    "C",
    "CC",
    "CC=O"
  ],
  "solvation-energies": [
    [
      0.3075838088989258,
      0.5105352401733398,
      0.9096126556396484,
      1.0906600952148438,
      2.1758193969726562
    ],
    [
      -1.2038593292236328,
      -0.5588760375976562,
      -0.6135234832763672,
      -0.34089183807373047,
      1.6689720153808594
    ],
    [
      -6.803349494934082,
      -6.5320281982421875,
      -6.589166164398193,
      -6.29440975189209,
      -5.189798355102539
    ]
  ],
  "dielectric-constants": [
    20.493,
    35.688,
    46.826,
    24.852,
    78.3553
  ],
  "training-set-distance": [
    0.9882038873152169,
    0.8182617387674459,
    0.5418242643337519
  ],
  "expected-error": [
    1.2369891114952873,
    0.9572314552138276,
    0.6307983394641673
  ],
  "likelihood-error-above-1kcal/mol": [
    0.3958329444154754,
    0.2861134888650363,
    0.1526715392416573
  ]
}

```

Note how the model produces predictions for the solvation energy in multiple solvents (differentiated by their dielectric constants) and estimates for how large the errors should be.

## Installation

The environment for this repository is described using Anaconda. Install it using:

```bash
conda env create --file environment.yml --force
```

The only dependency not included in the environment is [Horovod](https://horovod.readthedocs.io/en/latest/),
which we use for data parallel training.
Data parallel training let us train our models quickly on supercomputers, but is not necessary to recreate our results.

We recommend you train models without Horovod and also avoid using our complicated training scripts,
or consult your supercompuing center for installing Horovod and using it with our environment.

## Repository Organization

Our repository is broken into several key folders:

- `data/input`: Data as provided
- `data/output`: Input data processed into easily processible forms
- `jcesr_ml`: Utility functions used in the notebooks

The other folders represent different steps of the analysis.
Each should contain its own README describing the purpose of the tests.
Notebooks in each folder are labeled based on the order in which they should be executed.
