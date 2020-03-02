# Predicting Solvation Energy for Small Molecules with Machine Learning

This repository contains the code associated with a recent paper by Ward et al., 
[Title TBD](http://arxiv.org/)

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
