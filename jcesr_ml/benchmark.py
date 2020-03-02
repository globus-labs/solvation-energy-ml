import pandas as pd
import os

default_dataset = os.path.join(os.path.dirname(__file__), '..', 'data',
                               'output', 'g4mp2_data.json.gz')


dielectric_constants = {
    'sol_acetone': 20.493,
    'sol_acn': 35.688,
    'sol_dmso': 46.826,
    'sol_ethanol': 24.852,
    'sol_water': 78.3553
}
"""Dielectric constants of different solvents.

Taken from: https://gaussian.com/scrf/"""


def load_benchmark_data(path=default_dataset):
    """Load the benchmark dataset
    
    Args:
        path (str): Path to the benchmark data
    Returns:
        - (pd.DataFrame): Training data
        - (pd.DataFrame): Hold-out data
    """
    
    data = pd.read_json(path, lines=True)
    train_data = data.query('not in_holdout')
    holdout_data = data.query('in_holdout')
    return train_data, holdout_data
