from keras.models import Model, load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import NearestNeighbors
from jcesr_ml.mpnn import set_custom_objects, run_model, SolvationPreprocessor
from typing import List, Iterable, Dict, Optional
import pickle as pkl
import numpy as np
import gzip
import json
import os

set_custom_objects()

# Set the batch size
#  TODO (wardlt): Allow ability for users to configure, important for high-throughput screening
#   but not most uses
batch_size = 128

# Location for the global variables
#  Large objects that I only want to read from disk once!
model: Optional[Model] = None  # MPNN model used to predict solvation energy
converter: Optional[SolvationPreprocessor] = None  # Tool used to compute inputs for MPNNs
error_model: Optional[LinearRegression] = None  # Predicts error given distance from training set
error_model_prob: Optional[LogisticRegression] = None  # Predicts likelihood of error over 1 kcal/mol
error_rep_model: Optional[Model] = None  # Compute molecule representation
error_rep_nn_scaler: Optional[MinMaxScaler] = None  # Scales the representation to [0, 1]
error_rep_nn_computer: Optional[NearestNeighbors] = None  # Computes the the distance from the training set
error_rep_nn_dist_scale: Optional[Dict] = None  # Values used to scale the distances
dielectric_values: Optional[Iterable[float]] = None  # Dielectric constants used in search


def _load_model():
    global model, converter, error_model, error_rep_model, error_model_prob
    global error_rep_nn_scaler, error_rep_nn_computer, dielectric_values, error_rep_nn_dist_scale
    if model is not None:
        return

    # Load in the model
    model = load_model(os.path.join('model', 'best_model.h5'))

    # Load in the converter
    with open(os.path.join('model', 'converter.pkl'), 'rb') as fp:
        converter = pkl.load(fp)
    converter.atom_tokenizer.train = converter.bond_tokenizer.train = False

    # Get the list of dielectric constants used to train the model
    dielectric_values = converter.dielectric_cnsts

    # Load in the error-model-related stuff
    with open('error-models/train_dist_mol_64_mag.pkl', 'rb') as fp:
        error_model = pkl.load(fp)
    with open('error-models/train_dist_mol_64_prob.pkl', 'rb') as fp:
        error_model_prob = pkl.load(fp)
    error_rep_model = load_model('error-models/molecule-rep-model.h5')
    with open('error-models/molecule-nn-scaler.pkl', 'rb') as fp:
        error_rep_nn_scaler = pkl.load(fp)
    with gzip.open('error-models/molecule-nn-computer.pkl.gz') as fp:
        error_rep_nn_computer = pkl.load(fp)
    with open('error-models/train_dist_mol_64-dist-scaling.json') as fp:
        error_rep_nn_dist_scale = json.load(fp)


def evaluate_molecules(molecules: List[str], dielectric_constants: Optional[Iterable[float]] = None) -> Dict:
    """Compute the atomization energy of molecules

    Args:
        molecules ([str]): XYZ-format molecular structures. Assumed to be
            fully-relaxed
        dielectric_constants ([float]): List of dielectric constants
    Returns:
        ([float]): Estimated G4MP2 atomization energies of molecules
    """
    _load_model()

    # Add default arguments
    if dielectric_constants is None:
        dielectric_constants = dielectric_values

    # Prepare the output dict
    output = dict(smiles=molecules)

    # Set the dielectirc constants in the converter
    converter.dielectric_cnsts = list(dielectric_constants)
    if any([e < min(dielectric_values) or e > max(dielectric_values) for e in dielectric_constants]):
        output['warnings'] = [f'Some dielectrics outside of range in training set: {min(dielectric_values):.2f} '
                              f'- {max(dielectric_values):.2f}']

    # Compute the solvation energies
    solv_energies = run_model(model, converter, molecules, chunk_size=batch_size)
    output['solvation-energies'] = solv_energies.tolist()
    output['dielectric-constants'] = converter.dielectric_cnsts

    # Run the error estimation model
    mol_rep = run_model(error_rep_model, converter, molecules, chunk_size=batch_size)
    mol_rep_scaled = error_rep_nn_scaler.transform(mol_rep)
    mol_dists = error_rep_nn_computer.kneighbors(mol_rep_scaled)[0]
    mean_dists = mol_dists.mean(axis=1)

    # Scale the training set distances
    mean_dists = (mean_dists - error_rep_nn_dist_scale['min_dist']) / (
        error_rep_nn_dist_scale['max_dist'] - error_rep_nn_dist_scale['min_dist'])

    output['training-set-distance'] = mean_dists.tolist()

    # Run them through the error models
    output['expected-error'] = np.exp(error_model.predict(mean_dists[:, None])).tolist()
    output['likelihood-error-above-1kcal/mol'] = error_model_prob.predict_proba(mean_dists[:, None])[:, 1].tolist()

    return output


if __name__ == "__main__":
    # Get some data
    mols = ['C', 'CC', 'CCC']

    # Evaluate the energies
    pred = evaluate_molecules(mols)
    print(json.dumps(pred, indent=2))
