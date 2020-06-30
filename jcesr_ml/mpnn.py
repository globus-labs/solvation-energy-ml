"""Utilities related to running MPNN"""

from keras import Model
from keras.utils import get_custom_objects
from keras.callbacks import Callback
from keras.optimizers import Adam

from nfp.preprocessing import GraphSequence, SmilesPreprocessor
from nfp.layers import (MessageLayer, GRUStep, ReduceAtomToMol, Embedding2D, Squeeze, Set2Set, Embedding2DCompressed)
from nfp.models import GraphModel

from jcesr_ml.keras import set_custom_layers as general_keras_layers
from jcesr_ml.mpi import allgather_feature_calculation
from jcesr_ml.keras import KerasHorovodTrainer
from multiprocessing import Pool
from typing import List
import pickle as pkl
import pandas as pd
import numpy as np
import shutil
import json
import os


def atom_feature_element_only(atom) -> str:
    return str(atom.GetSymbol())


def bond_feature_type_only(bond, **kwargs) -> str:
    return str((bond.GetBondType(), bond.GetIsConjugated()))


def set_custom_objects():
    """Add custom objects to dictionary for loading Keras model"""
    general_keras_layers()
    custom_objs = get_custom_objects()

    # Add the new objects
    for cls in (MessageLayer, GRUStep, ReduceAtomToMol, Embedding2D,
                Embedding2DCompressed, Squeeze, GraphModel, Set2Set):
        custom_objs[cls.__name__] = cls


class SynchronousDataShuffler(Callback):
    """Ensures each MPI rank uses the same shuffling and different batches"""

    def __init__(self, generator):
        """
        Args:
             generator (GraphSequence): Data generator. Should be initialized with same seed,
                and an appropriate offset
             comm (mpi4py.mpi.MPIComm): MPI communicator
        """
        super().__init__()
        self.generator = generator

    def on_epoch_begin(self, epoch, logs=None):
        # Shuffling on batch beginning so that the first epoch is shuffled properly
        self.generator.on_epoch_end()
        
        
class AtomicPropertySequence(GraphSequence):
    """Graph sequence where output properties are per-atom rather than per-molecule.
    
    These properties are lists per entry, which must therefore be concatentated to work with TF"""
    
    def __getitem__(self, idx):
        if self._y is not None:
            X, y = super().__getitem__(idx)
            return X, np.hstack(y)


def dist_compute_graphs(mols, preprocessor: SmilesPreprocessor, comm, logger=None,
                        n_threads=None, max_chunksize=8192):
    """Compute all graphs, using MPI

    Args:
        mols ([str]): List of molecules to evaluate (all threads must have same copy)
        preprocessor (SmilesPreprocessor): Tool used to generate the graphs
        comm (mpi4py.mpi.Intracom): Tool for MPI communication
        logger: An initialized logger object
        n_threads (int): Number of threads per node
        max_chunksize (int): Maximum number of entries to reduce at one time
    Returns:
         ([dict]): Graphs for each molecule
    """
    return allgather_feature_calculation(mols, preprocessor.construct_feature_matrices,
                                         comm, max_chunksize, 1, logger)


def save_model_files(name, converter, model, output_props=['sol_water'],
                     overwrite=False, correct=False, normalize=True):
    """Save a converter and model pair to disk
    
    Args:
        name (str): Name of the model
        converter (SmilesPreprocessor): Tool used to convert molecule from smiles to features
        model (Model): Keras model, uncompiled
        output_props ([str]): List of output properties
        normalize (bool): Whether to normalize targets
        overwrite (bool): Whether to overwrite existing files
        correct (bool): Whether to correct for batch size variance
    """
    
    # Make the output directory
    output_path = os.path.join('networks', name)
    if os.path.isdir(output_path):
        if overwrite:
            shutil.rmtree(output_path)
        else:
            print('Already output. Skipping')
            return
    os.makedirs(output_path)
    
    # Save the converter
    with open(os.path.join(output_path, 'converter.pkl'), 'wb') as fp:
        pkl.dump(converter, fp)
                
    # Save the model
    model.save(os.path.join(output_path, 'architecture.h5'))
    
    # Save the options
    with open(os.path.join(output_path, 'options.json'), 'w') as fp:
        json.dump({
            'output_props': output_props,
            'normalize': normalize,
            'batch_correction': correct
        }, fp)


def run_model(model: Model, converter: SmilesPreprocessor, mols: List[str],
              chunk_size: int = 256, n_jobs=None,
              logger=None) -> np.ndarray:
    """Invoke a MPNN model on a series of molecules

    Args:
        model (Model): Keras model
        converter (SmilesPreprocessor): Tool to generate molecular graph
        mols ([str]): List of molecules to evaluate
        chunk_size (int): Number of molecules to process concurrently
        n_jobs (int): Number of jobs to use
        logger: An initialized logger object
    Returns:
          (np.ndarray) Predicted properties for each molecule
    """

    # Compute the molecular graphs
    if logger is not None:
        logger.info(f'Computing feature matrices on {n_jobs} threads')
    with Pool(n_jobs) as p:
        valid_graphs = p.map(converter.construct_feature_matrices, mols)

    # Make the graph generator
    generator = GraphSequence(valid_graphs, batch_size=chunk_size, final_batch=True,
                              shuffle=False)
    n_batch = len(generator)
    if logger:
        logger.info(f'Running model in chunks of {chunk_size}')

    # Generate all the predictions
    return np.vstack([model.predict_on_batch(b)
                      for _, b in zip(range(n_batch), generator)])


class SolvationPreprocessor(SmilesPreprocessor):
    """Preprocessor that includes the dielectric constants of solvents
    as inputs for a molecule graph
    """
    
    def __init__(self, dielectric_cnsts, **kwargs):
        """
        Args:
            dielectric_cnsts ([float]): Dielectric constants to use as inputs
        """
        super().__init__(**kwargs)
        self.dielectric_cnsts = dielectric_cnsts
        
    def construct_feature_matrices(self, smiles):
        output = super().construct_feature_matrices(smiles)
        output['dielectric_constants'] = np.array(self.dielectric_cnsts)[None, :]
        return output


class PartialChargesPreprocessor(SolvationPreprocessor):
    """Adds the partial charges and dielectric constants to the inputs"""

    def __init__(self, charges_lookup: dict, dielectric_cnts, **kwargs):
        """
        Args:
            charges_lookup (dict): Partial charges for each SMILES string
            dielectric_cnts ([float]): Dielectric constants to use as inputs
        """

        super().__init__(dielectric_cnts, **kwargs)
        self.charges_lookup = charges_lookup

    def construct_feature_matrices(self, smiles):
        output = super().construct_feature_matrices(smiles)
        output['partial_charges'] = self.charges_lookup[smiles]
        return output


class MPNNTrainer(KerasHorovodTrainer):
    """Utility for training MPNN models"""

    def __init__(self, comm, hvd, logger, net_dir, work_dir, train_size, batch_size,
                 lr_start, lr_patience, lr_decay, lr_min, lr_warmup,
                 ckpt_interval, max_epochs,
                 n_threads=None, valid_batch_size=1024, split_dir='splits',
                 ckpt_history=None, **kwargs):

        # Pass on the information needed for training
        super().__init__(comm, hvd, logger, net_dir, work_dir, train_size, batch_size,
                         lr_start, lr_patience, lr_decay, lr_min, lr_warmup,
                         ckpt_interval, max_epochs, n_threads, valid_batch_size,
                         'smiles_0', split_dir, ckpt_history, **kwargs)

        # Load the molecule preprocessor
        with open(os.path.join(net_dir, 'converter.pkl'), 'rb') as fp:
            self.converter = pkl.load(fp)

    def set_custom_objects(self):
        super().set_custom_objects()
        set_custom_objects()

    def make_train_loader(self, train_mols: list, train_target: np.ndarray,
                          batch_size: int):
        # Get the MPI pool information
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()

        # Determine the offset
        offset = rank * len(train_mols) // size

        # Compute molecular graphs
        train_graphs = dist_compute_graphs(train_mols, self.converter, self.comm,
                                           self.logger, self.n_threads)

        # Make the data loader
        self.logger.info(f'Making training set loader with offset of {offset}')
        return GraphSequence(train_graphs, train_target, batch_size,
                             final_batch=False, shuffle_offset=offset)

    def make_validation_loader(self, valid_mols: list, valid_target: np.ndarray,
                               batch_size: int):
        valid_graphs = list(map(self.converter.construct_feature_matrices, valid_mols))
        return GraphSequence(valid_graphs, valid_target, batch_size,
                             final_batch=True, shuffle=False)

    def add_callbacks(self, rank, callbacks, train_load: GraphSequence):
        callbacks.append(SynchronousDataShuffler(train_load))

    def make_optimizer(self, options):
        opt = options.get('optimizer', 'adam')
        if opt == 'adam':
            decay = 1 / (self.max_epochs * self.train_size / self.global_batch_size)
            return Adam(lr=self.lr_start, decay=decay, clipnorm=1)
        else:
            raise ValueError(f'Unrecognized optimizer: {opt}')
