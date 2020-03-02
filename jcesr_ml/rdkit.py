"""Tooling to make using RDKit easier"""

import os
import numpy as np
from rdkit.Chem import AllChem
from sklearn.base import BaseEstimator
from concurrent.futures import ThreadPoolExecutor


class MorganFingerprint(BaseEstimator):
    """Computes the Morgan Fingerprint for a series of molecules"""

    def __init__(self, n_bits=2048, radius=2, use_chirality=False, use_bond_types=True,
                 use_features=True, n_jobs=1):
        """
        Args:
             n_bits (int): Size of the representation
             radius (int): Number of steps in graph network to consider
             use_chirality (bool): Whether to consider chirality in the fingerprint
             use_bond_types (bool): Whether to use the bond order as a feature
             use_features (bool): Whether to use atomic features to describe atoms (e.g., if atom is a donor)
             n_jobs (int): Number of processors to use
        """

        super().__init__()
        self.n_bits = n_bits
        self.radius = radius
        self.use_chirality = use_chirality
        self.use_bond_types = use_bond_types
        self.use_features = use_features
        self.n_jobs = n_jobs

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        """Compute feature vectors for a list of RDKit molecules

        Args:
            X: List of RDKit molecule objects
        """

        # Define featurization function
        def func(x):
            return AllChem.GetMorganFingerprintAsBitVect(x, self.radius,
                                                         useFeatures=self.use_features,
                                                         nBits=self.n_bits,
                                                         useBondTypes=self.use_bond_types,
                                                         useChirality=self.use_chirality)

        if self.n_jobs == 1:
            return np.vstack(list(map(func, X)))
        else:
            n_procs = os.cpu_count() if self.n_jobs == -1 or self.n_jobs is None else self.n_jobs
            with ThreadPoolExecutor(n_procs) as t:
                # Functions are not picklable, but in C++. Still GIL problems with this
                return np.vstack(list(t.map(func, X, chunksize=len(X) // n_procs // 4)))
