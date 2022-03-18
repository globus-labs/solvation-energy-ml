"""Utility file used for compute pairwise distances"""

import numpy as np

def compute_dist_from_training_entry(mol_a, mol_b): 
    """Compute the distance between a molecule and a molecule from the training set
    
    Args:
        mol_a: Atomic features of a molecule
        mol_b: Atomic features of a molecule from the training set
    Returns:
        (float) Average distance of each molecule to the closest training entry
    """
    
    # Compute pairwise distances (using numpy broadcasting)
    diffs = mol_a[:, None, :] - mol_b[None, :, :] # num_a, num_b, n_feat
    dists = np.linalg.norm(diffs, 2, axis=-1)  # num_a, num_b
    
    # Compute the minimum distance for each atom from the closest atom in the training
    min_dists = dists.min(axis=1)
    return min_dists.mean()

if __name__ == "__main__":
    mol_a = np.array([[1, 1]])
    mol_b = np.array([[1, 1], [0, 0]])
    assert compute_dist_from_training_entry(mol_a, mol_b) == 0.0
    assert compute_dist_from_training_entry(mol_b, mol_a) == np.sqrt(2) / 2
