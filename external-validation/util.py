"""Utility file used for compute pairwise distances"""

from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem import AllChem, DataStructs
import numpy as np

def compute_dist_from_training_entry_mpnn(mol_a, mol_b): 
    """Compute the distance between a molecule and a molecule from the training set
    
    Args:
        mol_a: Atomic features of a molecule
        mol_b: Atomic features of a molecule from the training set
    Returns:
        (float) Average distance of each atom to the closest atom training entry
        (float)
    """
    
    # Compute pairwise distances (using numpy broadcasting)
    diffs = mol_a[:, None, :] - mol_b[None, :, :] # num_a, num_b, n_feat
    dists = np.linalg.norm(diffs, 2, axis=-1)  # num_a, num_b
    
    # Compute the minimum distance for each atom from the closest atom in the training
    min_dists = dists.min(axis=1)
    return min_dists.mean(), min_dists.max()


def compute_sim_to_training_entry_tanimoto(smiles, train_smiles, n_nearest): 
    """Compute similarities between a molecule and all molecules 
    in the training set.
    
    Uses Morgan Fingerprints with a radius of 4
    
    Args:
        mol_a str: SMILES string of a molecule
        train_mols ([str]): SMILES string of molecules from the training set
        n ([int]): Number of nearest molecules to measure
    Returns:
        (float) Average distance of each atom to the closest atom training entry
    """
    
    # Parse SMILES
    mol_a = Chem.MolFromSmiles(smiles)
    train_mols = [Chem.MolFromSmiles(s) for s in train_smiles]
    
    # Compute fingerprints
    fp_a = AllChem.GetMorganFingerprint(mol_a, 4)
    train_fps = [AllChem.GetMorganFingerprint(m, 4) for m in train_mols]
    
    # Compute the similiarities
    sims = [DataStructs.TanimotoSimilarity(fp_a, fp) for fp in train_fps]
    
    # Average the top ones
    k = np.max(n_nearest)
    sims = np.partition(sims, -k)[-k:]  # Partial sort
    sims = np.sort(sims)
    output = [np.mean(sims[-k:]) for k in n_nearest]
    return output


def has_stereoisomers(smiles):
    """Compute the number of stereoisomers for a molecule"""
    
    m = Chem.MolFromSmiles(smiles)
    opts = StereoEnumerationOptions(tryEmbedding=True, unique=True)
    isomers = iter(EnumerateStereoisomers(m, options=opts))
    i = 0
    for _ in zip(isomers, range(2)):
        i += 1
    return i > 1

if __name__ == "__main__":
    mol_a = np.array([[1, 1]])
    mol_b = np.array([[1, 1], [0, 0]])
    assert compute_dist_from_training_entry(mol_a, mol_b)[0] == 0.0
    assert compute_dist_from_training_entry(mol_b, mol_a)[0] == np.sqrt(2) / 2
    
    assert has_stereoisomers('BrC=CC1OC(C2)(F)C2(Cl)C1')
    assert not has_stereoisomers('C')
