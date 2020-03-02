"""Data needed for benchmark calculations"""
from ase.data import chemical_symbols
import numpy as np
import psutil
import json
import time

"""Total energy of isolated atoms (Ha)"""
atomic_ref_energies = {
    'g4mp2': {'C': -37.794203, 'H':  -0.502094, 'O':  -75.002483,
              'N':  -54.532825, 'Li': -7.434837, 'F':  -99.659686},
    'b3lyp': {'C': -37.846772, 'H': -0.500273, 'O':  -75.064579, 
              'N': -54.583861, 'Li': -7.491464, 'F':  -99.718730},
    'wB97Xd': {'C': -37.840258, 'H': -0.502668, 'O': -75.064433,
              'N': -54.583036, 'F': -99.732458}
}


def get_atomref_vector(method, max_z=10):
    """Generate a vector of atomrefs for use in schnetpack

    The atomic reference array where x[Z] is the reference for
    element with atomic number of Z. Note that this means that 
    that H is not in element zero.

    Args:
        method (str): name of the method
        max_z (int): total number of elements to include
    """
    output = np.zeros((max_z + 1,))
    for i in range(1, max_z+1):
        output[i] = atomic_ref_energies[method].get(chemical_symbols[i], 0)
    return output[:, None]


def compute_atomization_energy(atoms, u0, method):
    """Compute the atomization energy of a molecule
    
    Subtracts the atomic reference energies off of the molecular energy
    
    Args:
        atoms (ase.Atoms): Molecule to correct 
        u0 (float): Total energy of molecule at T=0K (Ha)
        method (string): Method used to compute energy
    Returns:
        (float): Atomization energy
    """
    
    # Subtract off the atomic contributions
    output = u0
    for a in atoms.get_chemical_symbols():
        output = output - atomic_ref_energies[method][a]
    return output


def monitor_usage(path, wait=5):
    """Print utilization information to disk in an infinite loop

    Args:
        wait (int): How long to wait between prints in seconds
        path (str): Path of the output file
    """

    psutil.cpu_percent()  # Starts the monitoring

    with open(path, 'w') as fp:
        while True:
            cpu_usage = psutil.cpu_percent(percpu=True)
            memory = psutil.virtual_memory()
            print(json.dumps({'cpu_usage': cpu_usage, 'total_mem': memory.total,
                'avail_mem': memory.available, 'time': time.ctime()}), file=fp)
            fp.flush()
            time.sleep(wait)
