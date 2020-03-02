"""Utilities related to the use of MPI4Py, etc"""
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np


def scatter_chunks(data, comm, max_chunksize=65536, root=0):
    """Perform an MPI scatter with a large list

    Really larger lists can `cause overflow issues <https://bitbucket.org/mpi4py/mpi4py/issues/57>`_

    Args:
        data (list): List of data to be scatter (ignored for non-root ranks)
        comm (mpi comm): MPI communication tool
        max_chunksize (int): Maximum number of entries to send per chunk
        root (int): Rank of process holding data to be scattered
    Returns:
        (list): List of data owned by this process
    """

    # Get my rank and pool size
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Make a fake list if not root
    if rank != root:
        data = []

    # Determine the number chunks per rank
    eff_chunksize = max(1, max_chunksize // size)  # Messages are chunksize * size
    chunks_per_rank = max(1, len(data) // size // eff_chunksize + 1)
    chunks_per_rank = comm.bcast(chunks_per_rank, root=root)

    # Make the chunks
    chunks = np.array_split(data, size * chunks_per_rank)

    # Scatter them to the MPI comm
    my_data = []
    for i in range(chunks_per_rank):
        my_data.append(comm.scatter(chunks[i * size:(i + 1) * size], root=root))
    return np.hstack(my_data)


def allgather_feature_calculation(mols, conv_func, comm, max_chunksize, n_threads, logger=None):
    """Compute features of molecules with an allgather approach

    Each rank has the same list of molecules and will collectively compute the features
    for each molecule.

    Uses multi-threaded MPI to have allgather occur asynchronously

    Args:
        mols (list): List of molecules to be processed
        conv_func: Function that computes features given list of molecules
        comm (mpi4py.mpi.Intercomm): MPI communicator
        max_chunksize (int): Maximum number of features to compute in a single batch.
            Used to keep message sizes smaller than MPI limit
        n_threads (int): Number of processes to use for computing features / allgather
        logger: Optional logger
    Returns:
        (list) Features for all molecules, in same order as provided
    """
    # Identifying my thread
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Determine number of batches so that max_chunksize is not exceeded
    n_batches = max(1, len(mols) // max_chunksize + 1)
    n_chunks = size * n_batches
    if logger:
        logger.info(f'Preparing to compute graphs of {len(mols)} molecules in {n_batches} batches')

    # Generate chunks
    chunks = np.array_split(mols, n_chunks)

    # Loop over batches
    all_graphs = []
    with ThreadPoolExecutor(n_threads) as t:
        with ProcessPoolExecutor(n_threads) as p:
            for batch in range(n_batches):
                my_mols = chunks[size * batch + rank]

                # Figure out which copy of the data this rank owns
                if logger:
                    logger.info(f'Processing batch {batch + 1} of {n_batches}.'
                                f' This rank owns {len(my_mols)}')

                # Convert them
                chunk_size = max(1, len(my_mols) // n_threads // 2)
                if logger:
                    logger.info(f'Computing graphs locally with {n_threads}'
                                f' processes in chunks of {chunk_size}')
                my_graphs = list(p.map(conv_func, my_mols, chunksize=chunk_size))

                # Everyone send everyone else the data
                if size > 1:
                    all_graphs.append(t.submit(comm.allgather, my_graphs))
                    logger.info(f'Submitted batch {batch} to be gathered')
                else:
                    all_graphs.append(my_graphs)

        # If more than one rank, we have to wait for the gathers to complete
        if size > 1:
            if logger:
                logger.info('Waiting to gather from all ranks')
            collected_graphs = []
            for r in all_graphs:
                collected_graphs.extend(r.result())
            all_graphs = collected_graphs
    return np.hstack(all_graphs)