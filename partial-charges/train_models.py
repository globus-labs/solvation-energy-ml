"""Script for training ML models for MEGNet"""

from jcesr_ml.utils import monitor_usage
from jcesr_ml.mpnn import MPNNTrainer, AtomicPropertySequence, dist_compute_graphs
from datetime import datetime
from threading import Thread
from keras import backend as K
import horovod.keras as hvd
from mpi4py import MPI
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import logging
import socket
import os

# Initialize Horovod
hvd.init()
size = hvd.size()
rank = hvd.rank()

# Get a MPI Comm for the world
comm = MPI.COMM_WORLD

# Get the environmental variables
n_threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count()))
jobid = os.environ.get('COBALT_JOBID', socket.gethostname())

# Set the Tensorflow settings
config = tf.ConfigProto()
config.allow_soft_placement = True
config.intra_op_parallelism_threads = n_threads
config.inter_op_parallelism_threads = 2
config.gpu_options.visible_device_list = str(hvd.local_rank())
sess = tf.Session(config=config)
K.set_session(sess)


# Make a class for training the charges
class ChargesMPNNTrainer(MPNNTrainer):
    
    def load_benchmark_data(self):
        return pd.read_pickle(self.benchmark_dataset)
    
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
        return AtomicPropertySequence(train_graphs, train_target, batch_size,
                                      final_batch=False, shuffle_offset=offset)

    def make_validation_loader(self, valid_mols: list, valid_target: np.ndarray,
                               batch_size: int):
        valid_graphs = list(map(self.converter.construct_feature_matrices, valid_mols))
        return AtomicPropertySequence(valid_graphs, valid_target, batch_size,
                                      final_batch=True, shuffle=False)



# Hard-coded options
validation_frac = 0.1

if __name__ == "__main__":
    # Make the parser
    parser = argparse.ArgumentParser()

    # Add the arguments to the parser
    parser.add_argument('--batch-size', '-t', default=8192, help='Batch size', type=int)
    parser.add_argument('--ckpt-interval', '-c', default=-1, help='Number of steps between validation/checkpointing', type=int)
    parser.add_argument('-l', '--learning-rate', default=1e-5,
                        help='Learning rate start (uncorrected by batch size)', type=float)
    parser.add_argument('name', help='Name of the model to be trained', type=str)
    parser.add_argument('train_size', help='Training set size', type=int)

    # Parse the arguments
    args = parser.parse_args()


    # Get the training set and batch size, and adjust LR accordingly
    train_size = args.train_size
    batch_size = args.batch_size  # Batch size is adjusted in train_models function
    name = args.name
    lr_warmup = args.learning_rate
    lr_start = args.learning_rate * args.batch_size / 128  # Linearly increase learning with with batch size
    lr_decay = 0.5
    lr_min = lr_start / 100
    lr_patience = max(25, 25 * 100000 // train_size * batch_size // 1024)
    ckpt_interval = max(1, 100000 // train_size) if args.ckpt_interval == -1 else args.ckpt_interval
    max_epochs = 8000 * 100000 // train_size

    # Determine the working directories
    net_dir = os.path.join('networks', name)
    work_dir = os.path.join('networks',
                            name,
                            '{}-entries'.format(train_size),
                            '{}-nodes'.format(size),
                            '{}-batch_size'.format(args.batch_size),
                            '{:.2e}-learning_rate'.format(args.learning_rate))

    # Make directories, if they do not exist
    if rank == 0:
        if not os.path.isdir(work_dir):
            os.makedirs(work_dir)
    comm.barrier()

    # Configure the logger
    for h in logging.root.handlers:
        logging.root.removeHandler(h)
    if rank <= 1:
        logging.basicConfig(filemode='a', filename=os.path.join(work_dir, 'train.log'),
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            level=logging.INFO)
    else:
        logging.basicConfig(filename=os.devnull)
    logger = logging.getLogger('{}.{}'.format(jobid, rank))
    logger.info('Training {} on a training set size of {} on {} ranks. Host: {}. Rank: {}'.format(name, train_size, size,
                                                                                                  socket.gethostname(), hvd.local_rank()))

    # Start the training
    if rank < 2:
        usage_path = os.path.join(work_dir, 'usage-rank{}.json'.format(rank))
        monitor_proc = Thread(target=monitor_usage, args=(usage_path, 15), daemon=True)
        monitor_proc.start()
    trainer = ChargesMPNNTrainer(comm, hvd, logger, net_dir, work_dir,
                                 train_size, batch_size,
                                 lr_start, lr_patience, lr_decay, lr_min, lr_warmup,
                                 ckpt_interval, max_epochs,
                                 n_threads, valid_batch_size=2048,
                                 benchmark_dataset='mapped_charges_dataset.pkl.gz')
    trainer.train_on_benchmark()
