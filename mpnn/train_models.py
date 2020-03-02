"""Script for training ML models for MEGNet"""

from jcesr_ml.utils import monitor_usage
from jcesr_ml.mpnn import MPNNTrainer
from threading import Thread
from keras import backend as K
import horovod.keras as hvd
from mpi4py import MPI
import tensorflow as tf
import argparse
import logging
import socket
import os

# Get the environmental variables
n_threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count()))
jobid = os.environ.get('COBALT_JOBID', socket.gethostname())

# Get a MPI Comm for the world
comm = MPI.COMM_WORLD

# Initialize Horovod
hvd.init()
size = hvd.size()
rank = hvd.rank()

# Set the Tensorflow settings
config = tf.ConfigProto()
config.allow_soft_placement = True
config.intra_op_parallelism_threads = n_threads
config.inter_op_parallelism_threads = 2
config.gpu_options.visible_device_list = str(hvd.local_rank())
sess = tf.Session(config=config)
K.set_session(sess)


# Hard-coded options
validation_frac = 0.1

if __name__ == "__main__":
    # Make the parser
    parser = argparse.ArgumentParser()

    # Add the arguments to the parser
    parser.add_argument('--batch-size', '-t', default=8192, help='Batch size', type=int)
    parser.add_argument('--ckpt-interval', '-c', default=-1, help='Number of steps between validation/checkpointing', type=int)
    parser.add_argument('--valid-batch', default=2048, help='Batch size during validation', type=int)
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
    logger.info('Training {} on a training set size of {} on {} ranks. Host: {}. Local rank: {}'.format(name, train_size, size,
                                                                                        socket.gethostname(), hvd.local_rank()))

    # Start the training
    if rank < 2:
        usage_path = os.path.join(work_dir, 'usage-rank{}.json'.format(rank))
        monitor_proc = Thread(target=monitor_usage, args=(usage_path, 15), daemon=True)
        monitor_proc.start()
    trainer = MPNNTrainer(comm, hvd, logger, net_dir, work_dir,
                          train_size, batch_size,
                          lr_start, lr_patience, lr_decay, lr_min, lr_warmup,
                          ckpt_interval, max_epochs,
                          n_threads, valid_batch_size=args.valid_batch)
    trainer.train_on_benchmark()
