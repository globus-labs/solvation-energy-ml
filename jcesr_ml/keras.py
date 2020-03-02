"""Utilities for training models with Keras"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from time import perf_counter
from datetime import datetime
from pandas.errors import EmptyDataError
from functools import partial
from jcesr_ml.benchmark import default_dataset
from keras import backend as K
from keras.layers import Layer
from keras.utils import Sequence, get_custom_objects
from keras.models import load_model
from keras.losses import mean_squared_error
from keras.callbacks import Callback, CSVLogger, ModelCheckpoint, ReduceLROnPlateau, \
    TerminateOnNaN, LearningRateScheduler
from sklearn.model_selection import train_test_split


from jcesr_ml.benchmark import load_benchmark_data


def set_custom_layers():
    custom_objs = get_custom_objects()
    custom_objs.update({'DenormalizeLayer': DenormalizeLayer,
                        'corrected_loss': mean_squared_error, 
                        'tf': tf})


def _make_corrected_loss(input_batch):
    func = partial(corrected_loss, input_batch=input_batch)
    func.__name__ = 'corrected_loss'
    return func


def corrected_loss(y_pred, y_true, input_batch=None):
    from horovod import tensorflow as hvd_tf
    my_batch_size = K.shape(input_batch[0])[0]
    global_size = hvd_tf.allreduce(my_batch_size,
                                   average=True)
    my_frac = K.cast(my_batch_size / global_size, K.floatx())
    return my_frac * mean_squared_error(y_true, y_pred)


def cartesian_product(inputs):
    """Computes a Cartesian product of two 2D arrays

    Args:
        inputs ([Tensor]): Two 2D tensors with dimensions [a, b] and [c, d]
    Returns:
        (Tensor) Tensor of shape [a, c, b + d]
    """

    a, b = inputs

    # Expand both arrays to (M, N, x) arrays
    a_tiled = K.tile(K.expand_dims(a, 1), [1, K.shape(b)[0], 1])
    b_tiled = K.tile(K.expand_dims(b, 0), [K.shape(a)[0], 1, 1])

    return K.concatenate([a_tiled, b_tiled], axis=2)


class StopOnLRMin(Callback):
    """Stop model training when the learning rate has decayed to a certain value"""

    def __init__(self, min_lr=1e-6, min_epochs=100):
        """
        Args:
             min_lr (float): Minimum learning rate
             min_epochs (int): Minimum number of epochs before stopping can occur
        """
        super().__init__()
        self.min_lr = min_lr
        self.min_epochs = min_epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.min_epochs:
            lr = float(K.get_value(self.model.optimizer.lr))
            if lr <= self.min_lr:
                self.model.stop_training = True


class LRLogger(Callback):
    """Add the LR to the logs

    Must be before any log writers in the callback list"""

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs['lr'] = float(K.get_value(self.model.optimizer.lr))


class EpochTimeLogger(Callback):
    """Adds the epoch time to the logs

    Must be before any log writers in the callback list"""

    def __init__(self):
        super().__init__()
        self.time = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.time = perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs['epoch_time'] = perf_counter() - self.time


class DistributedValidation(Callback):
    """Average the results of validation from all MPI ranks

    **NOTE**: Must be before the logging step in the callback list
    """

    def __init__(self, comm, val_size):
        """
        Args:
             comm (mpi4py Intercomm): Communicator to all MPI ranks
             val_size (int): Number of molecules in this rank's validation set
        """
        super().__init__()

        # Get the validation sizes from all ranks
        self.comm = comm
        size = self.comm.Get_size()
        sendbuf = val_size * np.ones((size,), dtype=np.float)
        self.val_fracs = np.empty((size,), dtype=np.float)
        self.comm.Alltoall(sendbuf, self.val_fracs)
        self.val_fracs = np.divide(self.val_fracs, np.sum(self.val_fracs))

    def on_epoch_end(self, epoch, logs=None):
        size = self.comm.Get_size()
        if logs:
            for key in logs:
                if key.startswith('val_'):
                    # Everyone send each other the result
                    sendbuf = logs[key] * np.ones(size, dtype=np.float)
                    results = np.empty(size, dtype=np.float)
                    self.comm.Alltoall(sendbuf, results)

                    # Compute weighted average
                    logs[key] = np.dot(self.val_fracs, results)


class DenormalizeLayer(Layer):
    """Layer to scale the output layer to match the input data distribution"""

    def __init__(self, mean=None, std=None, **kwargs):
        """
        Args:
             mean (ndarray): Mean of output variable(s)
             std (ndarray): Standard deviation of output variable(s)
        """
        self.mean = mean
        self.std = std
        self.mean_tensor = K.constant(mean, dtype=K.floatx())
        self.std_tensor = K.constant(std, dtype=K.floatx())
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['mean'] = self.mean
        config['std'] = self.std
        return config

    def call(self, inputs, **kwargs):
        return inputs * self.std_tensor + self.mean_tensor


class KerasHorovodTrainer:
    """Utility for running distributed training, logged in a way that works with the
    directory structure for the benchmarking project"""

    def __init__(self, comm, hvd, logger, net_dir, work_dir, train_size, batch_size,
                 lr_start, lr_patience, lr_decay, lr_min, lr_warmup,
                 ckpt_interval, max_epochs, n_threads=None, valid_batch_size=1024,
                 mol_column='xyz', split_dir='splits', ckpt_history=None,
                 benchmark_dataset=default_dataset):
        """Initialize the trainer and create directories

        Args:
            comm: MPI communicator
            hvd: Initialized Horovod communicator
            logger: Logger used to monitor status of training
            net_dir (str): Directory containing the base network architecture
            work_dir (str): Directory where checkpoints and logs with be saved
            train_size (int): Size of the training set
            batch_size (int): Number of entries per global batch
            lr_start (float): Initial learning rate
            lr_patience (float): How many epochs without improvement until learning rate is decayed
            lr_decay (float): How much to decay learning rate on plateaus
            lr_min (float); Minimum allowed learning rate
            lr_warmup (float): Learning rate to use in the first epoch
            ckpt_interval (int): Number of epochs between checkpoints
            max_epochs (int): Maximum number of epochs during training
            n_threads (int): Number of threads to use for preprocessing, etc
            valid_batch_size (int): Batch size for validator
            mol_column (str): Column of dataframe that contains molecule information
            split_dir (str): Directory storing the splits used for this model
            ckpt_history (int): After how many epochs to save a model checkpoint
            benchmark_dataset (str): Path of the dataset dataset
        """

        # Get the number of threads, if unset
        if n_threads is None:
            n_threads = os.cpu_count()
        self.comm = comm
        self.hvd = hvd
        self.net_dir = net_dir
        self.work_dir = work_dir
        self.train_size = train_size
        self.global_batch_size = batch_size
        self.ckpt_interval = ckpt_interval
        self.max_epochs = max_epochs
        self.logger = logger
        self.n_threads = n_threads
        self.valid_batch_size = valid_batch_size
        self.mol_column = mol_column
        self.split_dir = split_dir
        self.ckpt_history = ckpt_history
        self.benchmark_dataset = benchmark_dataset

        # Store the learning rate information
        self.lr_start = lr_start
        self.lr_decay = lr_decay
        self.lr_min = lr_min
        self.lr_patience = lr_patience

        # Create the learning rate warmup function
        def lr_sched(epoch, lr, warmup_epochs=5):
            if epoch < warmup_epochs:
                return lr_warmup + epoch / warmup_epochs * (lr_start - lr_warmup)
            return lr
        self.lr_sched = lr_sched

        # Caches for the data loaders
        self._train_hash = None
        self._train_load = self._valid_load = None

    def set_custom_objects(self):
        """Add custom objects to Keras dictionary"""
        set_custom_layers()

    def make_optimizer(self, options):
        """Generate an Optimizer for this training.

        Args:
            options (dict): Any options for the model
        Returns:
            Optimizer
        """
        raise NotImplementedError()

    def make_train_loader(self, train_mols: list, train_target: np.ndarray,
                          batch_size: int) -> Sequence:
        """Make a data loader for the training set

        Args:
            train_mols ([str]): List of molecules ot use as input
            train_target ([float]): Property to be predicted
            batch_size (int): Number of molecules per batch
        Returns:
            (Sequence) Training data generator with proper batch size
        """
        raise NotImplementedError()

    def make_validation_loader(self, valid_mols: list, valid_target: np.ndarray,
                               batch_size: int) -> Sequence:
        """Make a data loader for the validation set

        Args:
            valid_mols ([str]): List of molecules ot use as input
            valid_target ([float]): Property to be predicted
            batch_size (int): Number of molecules per batch
        Returns:
            (Sequence) Training data generator with proper batch size
        """

        raise NotImplementedError()

    def add_callbacks(self, rank, callbacks, train_load):
        """Add any model-specific callbacks

        Args:
            rank (int): Rank of this worker
            callbacks (list): List of callbacks to be modified
            train_load (Sequence): Training data generator
        """
        raise NotImplementedError()

    def is_finished(self):
        return os.path.isfile(os.path.join(self.work_dir, 'finished'))
    
    def load_benchmark_data(self):
        full_train_data, _ = load_benchmark_data(self.benchmark_dataset)
        return full_train_data

    def train_on_benchmark(self):
        """Trains the model on the benchmark dataset"""
        
        # Load in the training data
        full_train_data = self.load_benchmark_data()

        # Get the MPI pool information
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()

        # Set the per-rank batch size
        batch_size = max(1, self.global_batch_size // size)
        if self.global_batch_size % size != 0:
            self.logger.warn('Global batch size not a multiple of number of replicas')

        # Get the total training set size
        self.logger.info('Loaded the training data: Size = {}'.format(len(full_train_data)))

        # Make sure model hasn't finished training already
        if self.is_finished():
            self.logger.info('Model has already finished training')
            return

        # Load in the model options
        with open(os.path.join(self.net_dir, 'options.json')) as fp:
            options = json.load(fp)

        # Get the training/validation set split
        indices = np.arange(len(full_train_data))
        self.comm.barrier()
        if not os.path.exists(self.split_dir):
            if rank == 0:
                os.makedirs(self.split_dir)
            self.comm.barrier()
        split_file = os.path.join(self.split_dir, '{}.npz'.format(self.train_size))
        if os.path.isfile(split_file):
            inds = np.load(split_file)  # Use the old splits
            train_inds = inds['train']
            valid_inds = inds['valid']
            del inds
        else:
            if rank == 0:
                # Perform the splitting
                all_train_inds = np.random.choice(indices, self.train_size, replace=False)
                train_inds, valid_inds = train_test_split(all_train_inds, test_size=0.1)
                np.savez(split_file, train=train_inds, valid=valid_inds)
            else:
                train_inds = valid_inds = None
            train_inds = self.comm.bcast(train_inds)
            valid_inds = self.comm.bcast(valid_inds)

        # Get the training molecules and convert them a graph representation
        train_mols = full_train_data.iloc[train_inds][self.mol_column].tolist()
        train_target = np.squeeze(full_train_data.iloc[train_inds][options['output_props']])
        self.logger.info(f'Loaded {len(train_mols)} training molecules')

        # Make the training data loader
        train_load = self.make_train_loader(train_mols, train_target, batch_size)
        self.logger.info(f'Made training set loader.'
                         f' Train Size={len(train_target)} Batch Size={batch_size}')

        # Make the validation loader
        valid_inds = np.array_split(valid_inds, size)[rank]  # Only those for this rank
        valid_mols = full_train_data.iloc[valid_inds][self.mol_column].tolist()
        valid_target = np.squeeze(full_train_data.iloc[valid_inds][options['output_props']])

        valid_load = self.make_validation_loader(valid_mols, valid_target,
                                                 self.valid_batch_size)

        # Make the number of steps per epoch
        #   (noting that this set by global and local batch size)
        steps_per_epoch = max(1, len(train_mols) // self.global_batch_size)

        # Loading the model from disk
        self.set_custom_objects()  # MEGNet-specific layers
        ckpt_path = os.path.join(self.work_dir, 'checkpoint.h5')
        if os.path.isfile(ckpt_path):
            # Loads the model, optimizer, and compiles it
            model = self.hvd.load_model(ckpt_path)
            if options.get('batch_correction', False):
                # Make the corrected loss function and re-compile
                loss = _make_corrected_loss(model.input)  # Fix the loss function
                optimizer = model.optimizer
                model.compile(optimizer, loss)
            self.logger.info('Loaded model from checkpoint {}'.format(ckpt_path))

            # Load the model configuration information
            with open(os.path.join(self.work_dir, 'config.json')) as fp:
                config = json.load(fp)

            # Check to make sure this checkpoint wasn't created with a different size
            if config['size'] != size:
                raise RuntimeError(f"Checkpoint made with different size: {size}!={config['size']}")
        else:
            # Get the base model and compile it
            model = load_model(os.path.join(self.net_dir, 'architecture.h5'))

            # If desired, add layers to "denormalize the data"
            if options.get('normalize', False):
                # Get the class of the model
                cls = model.__class__

                # Graph the input and output layers
                input_layers = model.input
                output_layers = model.output

                # Add deserialization layers
                output_layers = DenormalizeLayer(np.array(train_target.mean(axis=0)),
                                                 np.array(train_target.std(axis=0)))(output_layers)
                model = cls(input_layers, output_layers)

            # Make the optimizer
            opt = self.make_optimizer(options)
            opt = self.hvd.DistributedOptimizer(opt)

            # Add a batch-size corrector
            loss = 'mse'
            if options.get('batch_correction', False):
                loss = _make_corrected_loss(model.input)

            # Save the number of nodes this is for
            with open(os.path.join(self.work_dir, 'config.json'), 'w') as fp:
                json.dump({'size': size}, fp)
            model.compile(opt, loss)
            self.logger.info('Created model from template in {}'.format(self.net_dir))

        # Make the callbacks for all processes
        redlr_callback = ReduceLROnPlateau(factor=self.lr_decay, min_lr=self.lr_min,
                                           patience=self.lr_patience)

        # Update the learning policy, if desired
        lr_sched = self.lr_sched
        if options.get('lr_poly', False):
            def lr_sched(epoch, lr):
                return self.lr_start * (1 - epoch / self.max_epochs) ** 2

        callbacks = [
            StopOnLRMin(min_lr=self.lr_min),
            TerminateOnNaN(),
            self.hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            DistributedValidation(self.comm, len(valid_inds)),
            LearningRateScheduler(lr_sched),
            redlr_callback
        ]

        # Add any new callabcks specific to a particular model
        self.add_callbacks(rank, callbacks, train_load)

        # Defining the maximum checkpoint interval and epoch count
        max_epochs = self.max_epochs
        ckpt_interval = self.ckpt_interval

        # Make the callbacks for the root thread
        log_path = os.path.join(self.work_dir, 'log.csv')
        if rank == 0:
            callbacks.extend([
                ModelCheckpoint(ckpt_path, period=ckpt_interval),
                ModelCheckpoint(os.path.join(self.work_dir, 'best_model.h5'),
                                save_weights_only=False, save_best_only=True),
                LRLogger(),
                EpochTimeLogger(),
                CSVLogger(log_path, append=True),
            ])

            if self.ckpt_history is not None:
                callbacks.append(
                    ModelCheckpoint(os.path.join(self.work_dir,
                                                 'history-{epoch}.h5'),
                                    save_weights_only=False,
                                    period=self.ckpt_history)
                )

        # Get the current epoch number and number of steps since last improvement
        if os.path.isfile(log_path):
            epoch = None
            since_best = None
            if rank == 0:
                try:
                    log = pd.read_csv(log_path)
                    epoch = len(log)

                    # Determine number of steps since loss has decreased by >= delta
                    best_loss = log['val_loss'].min()
                    last_epoch = log.query(f'val_loss <= {best_loss} + '
                                           f'{redlr_callback.min_delta}')['epoch'].min()
                    since_best = epoch - last_epoch
                except EmptyDataError:
                    epoch = 0
                    since_best = 0
            epoch = self.comm.bcast(epoch)
            since_best = self.comm.bcast(since_best)
            redlr_callback.wait = since_best
            self.logger.info(f'Resuming from epoch {epoch}, {since_best} epochs since improvement.')
        else:
            epoch = 0
            self.logger.info('Start training from scratch')

        # Create the trainer
        self.logger.info(f'Beginning training. {steps_per_epoch} train steps per epoch.'
                         f' {len(valid_load)} validation steps')
        model.fit_generator(train_load, steps_per_epoch=steps_per_epoch, initial_epoch=epoch,
                            validation_data=valid_load, verbose=0, shuffle=False,
                            validation_steps=len(valid_load),
                            callbacks=callbacks, epochs=max_epochs, workers=0)

        # Mark training as complete
        with open(os.path.join(self.work_dir, 'finished'), 'w') as fp:
            print(str(datetime.now()), file=fp)
        self.logger.info('Training finished')
