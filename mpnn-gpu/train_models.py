"""Script for training ML models for MEGNet"""
import pickle as pkl
import hashlib
import json
import argparse
import logging
import os

from keras.models import load_model
from keras import callbacks as cb
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from nfp.preprocessing import GraphSequence
import pandas as pd
import numpy as np


from jcesr_ml.benchmark import load_benchmark_data
from jcesr_ml.mpnn import set_custom_objects
from jcesr_ml.keras import LRLogger, EpochTimeLogger

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

# Hard-coded options
validation_frac = 0.1
set_custom_objects()

if __name__ == "__main__":
    # Make the parser
    parser = argparse.ArgumentParser()

    # Add the arguments to the parser
    parser.add_argument('--batch-size', '-t', default=64, help='Batch size', type=int)
    parser.add_argument('--num-epochs', '-n', help='Number of epochs to run during training', type=int, default=128)
    parser.add_argument('--seed', '-S', help='Random state', type=int, default=1)
    parser.add_argument('--lr-start', help='Starting learning rate', type=float, default=1e-3)
    parser.add_argument('--lr-final', help='Final learning rate', type=float, default=1e-6)
    parser.add_argument('name', help='Name of the model to be trained', type=str)
    parser.add_argument('train_size', help='Training set size', type=int)

    # Parse the arguments
    args = parser.parse_args()
    run_params = args.__dict__

    # Determine the working directories
    net_dir = os.path.join('networks', args.name)
    run_hash = hashlib.sha256(json.dumps(run_params).encode()).hexdigest()[:6]
    run_name = f'B{args.batch_size}-N{args.num_epochs}-L{args.lr_start: .3e}-{run_hash}'
    work_dir = os.path.join(net_dir, run_name)

    # Make directories
    os.makedirs(work_dir, exist_ok=True)

    # Configure the logger
    logger = logging.getLogger('main-thread')
    logger.info(f'Training {args.name} on a training set size of {args.train_size}')
    logger.info(f'Saving results to {work_dir}')

    # Save the configuration
    with open(os.path.join(work_dir, 'config.json'), 'w') as fp:
        json.dump(run_params, fp)

    # Load in the training details
    with open(os.path.join(net_dir, 'options.json')) as fp:
        options = json.load(fp)

    # Load in the model
    model = load_model(os.path.join(net_dir, 'architecture.h5'))
    logger.info(f'Created ML model')

    # Load in the training set
    train_data, test_data = load_benchmark_data()
    train_data = train_data.sample(args.train_size, random_state=args.seed)
    logger.info(f'Loaded {len(train_data)} training and {len(test_data)} test entries')

    # Preprocess the input data
    with open(os.path.join(net_dir, 'converter.pkl'), 'rb') as fp:
        conv = pkl.load(fp)
    train_data['inputs'] = train_data['smiles_0'].apply(conv.construct_feature_matrices)
    test_data['inputs'] = test_data['smiles_0'].apply(conv.construct_feature_matrices)
    logger.info('Computed input dictionaries for the train and test data')

    # Make the data loaders
    train_data, valid_data = train_test_split(train_data, test_size=0.1, random_state=args.seed)
    train_loader = GraphSequence(train_data['inputs'].values, train_data[options['output_props']].values,
                                 batch_size=args.batch_size)
    val_loader = GraphSequence(valid_data['inputs'].values, valid_data[options['output_props']].values, shuffle=False,
                               batch_size=args.batch_size)
    test_loader = GraphSequence(test_data['inputs'].values, test_data[options['output_props']].values, shuffle=False,
                                batch_size=args.batch_size)
    logger.info('Made train, validation amd test loaders')

    # Train the model
    final_learn_rate = args.lr_final
    init_learn_rate = args.lr_start
    decay_rate = (final_learn_rate / init_learn_rate) ** (1. / (args.num_epochs - 1))

    def lr_schedule(epoch, lr):
        return lr * decay_rate

    model.compile(Adam(init_learn_rate), 'mean_squared_error', metrics=['mean_absolute_error'])
    history = model.fit(
        train_loader, validation_data=val_loader, epochs=args.num_epochs, verbose=True,
        shuffle=False, callbacks=[
            LRLogger(),
            EpochTimeLogger(),
            cb.LearningRateScheduler(lr_schedule),
            cb.ModelCheckpoint(os.path.join(work_dir, 'best_model.h5'), save_best_only=True),
            cb.EarlyStopping(patience=args.num_epochs, restore_best_weights=True),
            cb.CSVLogger(os.path.join(work_dir, 'train_log.csv')),
            cb.TerminateOnNaN()
        ]
    )

    # Compute the performance on the test set
    y_true = test_data[options['output_props']]
    y_pred = model.predict(test_loader)

    #  Save it as a CSV file
    y_true.reset_index(inplace=True, drop=True)
    pred_cols = [f'{x}_pred' for x in options['output_props']]
    y_pred = pd.DataFrame(y_pred, columns=pred_cols)
    test_results = pd.concat([y_true, y_pred], axis=1)
    test_results['smiles'] = test_data['smiles_0'].values
    test_results.to_csv(os.path.join(work_dir, 'test_results.csv'), index=False)

