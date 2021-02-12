"""Command-line script for publishing SchNet models to DLHub"""

from dlhub_sdk.models.servables.python import PythonStaticMethodModel
from dlhub_sdk.utils.types import compose_argument_block
from dlhub_sdk.client import DLHubClient
from dlhub_app import evaluate_molecules
from argparse import ArgumentParser
from time import sleep, perf_counter
import numpy as np
import logging
import yaml

# Make a logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Define test input data
mols = ['C', 'CC', 'CCC']
authors = ["Ward, Logan", "Dandu, Naveen", "Blaiszik, Ben",
           "Narayanan, Badri", "Assary, Rajeev S.", "Redfern, Paul C.",
           "Foster, Ian", "Curtiss, Larry A."]
affil = ["Argonne National Laboratory"] * 3 + ["University of Louisville"] + ["Argonne National Laboratory"] * 4

# Parse the user input
parser = ArgumentParser(description='''Post a MPNN-based solvation energy model to DLHub.''')
parser.add_argument('--test', help='Just test out the submission and print the metadata', action='store_true')
args = parser.parse_args()

# Write out the generic components of the model
model = PythonStaticMethodModel.from_function_pointer(evaluate_molecules)

#   Descriptions of the model interface
model.set_outputs(
    'dict', 'Solvation energy predictions',
    properties={
        'smiles': compose_argument_block('list', 'List of molecules run with the model', item_type='string'),
        'solvation-energies': compose_argument_block(
            'ndarray', 'Predicted solvation energy for each molecule in each solvent', shape=[None, None]
        ),
        'dielectric-constants': compose_argument_block('list', 'Dielectric constants for solvents', item_type='float'),
        'training-set-distance': compose_argument_block('list', 'Distance to nearest molecules in training set.'
                                                                ' Normalized based on the distances in the test set',
                                                        item_type='float'),
        'expected-error': compose_argument_block('list', 'Estimated uncertainty in the prediction based on distance'
                                                         ' from training set',
                                                 item_type='float'),
        'likelihood-error-above-1kcal/mol': compose_argument_block('list',
                                                                   'Probability of the error being above 1kcal/mol, '
                                                                   'based on based on distance from training set',
                                                 item_type='float'),
        'warnings': compose_argument_block('list', 'Warnings about predictions', item_type='string')
    })
model.set_inputs(
    'list', 'List of SMILES strings', item_type='string'
)
#model['servable']['methods']['run']['parameters'] = {
#    'dielectric_constants': None #compose_argument_block('list', 'Dielectric constants of solvent', item_type='float')
#}

#   Provenance information for the model
model.add_related_identifier("10.1021/acs.jctc.8b00908", "DOI", "Requires")

#   DLHub search tools
logging.info('Initialized model')

model.add_file('dlhub_app.py')
for m in ['train_dist_mol_64_mag.pkl', 'train_dist_mol_64_prob.pkl', 'molecule-nn-scaler.pkl',
          'molecule-nn-computer.pkl.gz', 'train_dist_mol_64-dist-scaling.json', 'molecule-rep-model.h5']:
    model.add_file(f'error-models/{m}')
model.add_directory('model/', recursive=True)
model.add_directory('jcesr_ml', recursive=True, include='*.py')
model.parse_repo2docker_configuration()

# Add in the model-specific data

#   Model name
model.set_name(f'mpnn-solvation-energy')
model.set_title(f'MPNN Model to Predict Solvation Energy of Molecules')
logging.info(f'Defined model name: {model.name}')

#   Model provenance information
model.set_authors(authors, affil)

#  model.add_related_identifier(paper-doi, 'DOI', 'IsDescribedBy')  # Need DOI
model.add_related_identifier("https://github.com/globus-labs/solvation-energy-ml", 'URL', 'IsDescribedBy')
logging.info('Added model-specific metadata')

# If desired, print out the metadata
if args.test:
    logging.info(f'Metadata:\n{yaml.dump(model.to_dict(), indent=2)}')

    logging.info('Running function')
    local_result = evaluate_molecules(mols)
    logging.info(f'Success! Output: {local_result}')
    exit()

# Publish servable to DLHub

#   Make the client and submit task
client = DLHubClient(http_timeout=-1)
task_id = client.publish_servable(model)
logging.info(f'Started publication. Task ID: {task_id}')
