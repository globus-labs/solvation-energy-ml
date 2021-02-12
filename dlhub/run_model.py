from argparse import ArgumentParser
import json

from dlhub_sdk import DLHubClient

# Make the argument parser
parser = ArgumentParser()
parser.add_argument("--servable", default="loganw_globusid/mpnn-solvation-energy",
                    help="Name of the servable to run")
parser.add_argument("smiles", nargs="+", help="List of SMILES strings to evaluate")
args = parser.parse_args()

# Make the DLHub client
client = DLHubClient()
result = client.run(args.servable, args.smiles)
print(json.dumps(result))
