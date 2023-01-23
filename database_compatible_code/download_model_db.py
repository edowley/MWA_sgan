###############################################################################
#
# Downloads and unzips the files for a particular SGAN model.
# 
#    1. Takes as arguments the path of the data directory and the name of the
#       SGAN model (AlgorithmSetting) to download.
#         - The data directory is the parent of the 'saved_models/' directory
#    2. Downloads the config_file of the AlgorithmSetting (a .tar.gz), extracts
#       its contents to a directory of the same name, and deletes the .tar.gz.
#         - Failed downloads or extractions will prompt a warning message
#
###############################################################################

import argparse, os, sys
from glob import glob
import requests
from urllib.parse import urljoin
from urllib.request import urlretrieve

# Constants
SMART_BASE_URL = os.environ.get('SMART_BASE_URL', 'http://localhost:8000/')
SMART_TOKEN = os.environ.get('SMART_TOKEN', 'fagkjfasbnlvasfdfwjf783YDF')

# Parse arguments
parser = argparse.ArgumentParser(description='Download pfd files and extract as numpy array files.')
parser.add_argument('-d', '--data_directory', help='Absolute path of the data directory (contains the candidates/ subdirectory)', default='/data/SGAN_Test_Data/')
parser.add_argument('-m', '--model_name', help='Name of the AlgorithmSetting (containing the model) to download', default="")
parser.add_argument('-l', '--base_url', help='Base URL for the database', default=SMART_BASE_URL)
parser.add_argument('-t', '--token', help='Authorization token for the database', default=SMART_TOKEN)

args = parser.parse_args()
path_to_data = args.data_directory
model_name = args.model_name
base_url = args.base_url
token = args.token

# Ensure that the data path ends with a slash
if path_to_data[-1] != '/':
    path_to_data += '/'

# Make the target directory, if it doesn't already exist
os.makedirs(path_to_data + 'saved_models/', exist_ok=True)