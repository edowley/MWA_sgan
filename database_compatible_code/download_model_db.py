###############################################################################
#
# Downloads and unzips the files for a particular SGAN model.
# 
#    1. Takes as arguments the path of the data directory and the name of the
#       SGAN model (AlgorithmSetting) to download.
#         - The data directory is the parent of the 'saved_models/' directory
#    2. Downloads the config_file of the AlgorithmSetting (a .tar.gz), extracts
#       its contents to a directory of the same name, and deletes the .tar.gz.
#         - Failed downloads will prompt a warning message
#
###############################################################################

import argparse, os, sys
import numpy as np
import requests
import tarfile
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
path_to_models = path_to_data + 'saved_models/'
os.makedirs(path_to_models, exist_ok=True)

# Database token
class TokenAuth(requests.auth.AuthBase):
    def __init__(self, token):
        self.token = token

    def __call__(self, r):
        r.headers['Authorization'] = "Token {}".format(self.token)
        return r

# Start authorised session
my_session = requests.session()
my_session.auth = TokenAuth(token)


########## Function Definitions ##########

# Queries a url and returns the requested column of the result as a numpy array
def get_column(url=urljoin(base_url, 'api/candidates/'), param=None, field='id'):
    try:
        table = my_session.get(url, params=param)
        table.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
    try:
        entries = [row[field] for row in table.json()]
    except KeyError as err:
        print(err)
        print(f"This table has no '{field}' column")
    return np.array(entries)

# Checks if an SGAN model exists (in the AlgorithmSetting table)
def check_model_existence(name):
    if name in sgan_model_names:
        return True
    elif name == "":
        return False
    else:
        print(f"The name {name} doesn't match an existing SGAN model")
        return False


########## Download Model ##########

# Get the list of all SGAN model names in the AlgorithmSetting table
sgan_model_names = get_column(urljoin(base_url, 'api/algorithm_settings/?algorithm_parameter=SGAN_files'), field='value')

# Ensure that the requested SGAN model exists
exists = check_model_existence(model_name)
while not exists:
    model_name = input("Enter the name of the SGAN model to download: ")
    exists = check_model_existence(model_name)

# If the model files have already been downloaded/extracted, exit
if os.path.isdir(path_to_models + model_name):
    print(f"The files for model {model_name} are already present")
    sys.exit()

# Download the model .tar.gz file
sgan_targz = get_column(urljoin(base_url, f'api/algorithm_settings/?value={model_name}'), field='config_file')[0]
full_url = urljoin(base_url, sgan_targz)
temp_path = path_to_models + os.path.basename(sgan_targz)
try:
    urlretrieve(full_url, temp_path)
except Exception as e:
    print(f"Download failed: {full_url}, {e}")

# Extract the contents of the .tar.gz file to a subfolder of saved_models/
tar = tarfile.open(temp_path)
tar.extractall(path_to_models + model_name)
tar.close()

# Delete the .tar.gz file
os.unlink(temp_path)

my_session.close()

print("All done!")