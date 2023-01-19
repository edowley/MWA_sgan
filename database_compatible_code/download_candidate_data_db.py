###############################################################################
#
# Downloads and processes the Candidate data for a particular MlTrainingSetCollection.
# 
#    1. Takes as arguments the path of the data directory and the name of the
#       MlTrainingSetCollection to download.
#         - The data directory is the parent of the 'candidates/' directory
#    2. Downloads the PFD files of all Candidates in the collection, extracts
#       their contents to NumPy array files, and deletes the PFDs.
#         - Failed downloads or extractions will prompt a warning message
#         - Differences from the old version (pre database update):
#             - The file names (from the database) now begin with 'candidates/'
#             - No label files are generated
#
###############################################################################

import argparse, os, sys
import concurrent.futures as cf
from glob import glob
from math import floor
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
import requests
from time import time
from ubc_AI.training import pfddata
from urllib.request import urlretrieve

# Constants
DATABASE_URL = 'https://apps.datacentral.org.au/smart/media/'
SMART_BASE_URL = os.environ.get('SMART_BASE_URL', 'http://localhost:8000/api/')
SMART_TOKEN = os.environ.get('SMART_TOKEN', 'fagkjfasbnlvasfdfwjf783YDF')
NUM_CPUS = cpu_count()
N_FEATURES = 4

# Parse arguments
parser = argparse.ArgumentParser(description='Download pfd files and extract as numpy array files.')
parser.add_argument('-d', '--data_directory', help='Absolute path of the data directory (contains the candidates/ subdirectory)', default='/data/SGAN_Test_Data/')
parser.add_argument('-n', '--collection_name', help='Name of the MlTrainingSetCollection to download', default="")
parser.add_argument('-l', '--base_url', help='Base URL for the database', default=SMART_BASE_URL)
parser.add_argument('-t', '--token', help='Authorization token for the database', default=SMART_TOKEN)

args = parser.parse_args()
path_to_data = args.data_directory
collection_name = args.collection_name
base_url = args.base_url
token = args.token

# Ensure that the base url ends with a slash
if base_url[-1] != '/':
    base_url += '/'

# Ensure that the data path ends with a slash
if path_to_data[-1] != '/':
    path_to_data += '/'

# Make the target directory, if it doesn't already exist
os.makedirs(path_to_data, exist_ok=True)

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

# Queries the requested url and returns the result as a pandas dataframe
def get_dataframe(url=f'{base_url}candidates/', param=None):
    try:
        table = my_session.get(url, params=param)
        table.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
    return pd.read_json(table.json())

# Queries a url and returns the primary keys of the result as a numpy array
# (Only works if this is a column called 'id' or 'name')
def get_keys(url=f'{base_url}candidates/', param=None):
    try:
        table = my_session.get(url, params=param)
        table.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
    try:
        keys = [row['id'] for row in table.json()]
    except KeyError:
        try:
            keys = [row['name'] for row in table.json()]
        except KeyError as err:
            print(err)
            print("This table has no 'id' or 'name' column.")
    return np.array(keys)

# Checks if a particular MlTrainingSetCollection exists
def check_collection_existence(name):
    if name in set_collections:
        return True
    elif name == "":
        return False
    else:
        print(f"The name {name} doesn't match an existing MlTrainingSetCollection")
        return False

# Downloads pfd files from the database to the candidates/ directory
# Returns False and prints a message if the download fails, otherwise returns True
def download_pfd(pfd_name):
    if len(glob(path_to_data + pfd_name[:-4] + '*')) == 0:
        try:
            urlretrieve(DATABASE_URL + pfd_name, path_to_data + pfd_name)
            return True
        except Exception as e:
            print(f"Download failed: {pfd_name}, {e}")
            return False
    else:
        # If the target pfd file or associated numpy files already exist, skip download
        return True

# Executes the downloads in parallel (threads)
# Returns a mask for the successful downloads and prints the time taken
def parallel_download(download_list):
    start = time()
    successes = []
    with cf.ThreadPoolExecutor(NUM_CPUS) as executor:
        for result in executor.map(download_pfd, download_list):
            successes.append(result)
    total_time = time() - start
    print(f"Download time: {total_time}")
    return successes

# Extracts pfd files to numpy array files in the candidates/ directory and deletes the pfd files
# Returns False and prints a message if the extraction fails, otherwise returns True
def extract_from_pfd(pfd_name):
    if not len(glob(path_to_data + pfd_name[:-4] + '*')) == N_FEATURES:
        try:
            data_obj = pfddata(path_to_data + pfd_name)
            time_phase_data = data_obj.getdata(intervals=48)
            freq_phase_data = data_obj.getdata(subbands=48)
            dm_curve_data = data_obj.getdata(DMbins=60)
            profile_data = data_obj.getdata(phasebins=64)

            np.save(path_to_data + pfd_name[:-4] + '_time_phase.npy', time_phase_data)
            np.save(path_to_data + pfd_name[:-4] + '_freq_phase.npy', freq_phase_data)
            np.save(path_to_data + pfd_name[:-4] + '_dm_curve.npy', dm_curve_data)
            np.save(path_to_data + pfd_name[:-4] + '_pulse_profile.npy', profile_data)

            os.unlink(path_to_data + pfd_name)
            return True
        except ValueError: 
            print(f"Extraction failed: {pfd_name}")
             # If the extraction fails, delete the pfd anyway
            os.unlink(path_to_data + pfd_name)
            return False
    else:
        # If the numpy array files already exist, skip extraction
        return True

# Executes the extractions in parallel (threads)
# Returns a mask for the successful extractions and prints the time taken
def parallel_extraction(extraction_list):
    start = time()
    successes = []
    with cf.ThreadPoolExecutor(NUM_CPUS) as executor:
        for result in executor.map(extract_from_pfd, extraction_list):
            successes.append(result)
    total_time = time() - start
    print(f"Extraction time: {total_time}")
    return successes


########## Download Candidate Files ##########

# Get the list of all MlTrainingSetCollection names
set_collections = get_keys(f'{base_url}ml_training_set_collections/')

# Ensure that the requested MlTrainingSetCollection exists
exists = check_collection_existence(collection_name)
while not exists:
    collection_name = input("Enter the name of the MlTrainingSetCollection to download: ")
    exists = check_collection_existence(collection_name)

# Get the names of the MlTrainingSets associated with the MlTrainingSetCollection
URL = f'{base_url}ml_training_sets/?types__collections={collection_name}'
training_sets = get_keys(URL)
num_of_sets = len(training_sets)

num_failed_downloads = []
num_failed_extractions = []
# Download and extract the data for all Candidates associated with each MlTrainingSet
for i in range(num_of_sets):
    print(f"Starting work on set {i}/{num_of_sets}, {training_sets[i]}.")
    URL = f'{base_url}candidates/?ml_training_sets={training_sets[i]}'
    # Retrieve the file names for the relevant Candidates
    candidates = get_dataframe(URL)
    list_of_pfd_paths = candidates['file'].to_numpy()
    # Download the pfd files and keep track of failed downloads
    download_successes = parallel_download(list_of_pfd_paths)
    # Extract the pfds to numpy arrays, delete the pfds, and track failed extractions
    extraction_successes = parallel_extraction(list_of_pfd_paths)
    # Count the number of failed downloads/extractions
    num_failed_downloads.append(np.count_nonzero(download_successes=False))
    num_failed_extractions.append(np.count_nonzero(extraction_successes=False))
    if (num_failed_downloads != 0) or (num_failed_extractions != 0):
        print(f"Warning: MlTrainingSet {training_sets[i]} had {num_failed_downloads} failed downloads and {num_failed_extractions} failed extractions.")

# Print warnings about failed downloads/extractions, if any
if (np.count_nonzero(num_failed_downloads) != 0) or (np.count_nonzero(num_failed_extractions) != 0):
    print("Warning: Some failed downloads or extractions were detected.")
    for i in range(num_of_sets):
        if (num_failed_downloads[i] != 0) or (num_failed_extractions[i] != 0):
            print(f"MlTrainingSet {training_sets[i]} had {num_failed_downloads[i]} failed downloads and {num_failed_extractions[i]} failed extractions.")

print("All done!")