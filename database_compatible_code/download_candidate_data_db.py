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
from urllib.parse import urljoin
from urllib.request import urlretrieve

# Constants
SMART_BASE_URL = os.environ.get('SMART_BASE_URL', 'http://localhost:8000/')
SMART_TOKEN = os.environ.get('SMART_TOKEN', 'fagkjfasbnlvasfdfwjf783YDF')
NUM_CPUS = cpu_count()
N_FEATURES = 4

# Parse arguments
parser = argparse.ArgumentParser(description='Download pfd files and extract as numpy array files.')
parser.add_argument('-d', '--data_directory', help='Absolute path of the data directory (contains the candidates/ subdirectory)', default='/data/SGAN_Test_Data/')
parser.add_argument('-n', '--name', help='Name of the MlTrainingSetCollection or MlTrainingSet to download', default="")
parser.add_argument('-s', '--set_only', help='Set True to download a single MlTrainingSet, otherwise downloads an entire MlTrainingSetCollection', default=False)
parser.add_argument('-l', '--base_url', help='Base URL for the database', default=SMART_BASE_URL)
parser.add_argument('-t', '--token', help='Authorization token for the database', default=SMART_TOKEN)

args = parser.parse_args()
path_to_data = args.data_directory
name = args.name
set_only = args.set_only
base_url = args.base_url
token = args.token

# Convert to boolean
if (set_only == "True") or (set_only == "true") or (set_only == "1"):
    set_only = True
else:
    set_only = False

# Assign the appropriate variable
if set_only:
    set_name = name
else:
    collection_name = name

# Ensure that the data path ends with a slash
if path_to_data[-1] != '/':
    path_to_data += '/'

# Make the target directory, if it doesn't already exist
os.makedirs(path_to_data + 'candidates/', exist_ok=True)

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

# Checks if an MlTrainingSetCollection exists
def check_collection_existence(name):
    if name in set_collections:
        return True
    elif name == "":
        return False
    else:
        print(f"The name {name} doesn't match an existing MlTrainingSetCollection")
        return False

# Checks if an MlTrainingSet exists
def check_set_existence(name):
    if name in training_sets:
        return True
    elif name == "":
        return False
    else:
        print(f"The name {name} doesn't match an existing MlTrainingSet")
        return False

# Downloads pfd files from the database to the candidates/ directory
# Returns False and prints a message if the download fails, otherwise returns True
def download_pfd(pfd_url):
    pfd_path = path_to_data + pfd_url.partition('media/')[2]
    full_url = urljoin(base_url, pfd_url)
    if len(glob(pfd_path[:-4] + '*')) == 0:
        try:
            urlretrieve(full_url, pfd_path)
            return True
        except Exception as e:
            print(f"Download failed: {full_url}, {e}")
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
def extract_from_pfd(pfd_url):
    pfd_path = path_to_data + pfd_url.partition('media/')[2]
    if not len(glob(pfd_path[:-4] + '*')) == N_FEATURES:
        try:
            data_obj = pfddata(pfd_path)
            time_phase_data = data_obj.getdata(intervals=48)
            freq_phase_data = data_obj.getdata(subbands=48)
            dm_curve_data = data_obj.getdata(DMbins=60)
            profile_data = data_obj.getdata(phasebins=64)

            np.save(pfd_path[:-4] + '_time_phase.npy', time_phase_data)
            np.save(pfd_path[:-4] + '_freq_phase.npy', freq_phase_data)
            np.save(pfd_path[:-4] + '_dm_curve.npy', dm_curve_data)
            np.save(pfd_path[:-4] + '_pulse_profile.npy', profile_data)

            os.unlink(pfd_path)
            return True
        except ValueError: 
            print(f"Extraction failed: {pfd_path}")
             # If the extraction fails, delete the pfd anyway
            os.unlink(pfd_path)
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

# If downloading only an MlTrainingSet:
if set_only:
    # Get the list of all MlTrainingSet names
    training_sets = get_column(urljoin(base_url, 'api/ml_training_sets/'), field='name')

    # Ensure that the requested MlTrainingSet exists
    exists = check_set_existence(set_name)
    while not exists:
        set_name = input("Enter the name of the MlTrainingSet to use: ")
        exists = check_set_existence(set_name)
    
    # Download and extract the data for all Candidates in the MlTrainingSet
    print(f"Starting work on {set_name}.")
    URL = urljoin(base_url, f'api/candidates/?ml_training_sets={set_name}')
    # Retrieve the file names for the relevant Candidates
    list_of_files = get_column(URL, field='file')
    # Download the pfd files and keep track of failed downloads
    download_successes = parallel_download(list_of_files)
    # Extract the pfds to numpy arrays, delete the pfds, and track failed extractions
    extraction_successes = parallel_extraction(list_of_files)
    # Count the number of failed downloads/extractions
    num_failed_downloads = np.count_nonzero(download_successes == False)
    num_failed_extractions = np.count_nonzero(extraction_successes == False)
    if (num_failed_downloads != 0) or (num_failed_extractions != 0):
        print(f"Warning: MlTrainingSet {set_name} had {num_failed_downloads} failed downloads and {num_failed_extractions} failed extractions.")

# If downloading an MlTrainingSetCollection:
else:
    # Get the list of all MlTrainingSetCollection names
    set_collections = get_column(urljoin(base_url, 'api/ml_training_set_collections/'), field='name')

    # Ensure that the requested MlTrainingSetCollection exists
    exists = check_collection_existence(collection_name)
    while not exists:
        collection_name = input("Enter the name of the MlTrainingSetCollection to download: ")
        exists = check_collection_existence(collection_name)

    # Get the names of the MlTrainingSets associated with the MlTrainingSetCollection
    URL = urljoin(base_url, f'api/ml_training_sets/?types__collections={collection_name}')
    training_sets = get_column(URL, field='name')
    num_of_sets = len(training_sets)

    num_failed_downloads = []
    num_failed_extractions = []
    # Download and extract the data for all Candidates associated with each MlTrainingSet
    for i in range(num_of_sets):
        print(f"Starting work on set {i+1}/{num_of_sets}, {training_sets[i]}:")
        URL = urljoin(base_url, f'api/candidates/?ml_training_sets={training_sets[i]}')
        # Retrieve the file names for the relevant Candidates
        list_of_files = get_column(URL, field='file')
        # Download the pfd files and keep track of failed downloads
        download_successes = parallel_download(list_of_files)
        # Extract the pfds to numpy arrays, delete the pfds, and track failed extractions
        extraction_successes = parallel_extraction(list_of_files)
        # Count the number of failed downloads/extractions
        num_failed_downloads.append(np.count_nonzero(download_successes == False))
        num_failed_extractions.append(np.count_nonzero(extraction_successes == False))
    
    # Print warnings about failed downloads/extractions, if any
    if (np.count_nonzero(num_failed_downloads) != 0) or (np.count_nonzero(num_failed_extractions) != 0):
        print("Warning: Some failed downloads or extractions were detected")
        for i in range(num_of_sets):
            if (num_failed_downloads[i] != 0) or (num_failed_extractions[i] != 0):
                print(f"MlTrainingSet {training_sets[i]} had {num_failed_downloads[i]} failed downloads and {num_failed_extractions[i]} failed extractions")

my_session.close()

print("All done!")