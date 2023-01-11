###############################################################################
#
# Database-compatible version (WIP). 
#
# This file contains code that will download and process the required files.
# 
#       1. This description,
#          has not been written yet.
#           - Come back later
#
###############################################################################

import argparse
import concurrent.futures as cf
from glob import glob
import json
from math import floor
from multiprocessing import cpu_count
import numpy as np
import os
import pandas as pd
import requests
import sys
from time import time
from ubc_AI.training import pfddata
from urllib.request import urlretrieve


# Constants
NUM_CPUS = cpu_count()
N_FEATURES = 4

# Parse arguments
parser = argparse.ArgumentParser(description='Download pfd files, label, and extract as numpy array files.')
parser.add_argument('-d', '--data_directory', help='Absolute path of output directory for candidate data', default='/data/SGAN_Data/candidates/')
parser.add_argument('-n', '--collection_name', help='Name of the MlTrainingSetCollection', default="", type=str)

args = parser.parse_args()
path_to_data = args.data_directory
collection_name = args.collection_name


# Database stuff
class TokenAuth(requests.auth.AuthBase):
    def __init__(self, token):
        self.token = token

    def __call__(self, r):
        r.headers['Authorization'] = "Token {}".format(self.token)
        return r

my_session = requests.session()
my_session.auth = TokenAuth("fagkjfasbnlvasfdfwjf783YDF")


# Downloads the requested json file and returns it as a pandas dataframe
def get_dataframe(url='http://localhost:8000/api/candidates/?ml_ready_pulsars=true', param=None):
    try:
        table = my_session.get(url, params=param)
        table.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
    return pd.read_json(table.json)

# Downloads the requested json file and returns the primary keys as a numpy array
# Only works if the pk column is called 'id' or 'name'
def get_keys(url='http://localhost:8000/api/candidates/?ml_ready_pulsars=true', param=None):
    try:
        table = my_session.get(url, params=param)
        table.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
    try:
        keys = [row['id'] for row in response.json()]
    except KeyError:
        try:
            keys = [row['name'] for row in response.json()]
        except KeyError as err:
            print(err)
            print("This table has no 'id' or 'name' column.")
    return np.array(keys)


# Ensure that the MlTrainingSetCollection name is valid:

set_collections = get_keys('http://localhost:8000/api/ml-training-set-collections/')

def check_name_validity(name):
    if name in set_collections:
        return True
    elif name == "":
        return False
    else:
        print(f"The name {name} doesn't match a MlTrainingSetCollection")
        return False

valid = check_name_validity(collection_name)
while not valid:
    collection_name = input("Enter the name of the MlTrainingSetCollection to use: ")
    valid = check_name_validity


# Get the requested MlTrainingSetCollection
url = f'http://localhost:8000/api/ml-training-set-collections/?name={collection_name}'
collection = get_dataframe(url)

# Get the associated MlTrainingSetTypes
set_type_ids = collection['ml_training_set_types']
set_types = []
for set_type_id in set_type_ids:
    url = f'http://localhost:8000/api/ml-training-set-types/?id={set_type_id}'
    set_types.append(get_dataframe(url))

# Get the associated MlTrainingSets
set_names = [set_type['ml_training_set'] for set_type in set_types]
training_sets = []
for set_name in set_names:
    url = f'http://localhost:8000/api/ml-training-sets/?name={set_name}'
    training_sets.append(get_dataframe(url))


# Make the target directory, if it doesn't already exist
os.makedirs(path_to_data, exist_ok=True)


# Downloads pfd files from the database to the candidates directory
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
        # If the target pfd file or associated numpy files already exist, download is skipped
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

# Extracts pfd files to numpy array files in the candidates directory and deletes the pfd files
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
        # If the numpy array files already exist, extraction is skipped
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



#################### WIP beyond this point ####################

num_of_sets = len(training_sets)

# NOTE There is definitely a better way of getting the candidates in a particular collections
# The "working backwards" approach used above is okay if there are only a few items,
# but with hundreds of candidates it's obviously stupid

for i in range(num_of_sets):
    print(f"Starting work on set {i} of {num_of_sets}.")
    list_of_pfd_paths = ??????
    # Download the pfd files and keep track of failed downloads
    download_successes = parallel_download(list_of_pfd_paths)
    # Remove failed downloads from the dataframe (so it matches the directory contents)
    training_set = training_set[download_successes]
    # Extract the pfds to numpy arrays, delete the pfds, and track failed extractions
    extraction_successes = parallel_extraction(list_of_pfd_paths)
    # Remove failed extractions from the dataframe
    training_set = training_set[extraction_successes]

# NOTE Might also want to change the way failed downloads/extractions are treated,
# now that the MlTrainingSets are stored in the database

print("All done!")