###############################################################################
#
# Database-compatible version (WIP). 
#
# This file contains code that will create appropriate MlTrainingSets, etc.
# 
#       1. Takes as arguments the desired number of (labelled) pulsars, number of unlabelled
#          candidates, and validation/training set size ratio.
#       2. Queries the database for pulsars, noise, RFI and unlabelled candidates.
#           - Pulsars: Avg rating >= 4, no RFI
#           - Noise: Avg rating <= 2, no RFI
#           - RFI: Avg rating <= 2, RFI
#           - Unlabelled: Any (not selected in the other sets)
#       3. Candidates are randomly chosen for each of the sets based on the available
#          number of each candidate type (pulsars, noise, and RFI), the requested number
#          of pulsars, and the following rules:
#           - The ratio of pulsars to non-pulsars will be 1:1
#           - The ratio of noise to RFI will be between 1:1 (preferred) and 2:1
#           - The default ratio of labelled training data to validation data is 4:1
#           - Any amount of unlabelled training data can be used
#       4. 
#
###############################################################################

import argparse
import json
from math import floor
import numpy as np
import os
import pandas as pd
import requests


# Default values
DEFAULT_NUM_PULSARS = 32
DEFAULT_NUM_UNLABELLED = 64
DEFAULT_VALIDATION_RATIO = 0.2

# Parse arguments
parser = argparse.ArgumentParser(description='Download pfd files, label, and extract as numpy array files.')
parser.add_argument('-p', '--num_pulsars', help='Number of (labelled) pulsars (and non-pulsars) to use', default=DEFAULT_NUM_PULSARS, type=int)
parser.add_argument('-u', '--num_unlabelled', help='Number of unlabelled candidates to use', default=DEFAULT_NUM_UNLABELLED, type=int)
parser.add_argument('-v', '--validation_ratio', help='Proportion of labelled candidates to use in the validation set', default=DEFAULT_VALIDATION_RATIO, type=float)
parser.add_argument('-x', '--set_collection_name', help='Name of the MlTrainingSetCollection', default="", type=str)

args = parser.parse_args()
num_pulsars = args.num_pulsars
num_unlabelled = args.num_unlabelled
validation_ratio = args.validation_ratio
set_collection_name = args.set_collection_name


# Database token
class TokenAuth(requests.auth.AuthBase):
    def __init__(self, token):
        self.token = token

    def __call__(self, r):
        r.headers['Authorization'] = "Token {}".format(self.token)
        return r

my_session = requests.session()
my_session.auth = TokenAuth("fagkjfasbnlvasfdfwjf783YDF")

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

# Query parameter 'ml_ready_pulsars=true' means that candidates associated with
# the same pulsar are guaranteed to come from different observations

# Get array of pulsar candidates (avg rating >= 4, no RFI)
url = 'http://localhost:8000/api/candidates/?avg_rating__gte=4&ml_ready_pulsars=true'
param = {'rfi': False}
all_pulsars = get_keys(url, param)

# Get array of noise candidates (avg rating <= 2, no RFI)
url = 'http://localhost:8000/api/candidates/?avg_rating__lte=2&ml_ready_pulsars=true'
param = {'rfi': False}
all_noise = get_keys(url, param)

# Get array of RFI candidates (avg rating <= 2, RFI)
url = 'http://localhost:8000/api/candidates/?avg_rating__lte=2&ml_ready_pulsars=true'
param = {'rfi': True}
all_RFI = get_keys(url, param)

# Get array of all candidates
all_cands = get_keys()


# The total number of each candidate type available
total_num_pulsars = len(all_pulsars)
total_num_noise = len(all_noise)
total_num_RFI = len(all_RFI)
total_num_cands = len(all_cands)

# The number of each candidate type to use
# Based on the selection rules described at the top
num_pulsars = min(num_pulsars, total_num_pulsars, 2*total_num_noise, 3*total_num_RFI)
num_RFI = min(floor(num_pulsars/2), total_num_RFI)
num_noise = num_pulsars - num_RFI

# Randomly sample the required number of each candidate type
all_pulsars = np.random.choice(all_pulsars, size=num_pulsars, replace=False)
all_noise = np.random.choice(all_noise, size=num_noise, replace=False)
all_RFI = np.random.choice(all_RFI, size=num_RFI, replace=False)

# The number of pulsar, RFI and noise candidates in the labelled training set
# and labelled validation set
num_training_pulsars = floor(num_pulsars * (1 - validation_ratio))
num_training_noise = floor(num_noise * (1 - validation_ratio))
num_training_RFI = floor(num_RFI * (1 - validation_ratio))
num_validation_pulsars = num_pulsars - num_training_pulsars
num_validation_noise = num_noise - num_training_noise
num_validation_RFI = num_RFI - num_training_RFI 

# Filter out candidates assigned to the labelled sets from the unlabelled set
all_unlabelled = np.setdiff1d[all_unlabelled, all_pulsars.append(all_noise.append(all_RFI)), assume_unique=True]
# Randomly sample the required number of unlabelled candidates
total_num_unlabelled = len(all_unlabelled)
num_unlabelled = min(num_unlabelled, total_num_unlabelled)
all_unlabelled = np.random.choice(all_unlabelled, size=num_unlabelled, replace=False)

# Print the number of candidates in each set
print(f"Number of training pulsar candidates: {num_training_pulsars}")
print(f"Number of training noise candidates: {num_training_noise}")
print(f"Number of training RFI candidates: {num_training_RFI}")
print(f"Number of validation pulsar candidates: {num_validation_pulsars}")
print(f"Number of validation noise candidates: {num_validation_noise}")
print(f"Number of validation RFI candidates: {num_validation_RFI}")
print(f"Number of unlabelled training candidates: {num_unlabelled}")


# Ensure that the MlTrainingSetCollection has a valid name:

set_collections = get_keys('http://localhost:8000/api/ml-training-set-collections/')

def check_name_validity(name):
    if name == "":
        return False
    elif name in set_collections:
        print(f"The name {set_collection_name} is already in use")
        return False
    else:
        return True

valid = check_name_validity(set_collection_name)
while not valid:
    set_collection_name = input("Enter a name for the MlTrainingSetCollection: ")
    valid = check_name_validity


# Create json files for the MlTrainingSets:

def create_Ml_Training_Set(name, candidates):
    data = {}
    data['name'] = name
    data['candidates'] = candidates
    return json.dumps(data)

set_type_suffixes = ["_tp", "_tn", "_tr", "_vp", "_vn", "_vr", "_u"]
set_names = set_collection_name + set_type_suffixes

training_pulsars = create_Ml_Training_Set(set_names[0], all_pulsars[:num_training_pulsars])
training_noise = create_Ml_Training_Set(set_names[1], all_noise[:num_training_noise])
training_RFI = create_Ml_Training_Set(set_names[2], all_RFI[:num_training_RFI])
validation_pulsars = create_Ml_Training_Set(set_names[3], all_pulsars[num_training_pulsars+1:])
validation_noise = create_Ml_Training_Set(set_names[4], all_noise[num_training_noise+1:])
validation_RFI = create_Ml_Training_Set(set_names[5], all_RFI[num_training_RFI+1:])
unlabelled_training = create_Ml_Training_Set(set_names[6], all_unlabelled)

# Post the MlTrainingSets
my_session.post('http://localhost:8000/api/ml-training-sets/', json=training_pulsars)
my_session.post('http://localhost:8000/api/ml-training-sets/', json=training_noise)
my_session.post('http://localhost:8000/api/ml-training-sets/', json=training_RFI)
my_session.post('http://localhost:8000/api/ml-training-sets/', json=validation_pulsars)
my_session.post('http://localhost:8000/api/ml-training-sets/', json=validation_noise)
my_session.post('http://localhost:8000/api/ml-training-sets/', json=validation_RFI)
my_session.post('http://localhost:8000/api/ml-training-sets/', json=unlabelled_training)

# Create json files for the MlTrainingSetTypes:

def create_Ml_Training_Set_Type(training_set, label):
    data = {}
    data['ml_training_set'] = training_set
    data['type'] = label
    return json.dumps(data)

tp_type = create_Ml_Training_Set_Type(set_names[0], "TRAINING PULSARS")
tn_type = create_Ml_Training_Set_Type(set_names[1], "TRAINING NOISE")
tr_type = create_Ml_Training_Set_Type(set_names[2], "TRAINING RFI")
vp_type = create_Ml_Training_Set_Type(set_names[3], "VALIDATION PULSARS")
vn_type = create_Ml_Training_Set_Type(set_names[4], "VALIDATION NOISE")
vr_type = create_Ml_Training_Set_Type(set_names[5], "VALIDATION RFI")
u_type = create_Ml_Training_Set_Type(set_names[6], "UNLABELLED")

# Post the MlTrainingSetTypes and store their autoincremented ids
tp_id = my_session.post('http://localhost:8000/api/ml-training-set-types/', json=tp_type).json['id']
tn_id = my_session.post('http://localhost:8000/api/ml-training-set-types/', json=tn_type).json['id']
tr_id = my_session.post('http://localhost:8000/api/ml-training-set-types/', json=tr_type).json['id']
vp_id = my_session.post('http://localhost:8000/api/ml-training-set-types/', json=vp_type).json['id']
vn_id = my_session.post('http://localhost:8000/api/ml-training-set-types/', json=vn_type).json['id']
vr_id = my_session.post('http://localhost:8000/api/ml-training-set-types/', json=vr_type).json['id']
u_id = my_session.post('http://localhost:8000/api/ml-training-set-types/', json=u_type).json['id']

# Create the json files for the MlTrainingSetCollection

def create_Ml_Training_Set_Collection(name, set_types):
    data = {}
    data['name'] = name
    data['ml_training_set_types'] = set_types
    return json.dumps(data)

set_types_list = [tp_id, tn_id, tr_id, vp_id, vn_id, vr_id, u_id]
finished_collection = create_MlTraining_Set_Collection(collection_name, set_types_list)

# Post the MlTrainingSetCollection
my_session.post('http://localhost:8000/api/ml-training-set-collections/', json=finished_collection)

my_session.close()

print("All done!")