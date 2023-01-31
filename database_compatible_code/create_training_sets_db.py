###############################################################################
#
# Randomly selects appropriate Candidates for a complete MlTrainingSetCollection
# (consisting of 7 MlTrainingSets with their associated MlTrainingSetTypes)
# and updates the database accordingly.
# 
#    1. Takes as arguments the desired number of pulsars, number of unlabelled
#       candidates, and validation/training set size ratio; as well as a name
#       for the new MlTrainingSetCollection.
#         - The default values are at the start of the code
#    2. Queries the database for pulsars, noise, RFI and unlabelled candidates.
#         - Pulsars: Avg rating >= 4, no RFI
#         - Noise: Avg rating <= 2, no RFI
#         - RFI: Avg rating <= 2, RFI
#         - Unlabelled: All
#    3. Candidates are randomly chosen for each set based on the following rules:
#         - The ratio of pulsars to non-pulsars will be 1:1
#         - The ratio of noise to RFI will be between 1:1 (preferred) and 4:1
#         - Candidates associated with the same pulsar must come from different
#           observations
#         - Any Candidate not chosen for another set can be in the unlabelled set
#         - Candidates must not have an empty 'file' field
#    4. The names of the new MlTrainingSets consist of the MlTrainingSetCollection
#       name, plus a suffix indicating set type (e.g. my_collection_tp).
#
###############################################################################

import argparse, os, requests
from math import floor
import numpy as np
import pandas as pd
from urllib.parse import urljoin

# Constants
DEFAULT_NUM_PULSARS = 256
DEFAULT_NUM_UNLABELLED = 512
DEFAULT_VALIDATION_RATIO = 0.2
SMART_BASE_URL = os.environ.get('SMART_BASE_URL', 'http://localhost:8000/')
SMART_TOKEN = os.environ.get('SMART_TOKEN', 'fagkjfasbnlvasfdfwjf783YDF')

# Parse arguments
parser = argparse.ArgumentParser(description='Randomly generate MlTrainingSets/SetTypes/SetCollection.')
parser.add_argument('-p', '--num_pulsars', help='Number of (labelled) pulsars (and non-pulsars) to use', default=DEFAULT_NUM_PULSARS, type=int)
parser.add_argument('-u', '--num_unlabelled', help='Number of unlabelled candidates to use', default=DEFAULT_NUM_UNLABELLED, type=int)
parser.add_argument('-v', '--validation_ratio', help='Proportion of labelled candidates to use in the validation set', default=DEFAULT_VALIDATION_RATIO, type=float)
parser.add_argument('-n', '--collection_name', help='Name to use for the new MlTrainingSetCollection', default="")
parser.add_argument('-l', '--base_url', help='Base URL for the database', default=SMART_BASE_URL)
parser.add_argument('-t', '--token', help='Authorization token for the database', default=SMART_TOKEN)

args = parser.parse_args()
num_pulsars = args.num_pulsars
num_unlabelled = args.num_unlabelled
validation_ratio = args.validation_ratio
collection_name = args.collection_name
base_url = args.base_url
token = args.token

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

# Checks if an MlTrainingSetCollection name is valid
def check_name_validity(name):
    if name == "":
        return False
    elif name in set_collections:
        print(f"The name {name} is already in use")
        return False
    else:
        return True


########## Candidate Selection ##########

# 'has_file=1' : Candidate 'file' field is not null
URL = urljoin(base_url, 'api/candidates/?has_file=1')

# Get array of ids of Candidates rated as pulsars (avg rating >= 4, no RFI)
# 'ml_ready_pulsars=1' : no more than one Candidate per Pulsar per Observation
# NOTE set 'ml_ready_pulsars=1' once enough Candidates have been assigned to Pulsars
PARAMS = {'ml_ready_pulsars': 0, 'avg_rating__gte': 4, 'ratings__rfi': 0}
all_pulsars = get_column(URL, PARAMS, field='id')

# Get array of ids of Candidates rated as noise (avg rating <= 2, no RFI)
PARAMS = {'avg_rating__lte': 2, 'ratings__rfi': 0}
all_noise = get_column(URL, PARAMS, field='id')

# Get array of ids of Candidates rated as RFI (avg rating <= 2, RFI)
PARAMS = {'avg_rating__lte': 2, 'ratings__rfi': 1}
all_RFI = get_column(URL, PARAMS, field='id')

# Get array of all Candidate ids
all_cands = get_column(URL, field='id')

## NOTE Alternative to all_cands:
# PARAMS = {'avg_rating__isnull': 1}
# all_unlabelled = get_column(URL, PARAMS, field='id')
## Would save time/memory and eliminate some later steps, but...
## Relies on a large number of Candidates not receiving Ratings

# The total number of each candidate type available
total_num_pulsars = len(all_pulsars)
print(f"Total number of labelled pulsars to choose from: {total_num_pulsars}")
total_num_noise = len(all_noise)
print(f"Total number of labelled noise to choose from: {total_num_noise}")
total_num_RFI = len(all_RFI)
print(f"Total number of labelled RFI to choose from: {total_num_RFI}")
total_num_cands = len(all_cands)
print(f"Total number of Candidates (including unlabelled): {total_num_cands}")

# The number of each candidate type to use
# Based on the selection rules described at the top
num_pulsars = min(num_pulsars, total_num_pulsars, 2*total_num_noise, 5*total_num_RFI)
num_RFI = min(floor(num_pulsars/2), total_num_RFI)
num_noise = min(num_pulsars - num_RFI, total_num_noise)

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
all_unlabelled = np.setdiff1d(all_cands, np.concatenate((all_pulsars, all_noise, all_RFI)))
# Randomly sample the required number of unlabelled candidates
total_num_unlabelled = len(all_unlabelled)
num_unlabelled = min(num_unlabelled, total_num_unlabelled)
all_unlabelled = np.random.choice(all_unlabelled, size=num_unlabelled, replace=False)

# Print the number of candidates in each set
print(f"Number of training pulsar candidates selected: {num_training_pulsars}")
print(f"Number of training noise candidates selected: {num_training_noise}")
print(f"Number of training RFI candidates selected: {num_training_RFI}")
print(f"Number of validation pulsar candidates selected: {num_validation_pulsars}")
print(f"Number of validation noise candidates selected: {num_validation_noise}")
print(f"Number of validation RFI candidates selected: {num_validation_RFI}")
print(f"Number of unlabelled training candidates selected: {num_unlabelled}")


########## Database Object Creation ##########

# Get the list of all MlTrainingSetCollection names
set_collections = get_column(urljoin(base_url, 'api/ml_training_set_collections/'), field='name')

# Ensure that the chosen MlTrainingSetCollection name is valid
valid = check_name_validity(collection_name)
while not valid:
    collection_name = input("Enter a name for the MlTrainingSetCollection: ")
    valid = check_name_validity(collection_name)

##### MlTrainingSets #####

# The names of the new MlTrainingSets
set_type_suffixes = ["_tp", "_tn", "_tr", "_vp", "_vn", "_vr", "_u"]
set_names = [collection_name + suffix for suffix in set_type_suffixes]

# Create the MlTrainingSets
training_pulsars = {'name': set_names[0], 'candidates': [int(x) for x in all_pulsars[:num_training_pulsars]]}
training_noise = {'name': set_names[1], 'candidates': [int(x) for x in all_noise[:num_training_noise]]}
training_RFI = {'name': set_names[2], 'candidates': [int(x) for x in all_RFI[:num_training_RFI]]}
validation_pulsars = {'name': set_names[3], 'candidates': [int(x) for x in all_pulsars[num_training_pulsars+1:]]}
validation_noise = {'name': set_names[4], 'candidates': [int(x) for x in all_noise[num_training_noise+1:]]}
validation_RFI = {'name': set_names[5], 'candidates': [int(x) for x in all_RFI[num_training_RFI+1:]]}
unlabelled_training = {'name': set_names[6], 'candidates': [int(x) for x in all_unlabelled]}

# Post the MlTrainingSets
URL = urljoin(base_url, 'api/ml_training_sets/')
my_session.post(URL, json=training_pulsars)
my_session.post(URL, json=training_noise)
my_session.post(URL, json=training_RFI)
my_session.post(URL, json=validation_pulsars)
my_session.post(URL, json=validation_noise)
my_session.post(URL, json=validation_RFI)
my_session.post(URL, json=unlabelled_training)

##### MlTrainingSetTypes #####

# Create the MlTrainingSetTypes
tp_type = {'ml_training_set': set_names[0], 'type': "TRAINING PULSARS"}
tn_type = {'ml_training_set': set_names[1], 'type': "TRAINING NOISE"}
tr_type = {'ml_training_set': set_names[2], 'type': "TRAINING RFI"}
vp_type = {'ml_training_set': set_names[3], 'type': "VALIDATION PULSARS"}
vn_type = {'ml_training_set': set_names[4], 'type': "VALIDATION NOISE"}
vr_type = {'ml_training_set': set_names[5], 'type': "VALIDATION RFI"}
u_type = {'ml_training_set': set_names[6], 'type': "UNLABELLED"}

# Post the MlTrainingSetTypes and store their autoincremented ids
URL = urljoin(base_url, 'api/ml_training_set_types/')
tp_id = my_session.post(URL, json=tp_type).json()['id']
tn_id = my_session.post(URL, json=tn_type).json()['id']
tr_id = my_session.post(URL, json=tr_type).json()['id']
vp_id = my_session.post(URL, json=vp_type).json()['id']
vn_id = my_session.post(URL, json=vn_type).json()['id']
vr_id = my_session.post(URL, json=vr_type).json()['id']
u_id = my_session.post(URL, json=u_type).json()['id']

##### MlTrainingSetCollection #####

# The list of ids of the MlTrainingSetTypes in this collection
set_types_list = [tp_id, tn_id, tr_id, vp_id, vn_id, vr_id, u_id]

# Create and Post the MlTrainingSetCollection
finished_collection = {'name': collection_name, 'ml_training_set_types': set_types_list}
my_session.post(urljoin(base_url, 'api/ml_training_set_collections/'), json=finished_collection)

my_session.close()

print("All done!")