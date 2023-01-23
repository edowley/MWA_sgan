###############################################################################
# 
# This file contains code that will use existing models to classify candidates.
# 
#       1. This description,
#          has not been written yet.
#           - Come back later
#
###############################################################################

import argparse, errno, os, pickle, requests, sys
import concurrent.futures as cf
from glob import glob
from keras.utils import to_categorical
from keras.models import load_model
from multiprocessing import cpu_count
import numpy as np
from sklearn.ensemble import StackingClassifier
import tarfile
from time import time
from urllib.parse import urljoin

# Constants
NUM_CPUS = cpu_count()
N_FEATURES = 4
SMART_BASE_URL = os.environ.get('SMART_BASE_URL', 'http://localhost:8000/')
SMART_TOKEN = os.environ.get('SMART_TOKEN', 'fagkjfasbnlvasfdfwjf783YDF')

class NotADirectoryError(Exception):
    pass

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError("Directory path is not valid.")

# Parse arguments
parser = argparse.ArgumentParser(description='Classify Candidates using an SGAN model.')
parser.add_argument('-d', '--data_directory', help='Absolute path of the data directory (contains the candidates/ and saved_models/ subdirectories)', default='/data/SGAN_Test_Data/')
parser.add_argument('-s', '--set_name', help='Name of the MlTrainingSet to classify', default="")
parser.add_argument('-m', '--model_name', help='Name of the SGAN model to use', default="")
parser.add_argument('-b', '--batch_size', help='No. of pfd files that will be read in one batch', default='16', type=int)
parser.add_argument('-l', '--base_url', help='Base URL for the database', default=SMART_BASE_URL)
parser.add_argument('-t', '--token', help='Authorization token for the database', default=SMART_TOKEN)

args = parser.parse_args()
path_to_data = args.data_directory
set_name = args.set_name
model_name = args.model_name
batch_size = args.batch_size
base_url = args.base_url
token = args.token

# Ensure that the data path ends with a slash
if path_to_data[-1] != '/':
    path_to_data += '/'

# Check that the specified input directories exist
dir_path(path_to_data + 'candidates/')
path_to_models = path_to_data + 'saved_models/'
dir_path(path_to_models)

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

# Queries a url and returns the result as a pandas dataframe
def get_dataframe(url=urljoin(base_url, 'api/candidates/'), param=None):
    try:
        table = my_session.get(url, params=param)
        table.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
    return pd.DataFrame(table.json())

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

# Checks if an MlTrainingSet exists
def check_set_existence(name):
    if name in training_sets:
        return True
    elif name == "":
        return False
    else:
        print(f"The name {name} doesn't match an existing MlTrainingSet")
        return False

# Checks if an SGAN model exists (in the AlgorithmSetting table)
def check_model_existence(name):
    if name in sgan_model_names:
        return True
    elif name == "":
        return False
    else:
        print(f"The name {name} doesn't match an existing SGAN model")
        return False

# Checks if the required candidate files have been downloaded/extracted
def check_file(pfd_url):
    pfd_path = path_to_data + pfd_url.partition('media/')[2]
    if not len(glob(pfd_path[:-4] + '*')) == N_FEATURES:
        print(f"Warning: Missing files for candidate {pfd_path[:-4]}")
        return False
    else:
        return True

# Executes the file checks in parallel (threads)
# Returns a mask for the files that exist
def parallel_file_check(file_list):
    successes = []
    with cf.ThreadPoolExecutor(NUM_CPUS) as executor:
        for result in executor.map(check_file, file_list):
            successes.append(result)
    total_time = time() - start
    return successes   


########## Get Filenames and Labels ##########

# Get the list of all SGAN model names in the AlgorithmSetting table
sgan_model_names = get_column(urljoin(base_url, 'api/algorithm_settings/?algorithm_parameter=SGAN_files'), field='value')

# Ensure that the requested SGAN model exists
exists = check_model_existence(model_name)
while not exists:
    model_name = input("Enter the name of the SGAN model to download: ")
    exists = check_model_existence(model_name)

# Check that the model files have been downloaded/extracted (otherwise exit)
if not os.isdir(path_to_models + name):
    print(f"Warning: Missing files for model {name}")
    sys.exit()

# Get the list of all MlTrainingSet names
training_sets = get_column(urljoin(base_url, 'api/ml_training_sets/'), field='name')

# Ensure that the requested MlTrainingSet exists
exists = check_set_existence(set_name)
while not exists:
    set_name = input("Enter the name of the MlTrainingSet to use: ")
    exists = check_set_existence(set_name)

# Get the ids and file names for all Candidates in the MlTrainingSet
URL = urljoin(base_url, f'candidates/?ml_training_sets={set_name}')
candidates = get_dataframe(URL)
candidate_ids = candidates['id'].to_numpy()
candidate_files = candidates['file'].to_numpy()
# Check that all required files have been downloaded/extracted (otherwise exit)
file_successes = parallel_file_check(candidate_files)
num_file_failures = np.count_nonzero(file_successes=False)
num_of_sets += 1
if num_file_failures != 0:
    print(f"Warning: Files not found for {num_file_failures} candidates in this set.")
    print(f"Try using download_candidate_data_db.py with -n {set_name} -s 1")
    sys.exit()

# Convert the list of pfd urls (database) into a list of absolute paths (local)
candidate_files = path_to_data + np.array([x.partition('media/')[2] for x in candidate_files])


########## Calculate Scores ##########

# Load the best of the models
dm_curve_model = load_model(path_to_models + 'dm_curve_best_discriminator_model.h5')
freq_phase_model = load_model(path_to_models + 'freq_phase_best_discriminator_model.h5')
pulse_profile_model = load_model(path_to_models + 'pulse_profile_best_discriminator_model.h5')
time_phase_model = load_model(path_to_models + 'time_phase_best_discriminator_model.h5')

logistic_model = pickle.load(open(path_to_models + 'sgan_retrained.pkl', 'rb'))

# Load data (using [:-4] to remove the '.pfd' file extension from the name)
dm_curve_combined_array = [np.load(pfd_path[:-4] + '_dm_curve.npy') for pfd_path in candidate_files]
pulse_profile_combined_array = [np.load(pfd_path[:-4] + '_pulse_profile.npy') for pfd_path in candidate_files]
freq_phase_combined_array = [np.load(pfd_path[:-4] + '_freq_phase.npy') for pfd_path in candidate_files]
time_phase_combined_array = [np.load(pfd_path[:-4] + '_time_phase.npy') for pfd_path in candidate_files]

# Reshape the data for the neural nets to read
reshaped_time_phase = [np.reshape(f,(48,48,1)) for f in time_phase_combined_array]
reshaped_freq_phase = [np.reshape(f,(48,48,1)) for f in freq_phase_combined_array]
reshaped_pulse_profile = [np.reshape(f,(64,1)) for f in pulse_profile_combined_array]
reshaped_dm_curve = [np.reshape(f,(60,1)) for f in dm_curve_combined_array]

# Rescale the data between -1 and +1
dm_curve_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_dm_curve])
pulse_profile_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_pulse_profile])
freq_phase_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_freq_phase])
time_phase_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_time_phase])

print('Data loaded, making predictions...')

# Make predictions
predictions_freq_phase = freq_phase_model.predict([freq_phase_data])
predictions_time_phase = time_phase_model.predict([time_phase_data])
predictions_dm_curve = dm_curve_model.predict([dm_curve_data])
predictions_pulse_profile = pulse_profile_model.predict([pulse_profile_data])

# Process the predictions into numerical scores:

predictions_time_phase = np.rint(predictions_time_phase)
predictions_time_phase = np.argmax(predictions_time_phase, axis=1)
predictions_time_phase = np.reshape(predictions_time_phase, len(predictions_time_phase))

predictions_dm_curve = np.rint(predictions_dm_curve)
predictions_dm_curve = np.argmax(predictions_dm_curve, axis=1)
predictions_dm_curve = np.reshape(predictions_dm_curve, len(predictions_dm_curve))

predictions_pulse_profile = np.rint(predictions_pulse_profile)
predictions_pulse_profile = np.argmax(predictions_pulse_profile, axis=1)
predictions_pulse_profile = np.reshape(predictions_pulse_profile, len(predictions_pulse_profile))

predictions_freq_phase = np.rint(predictions_freq_phase)
predictions_freq_phase = np.argmax(predictions_freq_phase, axis=1)
predictions_freq_phase = np.reshape(predictions_freq_phase, len(predictions_freq_phase))

stacked_predictions = np.stack((predictions_freq_phase, predictions_time_phase, predictions_dm_curve, predictions_pulse_profile), axis=1)
stacked_predictions = np.reshape(stacked_predictions, (len(dm_curve_data),4))

'''
# Un-comment to get the predicted labels (1/0 for pulsar/non-pulsar)
classified_results = logistic_model.predict(stacked_predictions)
'''
# Use by default: estimated likelihood of being a pulsar (between 0 and 1)
classified_results = logistic_model.predict_proba(stacked_predictions)[:,1]

# Create a list of dictionaries for the new MlScores
scores_df = pd.DataFrame()
scores_df['candidate'] = candidate_ids
scores_df['score'] = classified_results
scores_json = scores_df.to_dict(orient='records')

# Upload the new MlScores
my_session.post(urljoin(base_url, 'api/ml_scores/'), json=scores_json)

my_session.close()

print('All done!')