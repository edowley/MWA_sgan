###############################################################################
#
# This file contains code that will measure the performance of a Model against
# the validation sets of a particular MlTrainingSetCollection.
# 
#       1. This description,
#          has not been written yet.
#           - Come back later
# 
###############################################################################

import argparse, errno, math, os, pickle, requests, sys
from glob import glob
from keras.utils import to_categorical
from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from time import time
from urllib.parse import urljoin

# Constants
SMART_BASE_URL = os.environ.get('SMART_BASE_URL', 'http://localhost:8000/')
SMART_TOKEN = os.environ.get('SMART_TOKEN', 'fagkjfasbnlvasfdfwjf783YDF')

class NotADirectoryError(Exception):
    pass

def dir_path(string):
    if os.path.isdir(string) and string[-1] == "/":
        return string
    else:
        raise NotADirectoryError("Directory path is not valid.")

# Parse arguments
parser = argparse.ArgumentParser(description='Calculate performance of the retrained SGAN model against a validation set.')
parser.add_argument('-d', '--data_directory', help='Absolute path of the data directory (contains the candidates/ and saved_models/ subdirectories)', default='/data/SGAN_Test_Data/')
parser.add_argument('-n', '--collection_name', help='Name of the MlTrainingSetCollection to get validation sets from', default="")
parser.add_argument('-m', '--model_name', help='Name of the AlgorithmSettings object that the SGAN model is stored in', default="")
parser.add_argument('-i', '--individual_stats', help='Also get stats for each model individually (dm_curve, freq_phase, etc.)', default=True)
parser.add_argument('-l', '--base_url', help='Base URL for the database', default=SMART_BASE_URL)
parser.add_argument('-t', '--token', help='Authorization token for the database', default=SMART_TOKEN)

args = parser.parse_args()
path_to_data = args.data_directory
collection_name = args.collection_name
model_name = args.model_name
individual_stats = args.individual_stats
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

# Checks if an SGAN model exists (in the AlgorithmSetting table)
def check_model_existence(name):
    if name in sgan_model_names:
        return True
    elif name == "":
        return False
    else:
        print(f"The name {name} doesn't match an existing SGAN model")
        return False

# Checks if a MlTrainingSetCollection exists
def check_collection_existence(name):
    if name in set_collections:
        return True
    elif name == "":
        return False
    else:
        print(f"The name {name} doesn't match an existing MlTrainingSetCollection")
        return False

# Checks if the required candidate files have been downloaded/extracted
def check_file(pfd_url):
    pfd_path = path_to_data + pfd_url.partition('media/')[2]
    if not len(glob(pfd_path[:-4] + '*')) == N_FEATURES:
        print(f"Warning: Missing files for candidate {pfd_path[:-4]}")
        return False
    else:
        return True

# Executes the candidate file checks in parallel (threads)
# Returns a mask for the files that exist
def parallel_file_check(file_list):
    successes = []
    with cf.ThreadPoolExecutor(NUM_CPUS) as executor:
        for result in executor.map(check_file, file_list):
            successes.append(result)
    total_time = time() - start
    return successes   


########## Get Filenames and Labels ##########

# Get the list of all MlTrainingSetCollection names
set_collections = get_column(urljoin(base_url, 'api/ml_training_set_collections/'), field='name')

# Ensure that the requested MlTrainingSetCollection exists
exists = check_collection_existence(collection_name)
while not exists:
    collection_name = input("Enter the name of the MlTrainingSetCollection to download: ")
    exists = check_collection_existence(collection_name)

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

# Get the MlTrainingSetTypes associated with the MlTrainingSetCollection
URL = urljoin(base_url, f'api/ml_training_set_types/?collections={collection_name}')
set_types = get_dataframe(URL)
set_type_ids = set_types['id'].to_numpy()
set_type_labels = set_types['type'].to_numpy()

# Get the filenames for all the Candidates in each MlTrainingSetType
# Check that all required files have been downloaded/extracted (otherwise exit)
num_of_sets = 0
start = time()
for index in range(len(set_type_ids)):
    URL = urljoin(base_url, f"api/candidates/?ml_training_sets__types={set_type_ids[index]}")
    # Check files for validation pulsars
    if set_type_labels[index] == "VALIDATION PULSARS":
        validation_pulsars = get_filenames(URL)
        file_successes = parallel_file_check(validation_pulsars)
        num_file_failures = np.count_nonzero(file_successes == False)
        num_of_sets += 1
        if num_file_failures != 0:
            print(f"Warning: Files not found for {num_file_failures} candidates in the VALIDATION PULSARS set.")
            sys.exit()
    # Check files for validation noise
    elif set_type_labels[index] == "VALIDATION NOISE":
        validation_noise = get_filenames(URL)
        file_successes = parallel_file_check(validation_noise)
        num_file_failures = np.count_nonzero(file_successes == False)
        num_of_sets += 1
        if num_file_failures != 0:
            print(f"Warning: Files not found for {num_file_failures} candidates in the VALIDATION NOISE set.")
            sys.exit()
    # Check files for validation RFI
    elif set_type_labels[index] == "VALIDATION RFI":
        validation_RFI = get_filenames(URL)
        file_successes = parallel_file_check(validation_RFI)
        num_file_failures = np.count_nonzero(file_successes == False)
        num_of_sets += 1
        if num_file_failures != 0:
            print(f"Warning: Files not found for {num_file_failures} candidates in the VALIDATION RFI set.")
            sys.exit()
# Check that the required number of sets were found
if num_of_sets != 3:
    print(f"Warning: One or more MlTrainingSets are missing from this MlTrainingSetCollection (expected 7, found {num_of_sets}).")
    sys.exit()
# Print the time taken to do the above steps
total_time = time() - start
print(f"Time taken to get filenames and check file existence: {total_time}")

# Remove the unwanted parts of the pfd urls
validation_pulsars = np.array([x.partition('media/')[2] for x in validation_pulsars])
validation_noise = np.array([x.partition('media/')[2] for x in validation_noise])
validation_RFI = np.array([x.partition('media/')[2] for x in validation_RFI])

# Create the combined validation set and its labels
candidate_files = path_to_data + np.concatenate((validation_pulsars, validation_noise, validation_RFI))
true_labels = np.tile(1, len(validation_pulsars)) + np.tile(0, len(validation_noise)+len(validation_RFI))


########## Rate Model Performance ##########

# Load the best of the models
dm_curve_model = load_model(path_to_models + 'best_retrained_models/dm_curve_best_discriminator_model.h5')
pulse_profile_model = load_model(path_to_models + 'best_retrained_models/pulse_profile_best_discriminator_model.h5')
freq_phase_model = load_model(path_to_models + 'best_retrained_models/freq_phase_best_discriminator_model.h5')
time_phase_model = load_model(path_to_models + 'best_retrained_models/time_phase_best_discriminator_model.h5')

logistic_model = pickle.load(open(path_to_models + 'best_retrained_models/sgan_retrained.pkl', 'rb'))

# Load data (using [:-4] to remove the '.pfd' file extension from the name)
dm_curve_combined_array = [np.load(pfd_path[:-4] + '_dm_curve.npy') for pfd_path in candidate_files]
pulse_profile_combined_array = [np.load(pfd_path[:-4] + '_pulse_profile.npy') for pfd_path in candidate_files]
freq_phase_combined_array = [np.load(pfd_path[:-4] + '_freq_phase.npy') for pfd_path in candidate_files]
time_phase_combined_array = [np.load(pfd_path[:-4] + '_time_phase.npy') for pfd_path in candidate_files]

# Reshape the data for the neural nets to read
reshaped_dm_curve = [np.reshape(f,(60,1)) for f in dm_curve_combined_array]
reshaped_pulse_profile = [np.reshape(f,(64,1)) for f in pulse_profile_combined_array]
reshaped_freq_phase = [np.reshape(f,(48,48,1)) for f in freq_phase_combined_array]
reshaped_time_phase = [np.reshape(f,(48,48,1)) for f in time_phase_combined_array]

# Rescale the data between -1 and +1
dm_curve_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_dm_curve])
pulse_profile_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_pulse_profile])
freq_phase_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_freq_phase])
time_phase_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_time_phase])

print('Test data loaded')

# Make predictions
predictions_dm_curve = dm_curve_model.predict([dm_curve_data])
predictions_pulse_profile = pulse_profile_model.predict([pulse_profile_data])
predictions_freq_phase = freq_phase_model.predict([freq_phase_data])
predictions_time_phase = time_phase_model.predict([time_phase_data])

# Process the predictions into numerical scores:

predictions_dm_curve = np.rint(predictions_dm_curve)
predictions_dm_curve = np.argmax(predictions_dm_curve, axis=1)
predictions_dm_curve = np.reshape(predictions_dm_curve, len(predictions_dm_curve))

predictions_pulse_profile = np.rint(predictions_pulse_profile)
predictions_pulse_profile = np.argmax(predictions_pulse_profile, axis=1)
predictions_pulse_profile = np.reshape(predictions_pulse_profile, len(predictions_pulse_profile))

predictions_freq_phase = np.rint(predictions_freq_phase)
predictions_freq_phase = np.argmax(predictions_freq_phase, axis=1)
predictions_freq_phase = np.reshape(predictions_freq_phase, len(predictions_freq_phase))

predictions_time_phase = np.rint(predictions_time_phase)
predictions_time_phase = np.argmax(predictions_time_phase, axis=1)
predictions_time_phase = np.reshape(predictions_time_phase, len(predictions_time_phase))

stacked_predictions = np.stack((predictions_freq_phase, predictions_time_phase, predictions_dm_curve, predictions_pulse_profile), axis=1)
stacked_predictions = np.reshape(stacked_predictions, (len(dm_curve_data), 4))

classified_results = logistic_model.predict(stacked_predictions)

if individual_stats:
    # DM CURVE
    print('')
    print('DM Curve Stats: ')
    accuracy = accuracy_score(true_labels, predictions_dm_curve)
    recall = recall_score(true_labels, predictions_dm_curve)
    f1 = f1_score(true_labels, predictions_dm_curve)
    precision = precision_score(true_labels, predictions_dm_curve)
    print(f'Accuracy = {accuracy:.3f}, F1-score = {f1:.3f} | Precision = {precision:.3f}, Recall = {recall:.3f}')
    # FREQ-PHASE
    print('')
    print('Freq-Phase Stats: ')
    accuracy = accuracy_score(true_labels, predictions_freq_phase)
    recall = recall_score(true_labels, predictions_freq_phase)
    f1 = f1_score(true_labels, predictions_freq_phase)
    precision = precision_score(true_labels, predictions_freq_phase)
    print(f'Accuracy = {accuracy:.3f}, F1-score = {f1:.3f} | Precision = {precision:.3f}, Recall = {recall:.3f}')
    # PULSE PROFILE
    print('')
    print('Pulse Profile Stats: ')
    accuracy = accuracy_score(true_labels, predictions_pulse_profile)
    recall = recall_score(true_labels, predictions_pulse_profile)
    f1 = f1_score(true_labels, predictions_pulse_profile)
    precision = precision_score(true_labels, predictions_pulse_profile)
    print(f'Accuracy = {accuracy:.3f}, F1-score = {f1:.3f} | Precision = {precision:.3f}, Recall = {recall:.3f}')
    # TIME-PHASE
    print('')
    print('Time-Phase Stats: ')
    accuracy = accuracy_score(true_labels, predictions_time_phase)
    recall = recall_score(true_labels, predictions_time_phase)
    f1 = f1_score(true_labels, predictions_time_phase)
    precision = precision_score(true_labels, predictions_time_phase)
    print(f'Accuracy = {accuracy:.3f}, F1-score = {f1:.3f} | Precision = {precision:.3f}, Recall = {recall:.3f}')

# FINAL CLASSIFICATION
print('')
print('Final SGAN Classification: ')
accuracy = accuracy_score(true_labels, classified_results)
recall = recall_score(true_labels, classified_results)
f1 = f1_score(true_labels, classified_results)
precision = precision_score(true_labels, classified_results)
tn, fp, fn, tp = confusion_matrix(true_labels, classified_results).ravel()
specificity = tn/(tn + fp)
gmean = math.sqrt(specificity * recall)
fpr = fp/(tn + fp)
print(f"SGAN Model File: {path_to_models}best_retrained_models/sgan_retrained.pkl")
print(f'Accuracy = {accuracy:.3f}, F1-score = {f1:.3f} | Precision = {precision:.3f}, Recall = {recall:.3f}')
print(f"False Positive Rate: {fpr:.3f}, Specificity: {specificity:.3f}, G-Mean: {gmean:.3f}")

my_session.close()