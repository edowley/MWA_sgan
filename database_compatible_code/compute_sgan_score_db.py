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

# Constants
NUM_CPUS = cpu_count()
N_FEATURES = 4
SMART_BASE_URL = os.environ.get('SMART_BASE_URL', 'http://localhost:8000/api/')
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

# Ensure that the base url ends with a slash
if base_url[-1] != '/':
    base_url += '/'

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


# NOTE variables that need to be replaced:
# path_to_data may be different now
# output_path should be removed (no csv file anymore)
# path_to_models is different now


########## WIP from here on ##########


# Get the candidate file names from validation_labels.csv
with open(path_to_data + "validation_labels.csv") as f:
    candidate_files = [path_to_data + row.split(',')[1] for row in f]
candidate_files = candidate_files[1:] # The first entry is the header 
basename_candidate_files = [os.path.basename(filename) for filename in candidate_files]

dm_curve_model = load_model(path_to_models + 'dm_curve_best_discriminator_model.h5')
freq_phase_model = load_model(path_to_models + 'freq_phase_best_discriminator_model.h5')
pulse_profile_model = load_model(path_to_models + 'pulse_profile_best_discriminator_model.h5')
time_phase_model = load_model(path_to_models + 'time_phase_best_discriminator_model.h5')

logistic_model = pickle.load(open(path_to_models + 'sgan_retrained.pkl', 'rb'))

dm_curve_combined_array = [np.load(filename[:-4] + '_dm_curve.npy') for filename in candidate_files]
pulse_profile_combined_array = [np.load(filename[:-4] + '_pulse_profile.npy') for filename in candidate_files]
freq_phase_combined_array = [np.load(filename[:-4] + '_freq_phase.npy') for filename in candidate_files]
time_phase_combined_array = [np.load(filename[:-4] + '_time_phase.npy') for filename in candidate_files]

reshaped_time_phase = [np.reshape(f,(48,48,1)) for f in time_phase_combined_array]
reshaped_freq_phase = [np.reshape(f,(48,48,1)) for f in freq_phase_combined_array]
reshaped_pulse_profile = [np.reshape(f,(64,1)) for f in pulse_profile_combined_array]
reshaped_dm_curve = [np.reshape(f,(60,1)) for f in dm_curve_combined_array]

dm_curve_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_dm_curve])
pulse_profile_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_pulse_profile])
freq_phase_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_freq_phase])
time_phase_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_time_phase])

predictions_freq_phase = freq_phase_model.predict([freq_phase_data])
predictions_time_phase = time_phase_model.predict([time_phase_data])
predictions_dm_curve = dm_curve_model.predict([dm_curve_data])
predictions_pulse_profile = pulse_profile_model.predict([pulse_profile_data])

# print(predictions_freq_phase)
# print(predictions_time_phase)
# print(predictions_dm_curve)
# print(predictions_pulse_profile)

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
# print(stacked_predictions)
# classified_results = logistic_model.predict(stacked_predictions) # If you want a classification score
classified_results = logistic_model.predict_proba(stacked_predictions)[:,1] # If you want a regression score
# print(logistic_model.predict_proba(stacked_predictions))
print(classified_results)

with open(output_path+'sgan_ai_score.csv', 'w') as f:
    f.write('Filename,SGAN_score' + '\n') 
    for i in range(len(candidate_files)):
        f.write(basename_candidate_files[i] + ',' + str(classified_results[i]) + '\n')
