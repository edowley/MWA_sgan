###############################################################################
#
# This file contains code that will retrain the SGAN using the chosen
# MlTrainingSetCollection, as long as the files have been downloaded.
# 
#       1. This description,
#          has not been written yet.
#           - Come back later
#
###############################################################################

import argparse, errno, glob, itertools, math, os, pickle, sys
from classifiers import Train_SGAN_DM_Curve, Train_SGAN_Pulse_Profile, Train_SGAN_Freq_Phase, Train_SGAN_Time_Phase
import concurrent.futures as cf
import json
from keras import backend
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.layers import Activation, BatchNormalization, concatenate, Conv1D, Conv2D, Dense, Dropout, Flatten, Input, LeakyReLU, MaxPooling2D, MaxPooling1D, Reshape, UpSampling2D, ZeroPadding2D
from keras.models import load_model, Model, Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, log_loss
from sklearn.model_selection import train_test_split
import sys
import tensorflow as tf
from time import time

# Constants
NUM_CPUS = cpu_count()
N_FEATURES = 4

class NotADirectoryError(Exception):
    pass

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError("Directory path is not valid.")

'''
# Class Labels
1 -> Pulsar
0 -> Non-Pulsar
-1 -> Unlabelled Candidate
'''
# Parse arguments
parser = argparse.ArgumentParser(description='Re-train SGAN machine learning model using files sourced by get_data.py')
parser.add_argument('-d', '--data_directory', help='Absolute path of the data directory (contains the candidates/ and models/ subdirectories)', default='/data/SGAN_Test_Data/')
parser.add_argument('-n', '--collection_name', help='Name of the MlTrainingSetCollection to download', default="")
parser.add_argument('-b', '--batch_size', help='No. of pfd files that will be read in one batch', default='16', type=int)
parser.add_argument('-e', '--num_epochs', help='No. of epochs to train', default='20', type=int)

args = parser.parse_args()
path_to_data = args.data_directory
path_to_models = path_to_data + 'models/'
collection_name = args.collection_name
batch_size = args.batch_size
num_epochs = args.num_epochs

# Check that the specified input and output directories exist
dir_path(path_to_data + 'candidates/')
os.makedirs(path_to_models, exist_ok=True)

# Check that the output subdirectories exist
os.makedirs(path_to_models + 'training_logs/', exist_ok=True)
os.makedirs(path_to_models + 'intermediate_models/', exist_ok=True)
os.makedirs(path_to_models + 'MWA_best_retrained_models/', exist_ok=True)


# Database token
class TokenAuth(requests.auth.AuthBase):
    def __init__(self, token):
        self.token = token

    def __call__(self, r):
        r.headers['Authorization'] = "Token {}".format(self.token)
        return r

# Start authorised session
my_session = requests.session()
my_session.auth = TokenAuth("fagkjfasbnlvasfdfwjf783YDF")


########## Function Definitions ##########

# Downloads the requested json file and returns it as a pandas dataframe
def get_dataframe(url='http://localhost:8000/api/candidates/', param=None):
    try:
        table = my_session.get(url, params=param)
        table.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
    return pd.read_json(table.json)

# Downloads the requested json file and returns the primary keys as a numpy array
# Only works if the pk column is called 'id' or 'name'
def get_keys(url='http://localhost:8000/api/candidates/', param=None):
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

# Downloads the requested json file and returns the file names as a numpy array
# Only works if there is a column called 'file'
def get_filenames(url='http://localhost:8000/api/candidates/', param=None):
    try:
        table = my_session.get(url, params=param)
        table.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
    try:
        filenames = [row['file'] for row in response.json()]
    except KeyError as err:
        print(err)
        print("This table has no 'file' column.")
    return np.array(filenames)

# Checks if a MlTrainingSetCollection exists
def check_collection_existence(name):
    if name in set_collections:
        return True
    elif name == "":
        return False
    else:
        print(f"The name {name} doesn't match an existing MlTrainingSetCollection")
        return False

# Checks if the required files have been downloaded/extracted
def check_file(pfd_name):
    if not len(glob(path_to_data + pfd_name[:-4] + '*')) == N_FEATURES:
        print(f"Warning: Missing files for {pfd_name[:-4]}.")
        return False
    else:
        return True

# Executes the file checks in parallel (threads)
# Returns a mask for the files that exist
def parallel_file_check(file_list):
    successes []
    with cf.ThreadPoolExecutor(NUM_CPUS) as executor:
        for result in executor.map(check_file, file_list):
            successes.append(result)
    total_time = time() - start
    return successes   


########## Get Filenames and Labels ##########

# Get the list of all MlTrainingSetCollection names
set_collections = get_keys('http://localhost:8000/api/ml-training-set-collections/')

# Ensure that the requested MlTrainingSetCollection exists
exists = check_collection_existence(collection_name)
while not exists:
    collection_name = input("Enter the name of the MlTrainingSetCollection to download: ")
    exists = check_collection_existence(collection_name)

# Get the MlTrainingSetTypes associated with the MlTrainingSetCollection
URL = f'http://localhost:8000/api/ml-training-set-types/?collection={collection_name}'
set_types = get_dataframe(URL)

# Get the filenames for all the Candidates in each MlTrainingSetType
# Check that all required files have been downloaded/extracted (otherwise exit)
num_of_sets = 0
start = time()
for set_type in set_types:
    if set_type['type'] == "TRAINING PULSARS":
        URL = f'http://localhost:8000/api/candidates/?ml-training-sets__types={set_type['id']}'
        training_pulsars = get_filenames(URL)
        file_successes = parallel_file_check(training_pulsars)
        num_file_failures = np.count_nonzero(file_successes=False)
        num_of_sets += 1
        if num_file_failures != 0:
            print(f"Warning: Files not found for {num_file_failures} candidates in the TRAINING PULSARS set.")
            sys.exit()
    elif set_type['type'] == "TRAINING NOISE":
        URL = f'http://localhost:8000/api/candidates/?ml-training-sets__types={set_type['id']}'
        training_noise = get_filenames(URL)
        file_successes = parallel_file_check(training_noise)
        num_file_failures = np.count_nonzero(file_successes=False)
        num_of_sets += 1
        if num_file_failures != 0:
            print(f"Warning: Files not found for {num_file_failures} candidates in the TRAINING NOISE set.")
            sys.exit()
    elif set_type['type'] == "TRAINING RFI":
        URL = f'http://localhost:8000/api/candidates/?ml-training-sets__types={set_type['id']}'
        training_RFI = get_filenames(URL)
        file_successes = parallel_file_check(training_RFI)
        num_file_failures = np.count_nonzero(file_successes=False)
        num_of_sets += 1
        if num_file_failures != 0:
            print(f"Warning: Files not found for {num_file_failures} candidates in the TRAINING RFI set.")
            sys.exit()
    elif set_type['type'] == "VALIDATION PULSARS":
        URL = f'http://localhost:8000/api/candidates/?ml-training-sets__types={set_type['id']}'
        validation_pulsars = get_filenames(URL)
        file_successes = parallel_file_check(validation_pulsars)
        num_file_failures = np.count_nonzero(file_successes=False)
        num_of_sets += 1
        if num_file_failures != 0:
            print(f"Warning: Files not found for {num_file_failures} candidates in the VALIDATION PULSARS set.")
            sys.exit()
    elif set_type['type'] == "VALIDATION NOISE":
        URL = f'http://localhost:8000/api/candidates/?ml-training-sets__types={set_type['id']}'
        validation_noise = get_filenames(URL)
        file_successes = parallel_file_check(validation_noise)
        num_file_failures = np.count_nonzero(file_successes=False)
        num_of_sets += 1
        if num_file_failures != 0:
            print(f"Warning: Files not found for {num_file_failures} candidates in the VALIDATION NOISE set.")
            sys.exit()
    elif set_type['type'] == "VALIDATION RFI":
        URL = f'http://localhost:8000/api/candidates/?ml-training-sets__types={set_type['id']}'
        validation_RFI = get_filenames(URL)
        file_successes = parallel_file_check(validation_RFI)
        num_file_failures = np.count_nonzero(file_successes=False)
        num_of_sets += 1
        if num_file_failures != 0:
            print(f"Warning: Files not found for {num_file_failures} candidates in the VALIDATION RFI set.")
            sys.exit()
    elif set_type['type'] == "UNLABELLED":
        URL = f'http://localhost:8000/api/candidates/?ml-training-sets__types={set_type['id']}'
        unlabelled_cands = get_filenames(URL)
        file_successes = parallel_file_check(unlabelled_cands)
        num_file_failures = np.count_nonzero(file_successes=False)
        num_of_sets += 1
        if num_file_failures != 0:
            print(f"Warning: Files not found for {num_file_failures} candidates in the UNLABELLED set.")
            sys.exit()
# Check that the required number of sets were found
if num_of_sets != 7:
    print(f"Warning: One or more MlTrainingSets are missing from this MlTrainingSetCollection (expected 7, found {num_of_sets}).")
    sys.exit()
# Print the time taken to do the above steps
total_time = time() - start
print(f"Time taken to get filenames and check file existence: {total_time}")

# Create the combined training and validation sets (unlabelled set already done)
training_files = path_to_data + training_pulsars.append(training_noise.append(training_RFI))
validation_files = path_to_data + validation_pulsars.append(validation_noise.append(validation_RFI))
unlabelled_files = path_to_data + unlabelled_cands

# Create the lables for each combined set
training_lables = np.tile(1, len(training_pulsars)) + np.tile(0, len(training_noise)+len(training_RFI))
validation_lables = np.tile(1, len(validation_pulsars)) + np.tile(0, len(validation_noise)+len(validation_RFI))
unlabelled_lables = np.tile(-1, len(unlabelled_files))


########## Prepare Training Data ##########

''' Prepare the labelled training data for use by the neural nets '''

# Load data (using [:-4] to remove the '.pfd' file extension from the name)
dm_curve_combined_array = [np.load(filename[:-4] + '_dm_curve.npy') for filename in training_files]
pulse_profile_combined_array = [np.load(filename[:-4] + '_pulse_profile.npy') for filename in training_files]
freq_phase_combined_array = [np.load(filename[:-4] + '_freq_phase.npy') for filename in training_files]
time_phase_combined_array = [np.load(filename[:-4] + '_time_phase.npy') for filename in training_files]

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

print('Labelled training data loaded')

''' Repeat for the validation data '''

dm_curve_validation_combined_array = [np.load(filename[:-4] + '_dm_curve.npy') for filename in validation_files]
pulse_profile_validation_combined_array = [np.load(filename[:-4] + '_pulse_profile.npy') for filename in validation_files]
freq_phase_validation_combined_array = [np.load(filename[:-4] + '_freq_phase.npy') for filename in validation_files]
time_phase_validation_combined_array = [np.load(filename[:-4] + '_time_phase.npy') for filename in validation_files]

reshaped_dm_curve_validation = [np.reshape(f,(60,1)) for f in dm_curve_validation_combined_array]
reshaped_pulse_profile_validation = [np.reshape(f,(64,1)) for f in pulse_profile_validation_combined_array]
reshaped_freq_phase_validation = [np.reshape(f,(48,48,1)) for f in freq_phase_validation_combined_array]
reshaped_time_phase_validation = [np.reshape(f,(48,48,1)) for f in time_phase_validation_combined_array]

dm_curve_validation_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_dm_curve_validation])
pulse_profile_validation_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_pulse_profile_validation])
freq_phase_validation_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_freq_phase_validation])
time_phase_validation_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_time_phase_validation])

print('Validation data loaded')

''' Repeat for the unlabelled training data '''

dm_curve_unlabelled_combined_array = [np.load(filename[:-4] + '_dm_curve.npy') for filename in unlabelled_files]
pulse_profile_unlabelled_combined_array = [np.load(filename[:-4] + '_pulse_profile.npy') for filename in unlabelled_files]
freq_phase_unlabelled_combined_array = [np.load(filename[:-4] + '_freq_phase.npy') for filename in unlabelled_files]
time_phase_unlabelled_combined_array = [np.load(filename[:-4] + '_time_phase.npy') for filename in unlabelled_files]

reshaped_dm_curve_unlabelled = [np.reshape(f,(60,1)) for f in dm_curve_unlabelled_combined_array]
reshaped_pulse_profile_unlabelled = [np.reshape(f,(64,1)) for f in pulse_profile_unlabelled_combined_array]
reshaped_freq_phase_unlabelled = [np.reshape(f,(48,48,1)) for f in freq_phase_unlabelled_combined_array]
reshaped_time_phase_unlabelled = [np.reshape(f,(48,48,1)) for f in time_phase_unlabelled_combined_array]

dm_curve_unlabelled_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_dm_curve_unlabelled])
pulse_profile_unlabelled_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_pulse_profile_unlabelled])
freq_phase_unlabelled_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_freq_phase_unlabelled])
time_phase_unlabelled_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_time_phase_unlabelled])

print('Unlabelled training data loaded')


########## Train the Models ##########

# The learning rates can be fine-tuned
learning_rate_discriminator = [0.0008, 0.001, 0.0002, 0.0002] 
learning_rate_gan = [0.003, 0.001, 0.0002, 0.0002]

dm_curve_instance = Train_SGAN_DM_Curve(path_to_models, dm_curve_data, training_labels, dm_curve_validation_data, validation_labels, dm_curve_unlabelled_data, unlabelled_lables, batch_size, \
                    lr_dis = learning_rate_discriminator[0], lr_gan = learning_rate_gan[0])
pulse_profile_instance = Train_SGAN_Pulse_Profile(path_to_models, pulse_profile_data, training_labels, pulse_profile_validation_data, validation_labels, pulse_profile_unlabelled_data, unlabelled_lables, batch_size, \
                    lr_dis = learning_rate_discriminator[1], lr_gan = learning_rate_gan[1])
freq_phase_instance = Train_SGAN_Freq_Phase(path_to_models, freq_phase_data, training_labels, freq_phase_validation_data, validation_labels, freq_phase_unlabelled_data, unlabelled_lables, batch_size, \
                    lr_dis = learning_rate_discriminator[2], lr_gan = learning_rate_gan[2])
time_phase_instance = Train_SGAN_Time_Phase(path_to_models, time_phase_data, training_labels, time_phase_validation_data, validation_labels, time_phase_unlabelled_data, unlabelled_lables, batch_size, \
                    lr_dis = learning_rate_discriminator[3], lr_gan = learning_rate_gan[3])

'''
# Use default learning rates
dm_curve_instance = Train_SGAN_DM_Curve(path_to_models, dm_curve_data, training_labels, dm_curve_validation_data, validation_labels, dm_curve_unlabelled_data, unlabelled_lables, batch_size)
pulse_profile_instance = Train_SGAN_Pulse_Profile(path_to_models, pulse_profile_data, training_labels, pulse_profile_validation_data, validation_labels, pulse_profile_unlabelled_data, unlabelled_lables, batch_size)
freq_phase_instance = Train_SGAN_Freq_Phase(path_to_models, freq_phase_data, training_labels, freq_phase_validation_data, validation_labels, freq_phase_unlabelled_data, unlabelled_lables, batch_size)
time_phase_instance = Train_SGAN_Time_Phase(path_to_models, time_phase_data, training_labels, time_phase_validation_data, validation_labels, time_phase_unlabelled_data, unlabelled_lables, batch_size)
'''

# Train the DM Curve models
start = time()
d_model, c_model = dm_curve_instance.define_discriminator()
g_model = dm_curve_instance.define_generator()
gan_model = dm_curve_instance.define_gan(g_model, d_model)
dm_curve_instance.train(g_model, d_model, c_model, gan_model, n_epochs=num_epochs)
end = time()
print('DM Curve Model Retraining has been completed')
print(f'Time = {end-start:.3f} seconds')

# Train the Pulse Profile models
start = time()
d_model, c_model = pulse_profile_instance.define_discriminator()
g_model = pulse_profile_instance.define_generator()
gan_model = pulse_profile_instance.define_gan(g_model, d_model)
pulse_profile_instance.train(g_model, d_model, c_model, gan_model, n_epochs=num_epochs)
end = time()
print('Pulse Profile Model Retraining has been completed')
print(f'Time = {end-start:.3f} seconds')

# Train the Freq-Phase models
start = time()
d_model, c_model = freq_phase_instance.define_discriminator()
g_model = freq_phase_instance.define_generator()
gan_model = freq_phase_instance.define_gan(g_model, d_model)
freq_phase_instance.train(g_model, d_model, c_model, gan_model, n_epochs=num_epochs)
end = time()
print('Freq-Phase Model Retraining has been completed')
print(f'Time = {end-start:.3f} seconds')

# Train the Time-Phase models
start = time()
d_model, c_model = time_phase_instance.define_discriminator()
g_model = time_phase_instance.define_generator()
gan_model = time_phase_instance.define_gan(g_model, d_model)
time_phase_instance.train(g_model, d_model, c_model, gan_model, n_epochs=num_epochs)
end = time()
print('Time-Phase Model Retraining has been completed')
print(f'Time = {end-start:.3f} seconds')


########## Make Predictions ##########

# Load the best of the models
dm_curve_model = load_model(path_to_models + 'MWA_best_retrained_models/dm_curve_best_discriminator_model.h5')
pulse_profile_model = load_model(path_to_models + 'MWA_best_retrained_models/pulse_profile_best_discriminator_model.h5')
freq_phase_model = load_model(path_to_models + 'MWA_best_retrained_models/freq_phase_best_discriminator_model.h5')
time_phase_model = load_model(path_to_models + 'MWA_best_retrained_models/time_phase_best_discriminator_model.h5')

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


# Train the logistic regression
model = LogisticRegression()
stacked_results = np.stack((predictions_freq_phase, predictions_time_phase, predictions_dm_curve, predictions_pulse_profile), axis=1)
stacked_results = np.reshape(stacked_results, (len(predictions_freq_phase), 4))
model.fit(stacked_results, training_labels)
pickle.dump(model, open(path_to_models + 'MWA_best_retrained_models/sgan_retrained.pkl', 'wb'))

# logistic_model = pickle.load(open(path_to_models + 'MWA_best_retrained_models/sgan_retrained.pkl', 'rb'))


########## Make Database Objects ##########




print("All done!")