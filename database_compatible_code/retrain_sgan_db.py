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

import argparse, errno, itertools, math, os, pickle, requests, shutil, sys
from classifiers import Train_SGAN_DM_Curve, Train_SGAN_Pulse_Profile, Train_SGAN_Freq_Phase, Train_SGAN_Time_Phase
import concurrent.futures as cf
from glob import glob
from keras import backend
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.layers import Activation, BatchNormalization, concatenate, Conv1D, Conv2D, Dense, Dropout, Flatten, Input, LeakyReLU, MaxPooling2D, MaxPooling1D, Reshape, UpSampling2D, ZeroPadding2D
from keras.models import load_model, Model, Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import tarfile
import tensorflow as tf
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
parser = argparse.ArgumentParser(description='Retrain SGAN model using files from download_candidate_data_db.py')
parser.add_argument('-d', '--data_directory', help='Absolute path of the data directory (contains the candidates/ and working_models/ subdirectories)', default='/data/SGAN_Test_Data/')
parser.add_argument('-n', '--collection_name', help='Name of the MlTrainingSetCollection to download', default="")
parser.add_argument('-m', '--model_name', help='Name to save the retrained models under', default="")
parser.add_argument('-b', '--batch_size', help='No. of pfd files that will be read in one batch', default='16', type=int)
parser.add_argument('-e', '--num_epochs', help='No. of epochs to train', default='20', type=int)
parser.add_argument('-l', '--base_url', help='Base URL for the database', default=SMART_BASE_URL)
parser.add_argument('-t', '--token', help='Authorization token for the database', default=SMART_TOKEN)

args = parser.parse_args()
path_to_data = args.data_directory
collection_name = args.collection_name
model_name = args.model_name
batch_size = args.batch_size
num_epochs = args.num_epochs
base_url = args.base_url
token = args.token

# Ensure that the base url ends with a slash
if base_url[-1] != '/':
    base_url += '/'

# Ensure that the data path ends with a slash
if path_to_data[-1] != '/':
    path_to_data += '/'

# Check that the specified input and output directories exist
dir_path(path_to_data + 'candidates/')
path_to_models = path_to_data + 'working_models/'
os.makedirs(path_to_models, exist_ok=True)

# Check that the output subdirectories exist
os.makedirs(path_to_models + 'training_logs/', exist_ok=True)
os.makedirs(path_to_models + 'intermediate_models/', exist_ok=True)
os.makedirs(path_to_models + 'best_retrained_models/', exist_ok=True)

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
def get_dataframe(url=f'{base_url}candidates/', param=None):
    try:
        table = my_session.get(url, params=param)
        table.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
    return pd.read_json(table.json())

# Queries a url and returns the requested column of the result as a numpy array
def get_column(url=f'{base_url}candidates/', param=None, field='id'):
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

# Checks if a model name is valid
def check_name_validity(name):
    if name == "":
        return False
    elif name in sgan_model_names:
        print(f"The name {name} is already in use")
        return False
    else:
        return True

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
set_collections = get_column(f'{base_url}ml_training_set_collections/', field='name')

# Ensure that the requested MlTrainingSetCollection exists
exists = check_collection_existence(collection_name)
while not exists:
    collection_name = input("Enter the name of the MlTrainingSetCollection to download: ")
    exists = check_collection_existence(collection_name)

# Get the MlTrainingSetTypes associated with the MlTrainingSetCollection
URL = f'{base_url}ml_training_set_types/?collection={collection_name}'
set_types = get_dataframe(URL)

# Get the filenames for all the Candidates in each MlTrainingSetType
# Check that all required files have been downloaded/extracted (otherwise exit)
num_of_sets = 0
start = time()
for set_type in set_types:
    URL = f'{base_url}candidates/?ml_training_sets__types={set_type['id']}'
    # Check files for training pulsars
    if set_type['type'] == "TRAINING PULSARS":
        training_pulsars = get_column(URL, field='file')
        file_successes = parallel_file_check(training_pulsars)
        num_file_failures = np.count_nonzero(file_successes=False)
        num_of_sets += 1
        if num_file_failures != 0:
            print(f"Warning: Files not found for {num_file_failures} candidates in the TRAINING PULSARS set.")
            sys.exit()
    # Check files for training noise
    elif set_type['type'] == "TRAINING NOISE":
        training_noise = get_column(URL, field='file')
        file_successes = parallel_file_check(training_noise)
        num_file_failures = np.count_nonzero(file_successes=False)
        num_of_sets += 1
        if num_file_failures != 0:
            print(f"Warning: Files not found for {num_file_failures} candidates in the TRAINING NOISE set.")
            sys.exit()
    # Check files for training RFI
    elif set_type['type'] == "TRAINING RFI":
        training_RFI = get_column(URL, field='file')
        file_successes = parallel_file_check(training_RFI)
        num_file_failures = np.count_nonzero(file_successes=False)
        num_of_sets += 1
        if num_file_failures != 0:
            print(f"Warning: Files not found for {num_file_failures} candidates in the TRAINING RFI set.")
            sys.exit()
    # Check files for validation pulsars
    elif set_type['type'] == "VALIDATION PULSARS":
        validation_pulsars = get_column(URL, field='file')
        file_successes = parallel_file_check(validation_pulsars)
        num_file_failures = np.count_nonzero(file_successes=False)
        num_of_sets += 1
        if num_file_failures != 0:
            print(f"Warning: Files not found for {num_file_failures} candidates in the VALIDATION PULSARS set.")
            sys.exit()
    # Check files for validation noise
    elif set_type['type'] == "VALIDATION NOISE":
        validation_noise = get_column(URL, field='file')
        file_successes = parallel_file_check(validation_noise)
        num_file_failures = np.count_nonzero(file_successes=False)
        num_of_sets += 1
        if num_file_failures != 0:
            print(f"Warning: Files not found for {num_file_failures} candidates in the VALIDATION NOISE set.")
            sys.exit()
    # Check files for validation RFI
    elif set_type['type'] == "VALIDATION RFI":
        validation_RFI = get_column(URL, field='file')
        file_successes = parallel_file_check(validation_RFI)
        num_file_failures = np.count_nonzero(file_successes=False)
        num_of_sets += 1
        if num_file_failures != 0:
            print(f"Warning: Files not found for {num_file_failures} candidates in the VALIDATION RFI set.")
            sys.exit()
    # Check files for unlabelled
    elif set_type['type'] == "UNLABELLED":
        unlabelled_cands = get_column(URL, field='file')
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

# Create the labels for each combined set
training_labels = np.tile(1, len(training_pulsars)) + np.tile(0, len(training_noise)+len(training_RFI))
validation_labels = np.tile(1, len(validation_pulsars)) + np.tile(0, len(validation_noise)+len(validation_RFI))
unlabelled_labels = np.tile(-1, len(unlabelled_files))


########## Prepare Training Data ##########

##### Prepare the labelled training data for use by the neural nets #####

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

##### Repeat for the validation data #####

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

##### Repeat for the unlabelled training data #####

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

dm_curve_instance = Train_SGAN_DM_Curve(path_to_models, dm_curve_data, training_labels, dm_curve_validation_data, validation_labels, dm_curve_unlabelled_data, unlabelled_labels, batch_size, \
                    lr_dis = learning_rate_discriminator[0], lr_gan = learning_rate_gan[0])
pulse_profile_instance = Train_SGAN_Pulse_Profile(path_to_models, pulse_profile_data, training_labels, pulse_profile_validation_data, validation_labels, pulse_profile_unlabelled_data, unlabelled_labels, batch_size, \
                    lr_dis = learning_rate_discriminator[1], lr_gan = learning_rate_gan[1])
freq_phase_instance = Train_SGAN_Freq_Phase(path_to_models, freq_phase_data, training_labels, freq_phase_validation_data, validation_labels, freq_phase_unlabelled_data, unlabelled_labels, batch_size, \
                    lr_dis = learning_rate_discriminator[2], lr_gan = learning_rate_gan[2])
time_phase_instance = Train_SGAN_Time_Phase(path_to_models, time_phase_data, training_labels, time_phase_validation_data, validation_labels, time_phase_unlabelled_data, unlabelled_labels, batch_size, \
                    lr_dis = learning_rate_discriminator[3], lr_gan = learning_rate_gan[3])

'''
# To use default learning rates:
dm_curve_instance = Train_SGAN_DM_Curve(path_to_models, dm_curve_data, training_labels, dm_curve_validation_data, validation_labels, dm_curve_unlabelled_data, unlabelled_labels, batch_size)
pulse_profile_instance = Train_SGAN_Pulse_Profile(path_to_models, pulse_profile_data, training_labels, pulse_profile_validation_data, validation_labels, pulse_profile_unlabelled_data, unlabelled_labels, batch_size)
freq_phase_instance = Train_SGAN_Freq_Phase(path_to_models, freq_phase_data, training_labels, freq_phase_validation_data, validation_labels, freq_phase_unlabelled_data, unlabelled_labels, batch_size)
time_phase_instance = Train_SGAN_Time_Phase(path_to_models, time_phase_data, training_labels, time_phase_validation_data, validation_labels, time_phase_unlabelled_data, unlabelled_labels, batch_size)
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
dm_curve_model = load_model(path_to_models + 'best_retrained_models/dm_curve_best_discriminator_model.h5')
pulse_profile_model = load_model(path_to_models + 'best_retrained_models/pulse_profile_best_discriminator_model.h5')
freq_phase_model = load_model(path_to_models + 'best_retrained_models/freq_phase_best_discriminator_model.h5')
time_phase_model = load_model(path_to_models + 'best_retrained_models/time_phase_best_discriminator_model.h5')

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

stacked_results = np.stack((predictions_freq_phase, predictions_time_phase, predictions_dm_curve, predictions_pulse_profile), axis=1)
stacked_results = np.reshape(stacked_results, (len(predictions_freq_phase), 4))

# Train the logistic regression
model = LogisticRegression()
model.fit(stacked_results, training_labels)
pickle.dump(model, open(path_to_models + 'best_retrained_models/sgan_retrained.pkl', 'wb'))


########## Rate Model Performance ##########

# Make predictions on the validation set
predictions_dm_curve = dm_curve_model.predict([dm_curve_validation_data])
predictions_pulse_profile = pulse_profile_model.predict([pulse_profile_validation_data])
predictions_freq_phase = freq_phase_model.predict([freq_phase_validation_data])
predictions_time_phase = time_phase_model.predict([time_phase_validation_data])

# Process the predictions into numerical scores

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

# Use the logistic regression model
classified_results = model.predict(stacked_predictions)

if individual_stats:
    # DM CURVE
    print('')
    print('DM Curve Stats: ')
    accuracy = accuracy_score(validation_labels, predictions_dm_curve)
    recall = recall_score(validation_labels, predictions_dm_curve)
    f1 = f1_score(validation_labels, predictions_dm_curve)
    precision = precision_score(validation_labels, predictions_dm_curve)
    print(f'Accuracy = {accuracy:.3f}, F1-score = {f1:.3f} | Precision = {precision:.3f}, Recall = {recall:.3f}')
    # FREQ-PHASE
    print('')
    print('Freq-Phase Stats: ')
    accuracy = accuracy_score(validation_labels, predictions_freq_phase)
    recall = recall_score(validation_labels, predictions_freq_phase)
    f1 = f1_score(validation_labels, predictions_freq_phase)
    precision = precision_score(validation_labels, predictions_freq_phase)
    print(f'Accuracy = {accuracy:.3f}, F1-score = {f1:.3f} | Precision = {precision:.3f}, Recall = {recall:.3f}')
    # PULSE PROFILE
    print('')
    print('Pulse Profile Stats: ')
    accuracy = accuracy_score(validation_labels, predictions_pulse_profile)
    recall = recall_score(validation_labels, predictions_pulse_profile)
    f1 = f1_score(validation_labels, predictions_pulse_profile)
    precision = precision_score(validation_labels, predictions_pulse_profile)
    print(f'Accuracy = {accuracy:.3f}, F1-score = {f1:.3f} | Precision = {precision:.3f}, Recall = {recall:.3f}')
    # TIME-PHASE
    print('')
    print('Time-Phase Stats: ')
    accuracy = accuracy_score(validation_labels, predictions_time_phase)
    recall = recall_score(validation_labels, predictions_time_phase)
    f1 = f1_score(validation_labels, predictions_time_phase)
    precision = precision_score(validation_labels, predictions_time_phase)
    print(f'Accuracy = {accuracy:.3f}, F1-score = {f1:.3f} | Precision = {precision:.3f}, Recall = {recall:.3f}')

# FINAL CLASSIFICATION
print('')
print('Final SGAN Classification: ')
accuracy = accuracy_score(validation_labels, classified_results)
recall = recall_score(validation_labels, classified_results)
f1 = f1_score(validation_labels, classified_results)
precision = precision_score(validation_labels, classified_results)
tn, fp, fn, tp = confusion_matrix(validation_labels, classified_results).ravel()
specificity = tn/(tn + fp)
gmean = math.sqrt(specificity * recall)
fpr = fp/(tn + fp)
print(f"Accuracy = {accuracy:.3f}, F1-score = {f1:.3f} | Precision = {precision:.3f}, Recall = {recall:.3f}")
print(f"False Positive Rate: {fpr:.3f}, Specificity: {specificity:.3f}, G-Mean: {gmean:.3f}")


########## Make Database Objects ##########

# Get the list of all SGAN model names in the AlgorithmSetting table
sgan_model_names = get_column(f'{base_url}algorithm_settings/?algorithm_parameter=SGAN_files', field='value')

# Ensure that the chosen AlgorithmSetting name is valid
valid = check_name_validity(model_name)
while not valid:
    model_name = input("Enter a name for the retrained models: ")
    valid = check_name_validity(model_name)

# Copy the working_models/ files to a subdirectory of saved_models/ under the chosen model name
new_dir_path = f'{path_to_data}saved_models/{model_name}/'
try:
    shutil.copytree(path_to_models, new_dir_path)
    os.rmdir(new_dir_path + 'intermediate_models/')
except Exception as err:
    print(err)

# Put the model files in a tar gz file
with tarfile.open(f'{model_name}.tar.gz', "w:gz") as tar:
    tar.add(new_dir_path, arcname=os.path.basename(new_dir_path))
    # Create the AlgorithmSetting object to hold the new model
    my_data = {'algorithm_parameter': 'SGAN_files', 'value': model_name, 'ml_training_set_collection': collection_name, \
            'description': f'Accuracy on validation set: {accuracy}'}
    my_files = {'config_file': tar}
    my_session.post(f'{base_url}algorithm_settings/', data=my_data, files=my_files)


print("All done!")