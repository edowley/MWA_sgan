#!/usr/bin/env python
# coding: utf-8

###############################################################################
# 
# This file contains code that will retrain the SGAN using the labelled, unlabelled,
# and validation sets constructed by get_data.py. It assumes that the input directory
# is the same as the directory used by get_data.py.
# 
#       1. This description,
#          has not been written yet.
#           - Come back later
#
###############################################################################

import argparse, errno, glob, itertools, math, os, pickle, sys
from classifiers import Train_SGAN_DM_Curve, Train_SGAN_Pulse_Profile, Train_SGAN_Freq_Phase, Train_SGAN_Time_Phase
from keras import backend
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.layers import Activation, BatchNormalization, concatenate, Conv1D, Conv2D, Dense, Dropout, Flatten, Input, LeakyReLU, MaxPooling2D, MaxPooling1D, Reshape, UpSampling2D, ZeroPadding2D
from keras.models import load_model, Model, Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, log_loss
from sklearn.model_selection import train_test_split
import tensorflow as tf
from time import time


class NotADirectoryError(Exception):
    pass

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError("Directory path is not valid.")

'''
# Class Labels format
0 -> Non-Pulsar
1 -> Pulsar
-1 -> Unlabelled Candidate
'''
# Parse arguments
parser = argparse.ArgumentParser(description='Re-train SGAN machine learning model using files sourced by get_data.py')
parser.add_argument('-i', '--input_path', help='Absolute path of input directory for candidate data', default='/data/SGAN_Test_Data/')
parser.add_argument('-m', '--models', help='Absolute path of output directory for saving models',  default='/data/SGAN_Test_Data/models/')
parser.add_argument('-b', '--batch_size', help='No. of pfd files that will be read in one batch', default='16', type=int)
parser.add_argument('-e', '--num_epochs', help='No. of epochs to train', default='20', type=int)

args = parser.parse_args()
path_to_data = args.input_path
output_path = args.models
batch_size = args.batch_size
num_epochs = args.num_epochs

# Check that the specified input and output directories exist
dir_path(path_to_data)
os.makedirs(output_path, exist_ok=True)

# Absolute paths to important files and subdirectories
labelled_data_path = path_to_data + 'labelled/' 
validation_data_path = path_to_data + 'validation/'
unlabelled_data_path = path_to_data + 'unlabelled/'
training_labels_file = labelled_data_path + 'training_labels.csv'
validation_labels_file = validation_data_path + 'validation_labels.csv'
unlabelled_labels_file = unlabelled_data_path + 'unlabelled_labels.csv'

# Read the label files as pandas dataframes
training_labels = pd.read_csv(training_labels_file, header = 0, index_col = 0, \
                dtype = {'ID': int, 'Pfd path': 'string', 'Classification': int})
validation_labels = pd.read_csv(validation_labels_file, header = 0, index_col = 0, \
                dtype = {'ID': int, 'Pfd path': 'string', 'Classification': int})
unlabelled_labels = pd.read_csv(unlabelled_labels_file, header = 0, index_col = 0, \
                dtype = {'ID': int, 'Pfd path': 'string', 'Classification': int})

# Extract the absolute paths of the candidate pfd files
# (The pfd files no longer exist, but the numpy array files have similar names)
training_files = labelled_data_path + training_labels['Pfd path'].to_numpy()
validation_files = validation_data_path + validation_labels['Pfd path'].to_numpy()
unlabelled_files = unlabelled_data_path + unlabelled_labels['Pfd path'].to_numpy()

# Extract the labels, creating labels of -1 for the unlabelled candidates
training_labels = training_labels['Classification'].to_numpy()
validation_labels = validation_labels['Classification'].to_numpy()
unlabelled_lables = np.tile(-1, len(unlabelled_files))

# Print the number of each type of candidate in the labelled training set
print('Note: Labels 1, 0, and -1 represent pulsars, non-pulsars, and unlabelled candidates respectively.')
print('These counts are for the labelled training set only:')
labels, counts = np.unique(training_labels, return_counts = True)
for label, count in zip(labels, counts):
    print(f'The number of candidates with label {label} is {count}')

''' Prepare the labelled training data for use by the neural nets '''

# Load data (using [:-4] to remove the '.pfd' file extension)
dm_curve_combined_array = [np.load(filename[:-4] + '_dm_curve.npy') for filename in training_files]
pulse_profile_combined_array = [np.load(filename[:-4] + '_pulse_profile.npy') for filename in training_files]
freq_phase_combined_array = [np.load(filename[:-4] + '_freq_phase.npy') for filename in training_files]
time_phase_combined_array = [np.load(filename[:-4] + '_time_phase.npy') for filename in training_files]

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

print('Labelled training data loaded')

''' Repeat above steps for validation data'''

dm_curve_validation_combined_array = [np.load(filename[:-4] + '_dm_curve.npy') for filename in validation_files]
pulse_profile_validation_combined_array = [np.load(filename[:-4] + '_pulse_profile.npy') for filename in validation_files]
freq_phase_validation_combined_array = [np.load(filename[:-4] + '_freq_phase.npy') for filename in validation_files]
time_phase_validation_combined_array = [np.load(filename[:-4] + '_time_phase.npy') for filename in validation_files]

reshaped_time_phase_validation = [np.reshape(f,(48,48,1)) for f in time_phase_validation_combined_array]
reshaped_freq_phase_validation = [np.reshape(f,(48,48,1)) for f in freq_phase_validation_combined_array]
reshaped_pulse_profile_validation = [np.reshape(f,(64,1)) for f in pulse_profile_validation_combined_array]
reshaped_dm_curve_validation = [np.reshape(f,(60,1)) for f in dm_curve_validation_combined_array]

dm_curve_validation_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_dm_curve_validation])
pulse_profile_validation_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_pulse_profile_validation])
freq_phase_validation_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_freq_phase_validation])
time_phase_validation_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_time_phase_validation])

print('Validation data loaded')

''' Repeat above steps for unlabelled data '''

dm_curve_unlabelled_combined_array = [np.load(filename[:-4] + '_dm_curve.npy') for filename in unlabelled_files]
pulse_profile_unlabelled_combined_array = [np.load(filename[:-4] + '_pulse_profile.npy') for filename in unlabelled_files]
freq_phase_unlabelled_combined_array = [np.load(filename[:-4] + '_freq_phase.npy') for filename in unlabelled_files]
time_phase_unlabelled_combined_array = [np.load(filename[:-4] + '_time_phase.npy') for filename in unlabelled_files]

reshaped_time_phase_unlabelled = [np.reshape(f,(48,48,1)) for f in time_phase_unlabelled_combined_array]
reshaped_freq_phase_unlabelled = [np.reshape(f,(48,48,1)) for f in freq_phase_unlabelled_combined_array]
reshaped_pulse_profile_unlabelled = [np.reshape(f,(64,1)) for f in pulse_profile_unlabelled_combined_array]
reshaped_dm_curve_unlabelled = [np.reshape(f,(60,1)) for f in dm_curve_unlabelled_combined_array]

dm_curve_unlabelled_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_dm_curve_unlabelled])
pulse_profile_unlabelled_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_pulse_profile_unlabelled])
freq_phase_unlabelled_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_freq_phase_unlabelled])
time_phase_unlabelled_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_time_phase_unlabelled])

print('Unlabelled data loaded')


''' Train the models '''

# The learning rates can be fine-tuned
learning_rate_discriminator = [0.0008, 0.001, 0.0002, 0.0002] 
learning_rate_gan = [0.003, 0.001, 0.0002, 0.0002]

dm_curve_instance = Train_SGAN_DM_Curve(output_path, dm_curve_data, training_labels, dm_curve_validation_data, validation_labels, dm_curve_unlabelled_data, unlabelled_lables, batch_size \
                    lr_dis = learning_rate_discriminator[0], lr_gan = learning_rate_gan[0])
pulse_profile_instance = Train_SGAN_Pulse_Profile(output_path, pulse_profile_data, training_labels, pulse_profile_validation_data, validation_labels, pulse_profile_unlabelled_data, unlabelled_lables, batch_size \
                    lr_dis = learning_rate_discriminator[1], lr_gan = learning_rate_gan[1])
freq_phase_instance = Train_SGAN_Freq_Phase(output_path, freq_phase_data, training_labels, freq_phase_validation_data, validation_labels, freq_phase_unlabelled_data, unlabelled_lables, batch_size \
                    lr_dis = learning_rate_discriminator[2], lr_gan = learning_rate_gan[2])
time_phase_instance = Train_SGAN_Time_Phase(output_path, time_phase_data, training_labels, time_phase_validation_data, validation_labels, time_phase_unlabelled_data, unlabelled_lables, batch_size \
                    lr_dis = learning_rate_discriminator[3], lr_gan = learning_rate_gan[3])

'''
# Use default learning rates
dm_curve_instance = Train_SGAN_DM_Curve(output_path, dm_curve_data, training_labels, dm_curve_validation_data, validation_labels, dm_curve_unlabelled_data, unlabelled_lables, batch_size)
pulse_profile_instance = Train_SGAN_Pulse_Profile(output_path, pulse_profile_data, training_labels, pulse_profile_validation_data, validation_labels, pulse_profile_unlabelled_data, unlabelled_lables, batch_size)
freq_phase_instance = Train_SGAN_Freq_Phase(output_path, freq_phase_data, training_labels, freq_phase_validation_data, validation_labels, freq_phase_unlabelled_data, unlabelled_lables, batch_size)
time_phase_instance = Train_SGAN_Time_Phase(output_path, time_phase_data, training_labels, time_phase_validation_data, validation_labels, time_phase_unlabelled_data, unlabelled_lables, batch_size)
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


''' Make predictions '''

# Load the best of the models
freq_phase_model = load_model(output_path + 'MWA_best_retrained_models/freq_phase_best_discriminator_model.h5')
time_phase_model = load_model(output_path + 'MWA_best_retrained_models/time_phase_best_discriminator_model.h5')
dm_curve_model = load_model(output_path + 'MWA_best_retrained_models/dm_curve_best_discriminator_model.h5')
pulse_profile_model = load_model(output_path + 'MWA_best_retrained_models/pulse_profile_best_discriminator_model.h5')

# Make predictions
predictions_freq_phase = freq_phase_model.predict([freq_phase_data])
predictions_time_phase = time_phase_model.predict([time_phase_data])
predictions_dm_curve = dm_curve_model.predict([dm_curve_data])
predictions_pulse_profile = pulse_profile_model.predict([pulse_profile_data])

# Process the predictions into numerical scores

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

# Train the logistic regression
model = LogisticRegression()
stacked_results = np.stack((predictions_freq_phase, predictions_time_phase, predictions_dm_curve, predictions_pulse_profile), axis=1)
stacked_results = np.reshape(stacked_results, (len(predictions_freq_phase), 4))
model.fit(stacked_results, training_labels)
pickle.dump(model, open(output_path + 'MWA_best_retrained_models/sgan_retrained.pkl', 'wb'))

# logistic_model = pickle.load(open(output_path + 'MWA_best_retrained_models/sgan_retrained.pkl', 'rb'))

print("All done!")