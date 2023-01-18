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

import argparse, errno, glob, math, os, pickle, sys, time
from keras.utils import to_categorical
from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

class NotADirectoryError(Exception):
    pass

def dir_path(string):
    if os.path.isdir(string) and string[-1] == "/":
        return string
    else:
        raise NotADirectoryError("Directory path is not valid.")


parser = argparse.ArgumentParser(description='Calculate performance of the retrained SGAN model against a validation set.')
parser.add_argument('-d', '--data_directory', help='Absolute path of the data directory (contains the candidates/ and models/ subdirectories)', default='/data/SGAN_Test_Data/')
parser.add_argument('-n', '--collection_name', help='Name of the MlTrainingSetCollection to get validation sets from', default="")
parser.add_argument('-m', '--model_name', help='Name of the AlgorithmSettings object that the SGAN model is stored in', default="")
parser.add_argument('-i', '--individual_stats', help='Also get stats for each model individually (dm_curve, freq_phase, etc.)', default=True)

args = parser.parse_args()
path_to_data = args.data_directory
path_to_models = path_to_data + 'models/'
collection_name = args.collection_name
individual_stats = args.individual_stats

dir_path(path_to_data + 'candidates/')
dir_path(path_to_models)


# Get candidates and labels the same way as retrain_sgan but only the validation sets 


# Read test set labels file
test_set = pd.read_csv(label_file_name, header = 0, index_col = 0, \
                dtype = {'ID': int, 'Pfd path': 'string', 'Classification': int})

candidate_files = path_to_data + test_set['Pfd path'].to_numpy()
true_labels = test_set['Classification'].to_numpy()
basename_candidate_files = [os.path.basename(filename) for filename in candidate_files]


# Load the best of the models
dm_curve_model = load_model(path_to_models + 'best_retrained_models/dm_curve_best_discriminator_model.h5')
pulse_profile_model = load_model(path_to_models + 'best_retrained_models/pulse_profile_best_discriminator_model.h5')
freq_phase_model = load_model(path_to_models + 'best_retrained_models/freq_phase_best_discriminator_model.h5')
time_phase_model = load_model(path_to_models + 'best_retrained_models/time_phase_best_discriminator_model.h5')

logistic_model = pickle.load(open(path_to_models + 'best_retrained_models/sgan_retrained.pkl', 'rb'))

# Load data (using [:-4] to remove the '.pfd' file extension from the name)
dm_curve_combined_array = [np.load(filename[:-4] + '_dm_curve.npy') for filename in candidate_files]
pulse_profile_combined_array = [np.load(filename[:-4] + '_pulse_profile.npy') for filename in candidate_files]
freq_phase_combined_array = [np.load(filename[:-4] + '_freq_phase.npy') for filename in candidate_files]
time_phase_combined_array = [np.load(filename[:-4] + '_time_phase.npy') for filename in candidate_files]

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
stacked_predictions = np.reshape(stacked_predictions, (len(dm_curve_data),4))

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
