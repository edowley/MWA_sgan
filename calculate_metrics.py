'''

This is made to calculate the values of metrics for a particular model output (including regressor)
** ONLY FOR VALIDATION SET **

'''

import tensorflow as tf
import pickle
import numpy as np

#######################################################
# Working directory where everything should be stored #
base_dir = '/home/isaaccolleran/Documents/sgan/'      #
#######################################################

''' Step 1. Load the Validation Data '''
from load_MWA_data import get_files_list, load_feature_datasets

# validation files
path_to_validation = base_dir + 'MWA_validation/'
validation_files, validation_labels = get_files_list(path_to_validation, 'validation_labels.csv')
# validation_files, validation_labels = get_files_list(path_to_validation, 'training_labels.csv')

# loading the physical data
validation_dm_curve_data, validation_freq_phase_data, validation_pulse_profile_data, validation_time_phase_data = load_feature_datasets(validation_files)


''' Step 2. Load the Models '''
from keras.models import load_model

# Change this if wanting to load different discriminators
dir_to_model = base_dir + 'MWA_best_retrained_models/attempt_20/'

dm_curve_model = load_model(dir_to_model + 'dm_curve_best_discriminator_model.h5')
time_phase_model = load_model(dir_to_model + 'time_phase_best_discriminator_model.h5')
freq_phase_model = load_model(dir_to_model + 'freq_phase_best_discriminator_model.h5')
pulse_profile_model = load_model(dir_to_model + 'pulse_profile_best_discriminator_model.h5')
logistic_model = pickle.load(open(dir_to_model + 'sgan_retrained.pkl', 'rb'))

# logistic_model = pickle.load(open(base_dir + 'MWA_best_retrained_models/sgan_retrained.pkl', 'rb'))

# dm_curve_model = load_model('semi_supervised_trained_models/dm_curve_best_discriminator_model_labelled_50814_unlabelled_265172_trial_4.h5')
# freq_phase_model = load_model('semi_supervised_trained_models/freq_phase_best_discriminator_model_labelled_50814_unlabelled_265172_trial_4.h5')
# pulse_profile_model = load_model('semi_supervised_trained_models/pulse_profile_best_discriminator_model_labelled_50814_unlabelled_265172_trial_4.h5')
# time_phase_model = load_model('semi_supervised_trained_models/time_phase_best_discriminator_model_labelled_50814_unlabelled_265172_trial_4.h5')
# logistic_model = pickle.load(open('semi_supervised_trained_models/logistic_regression_labelled_50814_unlabelled_265172_trial_4.pkl', 'rb'))


''' Step 3. Feed the Validation Data Through the Models'''
predictions_freq_phase = freq_phase_model.predict([validation_freq_phase_data])
predictions_time_phase = time_phase_model.predict([validation_time_phase_data])
predictions_dm_curve = dm_curve_model.predict([validation_dm_curve_data])
predictions_pulse_profile = pulse_profile_model.predict([validation_pulse_profile_data])

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
# stacked_predictions = np.stack((predictions_freq_phase, predictions_dm_curve, predictions_time_phase), axis=1)
stacked_predictions = np.reshape(stacked_predictions, (len(predictions_pulse_profile),4))
classified_results = logistic_model.predict(stacked_predictions) # if you want a classification score
# classified_results = logistic_model.predict_proba(stacked_predictions)[:,1] # If you want a regression score

''' Step 4. Calculate the Metrics '''
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
# other useful metrics / plots can be produced using this library

# DM CURVE
print('')
print('DM Curve Stats: ')
accuracy = accuracy_score(validation_labels, predictions_dm_curve)
recall = recall_score(validation_labels, predictions_dm_curve)
f1 = f1_score(validation_labels, predictions_dm_curve)
precision = precision_score(validation_labels, predictions_dm_curve)
print('Accuracy = %.3f, F1-score = %.3f | Precision = %.3f, Recall = %.3f'%(accuracy, f1, precision, recall))

# FREQ-PHASE
print('')
print('Freq-Phase Stats: ')
accuracy = accuracy_score(validation_labels, predictions_freq_phase)
recall = recall_score(validation_labels, predictions_freq_phase)
f1 = f1_score(validation_labels, predictions_freq_phase)
precision = precision_score(validation_labels, predictions_freq_phase)
print('Accuracy = %.3f, F1-score = %.3f | Precision = %.3f, Recall = %.3f'%(accuracy, f1, precision, recall))

# PULSE PROFILE
print('')
print('Pulse Profile Stats: ')
accuracy = accuracy_score(validation_labels, predictions_pulse_profile)
recall = recall_score(validation_labels, predictions_pulse_profile)
f1 = f1_score(validation_labels, predictions_pulse_profile)
precision = precision_score(validation_labels, predictions_pulse_profile)
print('Accuracy = %.3f, F1-score = %.3f | Precision = %.3f, Recall = %.3f'%(accuracy, f1, precision, recall))

# TIME-PHASE
print('')
print('Time-Phase Stats: ')
accuracy = accuracy_score(validation_labels, predictions_time_phase)
recall = recall_score(validation_labels, predictions_time_phase)
f1 = f1_score(validation_labels, predictions_time_phase)
precision = precision_score(validation_labels, predictions_time_phase)
print('Accuracy = %.3f, F1-score = %.3f | Precision = %.3f, Recall = %.3f'%(accuracy, f1, precision, recall))

# LOGISTIC REGRESSOR - FINAL CLASSIFICATION
print('')
print('Final Classification: ')
accuracy = accuracy_score(validation_labels, classified_results)
recall = recall_score(validation_labels, classified_results)
f1 = f1_score(validation_labels, classified_results)
precision = precision_score(validation_labels, classified_results)
print('Accuracy = %.3f, F1-score = %.3f | Precision = %.3f, Recall = %.3f'%(accuracy, f1, precision, recall))

