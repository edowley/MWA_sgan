from keras.utils import to_categorical
from sklearn.ensemble import StackingClassifier
from keras.models import load_model
import time, sys, os, glob
import numpy as np
import argparse, pickle, errno

class NotADirectoryError(Exception):
    pass

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError("Directory path is not valid.")


parser = argparse.ArgumentParser(description='Score pfd files based on SGAN Machine Learning Model')
parser.add_argument('-i', '--input_path', help='Absolute path of input directory', default="/data/SGAN_Test_Data/validation/")
parser.add_argument('-o', '--output', help='Absolute path of output directory',  default="/data/SGAN_Test_Data/")
parser.add_argument('-m', '--models', help='Absolute path of models directory',  default="/MWA_sgan/MWA_best_retrained_models/attempt_20/")
parser.add_argument('-b', '--batch_size', help='No. of pfd files that will be read in one batch', default=1, type=int)
args = parser.parse_args()
path_to_data = args.input_path
output_path = args.output
path_to_models = args.models
batch_size = args.batch_size

dir_path(path_to_data)

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
# classified_results = logistic_model.predict(stacked_predictions) # if you want a classification score
classified_results = logistic_model.predict_proba(stacked_predictions)[:,1] # If you want a regression score
# print(logistic_model.predict_proba(stacked_predictions))
print(classified_results)

with open(output_path+'sgan_ai_score.csv', 'w') as f:
    f.write('Filename,SGAN_score' + '\n') 
    for i in range(len(candidate_files)):
        f.write(basename_candidate_files[i] + ',' + str(classified_results[i]) + '\n')
