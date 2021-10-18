import os, pickle, argparse
from sklearn.linear_model import LogisticRegression
from glob import glob
import pandas as pd
import numpy as np
from keras.models import load_model


class NotADirectoryError(Exception):
    pass

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError("Directory path is not valid.")


'''
#Class Labels format
0 -> Non-Pulsar
1 -> Pulsar
-1 -> Unlabelled Candidate
'''
parser = argparse.ArgumentParser(description='Re-train SGAN Machine Learning Model using User input PFD Files')
parser.add_argument('-i', '--input_path', help='Absolute path of Input directory', default="/home/isaaccolleran/Documents/sgan/MWA_cands/")
parser.add_argument('-o', '--output', help='Output path to save model',  default="/home/isaaccolleran/Documents/sgan/MWA_best_retrained_models/")
parser.add_argument('-l', '--labels', help='File with training data classification labels',  default="/home/isaaccolleran/Documents/sgan/MWA_cands/training_labels.csv")

# parsing input arguments
args = parser.parse_args()
path_to_data = args.input_path
output_path = args.output
training_labels_file = args.labels

# loading in labels
label_data = pd.read_csv(training_labels_file)
training_labels = label_data['Classification'].to_numpy()

# getting candidate files
base_pfd_files = label_data['Filename'].to_numpy()
pfd_files = [path_to_data + filename for filename in base_pfd_files]
# with open(path_to_data + "training_labels.csv") as f:
#     pfd_files = [path_to_data + row.split(',')[0] for row in f]
# pfd_files = pfd_files[1:] # first entry is title 


''' Load Data'''
dm_curve_combined_array = [np.load(filename[:-4] + '_dm_curve.npy') for filename in pfd_files]
pulse_profile_combined_array = [np.load(filename[:-4] + '_pulse_profile.npy') for filename in pfd_files]
freq_phase_combined_array = [np.load(filename[:-4] + '_freq_phase.npy') for filename in pfd_files]
time_phase_combined_array = [np.load(filename[:-4] + '_time_phase.npy') for filename in pfd_files]

# the next 2 steps are basically redundant, because they should already be done, but hey they do it so i'm gonna do it too
''' Reshaping the data for the neural-nets to read '''

reshaped_dm_curve = [np.reshape(f,(60,1)) for f in dm_curve_combined_array]
reshaped_pulse_profile = [np.reshape(f,(64,1)) for f in pulse_profile_combined_array]
reshaped_freq_phase = [np.reshape(f,(48,48,1)) for f in freq_phase_combined_array]
reshaped_time_phase = [np.reshape(f,(48,48,1)) for f in time_phase_combined_array]


''' Rescale the data between -1 and +1 '''

dm_curve_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_dm_curve])
pulse_profile_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_pulse_profile])
freq_phase_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_freq_phase])
time_phase_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_time_phase])


''' Loading models'''
labelled_samples = 50814
unlabelled_samples = 265172
attempt_no = 4
freq_phase_model = load_model('semi_supervised_trained_models/freq_phase_best_discriminator_model_labelled_%d_unlabelled_%d_trial_%d.h5'%(labelled_samples, unlabelled_samples,  attempt_no))
time_phase_model = load_model('semi_supervised_trained_models/time_phase_best_discriminator_model_labelled_%d_unlabelled_%d_trial_%d.h5'%(labelled_samples, unlabelled_samples,  attempt_no))
dm_curve_model = load_model('semi_supervised_trained_models/dm_curve_best_discriminator_model_labelled_%d_unlabelled_%d_trial_%d.h5'%(labelled_samples, unlabelled_samples,  attempt_no))
pulse_profile_model = load_model('semi_supervised_trained_models/pulse_profile_best_discriminator_model_labelled_%d_unlabelled_%d_trial_%d.h5'%(labelled_samples, unlabelled_samples,  attempt_no))

# freq_phase_model = load_model('MWA_best_retrained_models/from_scratch/attempt2/freq_phase_best_discriminator_model.h5')
# time_phase_model = load_model('MWA_best_retrained_models/from_scratch/attempt2/time_phase_best_discriminator_model.h5')
# dm_curve_model = load_model('MWA_best_retrained_models/from_scratch/attempt2/dm_curve_best_discriminator_model.h5')
# pulse_profile_model = load_model('MWA_best_retrained_models/from_scratch/attempt2/pulse_profile_best_discriminator_model.h5')

logistic_model = LogisticRegression()

''' Predictions '''
predictions_freq_phase = freq_phase_model.predict([freq_phase_data])
predictions_time_phase = time_phase_model.predict([time_phase_data])
predictions_dm_curve = dm_curve_model.predict([dm_curve_data])
predictions_pulse_profile = pulse_profile_model.predict([pulse_profile_data])

# without argmax - just second value =================
# predictions_freq_phase = predictions_freq_phase[:, 1]
# predictions_time_phase = predictions_time_phase[:, 1]
# predictions_dm_curve = predictions_dm_curve[:, 1]
# predictions_pulse_profile = predictions_pulse_profile[:, 1]


# with argmax ============================================
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

# stacked_results = np.stack((predictions_freq_phase, predictions_time_phase, predictions_dm_curve, predictions_pulse_profile), axis=1)
stacked_results = np.stack((predictions_freq_phase, predictions_dm_curve, predictions_time_phase), axis=1)
stacked_results = np.reshape(stacked_results, (len(predictions_freq_phase), 3))

# fitting the logistic regressor
logistic_model.fit(stacked_results, training_labels)

# saving the results
pickle.dump(logistic_model, open(output_path+'sgan_retrained.pkl', 'wb'))


