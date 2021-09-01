import pandas as pd
import numpy as np

def get_files_list(dir_path, data_labels_filename):
    # load in the csv file containing pfd file names and label
    pfd_label_data = pd.read_csv(dir_path + data_labels_filename)

    # splitting up into pfd filename and labels
    base_pfd_files = pfd_label_data['Filename'].to_numpy()
    pfd_labels = pfd_label_data['Classification'].to_numpy()

    # combining base file names with directory path
    pfd_files = [dir_path + filename for filename in base_pfd_files]

    return pfd_files, pfd_labels

def load_feature_datasets(pfd_files):
# this function loads the feature data for all 4 plots from their .npy files
# it will also reshape it to the right size and make sure that all entries are valid

    dm_curve_data = [np.load(filename[:-4] + '_dm_curve.npy') for filename in pfd_files]
    dm_curve_data = [np.reshape(f,(60,1)) for f in dm_curve_data]
    dm_curve_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in dm_curve_data])

    freq_phase_data = [np.load(filename[:-4] + '_freq_phase.npy') for filename in pfd_files]
    freq_phase_data = [np.reshape(f,(48,48,1)) for f in freq_phase_data]
    freq_phase_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in freq_phase_data])
    
    pulse_profile_data = [np.load(filename[:-4] + '_pulse_profile.npy') for filename in pfd_files]
    pulse_profile_data = [np.reshape(f,(64,1)) for f in pulse_profile_data]
    pulse_profile_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in pulse_profile_data])

    time_phase_data = [np.load(filename[:-4] + '_time_phase.npy') for filename in pfd_files]
    time_phase_data = [np.reshape(f,(48,48,1)) for f in time_phase_data]
    time_phase_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in time_phase_data])

    return dm_curve_data, freq_phase_data, pulse_profile_data, time_phase_data