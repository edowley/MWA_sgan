###############################################################################
# 
# This file contains the code that will extract candidates and put them into 
# a file system that is supported by the code in extract_pfd_features.py.
# 
# This is assuming that the current candidates are in the grading system that
# is used in the SMART survey (Swainston 2021, et. al). The important things 
# to grasp from this grading system is:
#       - Known pulsars are in a directory called grade_4
#       - Noise candidates are in a directory called grade_0
#       - RFI candidates are in a directory called grade_1
# Once the candidates are in these folders, this script will:
#       1. Count the number of pulsars (I guess should also cound non-pusars
#          just in case there is more non-pulsars than pulsars)
#       2. Copy across the candidates from smallest class (pulsar, non-pulsar)
#          and choose a way to select candidates from the other class
#           - If more non-pulsars, will need to consider how many RFI
#             and noise candidates to include
#           - Probably 50/50 is the optimum ratio to not exceed
#       3. Keep track of the file names of all the candidates and their 
#          respective labels (0 for non-pulsars, 1 for pulsars).
#       4. Copy across the other class candidates and append their file
#          names to the end of the list (including labels)
#       5. Use np.argsort(array) to obtain the indexes of the candidates
#          of the current array if they were to be arranged into alphabetical
#          order. 
#           - This is because we need to sort the class labels also
#       6. Use array.sort() to sort the candidate list into alphabetical 
#          order. Use the idices determined in the previous step to do the
#          same for the labels array.
#       7. Save the labels as 'validation_labels.csv' and candidates in a
#          directory called sample_data
#
# This process should be done for validation set also
#
###############################################################################

import numpy as np
from ubc_AI.training import pfddata
from glob import glob
from os.path import join, isdir
from os import listdir, chdir
from random import shuffle
from math import floor, ceil
import argparse

class NotADirectoryError(Exception):
    pass

def dir_path(string):
    if isdir(string):
        return string
    else:
        raise NotADirectoryError("Directory path is not valid.")

# parse arguments
parser = argparse.ArgumentParser(description='Extract pfd or ar files as numpy array files.')
parser.add_argument('-i', '--input_path', help='Path of candidates', default="/home/isaaccolleran/Desktop/candidates/")
parser.add_argument('-o', '--output', help='Output directory location',  default="/home/isaaccolleran/Documents/sgan/MWA_cands/")
parser.add_argument('-n', '--num_pulsars', help='Numer of pulsars (and also nonpulsars) to be read in', default='237', type=int)
args = parser.parse_args()
path_to_data = args.input_path
output_path = args.output
num_pulsars = args.num_pulsars

# constants / settings
to_shuffle = True
MAX_PULSARS = 237

if num_pulsars > MAX_PULSARS:
    num_pulsars = MAX_PULSARS


def save_npy_from_pfd (directory, pfd_files, save_to_path):
    # loading the data from the pfd file
    data = [pfddata(directory + f) for f in pfd_files]

    # extracting the appropriate plot data from the original data object
    time_phase_data = [file_data.getdata(intervals=48) for file_data in data]
    freq_phase_data = [file_data.getdata(subbands=48) for file_data in data]
    dm_curve_data = [file_data.getdata(DMbins=60) for file_data in data]
    profile_data = [file_data.getdata(intervals=64) for file_data in data]

    # saving each of the plot data to .npy files
    for i in range(len(pfd_files)):
        np.save(output_path + pfd_files[i][:-4] + '_time_phase.npy', time_phase_data[i])
        np.save(output_path + pfd_files[i][:-4] + '_freq_phase.npy', freq_phase_data[i])
        np.save(output_path + pfd_files[i][:-4] + '_dm_curve.npy', dm_curve_data[i])
        np.save(output_path + pfd_files[i][:-4] + '_pulse_profile.npy', profile_data[i])



# getting all the candidate file names
chdir(path_to_data + 'grade_4')
pulsar_filenames = glob('*.pfd')

chdir(path_to_data + 'grade_1')
RFI_filenames = glob('*.pfd')
num_RFI = len(RFI_filenames)

chdir(path_to_data + 'grade_0')
noise_filenames = glob('*.pfd')
num_noise = len(noise_filenames)

if to_shuffle:
    shuffle(RFI_filenames)
    shuffle(noise_filenames)
    shuffle(pulsar_filenames)

candidates = pulsar_filenames[0:num_pulsars]

if num_RFI < num_pulsars/2:
    # include all RFI candidates
    candidates += RFI_filenames
else:
    # include up to half of RFI candidates
    candidates += RFI_filenames[0:(num_pulsars - floor(num_pulsars / 2))]
    num_RFI = len(RFI_filenames[0:(num_pulsars - floor(num_pulsars / 2))])

if num_noise < (num_pulsars - num_RFI):
    # incldude all candidates
    candidates += noise_filenames
else:
    candidates += noise_filenames[0:(num_pulsars - num_RFI)]
    num_noise = len(noise_filenames[0:(num_pulsars - num_RFI)])

# creating the labels 
pulsar_labels = np.ones(num_pulsars)
nonpulsar_labels = np.zeros(num_noise + num_RFI)
# (num_noise + num_RFI should be equal to num_pulsars)

# moving these files to the correct directory


# correctly ordering arrays
order = np.arraysort()







