###############################################################################
# 
# This file contains the code that will extract candidates and put them into 
# a file system that is supported by the code in extract_pfd_features.py.
# 
# This is assuming that the current candidates are in the grading system that
# is used in the SMART survey (Swainston 2021, et. al). The important things 
# to grasp from this grading system is:
#       - Known pulsars are in a directory called grade_4
#       - Noise candidates are in a directory called grade_1
#       - RFI candidates are in a directory called grade_0
# Once the candidates are in these folders, this script will:
#       1. Count the number of pulsars (I guess should also count non-pusars
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
#          order. THIS IS NOT NEEDED BECAUSE WHEN SAVING LABELS YOU ALSO SHOULD 
#          SAVE THE NAME OF THE CANDIDATE FILE
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
from os.path import join, isdir, basename, isfile
from random import shuffle
from math import floor
import argparse

# constants / settings
TO_SHUFFLE = True
SAVE_ALL = False
# MAX_PULSARS = 207
MAX_PULSARS = 225
N_FEATURES = 4
TRAINING = 0
VALIDATION = 1
UNLABELLED = -1

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
parser.add_argument('-n', '--num_pulsars', help='Numer of pulsars (and also nonpulsars) to be read in', default=MAX_PULSARS, type=int)
parser.add_argument('-c', '--candidates', help='Type of candidate set to load. 0 for training set, 1 for validation set, -1 for unlabelled training set.', default=0, type=int)

args = parser.parse_args()
path_to_data = args.input_path
output_path = args.output
num_pulsars = args.num_pulsars
dataset_type = args.candidates

if num_pulsars > MAX_PULSARS:
    num_pulsars = MAX_PULSARS

def save_npy_from_pfd (directory, pfd_files, save_to_path):
    # loading the data from the pfd file

    ''' Processing a single candidate at a time '''
    for i, f in enumerate(pfd_files):
        # print(i, f, len(glob(save_to_path+f[:-4]+'*')))
        if not len(glob(save_to_path + f[:-4] + '*')) == N_FEATURES:
            try:
                data_obj = pfddata(directory + f)
                time_phase_data = data_obj.getdata(intervals=48)
                freq_phase_data = data_obj.getdata(subbands=48)
                dm_curve_data = data_obj.getdata(DMbins=60)
                profile_data = data_obj.getdata(phasebins=64)

                np.save(save_to_path + f[:-4] + '_time_phase.npy', time_phase_data)
                np.save(save_to_path + f[:-4] + '_freq_phase.npy', freq_phase_data)
                np.save(save_to_path + f[:-4] + '_dm_curve.npy', dm_curve_data)
                np.save(save_to_path + f[:-4] + '_pulse_profile.npy', profile_data)
            except ValueError: 
                print(f)
        # else:
            # print('{} skipped'.format(i))
        # else it has already been processed, so don't resave
        

        if (i == int(len(pfd_files) / 2)):
            print(' ... 50% done!')


    ''' This way will only work when you have access large amounts of memory storage '''
    # data = [pfddata(directory + f) for f in pfd_files]

    # # extracting the appropriate plot data from the original data object
    # time_phase_data = [file_data.getdata(intervals=48) for file_data in data]
    # # print(len(time_phase_data[1]))
    # freq_phase_data = [file_data.getdata(subbands=48) for file_data in data]
    # dm_curve_data = [file_data.getdata(DMbins=60) for file_data in data]
    # profile_data = [file_data.getdata(phasebins=64) for file_data in data]

    # # saving each of the plot data to .npy files
    # for i in range(len(pfd_files)):
    #     np.save(output_path + pfd_files[i][:-4] + '_time_phase.npy', time_phase_data[i])
    #     np.save(output_path + pfd_files[i][:-4] + '_freq_phase.npy', freq_phase_data[i])
    #     np.save(output_path + pfd_files[i][:-4] + '_dm_curve.npy', dm_curve_data[i])
    #     np.save(output_path + pfd_files[i][:-4] + '_pulse_profile.npy', profile_data[i])



# getting all the candidate file names
if dataset_type == VALIDATION: 
# the validation set files are in a subdirectory of the training set files so we have to account for that
    pulsar_filenames = glob(path_to_data + 'grade_4/validation/*.pfd')
    RFI_filenames = glob(path_to_data + 'grade_0/validation/*.pfd')
    noise_filenames = glob(path_to_data + 'grade_1/validation/*.pfd')

elif dataset_type == TRAINING:
    pulsar_filenames = glob(path_to_data + 'grade_4/*.pfd')
    RFI_filenames = glob(path_to_data + 'grade_0/*.pfd')
    noise_filenames = glob(path_to_data + 'grade_1/*.pfd')
elif dataset_type == UNLABELLED:
    pulsar_filenames = glob(path_to_data + 'unlabelled/*.pfd')
    RFI_filenames = []
    noise_filenames = []
    print(noise_filenames)
    
pulsar_filenames = [basename(filename) for filename in pulsar_filenames]
pulsar_filenames = sorted(pulsar_filenames)
num_pulsars = len(pulsar_filenames)

RFI_filenames = [basename(filename) for filename in RFI_filenames]
RFI_filenames = sorted(RFI_filenames)
num_RFI = len(RFI_filenames)

noise_filenames = [basename(filename) for filename in noise_filenames]
noise_filenames = sorted(noise_filenames)
num_noise = len(noise_filenames)

if TO_SHUFFLE:
    shuffle(RFI_filenames)
    shuffle(noise_filenames)
    shuffle(pulsar_filenames)

candidates = []
if not SAVE_ALL:
    candidates = pulsar_filenames[0:num_pulsars]

    # handles the instance where the user supplies a -n (num_pulsars) value that is less
    # than the MAX_PULSARS value.
    if num_pulsars < MAX_PULSARS:
        pulsar_filenames = pulsar_filenames[0:num_pulsars]

    # This is here because there are a small number of RFI candidates (comparative to noise).
    # Therefore, when the user supplies an n value small enough, we will have a 50/50 split
    # for the nonpulsar candidates (noise / RFI). 
    if not num_RFI < num_pulsars/2:
        # include up to half of RFI candidates
        RFI_filenames = RFI_filenames[0:(num_pulsars - floor(num_pulsars / 2))]
        num_RFI = len(RFI_filenames)

    num_noise = num_pulsars - num_RFI
    noise_filenames = noise_filenames[0:num_noise]
    num_noise = len(noise_filenames)


print ("pulsars: {}, RFI: {}, noise: {}".format(num_pulsars, num_RFI, num_noise))
candidates += RFI_filenames + noise_filenames

# creating the labels 
pulsar_labels = [1] * num_pulsars
nonpulsar_labels = [0] * (num_noise + num_RFI)
# (num_noise + num_RFI should be equal to num_pulsars)
labels = pulsar_labels + nonpulsar_labels

# saving the canddiate name and the label to .csv file
print("Saving training_labels.csv file")
if dataset_type == TRAINING: # training set
    with open(output_path+'training_labels.csv', 'w') as f:
        f.write('Filename,Classification' + '\n') 
        for i in range(len(candidates)):
            f.write(candidates[i] + ',' + str(labels[i]) + '\n')

    # processing candidate names into .npy files for each of the plots
    print("Processing pulsar pfd files")
    save_npy_from_pfd(path_to_data+'grade_4/', pulsar_filenames, output_path)
    print("Processing RFI pfd files")
    save_npy_from_pfd(path_to_data+'grade_0/', RFI_filenames, output_path)
    print("Processing noise pfd files")
    save_npy_from_pfd(path_to_data+'grade_1/', noise_filenames, output_path)

elif dataset_type == VALIDATION:
    output_path = '/home/isaaccolleran/Documents/sgan/MWA_validation/'
    with open(output_path + 'validation_labels.csv', 'w') as f:
        f.write('Filename,Classification' + '\n') 
        for i in range(len(candidates)):
            f.write(candidates[i] + ',' + str(labels[i]) + '\n')

    # processing candidate names into .npy files for each of the plots
    print("Processing pulsar pfd files")
    save_npy_from_pfd(path_to_data+'grade_4/validation/', pulsar_filenames, output_path)
    print("Processing RFI pfd files")
    save_npy_from_pfd(path_to_data+'grade_0/validation/', RFI_filenames, output_path)
    print("Processing noise pfd files")
    save_npy_from_pfd(path_to_data+'grade_1/validation/', noise_filenames, output_path)

elif dataset_type == UNLABELLED:
    output_path = '/home/isaaccolleran/Documents/sgan/MWA_unlabelled_cands/'
    with open(output_path + 'training_labels.csv', 'w') as f:
        f.write('Filename,Classification' + '\n') 
        for i in range(len(candidates)):
            f.write(candidates[i] + ',-1\n')

    # processing candidate names into .npy files for each of the plots
    print("Processing unlabelled pfd files")
    save_npy_from_pfd(path_to_data+'unlabelled/', candidates, output_path)













