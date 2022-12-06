###############################################################################
# 
# This file contains code that will do stuff. WIP
# 
# This description will be updated eventually:
#       1. Count the number of pulsars (I guess should also count non-pulsars
#          just in case there are more non-pulsars than pulsars)
#       2. Copy across the candidates from smallest class (pulsar, non-pulsar)
#          and choose a way to select candidates from the other class
#           - If more non-pulsars, will need to consider how many RFI
#             and noise candidates to include
#           - Probably 50/50 is the optimum ratio to not exceed
#       3. Keep track of the file names of all the candidates and their 
#          respective labels (0 for non-pulsars, 1 for pulsars).
#       4. Copy across the other class candidates and append their file
#          names to the end of the list (including labels)
#       6. Use array.sort() to sort the candidate list into alphabetical 
#          order. Use the indices determined in the previous step to do the
#          same for the labels array.
#       7. Save the labels as 'validation_labels.csv' and candidates in a
#          directory called MWA_cands or MWA_validation
#
###############################################################################

import numpy as np
from ubc_AI.training import pfddata
from glob import glob
from os.path import join, isdir, basename, isfile
from random import shuffle
from math import floor
import argparse
import pandas as pd
from urllib.request import urlretrieve

# constants / settings
TO_SHUFFLE = True
SAVE_ALL = False
MAX_PULSARS = 4096
MAX_UNLABELLED = 32768
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
parser = argparse.ArgumentParser(description='Extract pfd files as numpy array files.')
parser.add_argument('-i', '--input_path', help='Path of candidates', default="/data/SGAN_Test_Data/")
parser.add_argument('-o', '--output', help='Output directory location',  default="/data/SGAN_Test_Data/")
parser.add_argument('-n', '--num_pulsars', help='Number of pulsars (and also non-pulsars) to be used', default=MAX_PULSARS, type=int)
parser.add_argument('-u', '--num_unlabelled', help='Number of unlabelled candidates to be used', default=MAX_UNLABELLED, type=int)

args = parser.parse_args()
path_to_data = args.input_path
output_path = args.output
num_pulsars = args.num_pulsars
num_unlabelled = args.num_unlabelled

# THIS VARIABLE WILL BE DELETED SHORTLY
dataset_type = 0

dir_path(path_to_data)

if num_pulsars > MAX_PULSARS:
    num_pulsars = MAX_PULSARS
if num_unlabelled > MAX UNLABELLED:
    num_unlabelled = MAX_UNLABELLED


##################################################

# Download database.csv from the SMART database and read it as a pandas dataframe
# Contains candidate IDs, PFD URLs, notes, ratings, etc.
df_path = input_path + 'database.csv'
if not isfile(df_path):
    urlretrieve('https://apps.datacentral.org.au/smart/candidates/?_export=csv', input_path + 'database.csv')
df = pd.read_csv(input_path + 'database.csv')

print(df)

# Create masks for pulsars, noise, RFI and unlabelled candidates in the dataframe
# Only considers pulsars with an ID above 20000 (so the PFD is available)
# Would be nice if there was a boolean column for RFI, rather than relying on notes
labelled_mask = (20000 <= df['ID'].to_numpy()) & ~np.isnan(df['Avg rating'].to_numpy())
pulsar_mask = (df['Avg rating'].to_numpy() >= 4) & labelled_mask
noise_mask = (df['Avg rating'].to_numpy() <= 2) & (np.char.find(df['Notes'].to_numpy(), 'RFI') == -1) & labelled_mask
RFI_mask = (df['Avg rating'].to_numpy() <= 2) & (np.char.find(df['Notes'].to_numpy(), 'RFI') != -1) & labelled_mask
unlabelled_mask = (20000 <= df['ID'].to_numpy()) & np.isnan(df['Avg rating'].to_numpy())

total_num_pulsars = np.count_nonzero(pulsar_mask)
total_num_noise = np.count_nonzero(noise_mask)
total_num_RFI = np.count_nonzero(RFI_mask)
total_num_unlabelled = np.count_nonzero(unlabelled_mask)

# TO-DO:
# Choose 70-80% of pulsars for the training set
# Choose up to half that number of RFI, and fill the remaining with noise
# Set aside all remaining pulsars, noise and RFI for the validation set
# Also place all unlabelled candidates in the unlabelled training set















##################################################




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

'''
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

# saving the candidate name and the label to .csv file
print("Saving training_labels.csv file")
if dataset_type == TRAINING: # training set
    with open(output_path + 'MWA_cands/' + 'training_labels.csv', 'w') as f:
        f.write('Filename,Classification' + '\n') 
        for i in range(len(candidates)):
            f.write(candidates[i] + ',' + str(labels[i]) + '\n')

    # processing candidate names into .npy files for each of the plots
    print("Processing pulsar pfd files")
    save_npy_from_pfd(path_to_data + 'grade_4/', pulsar_filenames, output_path)
    print("Processing RFI pfd files")
    save_npy_from_pfd(path_to_data + 'grade_0/', RFI_filenames, output_path)
    print("Processing noise pfd files")
    save_npy_from_pfd(path_to_data + 'grade_1/', noise_filenames, output_path)

elif dataset_type == VALIDATION:
    with open(output_path + 'MWA_validation/' + 'validation_labels.csv', 'w') as f:
        f.write('Filename,Classification' + '\n') 
        for i in range(len(candidates)):
            f.write(candidates[i] + ',' + str(labels[i]) + '\n')

    # processing candidate names into .npy files for each of the plots
    print("Processing pulsar pfd files")
    save_npy_from_pfd(path_to_data + 'grade_4/validation/', pulsar_filenames, output_path)
    print("Processing RFI pfd files")
    save_npy_from_pfd(path_to_data + 'grade_0/validation/', RFI_filenames, output_path)
    print("Processing noise pfd files")
    save_npy_from_pfd(path_to_data + 'grade_1/validation/', noise_filenames, output_path)

elif dataset_type == UNLABELLED:
    with open(output_path + 'MWA_unlabelled_cands/' + 'training_labels.csv', 'w') as f:
        f.write('Filename,Classification' + '\n') 
        for i in range(len(candidates)):
            f.write(candidates[i] + ',-1\n')

    # processing candidate names into .npy files for each of the plots
    print("Processing unlabelled pfd files")
    save_npy_from_pfd(path_to_data + 'unlabelled/', candidates, output_path)

'''











