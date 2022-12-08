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

import argparse
from glob import glob
from math import floor
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import sys.exit
from time import time, sleep
from ubc_AI.training import pfddata
from urllib.request import urlretrieve

# constants / settings
NUM_CPUS = mp.cpu_count()
DEFAULT_PULSARS = 4096
DEFAULT_UNLABELLED = 32768
VALIDATION_RATIO = 0.2
N_FEATURES = 4
TRAINING = 0
VALIDATION = 1
UNLABELLED = -1


# parse arguments
parser = argparse.ArgumentParser(description='Download pfd files, label, and extract as numpy array files.')
parser.add_argument('-d', '--directory', help='Directory location to store data',  default="/data/SGAN_Test_Data/")
parser.add_argument('-n', '--num_pulsars', help='Number of pulsars (and also non-pulsars) to be used', default=DEFAULT_PULSARS, type=int)
parser.add_argument('-u', '--num_unlabelled', help='Number of unlabelled candidates to be used', default=DEFAULT_UNLABELLED, type=int)

args = parser.parse_args()
directory = args.directory # will contain 'database.csv', 'labelled/', 'validation/', and 'unlabelled/'
num_pulsars = args.num_pulsars
num_unlabelled = args.num_unlabelled
  

# Check if there is already data in the directory
if os.path.isdir(directory + 'validation/'):
    with os.scandir(directory + 'validation/') as it:
        if any(it):
            print("There appears to already be some data in the target directory.")
            sleep(2)
            print("Please empty the 'labelled/', 'validation/' and 'unlabelled/' subdirectories before proceeding.")
            sleep(3)
            while True:
                cont = input("Continue? (y/n) ")
                if cont == 'y':
                    break
                elif cont == 'n':
                    sys.exit() 


# Make the directory, if it doesn't already exist
os.makedirs(directory, exist_ok=True)

# Download database.csv, if it doesn't already exist, and read as a pandas dataframe
# Contains candidate IDs, PFD URLs, notes, ratings, etc.
database_path = directory + 'database.csv'
if not os.path.isfile(database_path):
    urlretrieve('https://apps.datacentral.org.au/smart/candidates/?_export=csv', database_path)
df = pd.read_csv(database_path)
df = df.set_index('ID')


print(df)


# Create masks for pulsars, noise, RFI and unlabelled candidates in the dataframe
# Only considers pulsars with an ID above 20000 (so the PFD is available)
# Would be nice if there was a boolean column for RFI, rather than relying on notes
labelled_mask = (20000 <= df['ID'].to_numpy()) & ~np.isnan(df['Avg rating'].to_numpy())
pulsar_mask = (df['Avg rating'].to_numpy() >= 4) & labelled_mask
noise_mask = (df['Avg rating'].to_numpy() <= 2) & (np.char.find(df['Notes'].to_numpy(), 'RFI') == -1) & labelled_mask
RFI_mask = (df['Avg rating'].to_numpy() <= 2) & (np.char.find(df['Notes'].to_numpy(), 'RFI') != -1) & labelled_mask
unlabelled_mask = (20000 <= df['ID'].to_numpy()) & np.isnan(df['Avg rating'].to_numpy())

# total_num_pulsars = np.count_nonzero(pulsar_mask)
# total_num_noise = np.count_nonzero(noise_mask)
# total_num_RFI = np.count_nonzero(RFI_mask)
# total_num_unlabelled = np.count_nonzero(unlabelled_mask)

# Dataframes separated by candidate type, containing the candidate ID and Pfd path
all_pulsars = df[pulsar_mask][['ID', 'Pfd path']]
all_noise = df[noise_mask][['ID', 'Pfd path']]
all_RFI = df[RFI_mask][['ID', 'Pfd path']]
all_unlabelled = df[unlablled_mask][['ID', 'Pfd path']]

# Add the labels (1 for pulsar, 0 for non-pulsar)
all_pulsars['Classification'] = 1
all_noise['Classification'] = 0
all_RFI['Classification'] = 0

# The total number of each candidate type available
total_num_pulsars = len(all_pulsars.index)
total_num_noise = len(all_noise.index)
total_num_RFI = len(all_RFI.index)
total_num_unlabelled = len(all_unlabelled.index)

# The number of each candidate type to use
num_pulsars = min(num_pulsars, total_num_pulsars, 2*total_num_noise, 3*total_num_RFI)
num_RFI = min(floor(num_pulsars/2), total_num_RFI)
num_noise = num_pulsars - num_RFI
num_unlabelled = min(num_unlabelled, total_num_unlabelled)

# Randomly sample the required number of each candidate type
# (This is actually an in-place operation)
all_pulsars = all_pulsars.sample(n = num_pulsars, random_state = 1)
all_noise = all_noise.sample(n = num_noise, random_state = 1)
all_RFI = all_RFI.sample(n = num_RFI, random_state = 1)
all_unlabelled = all_unlabelled.sample(n = num_unlabelled, random_state = 1)


# Construct the labelled training and validation sets
num_training_pulsars = floor(num_pulsars * (1 - VALIDATION_RATIO))
num_training_noise = floor(num_RFI * (1 - VALIDATION_RATIO))
num_training_RFI = floor(num_RFI * (1 - VALIDATION_RATIO))

print("Number of training pulsar candidates: " + num_training_pulsars)
print("Number of training noise candidates: " + num_training_noise)
print("Number of training RFI candidates: " + num_training_RFI)
print("Number of unlabelled training candidates: " + num_unlabelled)

training_set = pandas.concat([all_pulsars.iloc[:num_training_pulsars], \
                             all_noise.iloc[:num_training_noise], \
                             all_RFI.iloc[:num_training_RFI]])
validation_set = pandas.concat([all_pulsars.iloc[num_training_pulsars+1:], \
                             all_noise.iloc[num_training_noise+1:], \
                             all_RFI.iloc[num_training_RFI+1:]])



# Extracts pfd files to numpy array files and then deletes the pfd files
def extract_from_pfd(pfd_name):
    if not len(glob(WORKING_LOCATION + pfd_name[:-4] + '*')) == N_FEATURES:
        try:
            data_obj = pfddata(WORKING_LOCATION + pfd_name)
            time_phase_data = data_obj.getdata(intervals=48)
            freq_phase_data = data_obj.getdata(subbands=48)
            dm_curve_data = data_obj.getdata(DMbins=60)
            profile_data = data_obj.getdata(phasebins=64)

            np.save(WORKING_LOCATION + pfd_name[:-4] + '_time_phase.npy', time_phase_data)
            np.save(WORKING_LOCATION + pfd_name[:-4] + '_freq_phase.npy', freq_phase_data)
            np.save(WORKING_LOCATION + pfd_name[:-4] + '_dm_curve.npy', dm_curve_data)
            np.save(WORKING_LOCATION + pfd_name[:-4] + '_pulse_profile.npy', profile_data)

            os.unlink(WORKING_LOCATION + pfd_name)
        except ValueError: 
            print('Extraction failed: (' + pfd_name + ')')

# Executes the extractions in parallel
def parallel_extraction(extraction_list):
    start = time()
    mp.ThreadPool(NUM_CPUS - 1).imap_unordered(extract_from_pfd, extraction_list)
    total_time = time() - start
    print('Extraction time: ' + total_time)

# Downloads pfd files to the specified file paths and returns failed downloads
def download_pfd(pfd_name):
    download_path = 'https://apps.datacentral.org.au/smart/media/candidates/' + pfd_name
    try:
        urlretrieve(download_path, WORKING_LOCATION + pfd_name)
        return None
    except Exception as e:
        print('Download failed: (' + pfd_name + ') ' + e)
        return pfd_name

# Executes the downloads in parallel
def parallel_download(download_list):
    start = time()
    failures = mp.ThreadPool(NUM_CPUS - 1).imap_unordered(download_pfd, download_list)
    total_time = time() - start
    print('Download time: ' + total_time)
    return failures


# Specify the pfd files to be downloaded - currently not needed
# labelled_downloads = [(pfd_name, directory + 'labelled/' + pfd_name) for pfd_name in training_set['Pfd path'].values]
# validation_downloads = [(pfd_name, directory + 'validation/' + pfd_name) for pfd_name in validation_set['Pfd path'].values]
# unlabelled_downloads = [(pfd_name, directory + 'unlabelled/' + pfd_name) for pfd_name in all_unlabelled['Pfd path'].values]


# Make the target directories, if they don't already exist
os.makedirs(directory + 'labelled/', exist_ok=True)
os.makedirs(directory + 'validation/', exist_ok=True)
os.makedirs(directory + 'unlabelled/', exist_ok=True)


# Specify which set is currently being populated
WORKING_LOCATION = directory + 'labelled/'
# Download the pfd files and keep track of failed downloads
labelled_failures = parallel_download(training_set['Pfd path'].values)
# Remove failed downloads from the dataframes (so they match the directory contents)
training_set = training_set[training_set.Pfd_path not in labelled_failures]
# Extract the pfd files to numpy arrays and then delete the pfds
parallel_extraction(training_set['Pfd path'].values)

# Repeat for the other two sets:

WORKING_LOCATION = directory + 'validation/'
validation_failures = parallel_download(validation_set['Pfd path'].values)
validation_set = validation_set[validation_set.Pfd_path not in validation_failures]
parallel_extraction(validation_set['Pfd path'].values)


WORKING_LOCATION = directory + 'unlabelled/'
unlabelled_failures = parallel_download(all_unlabelled['Pfd path'].values)
all_unlabelled = all_unlabelled[all_unlabelled.Pfd_path not in unlabelled_failures]
parallel_extraction(all_unlabelled['Pfd path'].values)


# Write the dataframes to csv files
training_set.to_csv(directory + 'labelled/' + 'training_labels.csv')
validation_set.to_csv(directory + 'validation/' + 'validation_labels.csv')
all_unlabelled.to_csv(directory + 'unlabelled/' + 'unlabelled_labels.csv')

