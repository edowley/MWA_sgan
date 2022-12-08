###############################################################################
# 
# This file contains code that will select appropriate training and validation sets,
# download and process the required files, and output csv files containing the
# candidate IDs, file names, and labels (if applicable) for each set
# 
#       1. Takes as arguments the desired number of (labelled) pulsars to use,
#          the number of unlabelled candidates to use, and the data directory path
#           - Currently asks the user to remove any existing files from the directory
#       2. Downloads and reads a csv file of the SMART database as a pandas dataframe,
#          and creates masks for pulsars, noise, RFI and unlabelled candidates:
#           - Only candidates with an ID > 20000 are used (else the pfd is unavailable)
#           - Pulsars: Avg rating >= 4
#           - Noise: Avg rating <= 2, no mention of RFI
#           - RFI: Avg rating <= 2, "Notes" column mentions "RFI"
#           - Unlabelled: Avg rating is NaN
#          NB: It would be nice to have an RFI column with a 1 or a 0
#       3. Candidates are pseudo-randomly chosen for the three sets (labelled training,
#          unlabelled training, and validation) based on the available number of each
#          candidate type (pulsar, noise, and RFI), the requested number of pulsars,
#          and the following rules:
#           - The ratio of pulsars to non-pulsars will be 1:1
#           - The ratio of noise to RFI will be between 1:1 (preferred) and 2:1
#           - The ratio of labelled training data to validation data will be 4:1
#           - Any amount of unlabelled training data can be used
#          NB: The random seed is currently fixed for testing purposes
#       4. Each set has a dateframe which holds the candidate IDs and pfd file names,
#          plus labels if applicable (0 for non-pulsars, 1 for pulsars)
#       5. The selected candidate pfd files are downloaded to the 'labelled/',
#          'unlabelled/' or 'validation/' subdirectories, their contents extracted to
#          numpy array files, and then deleted
#           - Downloads are done in parallel, currently CPU only
#           - Ditto for extractions
#           - The sets are processed one at a time, to avoid cluttering with pfd files
#           - Failed downloads are tracked
#          NB: It would be cool to download set 2 in parallel with extracting set 1,
#          but I don't know how to implement that
#       6. Finally, the dataframes for each set are written to csv files for later use
#
###############################################################################

import argparse
from glob import glob
from math import floor
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import sys
from time import time, sleep
from ubc_AI.training import pfddata
from urllib.request import urlretrieve


# constants / settings
NUM_CPUS = mp.cpu_count()
DEFAULT_PULSARS = 32
DEFAULT_UNLABELLED = 64
# DEFAULT_PULSARS = 4096
# DEFAULT_UNLABELLED = 32768
VALIDATION_RATIO = 0.2
N_FEATURES = 4


# parse arguments
parser = argparse.ArgumentParser(description='Download pfd files, label, and extract as numpy array files.')
parser.add_argument('-d', '--directory', help='Directory location to store data',  default="/data/SGAN_Test_Data/")
parser.add_argument('-p', '--num_pulsars', help='Number of pulsars (and also non-pulsars) to be used', default=DEFAULT_PULSARS, type=int)
parser.add_argument('-u', '--num_unlabelled', help='Number of unlabelled candidates to be used', default=DEFAULT_UNLABELLED, type=int)

args = parser.parse_args()
directory = args.directory # will contain 'database.csv', 'labelled/', 'validation/', and 'unlabelled/'
num_pulsars = args.num_pulsars
num_unlabelled = args.num_unlabelled
  

# Check if there is already data in the target sub-directories
# This is just a temporary solution
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
df = df.set_index('ID', inplace=True)


print(df)


# Create masks for pulsars, noise, RFI and unlabelled candidates in the dataframe
# Only considers pulsars with an ID above 20000 (so the PFD is available)
# Would be nice if there was a boolean column for RFI, rather than relying on notes
labelled_mask = (20000 <= df.index) & ~np.isnan(df['Avg rating'].to_numpy())
pulsar_mask = (df['Avg rating'].to_numpy() >= 4) & labelled_mask
noise_mask = (df['Avg rating'].to_numpy() <= 2) & (np.char.find(df['Notes'].to_numpy(), 'RFI') == -1) & labelled_mask
RFI_mask = (df['Avg rating'].to_numpy() <= 2) & (np.char.find(df['Notes'].to_numpy(), 'RFI') != -1) & labelled_mask
unlabelled_mask = (20000 <= df.index) & np.isnan(df['Avg rating'].to_numpy())

# This code is no longer necessary
# total_num_pulsars = np.count_nonzero(pulsar_mask)
# total_num_noise = np.count_nonzero(noise_mask)
# total_num_RFI = np.count_nonzero(RFI_mask)
# total_num_unlabelled = np.count_nonzero(unlabelled_mask)

# Dataframes for each candidate type, containing the pfd path and candidate ID (index)
all_pulsars = df[pulsar_mask][['Pfd path']]
all_noise = df[noise_mask][['Pfd path']]
all_RFI = df[RFI_mask][['Pfd path']]
all_unlabelled = df[unlablled_mask][['Pfd path']]

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
# Based on the selection rules described at the top
num_pulsars = min(num_pulsars, total_num_pulsars, 2*total_num_noise, 3*total_num_RFI)
num_RFI = min(floor(num_pulsars/2), total_num_RFI)
num_noise = num_pulsars - num_RFI
num_unlabelled = min(num_unlabelled, total_num_unlabelled)

# Randomly sample the required number of each candidate type
# (This is apparently an in-place operation, despite what it looks like)
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

