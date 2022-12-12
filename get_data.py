###############################################################################
# 
# This file contains code that will select appropriate training and validation sets,
# download and process the required files, and output csv files containing the
# candidate IDs, file names, and labels (if applicable) for each set
# 
#       1. Takes as arguments the desired number of (labelled) pulsars to use,
#          the number of unlabelled candidates to use, and the data directory path.
#           - Currently asks the user to remove any existing files from the directory
#       2. Downloads and reads a csv file of the SMART database as a pandas dataframe,
#          and creates masks for pulsars, noise, RFI and unlabelled candidates.
#           - Only candidates with an ID > 20000 are used (else the pfd is unavailable)
#           - Pulsars: Avg rating >= 4
#           - Noise: Avg rating <= 2, no mention of RFI
#           - RFI: Avg rating <= 2, "Notes" column mentions "RFI"
#           - Unlabelled: Avg rating is NaN
#          NB: It would be nice to have an RFI column with a 1 or a 0.
#       3. Candidates are pseudo-randomly chosen for the three sets (labelled training,
#          unlabelled training, and validation) based on the available number of each
#          candidate type (pulsar, noise, and RFI), the requested number of pulsars,
#          and the following rules:
#           - The ratio of pulsars to non-pulsars will be 1:1
#           - The ratio of noise to RFI will be between 1:1 (preferred) and 2:1
#           - The ratio of labelled training data to validation data will be 4:1
#           - Any amount of unlabelled training data can be used
#          NB: The random seed is currently fixed for testing purposes.
#       4. Each set has a dateframe which holds the candidate IDs and pfd file names,
#          plus labels (1 for pulsar, 0 for non-pulsar, -1 for unlabelled).
#       5. The selected candidate pfd files are downloaded to the 'labelled/',
#          'unlabelled/' or 'validation/' subdirectories, their contents extracted to
#          numpy array files, and then deleted.
#           - Downloads are done in parallel, currently CPU only, failures are tracked
#           - Ditto for extractions
#           - The sets are processed one at a time, to avoid cluttering with pfd files
#          NB: It should be possible to speed up the extraction phase more.
#       6. Finally, the dataframes for each set are written to csv files for later use.
#
###############################################################################

import argparse
import concurrent.futures as cf
from glob import glob
from math import floor
from multiprocessing import cpu_count
import numpy as np
import os
import pandas as pd
import sys
from time import time, sleep
from ubc_AI.training import pfddata
from urllib.request import urlretrieve


# Constants
NUM_CPUS = cpu_count()
DATABASE_URL = 'https://apps.datacentral.org.au/smart/media/candidates/'
DATABASE_CSV_URL = 'https://apps.datacentral.org.au/smart/candidates/?_export=csv'
DEFAULT_NUM_PULSARS = 32
DEFAULT_NUM_UNLABELLED = 64
DEFAULT_VALIDATION_RATIO = 0.2
N_FEATURES = 4


# Parse arguments
parser = argparse.ArgumentParser(description='Download pfd files, label, and extract as numpy array files.')
parser.add_argument('-d', '--directory', help='Directory location to store data',  default="/data/SGAN_Test_Data/")
parser.add_argument('-p', '--num_pulsars', help='Number of pulsars (and also non-pulsars) to use', default=DEFAULT_NUM_PULSARS, type=int)
parser.add_argument('-u', '--num_unlabelled', help='Number of unlabelled candidates to use', default=DEFAULT_NUM_UNLABELLED, type=int)
parser.add_argument('-v', '--validation_ratio', help='Proportion of labelled candidates to use in the validation set', default=DEFAULT_VALIDATION_RATIO, type=float)

args = parser.parse_args()
path_to_data = args.directory
num_pulsars = args.num_pulsars
num_unlabelled = args.num_unlabelled
validation_ratio = arg.validation_ratio
if (validation_ratio < 0.01) or (validation_ratio > 0.99):
    validation_ratio = DEFAULT_VALIDATION_RATIO
    print("Validation ratio was invalid, using default instead")

# Absolute paths to important files and subdirectories
database_csv_path = path_to_data + 'database.csv'
labelled_data_path = path_to_data + 'labelled/' 
validation_data_path = path_to_data + 'validation/'
unlabelled_data_path = path_to_data + 'unlabelled/'
training_labels_file = labelled_data_path + 'training_labels.csv'
validation_labels_file = validation_data_path + 'validation_labels.csv'
unlabelled_labels_file = unlabelled_data_path + 'unlabelled_labels.csv'


# Makes directories, or checks that they are empty if they already exist (temporary solution)
def directory_setup(path):
    if os.path.isdir(path):
        with os.scandir(path) as it:
            if any(it):
                print("There appear to already be files in " + path)
                sleep(1)
                print("Please empty the 'labelled/', 'validation/' and 'unlabelled/' subdirectories before proceeding.")
                sleep(2)
                while True:
                    cont = input("Continue? (y/n) ")
                    if cont == 'y':
                        break
                    elif cont == 'n':
                        sys.exit()
    else:
       os.makedirs(path)

# Make the target directories, if they don't already exist
# Also checks that the subdirectories are empty (temporary solution)
os.makedirs(path_to_data, exist_ok=True)
directory_setup(labelled_data_path)
directory_setup(validation_data_path)
directory_setup(unlabelled_data_path)

# Download database.csv, if it doesn't already exist
# Contains candidate IDs, pfd URLs, notes, ratings, etc.
if not os.path.isfile(database_csv_path):
    urlretrieve(DATABASE_CSV_URL, database_csv_path)
# Read "ID", "Pfd path", "Notes" and "Avg rating" columns, set "ID" as the index
# Skips the first 6757 rows after the header (no pfd name / different name format)
df = pd.read_csv(database_csv_path, header = 0, index_col = 'ID', usecols = ['ID', 'Pfd path', 'Notes', 'Avg rating'], \
                dtype = {'ID': int, 'Pfd path': 'string', 'Notes': 'string', 'Avg rating': float}, \
                skiprows = range(1, 6758), on_bad_lines = 'warn')


# Create masks for pulsars, noise, RFI and unlabelled candidates in the dataframe
# Only considers pulsars with an ID above 20000 (so the pfd is available)
# Would be nice if there was a boolean column for RFI, rather than relying on notes
labelled_mask = (20000 <= df.index) & ~np.isnan(df['Avg rating'].to_numpy(dtype = float))
pulsar_mask = (df['Avg rating'].to_numpy(dtype = float) >= 4) & labelled_mask
noise_mask = (df['Avg rating'].to_numpy(dtype = float) <= 2) & (np.char.find(df['Notes'].to_numpy(dtype = 'str'), 'RFI') == -1) & labelled_mask
RFI_mask = (df['Avg rating'].to_numpy(dtype = float) <= 2) & (np.char.find(df['Notes'].to_numpy(dtype = 'str'), 'RFI') != -1) & labelled_mask
unlabelled_mask = (20000 <= df.index) & np.isnan(df['Avg rating'].to_numpy(dtype = float))

# Dataframes for each candidate type, containing the pfd name and candidate ID (index)
all_pulsars = df[pulsar_mask][['Pfd path']]
all_noise = df[noise_mask][['Pfd path']]
all_RFI = df[RFI_mask][['Pfd path']]
all_unlabelled = df[unlabelled_mask][['Pfd path']]

# Add the labels (1 for pulsar, 0 for non-pulsar, -1 for unlabelled)
all_pulsars['Classification'] = 1
all_noise['Classification'] = 0
all_RFI['Classification'] = 0
all_unlabelled['Classification'] = -1

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
all_pulsars = all_pulsars.sample(n = num_pulsars, random_state = 1)
all_noise = all_noise.sample(n = num_noise, random_state = 1)
all_RFI = all_RFI.sample(n = num_RFI, random_state = 1)
all_unlabelled = all_unlabelled.sample(n = num_unlabelled, random_state = 1)

# Print the number of pulsar, RFI and noise candidates in the labelled training set,
# plus the total number of candidates in the unlabelled and validation sets
num_training_pulsars = floor(num_pulsars * (1 - VALIDATION_RATIO))
num_training_noise = floor(num_noise * (1 - VALIDATION_RATIO))
num_training_RFI = floor(num_RFI * (1 - VALIDATION_RATIO))
num_validation = num_pulsars + num_noise + num_RFI - num_training_pulsars - num_training_noise - num_training_RFI 

print("Number of training pulsar candidates: " + str(num_training_pulsars))
print("Number of training noise candidates: " + str(num_training_noise))
print("Number of training RFI candidates: " + str(num_training_RFI))
print("Number of unlabelled training candidates: " + str(num_unlabelled))
print("Total number of validation candidates: " + str(num_validation))

# Construct the labelled training and validation sets
# The unlabelled training set is "all_unlabelled" (no changes required)
training_set = pd.concat([all_pulsars.iloc[:num_training_pulsars], \
                             all_noise.iloc[:num_training_noise], \
                             all_RFI.iloc[:num_training_RFI]])
validation_set = pd.concat([all_pulsars.iloc[num_training_pulsars+1:], \
                             all_noise.iloc[num_training_noise+1:], \
                             all_RFI.iloc[num_training_RFI+1:]])


# Downloads pfd files from the DATABASE_URL to the current WORKING_LOCATION directory
# Returns False and prints a message if the download fails, otherwise returns True
def download_pfd(pfd_name):
    try:
        urlretrieve(DATABASE_URL + pfd_name, WORKING_LOCATION + pfd_name)
        return True
    except Exception as e:
        print('Download failed: (' + pfd_name + ') ' + e)
        return False

# Executes the downloads in parallel (threads)
# Returns a mask for the successful downloads and prints the time taken
def parallel_download(download_list):
    start = time()
    successes = []
    with cf.ThreadPoolExecutor(NUM_CPUS) as executor:
        for result in executor.map(download_pfd, download_list):
            successes.append(result)
    total_time = time() - start
    print('Download time: ' + str(total_time))
    return successes

# Extracts pfd files to numpy array files in the WORKING_LOCATION and deletes the pfd files
# Returns False and prints a message if the extraction fails, otherwise returns True
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
            return True
        except ValueError: 
            print('Extraction failed: (' + pfd_name + ')')
             # If the extraction fails, delete the pfd anyway
            os.unlink(WORKING_LOCATION + pfd_name)
            return False

# Executes the extractions in parallel (threads)
# Returns a mask for the successful extractions and prints the time taken
def parallel_extraction(extraction_list):
    start = time()
    successes = []
    with cf.ThreadPoolExecutor(NUM_CPUS) as executor:
        for result in executor.map(extract_from_pfd, extraction_list):
            successes.append(result)
    total_time = time() - start
    print('Extraction time: ' + str(total_time))
    return successes


# Specify which set is currently being populated
WORKING_LOCATION = labelled_data_path
print("Starting work on the labelled training set...")
# Download the pfd files and keep track of failed downloads
download_successes = parallel_download(training_set['Pfd path'].values)
# Remove failed downloads from the dataframe (so it matches the directory contents)
training_set = training_set[download_successes]
# Extract the pfds to numpy arrays, delete the pfds, and track failed extractions
extraction_successes = parallel_extraction(training_set['Pfd path'].values)
# Remove failed extractions from the dataframe
training_set = training_set[extraction_successes]

# Repeat for the other two sets:

WORKING_LOCATION = validation_data_path
print("Starting work on the validation set...")
download_successes = parallel_download(validation_set['Pfd path'].values)
validation_set = validation_set[download_successes]
extraction_successes = parallel_extraction(validation_set['Pfd path'].values)
validation_set = validation_set[extraction_successes]


WORKING_LOCATION = unlabelled_data_path
print("Starting work on the unlabelled training set...")
download_successes = parallel_download(all_unlabelled['Pfd path'].values)
all_unlabelled = all_unlabelled[download_successes]
extraction_successes = parallel_extraction(all_unlabelled['Pfd path'].values)
all_unlabelled = all_unlabelled[extraction_successes]


# Write the dataframes to csv files
training_set.to_csv(training_labels_file)
validation_set.to_csv(validation_labels_file)
all_unlabelled.to_csv(unlabelled_labels_file)

print("All done!")