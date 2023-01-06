###############################################################################
# 
# This file contains code that will select appropriate training and validation sets,
# download and process the required files, and output csv files containing the
# candidate IDs, file names, and labels (if applicable) for each set
# 
#       1. Takes as arguments the desired number of (labelled) pulsars to use,
#          the number of unlabelled candidates to use, and the data directory path.
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
#          plus labels if applicable (1 for pulsar, 0 for non-pulsar).
#       5. The selected candidate pfd files are downloaded to the designated candidates
#          directory, their contents extracted to numpy array files, and then deleted.
#           - Downloads are done in parallel, currently CPU only, failures are tracked
#           - Ditto for extractions
#           - The sets are processed one at a time, to avoid cluttering with pfd files
#           - If a candidate's files are already present, download/extraction is skipped
#          NB: It should be possible to speed up the extraction phase more.
#       6. The dataframes for each set are written to csv files in the designated labels directory.
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
parser.add_argument('-c', '--candidates_path', help='Absolute path of output directory for candidate data', default='/data/SGAN_Test_Data/candidates/')
parser.add_argument('-l', '--labels_path', help='Absolute path of output directory for label csv files',  default='/data/SGAN_Test_Data/labels/')
parser.add_argument('-p', '--num_pulsars', help='Number of pulsars (and also non-pulsars) to use', default=DEFAULT_NUM_PULSARS, type=int)
parser.add_argument('-u', '--num_unlabelled', help='Number of unlabelled candidates to use', default=DEFAULT_NUM_UNLABELLED, type=int)
parser.add_argument('-v', '--validation_ratio', help='Proportion of labelled candidates to use in the validation set', default=DEFAULT_VALIDATION_RATIO, type=float)
parser.add_argument('-x', '--set_ID', help='Identifier to append to the label file names for these sets', default="", type=str)

args = parser.parse_args()
path_to_data = args.candidates_path
path_to_labels = args.labels_path
num_pulsars = args.num_pulsars
num_unlabelled = args.num_unlabelled
validation_ratio = args.validation_ratio
set_ID = args.set_ID

# Absolute paths to label csv files and the complete database csv file
database_csv_file = path_to_labels + 'database.csv'
training_labels_file = f"{path_to_labels}training_labels{set_ID}.csv"
validation_labels_file = f"{path_to_labels}validation_labels{set_ID}.csv"
unlabelled_labels_file = f"{path_to_labels}unlabelled_labels{set_ID}.csv"

# Make the target directories, if they don't already exist
os.makedirs(path_to_data, exist_ok=True)
os.makedirs(path_to_labels, exist_ok=True)

# Download database.csv, if it doesn't already exist
# Contains candidate IDs, pfd URLs, notes, ratings, etc.
if not os.path.isfile(database_csv_file):
    urlretrieve(DATABASE_CSV_URL, database_csv_file)

# Read "ID", "Pfd path", "Notes" and "Avg rating" columns, set "ID" as the index
# Skips the first 6757 rows after the header (no pfd name / different name format)
df = pd.read_csv(database_csv_file, header = 0, index_col = 'ID', usecols = ['ID', 'Pfd path', 'Notes', 'Avg rating'], \
                dtype = {'ID': int, 'Pfd path': 'string', 'Notes': 'string', 'Avg rating': float}, \
                skiprows = range(1, 6758), on_bad_lines = 'warn')


# Create masks for pulsars, noise, RFI and unlabelled candidates in the dataframe
# Only considers pulsars with an ID above 20000 (so the pfd is available)
# Would be nice if there was a boolean column for RFI, rather than relying on notes
labelled_mask = (20000 <= df.index) & ~np.isnan(df['Avg rating'].to_numpy(dtype = float))
pulsar_mask = (df['Avg rating'].to_numpy(dtype = float) >= 4) & labelled_mask
noise_mask = (df['Avg rating'].to_numpy(dtype = float) <= 2) & (np.char.find(df['Notes'].to_numpy(dtype = 'U'), 'RFI') == -1) & labelled_mask
RFI_mask = (df['Avg rating'].to_numpy(dtype = float) <= 2) & (np.char.find(df['Notes'].to_numpy(dtype = 'U'), 'RFI') != -1) & labelled_mask
unlabelled_mask = (20000 <= df.index) & np.isnan(df['Avg rating'].to_numpy(dtype = float))

# Dataframes for each candidate type, containing the pfd name and candidate ID (index)
all_pulsars = df[pulsar_mask][['Pfd path']]
all_noise = df[noise_mask][['Pfd path']]
all_RFI = df[RFI_mask][['Pfd path']]
all_unlabelled = df[unlabelled_mask][['Pfd path']]

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
all_pulsars = all_pulsars.sample(n = num_pulsars, random_state = 1)
all_noise = all_noise.sample(n = num_noise, random_state = 1)
all_RFI = all_RFI.sample(n = num_RFI, random_state = 1)
all_unlabelled = all_unlabelled.sample(n = num_unlabelled, random_state = 1)

# Print the number of pulsar, RFI and noise candidates in the labelled training set,
# plus the total number of candidates in the unlabelled and validation sets
num_training_pulsars = floor(num_pulsars * (1 - validation_ratio))
num_training_noise = floor(num_noise * (1 - validation_ratio))
num_training_RFI = floor(num_RFI * (1 - validation_ratio))
num_validation = num_pulsars + num_noise + num_RFI - num_training_pulsars - num_training_noise - num_training_RFI 

print(f"Number of training pulsar candidates: {num_training_pulsars}")
print(f"Number of training noise candidates: {num_training_noise}")
print(f"Number of training RFI candidates: {num_training_RFI}")
print(f"Number of unlabelled training candidates: {num_unlabelled}")
print(f"Total number of validation candidates: {num_validation}")

# Construct the labelled training and validation sets
# The unlabelled training set is "all_unlabelled" (no changes required)
training_set = pd.concat([all_pulsars.iloc[:num_training_pulsars], \
                             all_noise.iloc[:num_training_noise], \
                             all_RFI.iloc[:num_training_RFI]])
validation_set = pd.concat([all_pulsars.iloc[num_training_pulsars+1:], \
                             all_noise.iloc[num_training_noise+1:], \
                             all_RFI.iloc[num_training_RFI+1:]])


# Downloads pfd files from the DATABASE_URL to the candidates directory
# Returns False and prints a message if the download fails, otherwise returns True
def download_pfd(pfd_name):
    if len(glob(path_to_data + pfd_name[:-4] + '*')) == 0:
        try:
            urlretrieve(DATABASE_URL + pfd_name, path_to_data + pfd_name)
            return True
        except Exception as e:
            print(f"Download failed: {pfd_name}, {e}")
            return False
    else:
        # If the target pfd file or associated numpy files already exist, download is skipped
        return True

# Executes the downloads in parallel (threads)
# Returns a mask for the successful downloads and prints the time taken
def parallel_download(download_list):
    start = time()
    successes = []
    with cf.ThreadPoolExecutor(NUM_CPUS) as executor:
        for result in executor.map(download_pfd, download_list):
            successes.append(result)
    total_time = time() - start
    print(f"Download time: {total_time}")
    return successes

# Extracts pfd files to numpy array files in the candidates directory and deletes the pfd files
# Returns False and prints a message if the extraction fails, otherwise returns True
def extract_from_pfd(pfd_name):
    if not len(glob(path_to_data + pfd_name[:-4] + '*')) == N_FEATURES:
        try:
            data_obj = pfddata(path_to_data + pfd_name)
            time_phase_data = data_obj.getdata(intervals=48)
            freq_phase_data = data_obj.getdata(subbands=48)
            dm_curve_data = data_obj.getdata(DMbins=60)
            profile_data = data_obj.getdata(phasebins=64)

            np.save(path_to_data + pfd_name[:-4] + '_time_phase.npy', time_phase_data)
            np.save(path_to_data + pfd_name[:-4] + '_freq_phase.npy', freq_phase_data)
            np.save(path_to_data + pfd_name[:-4] + '_dm_curve.npy', dm_curve_data)
            np.save(path_to_data + pfd_name[:-4] + '_pulse_profile.npy', profile_data)

            os.unlink(path_to_data + pfd_name)
            return True
        except ValueError: 
            print(f"Extraction failed: {pfd_name}")
             # If the extraction fails, delete the pfd anyway
            os.unlink(path_to_data + pfd_name)
            return False
    else:
        # If the numpy array files already exist, extraction is skipped
        return True

# Executes the extractions in parallel (threads)
# Returns a mask for the successful extractions and prints the time taken
def parallel_extraction(extraction_list):
    start = time()
    successes = []
    with cf.ThreadPoolExecutor(NUM_CPUS) as executor:
        for result in executor.map(extract_from_pfd, extraction_list):
            successes.append(result)
    total_time = time() - start
    print(f"Extraction time: {total_time}")
    return successes


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

print("Starting work on the validation set...")
download_successes = parallel_download(validation_set['Pfd path'].values)
validation_set = validation_set[download_successes]
extraction_successes = parallel_extraction(validation_set['Pfd path'].values)
validation_set = validation_set[extraction_successes]

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