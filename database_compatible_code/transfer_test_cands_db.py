# Code to transfer test candidates to the new database
# Uses a dummy observation and dummy beam (376)

import concurrent.futures as cf
from glob import glob
import json
import numpy as np
from multiprocessing import cpu_count
import pandas as pd
import requests
import sys
from time import time
from urllib.request import urlretrieve

NUM_CPUS = cpu_count()

# Paths
DATABASE_URL = 'https://apps.datacentral.org.au/smart/media/candidates/'
path_to_data = '/data/SGAN_Test_Data/candidates/'
path_to_labels = '/data/SGAN_Test_Data/labels/'

# Absolute paths of all csv files
database_csv_file = f"{path_to_labels}database.csv"
training_labels_file = f"{path_to_labels}training_labels.csv"
validation_labels_file = f"{path_to_labels}validation_labels.csv"
unlabelled_labels_file = f"{path_to_labels}unlabelled_labels.csv"


# Read "ID", "Pfd path", "Notes" and "Avg rating" columns, set "ID" as the index
# Skips the first 6757 rows after the header (no pfd name / different name format)
database_df = pd.read_csv(database_csv_file, header = 0, index_col = 'ID', usecols = ['ID', 'Pfd path', 'Notes', 'Avg rating'], \
                dtype = {'ID': int, 'Pfd path': 'string', 'Notes': 'string', 'Avg rating': float}, \
                skiprows = range(1, 6758), on_bad_lines = 'warn')

# Read the label csv files
training_df = pd.read_csv(training_labels_file, header = 0, index_col = 'ID', on_bad_lines = 'warn')
validation_df = pd.read_csv(validation_labels_file, header = 0, index_col = 'ID', on_bad_lines = 'warn')
unlabelled_df = pd.read_csv(unlabelled_labels_file, header = 0, index_col = 'ID', on_bad_lines = 'warn')

'''
# Keep only the candidates already used as test data
relevant_ids = training_df.index.append(validation_df.index.append(unlabelled_df.index))
database_df = database_df[database_df.index.isin(relevant_ids)]
'''

# Keep only candidates not used previously, with an id 20000-60000, and a rating
labelled_mask = (22000 <= database_df.index) & (database_df.index <= 60000) & ~np.isnan(database_df['Avg rating'].to_numpy(dtype = float))
unused_mask = ~database_df.index.isin(training_df.index) & ~database_df.index.isin(validation_df.index) & ~database_df.index.isin(unlabelled_df.index)
pulsar_mask = (database_df['Avg rating'].to_numpy(dtype = float) >= 4) & labelled_mask & unused_mask
noise_mask = (database_df['Avg rating'].to_numpy(dtype = float) <= 2) & (np.char.find(database_df['Notes'].to_numpy(dtype = 'U'), 'RFI') == -1) & labelled_mask & unused_mask
RFI_mask = (database_df['Avg rating'].to_numpy(dtype = float) <= 2) & (np.char.find(database_df['Notes'].to_numpy(dtype = 'U'), 'RFI') != -1) & labelled_mask & unused_mask

pulsar_count = 0
noise_count = 0
for index in range(len(pulsar_mask)):
    if pulsar_mask[index] == True:
        pulsar_count += 1
        if pulsar_count % 5 != 0:
            pulsar_mask[index] = False
    if noise_mask[index] == True:
        noise_count += 1
        if noise_count % 35 != 0:
            noise_mask[index] = False

print("Number of pulsars found = " + str(np.count_nonzero(pulsar_mask)))
print("Number of noise found = " + str(np.count_nonzero(noise_mask)))
print("Number of RFI found = " + str(np.count_nonzero(RFI_mask)))

# 28548
# 28931

combined_mask = (pulsar_mask | noise_mask | RFI_mask) & (database_df.index != 28548) & (database_df.index != 28931)
database_df = database_df[combined_mask]

# Downloads pfd files from the database to the candidates directory
# Returns False and prints a message if the download fails, otherwise returns True
def download_pfd(pfd_name):
    if len(glob(path_to_data + pfd_name)) == 0:
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


pfd_names = database_df['Pfd path'].to_numpy()

# Download the pfd files
successful_downloads = parallel_download(pfd_names)
database_df = database_df[successful_downloads]

########## New Database Stuff ##########

# Database token
class TokenAuth(requests.auth.AuthBase):
    def __init__(self, token):
        self.token = token

    def __call__(self, r):
        r.headers['Authorization'] = "Token {}".format(self.token)
        return r

# Start authorised session
my_session = requests.session()
my_session.auth = TokenAuth("fagkjfasbnlvasfdfwjf783YDF")


# Creates and posts a new Candidate, returns the autoincremented id
def create_Candidate(filename):
    with open(path_to_data + filename, 'rb') as f:
        my_data = {'beam': 376}
        my_files = {'file': f}
        candidate_json = my_session.post('http://localhost:8000/api/candidates/', data=my_data, files=my_files).json()
    return candidate_json['id']

# Creates the Candidates in parallel (threads)
def parallel_Candidates(file_list):
    start = time()
    candidate_ids = []
    with cf.ThreadPoolExecutor(NUM_CPUS) as executor:
        for result in executor.map(create_Candidate, file_list):
            candidate_ids.append(result)
    total_time = time() - start
    print(f"Candidate creation and upload time: {total_time}")
    return candidate_ids

# Upload the new Candidates and record their autogenerated ids
candidate_ids = parallel_Candidates(pfd_names)


# Retrieve the Candidate ids NOTE the above method doesn't seem to work properly
candidate_ids = []
for pfd in pfd_names:
    my_json = my_session.get(f'http://localhost:8000/api/candidates/?file=candidates/{pfd}').json()
    candidate_ids.append(my_json[0]['id'])
print("Finished getting Candidate ids")


# Create dataframe for the new Ratings
ratings_df = pd.DataFrame()
ratings_df['candidate'] = candidate_ids
ratings_df['rating'] = np.round(database_df['Avg rating'].to_numpy(dtype=float))
ratings_df['user'] = 'edowley'
RFI_mask = np.char.find(database_df['Notes'].to_numpy(dtype='U'), 'RFI') != -1
ratings_df['rfi'] = RFI_mask

ratings_df = ratings_df[~ratings_df['rating'].isnull()]
ratings_df['rating'] = ratings_df['rating'].astype(int)

ratings_json = ratings_df.to_dict(orient='records')

# Upload the new Ratings 
my_session.post('http://localhost:8000/api/ratings/', json=ratings_json)

my_session.close()

print('All done!')