import numpy as np
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from tensorflow import keras
from glob import glob
import sys
import shutil, os
from os.path import isfile, join
from pathlib import Path
from keras.utils.vis_utils import plot_model
from ubc_AI.training import pfddata

home = str(Path.home())

# uncomment any of the next 3 blocks to choose which disciminator to use

'''
# this is for the "best discriminator"
directory = home + "/Documents/sgan/best_retrained_models/"
dm_model_filename = "dm_curve_best_discriminator_model.h5"
time_model_filename = "time_phase_best_discriminator_model.h5"
freq_model_filename = "freq_phase_best_discriminator_model.h5"

'''

# semi supervised model 
directory = home + "/Documents/sgan/semi_supervised_trained_models/"
dm_model_filename = "dm_curve_best_discriminator_model_labelled_50814_unlabelled_265172_trial_4.h5"
time_model_filename = "time_phase_best_discriminator_model_labelled_50814_unlabelled_265172_trial_4.h5"
freq_model_filename = "freq_phase_best_discriminator_model_labelled_50814_unlabelled_265172_trial_4.h5"

'''


#supervised model (pulsar/non-pulsar)
directory = home + "/Documents/sgan/supervised_trained_models/"
dm_model_filename = "best_model_ann_dm_curve_htru_labelled_30000_trial_1.h5"
time_model_filename = "best_model_cnn_time_phase_htru_labelled_30000_trial_1.h5"
freq_model_filename = "best_model_cnn_freq_phase_htru_labelled_30000_trial_1.h5"

'''


# loading models
DM_model = tf.keras.models.load_model(directory+dm_model_filename)
time_model = tf.keras.models.load_model(directory+time_model_filename)
freq_model = tf.keras.models.load_model(directory+freq_model_filename)



# declaring some constants / settings #############################################################
load_from_pfd = False # if this is false, then data are loaded from .npy files
if load_from_pfd:
    load_time_phase = True
    load_freq_phase = True
    load_DM_curve = True
    load_profile = True

classification = "RFI" # can either be "pulsar", "noise", or "RFI"

freq_bins = 48
time_bins = 48
phase_bins = 64
DM_bins = 60

cand_dir = home + "/Desktop/candidates/" # base directory where the candidate types are stored
###################################################################################################

if classification == "pulsar":
    classification_dir = cand_dir + "grade_4/"
elif classification == "RFI":
    classification_dir = cand_dir + "grade_0/"
elif classification == "noise":
    classification_dir = cand_dir + "grade_1/"

# actually loading the files
if load_from_pfd:
    print("Loading from PFD files:")
    
    dirs_to_load = glob(join(classification_dir, "*.pfd"))
    num_cand = len(dirs_to_load)

    time_phase_data = np.zeros((num_cand, time_bins, time_bins, 1))
    freq_phase_data = np.zeros((num_cand, freq_bins, freq_bins, 1))
    profile_data = np.zeros((num_cand, phase_bins, phase_bins, 1))
    DM_curve_data = np.zeros((num_cand, DM_bins, 1))

    # removing any numpy files that currently exist
    print("   - Removing any leftover numpy files: ", end='')
    for f in os.listdir(classification_dir+"numpy/"):
        os.remove(join(classification_dir+"numpy/", f))
    print("done")

    count = 0 # need this for an iterator / place keeper
    for filename in dirs_to_load:
        try:
            data = pfddata(filename)                                   # gets data cube
            
            # time-phase data
            time_phase_data[count, :, :,] = np.reshape(data.getdata(intervals=time_bins),\
                (time_bins, time_bins,1))                              # gets time_phase data from data cube 
                                                                       # (plus does downsizing)
            #time_phase_data[count, :, :] /= np.amax(time_phase_data[count, :, :]) # normalising to [-1,1]
            # don't need to do this type of normalisation because it is already normalised
            time_save_filename = filename[:-4] + "_time_phase.npy"     # gets new filename
            np.save(time_save_filename, time_phase_data[count, :, :])  # saves the data as a .npy file
            shutil.move(time_save_filename, classification_dir + "numpy/") # moving to /numpy subdirectory
            
            # freq-phase data
            freq_phase_data[count, :, :,] = np.reshape(data.getdata(subbands=freq_bins), (freq_bins, freq_bins, 1))
            freq_phase_filename = filename[:-4] + "_freq_phase.npy"
            np.save(freq_phase_filename, freq_phase_data[count, :, :])
            shutil.move(freq_phase_filename, classification_dir + "numpy/")
            
            # DM data
            DM_curve_data[count, :, ] = np.reshape(data.getdata(DMbins=DM_bins), (DM_bins, 1))
            DM_filename = filename[:-4] + "_DM_curve.npy"
            np.save(DM_filename, DM_curve_data[count, :, :])
            shutil.move(DM_filename, classification_dir + "numpy/")

        except ValueError:
            print("Error in the file: {}".format(filename))           # doesn't save any file with an error
                                                                      # and prints out the filename 
        count += 1
    # end for
else: # load directly from .npy files
    print("Loading from npy files:")

    print("   - Loading time-phase data: ", end='')
    time_phase_dirs = glob(join(classification_dir + "numpy/", "*time_phase.npy"))
    time_phase_data = [np.load(filename) for filename in time_phase_dirs]
    time_phase_data = np.array(time_phase_data)
    print("done")

    print("   - Loading freq-phase data: ", end='')
    freq_phase_dirs = glob(join(classification_dir + "numpy/", "*freq_phase.npy"))
    freq_phase_data = [np.load(filename) for filename in freq_phase_dirs]
    freq_phase_data = np.array(freq_phase_data)
    print("done")

    print("   - Loading DM curve data: ", end='')
    DM_curve_dirs = glob(join(classification_dir + "numpy/", "*DM_curve.npy"))
    DM_curve_data = [np.load(filename) for filename in DM_curve_dirs]
    DM_curve_data = np.array(DM_curve_data)
    print("done")
# end if
print("Complete!")

#print(time_phase_data[2,:,:])




def print_stats (predictions, ground_truth, title):

    # predictions = np.rint(predictions)                         # rounds all elements to integer
    # predictions = np.argmax(predictions, axis=1)
    # predictions = np.reshape(predictions, len(predictions))    # reshapes result

    # predictions = np.amax(predictions, axis=1)                 # max element of each row

    print("----------------------------")
    print("{} stats:".format(title))
    
    predictions = predictions[:, 1]                              # gets second column

    if ground_truth==1:
        correct = np.sum(predictions >= 0.5)
    else:
        correct = np.sum(predictions < 0.5)

    mean = np.mean(predictions)
    median = np.median(predictions)
    minimum = predictions.min() #min(predictions)

    print("TOTAL CORRECT = {} / {}".format(correct, len(predictions)))
    print("mean = {}".format(mean))
    print("median = {}".format(median))
    print("minimum = {}".format(minimum))
    print("----------------------------")


'''
print(time_phase_predictions.shape)
for i in range(len(time_phase_predictions)):
    print(time_phase_predictions[i])
'''


print("\n\n\n")
print("PRINTING SUMMARY")
print("")
print("----------------------------")
if classification == "pulsar":
    print("           PULSARS          ")
    ground_truth = 1
elif classification == "RFI":
    print("             RFI            ")
    ground_truth = 0
elif classification == "noise":
    print("           NOISE            ")
    ground_truth = 0
print("----------------------------")

# making predictions
print("Making precitions:")

print("   - Time-phase: ", end='')
time_phase_predictions = time_model.predict(time_phase_data)
print("done")
# print(time_phase_predictions)

print("   - Freq-phase: ", end='')
freq_phase_predictions = freq_model.predict(freq_phase_data)
print("done")

print("   - DM curve: ", end='')
DM_curve_predictions = DM_model.predict(DM_curve_data)
print("done")

print_stats(time_phase_predictions, ground_truth, "time-phase")
print_stats(freq_phase_predictions, ground_truth, "freq-phase")
print_stats(DM_curve_predictions, ground_truth, "DM curve")

'''
print("")
print("----------------------------")
print("           NOISE            ")
print("----------------------------")

# making predictions
print("Making precitions:")

print("   - Time-phase: ", end='')
#print(time_phase_data[1, :, :,].shape)
#time_phase_predictions = [time_model.predict(time_phase_data[ii, :, :,]) for ii in range(len(time_phase_data))]
time_phase_predictions = time_model.predict(time_phase_data)
print("done")

print_stats(time_phase_predictions, 1, "time-phase")

print("")
print("----------------------------")
print("             RFI            ")
print("----------------------------")

print_stats(time_phase_predictions, 1, "time-phase")
'''


# model summary stuff
#print(time_model.summary())
#plot_model(time_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
