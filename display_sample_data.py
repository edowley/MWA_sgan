from ubc_AI.training import pfddata
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import sys
import shutil, os
from os.path import isfile, join
from os import chdir

# cwd = os.getcwd() + "/sample_data/"

# cwd = '/home/isaaccolleran/Desktop/candidates/grade_4/'
# cwd = '/home/isaaccolleran/Desktop/candidates/grade_0/'
cwd = '/home/isaaccolleran/Desktop/candidates/grade_1/'

# cwd = '/home/isaaccolleran/Documents/sgan/sample_data/'
# cwd = '/home/isaaccolleran/Documents/sgan/validation_data/'
# cwd = '/home/isaaccolleran/Documents/sgan/unlabelled_data/'
dirs = glob(join(cwd, "*.pfd"))

print('Displaying {} Candidates'.format(len(dirs)))

def plotting (subbands, subints, profile, dm_curve, title):
    fig = plt.figure(figsize=(10, 11))

    ax1 = plt.subplot(2, 2, 1)
    subints = subints.reshape((48, 48))
    plt.imshow(subints, cmap='gray')
    ax1.title.set_text('subints')
    ax1.set_ylabel('time')
    
    ax2 = plt.subplot(2, 2, 2)
    subbands = subbands.reshape((48, 48))
    plt.imshow(subbands, cmap='gray')
    ax2.title.set_text('subband')
    ax2.set_ylabel('frequency')
    
    ax3 = plt.subplot(2, 2, 3)
    plt.plot(profile)
    ax3.title.set_text('profile')
    
    ax4 = plt.subplot(2, 2, 4)
    plt.plot(dm_curve)
    ax4.title.set_text('dm curve')
    
    fig.suptitle(title)    
    # plt.show()
    chdir('/home/isaaccolleran/Documents/sgan/cand_pngs/')
    plt.savefig(title +'.png')
# end def

for f in dirs:
    data = pfddata(f)

    time_phase_data = data.getdata(intervals=48)
    freq_phase_data = data.getdata(subbands=48)
    pulse_profile_data = data.getdata(phasebins=64)
    dm_curve_data = data.getdata(DMbins=60)

    plotting(freq_phase_data, time_phase_data, pulse_profile_data, dm_curve_data, f[51:-4])


