import numpy as np
from ubc_AI.training import pfddata

filename = 'Blind_1265470568_08:22:10.97_-21:55:47.81_DM4.16_ACCEL_0:9_630.01ms_Cand.pfd'
directory = '/home/isaaccolleran/Desktop/candidates/grade_1/'

data_obj = pfddata(directory + filename)
time_phase_data = data_obj.getdata(intervals=48)
freq_phase_data = data_obj.getdata(subbands=48)
dm_curve_data = data_obj.getdata(DMbins=60)
profile_data = data_obj.getdata(phasebins=64)

