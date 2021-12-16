

from load_MWA_data import get_files_list, load_feature_datasets

base_dir = '/home/isaaccolleran/Documents/sgan/'

# labelled training files
path_to_data = base_dir + 'MWA_cands/'
pfd_files, pfd_labels = get_files_list(path_to_data, 'training_labels.csv')

# validation files
path_to_validation = base_dir + 'MWA_validation/'
validation_files, validation_labels = get_files_list(path_to_validation, 'validation_labels.csv')

# unlabelled training files
path_to_unlabelled = base_dir + 'MWA_unlabelled_cands/'
unlabelled_files, unlabelled_labels = get_files_list(path_to_unlabelled, 'training_labels.csv')

# loading the physical data
dm_curve_data, freq_phase_data, pulse_profile_data, time_phase_data = load_feature_datasets(pfd_files)
validation_dm_curve_data, validation_freq_phase_data, validation_pulse_profile_data, validation_time_phase_data = load_feature_datasets(validation_files)
unlabelled_dm_curve_data, unlabelled_freq_phase_data, unlabelled_pulse_profile_data, unlabelled_time_phase_data = load_feature_datasets(unlabelled_files)

# combining labels and data
dm_curve_dataset = [dm_curve_data, pfd_labels]
dm_curve_validation_dataset = [validation_dm_curve_data, validation_labels]
dm_curve_unlabelled_dataset = [unlabelled_dm_curve_data, unlabelled_labels]


from classifiers import Train_SGAN_DM_Curve, Train_SGAN_Freq_Phase, Train_SGAN_Time_Phase, Train_SGAN_Pulse_Profile

batch_size = 16

dm_curve_instance = Train_SGAN_DM_Curve(dm_curve_data, pfd_labels, validation_dm_curve_data, validation_labels, unlabelled_dm_curve_data, unlabelled_labels, batch_size)
pulse_profile_instance = Train_SGAN_Pulse_Profile(pulse_profile_data, pfd_labels, validation_pulse_profile_data, validation_labels, unlabelled_pulse_profile_data, unlabelled_labels, batch_size)
freq_phase_instance = Train_SGAN_Freq_Phase(freq_phase_data, pfd_labels, validation_freq_phase_data, validation_labels, unlabelled_freq_phase_data, unlabelled_labels, batch_size)
time_phase_instance = Train_SGAN_Time_Phase(time_phase_data, pfd_labels, validation_time_phase_data, validation_labels, unlabelled_time_phase_data, unlabelled_labels, batch_size)


# retraining freq_phase model ##################

d_model, c_model = freq_phase_instance.define_discriminator()
generator = freq_phase_instance.define_generator()
gan = freq_phase_instance.define_gan(generator, d_model)
freq_phase_instance.train(generator, d_model, c_model, gan, n_epochs=25)

# d_model, c_model = time_phase_instance.define_discriminator()
# generator = time_phase_instance.define_generator()
# gan = time_phase_instance.define_gan(generator, d_model)
# time_phase_instance.train(generator, d_model, c_model, gan, n_epochs=25)