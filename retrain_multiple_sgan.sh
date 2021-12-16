#!/bin/bash
source ~/env/bin/activate.sh # activating environment
cd ~/Documents/sgan/ # navigating to directory

n_runs=20 # number of times to retrain
i=19
# for ((i=1; i<=${n_runs}; i++)); do
while [ "$i" -le ${n_runs} ]; do
    python retrain_sgan.py > output.log
    python compute_sgan_score.py

    mv output.log MWA_best_retrained_models/
    mv sgan_ai_score.csv MWA_best_retrained_models/
    cd MWA_best_retrained_models/

    dir="attempt_${i}"
    mkdir ${dir}

    # moving everything into the new directory
    mv output.log ${dir}/
    mv sgan_ai_score.csv ${dir}/
    mv dm_curve_best_discriminator_model.h5 ${dir}/
    mv dm_curve_best_generator_model.h5 ${dir}/
    mv freq_phase_best_discriminator_model.h5 ${dir}/
    mv freq_phase_best_generator_model.h5 ${dir}/
    mv time_phase_best_discriminator_model.h5 ${dir}/
    mv time_phase_best_generator_model.h5 ${dir}/
    mv pulse_profile_best_discriminator_model.h5 ${dir}/
    mv pulse_profile_best_generator_model.h5 ${dir}/
    mv sgan_retrained.pkl ${dir}/
    mv ../training_logs/dm_curve.png ${dir}/
    mv ../training_logs/freq_phase.png ${dir}/
    mv ../training_logs/pulse_profile.png ${dir}/
    mv ../training_logs/time_phase.png ${dir}/

    # going back to sgan directory
    cd ~/Documents/sgan/

    i=$(( i + 1 ))
done
