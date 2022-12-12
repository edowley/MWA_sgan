
# **Murchison Widefield Array Semi-Supervised Generative Adversarial Networks (SGAN).**

This is a forked repo. The original can be found at https://github.com/vishnubk/sgan

**Interpreting MWA_best_retrained_models Directory**

This directory contains 20 models that have been retrained using pulsar candidates produced from the MWA. Attempts 1-10 were produced by one validation set containing 60 candidates, and attempts 11-20 were produced by a validation set that was slightly different. The performance of the second 10 (attempts 11-20) was assessed using a "better" validation set. If I were to put forward a best network of models, I would suggest attempt 20. The saved models within that directory achieved an overall accuracy of 88% on validation data. 

**Training Set**

The training/validation set used to train this network is not yet publicly available. 

**Retraining With Your Own Data**

1. Set up an environment containing all the required packages and software. This includes an up-to-date PRESTO installation on the device (or at least access to the python wrappers within it), and ubc_ai software. The latter can be found at https://github.com/zhuww/ubc_AI. All of the useful bits of code within that link have been written in python2, so it may have to be converted to python3 (or a separate python2 environment could be made).
  - A helpful tip that I would recommend is using anaconda3 to set up and manage the environments. In particular, setting up tensorflow to run on local computer GPUs is made a lot simpler with a conda environment (all you need to do is type "conda install tensorflow-gpu" and it will sort out installing all the required libraries, like cuDNN and cuda)
2. Process the pfd files (saving feature plots as .npy files). You can either use extract_pfd_features.py or choose_candidates.py to do this task. If choosing to use choose_candidates.py, ensure the labelled pfd data is located in the correct directories -- the expected directory structure is outlined in the comment block at the start of the program.
3. Run retrain_sgan.py. If wanting to retrain multiple sgan networks successively, a bash script, retrain_multiple_sgan.sh, has also been included within the repo.
4. Evaluate the performance of the network. compute_sgan_score.py saves a csv file containing the networks predictions for all the files in the validation set. calculate_metrics.py loads up the desired network, makes predictions of the validation set and outputs the accuracy, f1-score, recall and precision. 
  
**Testing/Prediction Using SGAN Model**

As mentioned previously, I would recommend implementing the model found in the directory "MWA_best_retrained_models/attempt_20/" for prediction tasks (on MWA data). The procedure to implement this would be similar to that of retraining the model from scratch. To reiterate:
1. Set up environment
2. Load up model
3. (Optional) Extract feature plots from pfd files and save as npy files. This step is optional because saving the npy files is only important if you are planning on feeding the pfd file/s through the network more than once. It saves significant computation to load up 4 npy files instead of having to extract the data from the encoded pfd file each time running. 
4. Feed the data through the network and obtain predictions. This could be done by using compute_sgan_score.py and modifying the directory of the input data. (Would also need to obtain file names from a source other than validation_labels.csv.)

**New Contributions**

## Docker notes

The [Dockerfile](Dockerfile) can be used to create a Docker image with
```
docker build -t [image_name:tag] .
```

In order to have the GPUs on your system be visible from inside the container, you need to install [nvidia-container-runtime](https://nvidia.github.io/nvidia-container-runtime/) on your machine (i.e. not inside the container), as per the instructions [here](https://docs.docker.com/engine/reference/commandline/run/#access-an-nvidia-gpu).
Then, to add GPUs to the container when running something, use the `--gpus` option, e.g.
```
docker run --gpus all [IMAGE] nvidia-smi
```

Using the Docker image replaces the setup in step 1 of the previous sections.

**Run Image with Test Data**

1. Create a volume called sgan_data and add the test data to it:
```
docker container create --name dummy -v sgan_data:/root hello-world
docker cp ~/SGAN_Test_Data dummy:/root
docker rm dummy
```

2. Use the following command (in this directory) to construct a container from the sgan:1.1 image with sgan_data attached:
```
docker run --gpus all --name sgan_test -it -v $(pwd):/MWA_sgan -v sgan_data:/data -u $(id -u):$(id -g) sgan:1.1 bash
```

To re-open an existing container (e.g. sgan_test), use:
```
docker start [CONTAINER]
docker exec -it [CONTAINER] bash
```

Note that the default directory in the container is currently /code/presto/src, not /MWA_sgan.

## SMART Database
Candidates are currently sourced from https://apps.datacentral.org.au/smart/media/candidates/ using https://apps.datacentral.org.au/smart/candidates/?_export=csv as a reference.
This database will soon be receiving updates to make it more compatible with machine learning applications, which will necessitate changes to the code. In particular, the first half of get_data.py.
(Also note that other scripts, e.g. retrain_sgan.py, refer to columns in the label csv files by their header names, which could be affected by these changes.)
