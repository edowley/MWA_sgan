
# **Murchison Widefield Array Semi-Supervised Generative Adversarial Networks (SGAN)**

This is a forked repo. The original can be found at https://github.com/vishnubk/sgan
An intermediary version can be found at https://github.com/isaaccolleran/MWA_sgan


**Training Sets**

This SGAN is designed for use with data from the SMART Pulsar Survey.

**Retraining With Your Own Data**

1. Set up a container by following the instructions in the Docker section below
2. Use create_training_sets_db.py to pseudo-randomly select training and validation sets of a particular size
3. Use download_candidate_data_db.py to download and process the pfd files of all candidates in a particular collection of training sets
4. Use retrain_sgan_db.py to train models on a particular collection of training sets

**Testing/Prediction Using SGAN Model**

5. Use download_model_db.py to download the files for a particular model (if required)
6. Use check_model_performance_db.py to evaluate the performance of a particular model on any validation set (optional)
7. Use compute_sgan_score_db.py to make predictions of any set of candidates using a particular model


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

# Section needs modification

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

**Version information**
The provided Docker image uses:
* Tensorflow 2.11.0-gpu
* Ubuntu 20.04.5 LTS (from the Tensorflow image)
* CUDA 11.8 (compatible with driver versions < 525)
* PRESTO 3.0.1 (may eventually be updated to 4.0)

Potential compatibility issues:
The tensorflow:2.11.0-gpu image uses Ubuntu and CUDA versions that are about two years out of date, so the Dockerfile contains instructions to install CUDA 11.8 as well. This is necessary to support driver versions > 470. In the future, CUDA 11.8 will become similarly out-dated, and the Dockerfile will have to be updated (upgrading to a new tensorflow base image will likely not be sufficient). There is no CUDA 12.0 package available for Ubuntu 20.04 currently, but one may be available by the time this becomes an issue.
It is also necessary to have an appropriate version of nvidia-container-runtime installed locally, as mentioned at the beginning of the Docker notes.

## SMART Database
Pending details...


Clean up: the candidate numpy array files and model files will remain in the attached volume and will need to be manually deleted once the training has been finished.

