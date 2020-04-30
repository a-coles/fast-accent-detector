# Fast Speaker Accent Detection with Deep RL

US vs. UK speaker accent detector made low-latency via reinforcement learning. This repo is a course project for COMP 767 (Reinforcement Learning) with Doina Precup at McGill University, Winter 2020. The team for this project includes:

* Arlie Coles
* Vilayphone Vilaysouk

*NB: Cloning this repo may take a long time/a lot of space, since it contains full datasets needed for reproduction of results.*

## Prerequisites

We recommend [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) for environment creation to run this project. You can install Conda on Linux like this:

```
cd dir/to/install/miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -p dir/to/install/miniconda/miniconda3
./miniconda3/bin/conda init bash
```

The requirements file is included in `requirements.txt`. You can create an environment from this file by running this from the root of the repo:

```
conda create --name rl_env --file requirements.txt
conda activate rl_env
```

## Repo structure

This repo includes the following dirs:

* `datasets/` - contains preprocessed, MFCC-featurized US and UK speaker datasets, in normalized and non-normalized versions. *NB: original audio is not included due to excessive size, but can be supplied upon request.*
* `fast_accent_detector/` - contains the model source code and preprocessing code.
* `models/` - contains the two final saved PyTorch models (baseline and RL models).
* `results/` - stores .csv and graph results, if re-training is desired.
