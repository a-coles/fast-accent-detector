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
	* `config/` - contains .json configurations for model training and testing.
* `models/` - contains the two final saved PyTorch models (baseline and RL models).
* `results/` - stores .csv and graph results, if re-training is desired.


## Running

### Training

*NB: Re-training is not recommended, since it can take an excessively long time and runs the risk of clobbering existing saved models if you are not careful renaming.*

To train the baseline, run:

```
cd fast_accent_detector
python train_baseline.py config/baseline.json --name baseline
```

To train the RL model, run:

```
cd fast_accent_detector
python train_reinforce.py config/reinforce.json --name reinforce
```

If you would like to further train the model, you can provide the argument `--continue_model ../models/reinforce.pt`. Training results will be dumped to `results/`.

### Testing

To test the baseline, run:


```
cd fast_accent_detector
python test_baseline.py ../models/baseline.pt config/baseline.json --name baseline
```

To test the RL model, run:

```
cd fast_accent_detector
python test_reinforce.py ../models/reinforce.pt config/reinforce.json --name reinforce
```
