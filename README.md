# Language-Based Augmentation to Address Shortcut Learning in Object-Goal Navigation
Dennis Hoftijzer, Gertjan Burghouts, Luuk Spreeuwers

This repository contains the code supplement to reproduce the experiments in our paper. It is basically a fork of the [Allenact repo (v0.5.2)](https://github.com/allenai/allenact/tree/v0.5.2).


[Paper link][[1 min video](https://youtu.be/gTxGiogdhP4)]

![This is an image](gifs/Language-Based Augmentation.gif)

We design an experiment for inserting a shortcut bias in the appearance of training environments for ObjectNav. As an example, we associate room types to specific wall colors (e.g., bedrooms with green walls), and observe poor generalization of a [SOTA ObjectNav method](https://github.com/allenai/embodied-clip) to environments where this is not the case (e.g., bedrooms with blue walls). We find that shortcut learning is the root cause: the agent learns to navigate to target objects, by simply searching for the associated wall color of the target object’s room. To solve this, we propose Language-Based Augmentation (L-B).

## Overview
The main additions to the [Allenact repo (v0.5.2)](https://github.com/allenai/allenact/tree/v0.5.2) are:
- **[Experimental configurations:](projects/objectnav_baselines/experiments/procthor)**
  - EmbCLIP (closed-world): [`CW_objectnav_procthor_rgb_clipresnet50gru_ddppo.py`](projects/objectnav_baselines/experiments/procthor/CW_objectnav_procthor_rgb_clipresnet50gru_ddppo.py)
  - EmbCLIP (open_world): [`OW_objectnav_procthor_rgb_clipresnet50gru_ddppo.py`](projects/objectnav_baselines/experiments/procthor/OW_objectnav_procthor_rgb_clipresnet50gru_ddppo.py)
  - EmbCLIP (w/ L-B augmentations): [`LBaug_objectnav_procthor_rgb_clipresnet50gru_ddppo.py`](projects/objectnav_baselines/experiments/procthor/LBaug_objectnav_procthor_rgb_clipresnet50gru_ddppo.py)
  
  _Note:_ For our o.o.d. generalization test we use [EmbCLIP (closed-world)](https://github.com/allenai/embodied-clip). For integrating our L-B augmentations, we use [EmbCLIP (open_world)](https://github.com/allenai/embodied-clip/tree/zeroshot-objectnav).

- **[Dataset:](datasets/ProcTHOR)**
  - In total 25 [ProcTHOR-10k](https://github.com/allenai/procthor-10k) scene layouts, where we have applied our proposed interventions.
  - [Top_downs](datasets/ProcTHOR/Top_downs): Top down view of the houses.
  - [Test scenes](datasets/ProcTHOR/Test) are subdivided according to the number of wall color changes and appropriate target rooms. 
  - Each scene is identified by its index in ProcTHOR-10k e.g., 5902, and a combination of wall colors e.g., LR0_K1_BR2 represents a blue (0) livingroom, a red (1) kitchen and a green (2) bedroom.
    
- **[L-B Augmentation:](allenact_plugins/clip_plugin/clip_preprocessors_openworld.py)**
  - `ClipResNetPreprocessor_LB_Augmented` class.

- **[Procthor plugin:](allenact_plugins/procthor_plugin)**
  - Custom plugin for use of ProcTHOR scenes within Allenact.

- **[scripts.py:](scripts.py)**
  - Scripts for training, evaluation and vizualization.  

## Installation
To install, please follow the [instructions](https://allenact.org/installation/installation-allenact/) for installing the full AllenAct library using conda, but clone this repo instead:

1. Clone this repo:
```
git clone https://github.com/Dennishoftijzer/L-B_Augmentation.git
```
2. Create Conda environment:

   _Note:_ Please install the [conda-libmamba-solver](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) if you are impatient (like me). It solves the environment much faster.
```
cd L-B_Augmentation
export EMBCLIP_ENV_NAME=allenact
export CONDA_BASE="$(dirname $(dirname "${CONDA_EXE}"))"
export PIP_SRC="${CONDA_BASE}/envs/${EMBCLIP_ENV_NAME}/pipsrc"
conda env create --file ./conda/environment-base.yml --name $EMBCLIP_ENV_NAME
conda activate $EMBCLIP_ENV_NAME
```

3. Install additional requirements Allenact plugins and this repo:
```
conda env update --file allenact_plugins/ithor_plugin/extra_environment.yml --name $EMBCLIP_ENV_NAME
conda env update --file allenact_plugins/clip_plugin/extra_environment.yml --name $EMBCLIP_ENV_NAME
conda env update --file allenact_plugins/procthor_plugin/extra_environment.yml --name $EMBCLIP_ENV_NAME
``` 
4. Install Pytorch (the code is known to be compatible with pytorch 2.0.1) with appropriate CUDA version for your machine, e.g., for CUDA version 11.8:
```
conda uninstall pytorch pytorch-mutex torchvision --name $EMBCLIP_ENV_NAME
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
5. Download pretrained CLIP visual encoder:
```
python -c "import clip; clip.load('RN50')"
```

### Headless setup 
If you wish to run AI2-THOR on a machine without an attached display (headless), you will need to start an X server. See this [page](https://allenact.org/installation/installation-framework/) for further instructions. 

### Docker
A docker file will be made available.

## Training
Simply run [scripts.py](scripts.py) with the specified experimental config (run `python scripts.py -h` for help on additional optional arguments):
```
python scripts.py {closed_world_embclip,open_world_embclip,LBaug_embclip}
```
## Evaluation
Append the `--eval` flag:
```
python scripts.py {closed_world_embclip,open_world_embclip,LBaug_embclip} --eval
```

## Visualization
If you would like to vizualize episodes (agent egocentric view videos, trajectories, action probabilities): 
1. Add the `--viz`: flag:
    ```
    python scripts.py {closed_world_embclip,open_world_embclip,LBaug_embclip} --viz
    ```
2. Tensorboard:
    ```
    tensorboard --logdir viz_output/objectnav_procthor/tb
    ```
If you would like to vizualize episodes for different scenes than default:
- Generate episodes using [generate_episodes.py](generate_episodes.py) (run `python generate_episodes.py -h` to see how to specify a scene):
    ```
    python generate_episodes.py
    ```

- Set the [vizualization experimental config](projects/objectnav_baselines/experiments/procthor/viz_objectnav_procthor_rgb_clipresnet50gru_ddppo.py) accordingly:
    - Update `TEST_EPS_DIR` and `SCENE` to the specific scene you want to vizualize episodes for.
    - Set `viz_ep_ids` and `viz_video_ids` to the specific episode ids within the generated json file. 


## Citation
