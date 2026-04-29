#!/bin/bash

### TODO: learn2learn fails to compile with 3.13 headers.
### In order to update python, we have to fork it and fix the cython compile issue.
###

echo "This is a WIP script for Python 3.13"


### Simple helper script that replicates the environment using conda

### The environment is called meta-pde

if command -v micromamba &> /dev/null; 
then 
    CONDA="micromamba"
else
    CONDA="conda"
fi


$CONDA create -n inv-env-tmp python==3.11 -y
$CONDA activate inv-env-tmp 

# # Project packages
$CONDA install lightning matplotlib pandas -y
$CONDA install scipy scs 

# # pip pkgs
pip install torch torchvision torchaudio plotly
pip install tqdm h5py wandb hydra-core torch_harmonics moviepy imageio gitpython jaxtyping==0.2.28 neuraloperator==0.3.0
pip install scoringrules pyyaml typeguard==2.13.3
pip install scOT@git+https://github.com/divyam123-EECS-Physics/poseidon_fork@main