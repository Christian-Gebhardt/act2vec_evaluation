#!/bin/bash

# Create environment with required packages
conda env create -f environment.yml

# Activate the created environment
conda activate embeddings4pm

# Check if nvidia-sml is available
nvidia-smi

# Create necessary directory
mkdir -p $CONDA_PREFIX/etc/conda/activate.d

# Setup Tensorflow CudNN support
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

echo "Init commands executed successfully."
