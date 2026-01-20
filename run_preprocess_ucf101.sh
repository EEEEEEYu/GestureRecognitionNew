#!/bin/bash

# Script to run UCF101-DVS preprocessing
# Uses conda run to execute in the torch environment

echo "Starting UCF101-DVS preprocessing..."
echo "This will use the 'torch' conda environment"
echo ""

mamba run -n torch python data/UCF101_DVS/preprocess.py --config configs/config_ucf101.yaml

echo ""
echo "Preprocessing complete!"
