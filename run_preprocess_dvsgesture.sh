#!/bin/bash

# Script to run DVSGesture preprocessing
# Uses conda run to execute in the torch environment

echo "Starting DVSGesture preprocessing..."
echo "This will use the 'torch' conda environment"
echo ""

mamba run -n torch python data/preprocess_dvsgesture.py --config configs/config.yaml

echo ""
echo "Preprocessing complete!"
