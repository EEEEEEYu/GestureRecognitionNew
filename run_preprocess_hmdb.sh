#!/bin/bash

# Script to run HMDB-DVS preprocessing
# Uses conda run to execute in the torch environment

echo "Starting HMDB-DVS preprocessing..."
echo "This will use the 'torch' conda environment"
echo ""

mamba run -n torch python data/HMDB/preprocess.py --config configs/config_hmdb.yaml

echo ""
echo "Preprocessing complete!"
