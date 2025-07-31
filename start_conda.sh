#!/bin/bash

# Data Analysis & Labeling Service - Anaconda Start Script
echo "Starting Data Analysis & Labeling Service with Anaconda..."
echo "========================================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: Anaconda/Miniconda is not installed or not in PATH"
    exit 1
fi

# Check if environment exists
if ! conda env list | grep -q "data-analysis-labeling"; then
    echo "Conda environment 'data-analysis-labeling' not found."
    echo "Creating environment..."
    ./setup_conda_env.sh
    if [ $? -ne 0 ]; then
        exit 1
    fi
fi

# Activate conda environment
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate data-analysis-labeling

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment"
    exit 1
fi

# Check if required files exist
if [ ! -f "main.py" ]; then
    echo "Error: main.py not found"
    exit 1
fi

# Run the application
echo "Starting the application..."
echo "Web interface will be available at: http://localhost:8050"
echo "Press Ctrl+C to stop the application"
echo ""

python main.py
