#!/bin/bash

# Data Analysis & Labeling Service - Anaconda Environment Setup
echo "Setting up Anaconda environment for Data Analysis & Labeling Service..."
echo "================================================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: Anaconda/Miniconda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if environment.yml exists
if [ ! -f "environment.yml" ]; then
    echo "Error: environment.yml not found"
    exit 1
fi

# Create conda environment
echo "Creating conda environment 'data-analysis-labeling'..."
conda env create -f environment.yml

if [ $? -eq 0 ]; then
    echo "✅ Conda environment created successfully!"
    echo ""
    echo "To activate the environment and run the application:"
    echo "  conda activate data-analysis-labeling"
    echo "  python main.py"
    echo ""
    echo "Or use the provided start script:"
    echo "  ./start_conda.sh"
else
    echo "❌ Failed to create conda environment"
    exit 1
fi
