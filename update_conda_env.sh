#!/bin/bash

# Data Analysis & Labeling Service - Update Anaconda Environment
echo "Updating Anaconda environment for Data Analysis & Labeling Service..."
echo "=================================================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: Anaconda/Miniconda is not installed or not in PATH"
    exit 1
fi

# Check if environment exists
if ! conda env list | grep -q "data-analysis-labeling"; then
    echo "Conda environment 'data-analysis-labeling' not found."
    echo "Creating new environment..."
    ./setup_conda_env.sh
    exit 0
fi

# Activate conda environment
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate data-analysis-labeling

# Update environment from environment.yml
echo "Updating environment from environment.yml..."
conda env update -f environment.yml

if [ $? -eq 0 ]; then
    echo "✅ Conda environment updated successfully!"
    echo ""
    echo "To run the application:"
    echo "  conda activate data-analysis-labeling"
    echo "  python main.py"
    echo ""
    echo "Or use the start script:"
    echo "  ./start_conda.sh"
else
    echo "❌ Failed to update conda environment"
    exit 1
fi
