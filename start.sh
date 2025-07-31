#!/bin/bash

# Data Analysis & Labeling Service Startup Script

echo "Starting Data Analysis & Labeling Service..."
echo "=============================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if required files exist
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found"
    exit 1
fi

if [ ! -f "main.py" ]; then
    echo "Error: main.py not found"
    exit 1
fi

# Check if virtual environment exists, create if not
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the application
echo "Starting the application..."
echo "Web interface will be available at: http://localhost:8050"
echo "Press Ctrl+C to stop the application"
echo ""

python main.py
