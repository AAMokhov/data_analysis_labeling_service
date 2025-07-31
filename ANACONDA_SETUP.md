# Anaconda Setup Guide for Data Analysis & Labeling Service

## 🐍 Why Use Anaconda?

Anaconda provides several advantages for this project:
- **Isolated Environment**: Prevents conflicts with system Python packages
- **Optimized Packages**: Pre-compiled scientific computing packages
- **Easy Management**: Simple environment creation and dependency management
- **Cross-platform**: Works consistently across Windows, macOS, and Linux

## 🚀 Quick Start with Anaconda

### Prerequisites
- **Anaconda** or **Miniconda** installed
  - Download from: https://docs.conda.io/en/latest/miniconda.html
  - Or: https://www.anaconda.com/products/distribution

### One-Command Setup
```bash
# Navigate to project directory
cd data_analysis_labeling_service

# Set up environment and start application
./start_conda.sh
```

This script will:
1. ✅ Check if Anaconda is installed
2. ✅ Create the conda environment if it doesn't exist
3. ✅ Activate the environment
4. ✅ Start the application
5. ✅ Open browser to http://localhost:8050

## 📋 Manual Setup Steps

### Step 1: Create Environment
```bash
# Create conda environment from environment.yml
conda env create -f environment.yml
```

### Step 2: Activate Environment
```bash
# Activate the environment
conda activate data-analysis-labeling
```

### Step 3: Start Application
```bash
# Run the application
python main.py
```

### Step 4: Access Web Interface
Open your browser and navigate to: **http://localhost:8050**

## 🛠️ Environment Management

### Available Scripts

| Script | Purpose |
|--------|---------|
| `setup_conda_env.sh` | Create new conda environment |
| `start_conda.sh` | Start application with environment |
| `update_conda_env.sh` | Update existing environment |

### Manual Commands

#### Create Environment
```bash
conda env create -f environment.yml
```

#### Activate Environment
```bash
conda activate data-analysis-labeling
```

#### Update Environment
```bash
conda env update -f environment.yml
```

#### List Environments
```bash
conda env list
```

#### Remove Environment
```bash
conda env remove -n data-analysis-labeling
```

#### Export Environment
```bash
conda env export > environment_backup.yml
```

## 📦 Environment Contents

The `environment.yml` file includes:

### Core Scientific Packages
- **Python 3.9**: Main programming language
- **NumPy**: Numerical computing
- **SciPy**: Scientific computing
- **Pandas**: Data manipulation
- **Matplotlib**: Basic plotting

### Specialized Packages
- **Plotly**: Interactive visualizations
- **H5Py**: HDF5 file handling
- **PyWavelets**: Wavelet analysis (via pip)
- **Scikit-learn**: Machine learning utilities
- **PyArrow**: Fast data processing

### Web Framework (via pip)
- **Dash**: Web application framework
- **Dash Bootstrap Components**: UI components
- **Streamlit**: Alternative web framework

## 🔧 Troubleshooting

### Common Issues

#### 1. Conda Not Found
```bash
# Add conda to PATH (if needed)
export PATH="/path/to/anaconda3/bin:$PATH"
```

#### 2. Environment Creation Fails
```bash
# Try with conda-forge channel
conda env create -f environment.yml -c conda-forge
```

#### 3. Package Conflicts
```bash
# Remove and recreate environment
conda env remove -n data-analysis-labeling
conda env create -f environment.yml
```

#### 4. Port Already in Use
```bash
# Kill existing process
lsof -ti:8050 | xargs kill -9
```

### Environment Verification

#### Check Environment
```bash
# Verify environment is active
conda info --envs

# Check installed packages
conda list
```

#### Test Installation
```bash
# Run test suite
python test_installation.py
```

## 📊 Performance Benefits

### Anaconda vs Standard Python
- **Faster Installation**: Pre-compiled packages
- **Better Performance**: Optimized scientific libraries
- **Easier Management**: Environment isolation
- **Consistent Dependencies**: Version compatibility

### Memory Usage
- **Isolated Environment**: No system package conflicts
- **Optimized Libraries**: Better memory management
- **Clean Dependencies**: Only required packages installed

## 🔄 Updating the Environment

### Automatic Update
```bash
./update_conda_env.sh
```

### Manual Update
```bash
# Activate environment
conda activate data-analysis-labeling

# Update from environment.yml
conda env update -f environment.yml

# Or update specific packages
conda update numpy scipy pandas
```

## 🐳 Docker Alternative

If you prefer Docker over Anaconda:
```bash
# Build and run with Docker
docker-compose up --build
```

## 📝 Environment Customization

### Adding New Packages
```bash
# Activate environment
conda activate data-analysis-labeling

# Install new package
conda install package_name

# Or with pip
pip install package_name
```

### Updating environment.yml
```bash
# Export current environment
conda env export > environment.yml
```

## 🎯 Best Practices

### Environment Management
1. **Always activate environment** before running the application
2. **Use environment.yml** for reproducible setups
3. **Update regularly** to get latest security patches
4. **Backup environment** before major changes

### Development Workflow
1. **Create environment** once using `setup_conda_env.sh`
2. **Use start script** for daily development: `./start_conda.sh`
3. **Update when needed** using `update_conda_env.sh`
4. **Test changes** with `python test_installation.py`

## 🎉 Ready to Use!

Your Anaconda environment is now ready for the Data Analysis & Labeling Service. The isolated environment ensures consistent performance and prevents conflicts with other Python projects.

### Quick Commands Summary
```bash
# First time setup
./setup_conda_env.sh

# Daily use
./start_conda.sh

# Updates
./update_conda_env.sh

# Testing
python test_installation.py
```

Happy analyzing! 🚀
