# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Option 1: Anaconda (Recommended)
```bash
# 1. Set up environment
./setup_conda_env.sh

# 2. Start application
./start_conda.sh

# 3. Open browser
# Navigate to: http://localhost:8050
```

### Option 2: Standard Python
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run application
python main.py

# 3. Open browser
# Navigate to: http://localhost:8050
```

## 📊 What You Can Do

### Data Analysis
- **Load HDF5 files** with segmented current data
- **Navigate segments** using Previous/Next buttons
- **Perform spectral analysis** with one click
- **View multiple visualizations**:
  - Time series plots
  - FFT frequency spectra
  - Spectrograms (STFT)
  - Envelope analysis
  - Wavelet analysis
  - Comprehensive dashboard

### Labeling Interface
- **Categorize defects**:
  - Normal
  - Outer ring defect
  - Inner ring defect
  - Rolling element defect
  - Cage defect
  - Imbalance
  - Misalignment
  - Other

- **Set severity levels**:
  - Initial
  - Medium
  - High
  - Critical

- **Add metadata**:
  - Confidence score (0-1)
  - Analyst name
  - Comments and notes

### Data Management
- **Automatic saving** to HDF5 format
- **Export labels** to CSV
- **Progress tracking** with statistics
- **Backup support**

## 🎯 Example Workflow

1. **Select Data File**: Choose from dropdown (processed_current_1.h5, etc.)
2. **Load Segment**: Click "Load Segment" or select from dropdown
3. **Analyze**: Click "Analyze Segment" to perform spectral analysis
4. **Review**: Explore different visualization tabs
5. **Label**: Use the labeling interface to categorize the defect
6. **Save**: Click "Save Label" to store your annotation
7. **Navigate**: Use Previous/Next to move between segments
8. **Export**: Use "Export Labels" when finished

## 🔧 Troubleshooting

### Common Issues
- **Port 8050 in use**: Change port in `main.py` or kill existing process
- **Missing dependencies**: Run `pip install -r requirements.txt`
- **Data files not found**: Ensure files are in `app/data/` directory

### Test Installation
```bash
python test_installation.py
```

## 📁 File Structure
```
data_analysis_labeling_service/
├── app/
│   ├── data/                    # Your HDF5 data files
│   ├── data_loader.py           # Data loading
│   ├── spectral_analysis.py     # Signal processing
│   ├── label_manager.py         # Label management
│   ├── visualization.py         # Plotly charts
│   └── dash_app.py             # Web interface
├── main.py                      # Entry point
├── requirements.txt             # Dependencies
├── Dockerfile                   # Container config
├── docker-compose.yml           # Docker orchestration
└── README.md                    # Full documentation
```

## 🐳 Docker Alternative
```bash
docker-compose up --build
```

## 📈 Features Overview

### Spectral Analysis
- **FFT**: Fast Fourier Transform for frequency analysis
- **STFT**: Short-Time Fourier Transform for time-frequency analysis
- **Envelope Analysis**: Bearing defect detection
- **Wavelet Analysis**: Multi-scale signal analysis
- **Peak Detection**: Automatic spectral peak identification
- **Statistical Features**: Comprehensive feature extraction

### Interactive Visualization
- **Zoom and pan** on all plots
- **Hover information** with detailed data
- **Multiple view modes** for different analysis needs
- **Responsive design** for different screen sizes

### Label Management
- **Real-time saving** to prevent data loss
- **Progress tracking** with visual indicators
- **Statistics dashboard** for overview
- **Export capabilities** for ML training

## 🎉 Ready to Start!

Your Data Analysis & Labeling Service is ready to use. Start analyzing and labeling your electrical current data for machine learning training!
