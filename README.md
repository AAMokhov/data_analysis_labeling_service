# Data Analysis & Labeling Service

A comprehensive web-based service for analyzing segmented electrical current data and providing interactive labeling capabilities for machine learning training datasets.

## Features

### Data Loading
- **HDF5 Support**: Load segmented data from HDF5 files
- **Multi-file Support**: Handle multiple data files simultaneously
- **Segment Management**: Navigate through data segments efficiently

### Spectral Analysis
- **Fast Fourier Transform (FFT)**: Frequency domain analysis
- **Short-Time Fourier Transform (STFT)**: Time-frequency analysis
- **Envelope Analysis**: Bearing defect detection
- **Wavelet Analysis**: Non-stationary signal processing
- **Peak Detection**: Automatic identification of spectral peaks
- **Statistical Features**: Comprehensive feature extraction

### Interactive Visualization
- **Time Series Plots**: Original signal visualization
- **FFT Spectra**: Frequency domain visualization
- **Spectrograms**: Time-frequency heatmaps
- **Envelope Analysis**: Bearing defect indicators
- **Wavelet Analysis**: Multi-scale analysis visualization
- **Comprehensive Dashboard**: All analyses in one view

### Labeling Interface
- **Defect Categories**:
  - Normal
  - Outer ring defect
  - Inner ring defect
  - Rolling element defect
  - Cage defect
  - Imbalance
  - Misalignment
  - Other

- **Severity Levels**:
  - Initial
  - Medium
  - High
  - Critical

- **Additional Features**:
  - Confidence scoring (0-1)
  - Analyst attribution
  - Comments and notes
  - Timestamp tracking

### Data Management
- **HDF5 Storage**: Efficient label storage
- **CSV Export**: Easy data export
- **Progress Tracking**: Real-time labeling progress
- **Statistics Dashboard**: Label distribution analysis
- **Backup Support**: Automatic backup creation

## Installation

### Prerequisites
- Python 3.9+ or Anaconda/Miniconda
- Docker (optional)

### Option 1: Anaconda Installation (Recommended)

1. **Clone the repository**:
```bash
git clone <repository-url>
cd data_analysis_labeling_service
```

2. **Set up Anaconda environment**:
```bash
./setup_conda_env.sh
```

3. **Start the application**:
```bash
./start_conda.sh
```

4. **Access the web interface**:
Open your browser and navigate to `http://localhost:8050`

### Option 2: Standard Python Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd data_analysis_labeling_service
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
python main.py
```

4. **Access the web interface**:
Open your browser and navigate to `http://localhost:8050`

### Option 3: Docker Installation

1. **Build and run with Docker Compose**:
```bash
docker-compose up --build
```

2. **Or build and run manually**:
```bash
docker build -t data-analysis-labeling-service .
docker run -p 8050:8050 -v $(pwd)/app/data:/app/app/data data-analysis-labeling-service
```

## Anaconda Environment Management

### Creating Environment
```bash
./setup_conda_env.sh
```

### Starting Application
```bash
./start_conda.sh
```

### Updating Environment
```bash
./update_conda_env.sh
```

### Manual Environment Management
```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate data-analysis-labeling

# Update environment
conda env update -f environment.yml

# Remove environment (if needed)
conda env remove -n data-analysis-labeling
```

## Usage

### Getting Started

1. **Select Data File**: Choose an HDF5 file from the dropdown menu
2. **Load Segments**: The application will automatically load available segments
3. **Navigate Segments**: Use Previous/Next buttons or select from dropdown
4. **Analyze Data**: Click "Analyze Segment" to perform spectral analysis
5. **Review Visualizations**: Explore different analysis views in the tabs
6. **Apply Labels**: Use the labeling interface to categorize defects
7. **Save Labels**: Click "Save Label" to store your annotations
8. **Export Data**: Use "Export Labels" to download CSV file

### Data Format

The service expects HDF5 files with the following structure:
```
segments/
├── phase_current_T/
│   ├── current_T_000990/
│   │   └── data: [1024,] float64
│   ├── current_T_000991/
│   │   └── data: [1024,] float64
│   └── ...
```

### Analysis Parameters

- **Sample Rate**: Default 1000 Hz (configurable)
- **FFT Window**: Hann window (configurable)
- **STFT Parameters**: 256-point segments, 128-point overlap
- **Envelope Analysis**: 0.1 * Nyquist cutoff frequency
- **Wavelet Analysis**: Daubechies 4 wavelet, logarithmic scales

## Configuration

### Environment Variables
- `PYTHONUNBUFFERED=1`: Enable unbuffered output
- `DASH_DEBUG=False`: Disable debug mode for production
- `SAMPLE_RATE=1000`: Set default sample rate

### File Paths
- Data files: `app/data/`
- Label storage: `app/data/labeled_data.h5`
- Export files: `app/data/labels_export.csv`
- Logs: `app.log`

## API Reference

### DataLoader
```python
from app.data_loader import DataLoader

# Initialize loader
loader = DataLoader("path/to/data.h5")

# Get segment data
data = loader.get_segment_data("current_T_000990")

# Get segment info
info = loader.get_segment_info("current_T_000990")
```

### SpectralAnalyzer
```python
from app.spectral_analysis import SpectralAnalyzer

# Initialize analyzer
analyzer = SpectralAnalyzer(sample_rate=1000.0)

# Perform analysis
results = analyzer.analyze_segment(data)
```

### LabelManager
```python
from app.label_manager import LabelManager

# Initialize manager
manager = LabelManager("labels.h5")

# Add label
manager.add_label(
    segment_id="current_T_000990",
    defect_category="Outer ring defect",
    severity="Medium",
    confidence=0.8,
    analyst="John Doe",
    comments="Clear bearing fault signature"
)
```

## Troubleshooting

### Common Issues

1. **HDF5 File Not Found**:
   - Ensure data files are in the `app/data/` directory
   - Check file permissions

2. **Memory Issues**:
   - Large files may require more memory
   - Consider processing smaller batches

3. **Port Already in Use**:
   - Change port in `main.py` or Docker configuration
   - Kill existing processes on port 8050

4. **Dependencies Missing**:
   - Reinstall requirements: `pip install -r requirements.txt`
   - Check Python version compatibility

### Logs
Check `app.log` for detailed error messages and debugging information.

## Development

### Project Structure
```
data_analysis_labeling_service/
├── app/
│   ├── __init__.py
│   ├── data_loader.py          # HDF5 data loading
│   ├── spectral_analysis.py    # Signal processing
│   ├── label_manager.py        # Label management
│   ├── visualization.py        # Plotly visualizations
│   ├── dash_app.py            # Main web application
│   └── data/                  # Data files
├── main.py                    # Entry point
├── requirements.txt           # Dependencies
├── Dockerfile                 # Container configuration
├── docker-compose.yml         # Docker orchestration
└── README.md                  # This file
```

### Adding New Features

1. **New Analysis Methods**: Extend `SpectralAnalyzer` class
2. **New Visualizations**: Add methods to `SpectralVisualizer` class
3. **New Label Categories**: Update `LabelManager.DEFECT_CATEGORIES`
4. **UI Components**: Modify `dash_app.py` layout and callbacks

### Testing
```bash
# Run basic functionality test
python -c "from app.data_loader import DataLoader; print('DataLoader imported successfully')"
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review the logs
- Create an issue in the repository
- Contact the development team
