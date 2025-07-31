#!/usr/bin/env python3
"""
Test script to verify Data Analysis & Labeling Service installation
"""

import sys
import os
import logging
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing module imports...")

    try:
        from app.data_loader import DataLoader, MultiFileDataLoader
        print("✓ DataLoader imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import DataLoader: {e}")
        return False

    try:
        from app.spectral_analysis import SpectralAnalyzer
        print("✓ SpectralAnalyzer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import SpectralAnalyzer: {e}")
        return False

    try:
        from app.label_manager import LabelManager
        print("✓ LabelManager imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import LabelManager: {e}")
        return False

    try:
        from app.visualization import SpectralVisualizer
        print("✓ SpectralVisualizer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import SpectralVisualizer: {e}")
        return False

    try:
        from app.dash_app import app
        print("✓ Dash app imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Dash app: {e}")
        return False

    return True

def test_data_loading():
    """Test data loading functionality"""
    print("\nTesting data loading...")

    try:
        from app.data_loader import DataLoader

        # Check if data files exist
        data_files = [
            "app/data/processed_current_1.h5",
            "app/data/processed_current_2.h5",
            "app/data/processed_current_3.h5",
            "app/data/processed_data.h5"
        ]

        available_files = []
        for file_path in data_files:
            if os.path.exists(file_path):
                available_files.append(file_path)
                print(f"✓ Found data file: {file_path}")
            else:
                print(f"⚠ Data file not found: {file_path}")

        if not available_files:
            print("✗ No data files found")
            return False

        # Test loading the first available file
        test_file = available_files[0]
        loader = DataLoader(test_file)
        segment_ids = loader.get_all_segment_ids()

        if segment_ids:
            print(f"✓ Successfully loaded {len(segment_ids)} segments from {test_file}")

            # Test loading a segment
            test_segment = segment_ids[0]
            data = loader.get_segment_data(test_segment)
            print(f"✓ Successfully loaded segment {test_segment} with {len(data)} data points")

            return True
        else:
            print("✗ No segments found in data file")
            return False

    except Exception as e:
        print(f"✗ Error testing data loading: {e}")
        return False

def test_spectral_analysis():
    """Test spectral analysis functionality"""
    print("\nTesting spectral analysis...")

    try:
        from app.spectral_analysis import SpectralAnalyzer
        import numpy as np

        # Create test data
        sample_rate = 1000.0
        duration = 1.0  # 1 second
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Create a test signal with known frequency components
        test_signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)

        # Initialize analyzer
        analyzer = SpectralAnalyzer(sample_rate=sample_rate)

        # Test FFT
        fft_result = analyzer.compute_fft(test_signal)
        print("✓ FFT analysis completed successfully")

        # Test STFT
        stft_result = analyzer.compute_stft(test_signal)
        print("✓ STFT analysis completed successfully")

        # Test envelope analysis
        envelope_result = analyzer.compute_envelope_analysis(test_signal)
        print("✓ Envelope analysis completed successfully")

        # Test wavelet analysis
        wavelet_result = analyzer.compute_wavelet_analysis(test_signal)
        print("✓ Wavelet analysis completed successfully")

        # Test comprehensive analysis
        analysis_results = analyzer.analyze_segment(test_signal)
        print("✓ Comprehensive analysis completed successfully")

        return True

    except Exception as e:
        print(f"✗ Error testing spectral analysis: {e}")
        return False

def test_label_management():
    """Test label management functionality"""
    print("\nTesting label management...")

    try:
        from app.label_manager import LabelManager
        import tempfile
        import os

        # Create temporary file for testing
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            test_file = tmp_file.name

        try:
            # Initialize label manager
            manager = LabelManager(test_file)
            print("✓ LabelManager initialized successfully")

            # Test adding a label
            success = manager.add_label(
                segment_id="test_segment_001",
                defect_category="Normal",
                severity="Initial",
                confidence=0.9,
                analyst="Test Analyst",
                comments="Test label"
            )

            if success:
                print("✓ Label added successfully")
            else:
                print("✗ Failed to add label")
                return False

            # Test retrieving label
            label = manager.get_label("test_segment_001")
            if label and label['defect_category'] == "Normal":
                print("✓ Label retrieved successfully")
            else:
                print("✗ Failed to retrieve label")
                return False

            # Test statistics
            stats = manager.get_label_statistics()
            if stats['total_labels'] == 1:
                print("✓ Label statistics computed successfully")
            else:
                print("✗ Label statistics incorrect")
                return False

            return True

        finally:
            # Clean up temporary file
            if os.path.exists(test_file):
                os.unlink(test_file)

    except Exception as e:
        print(f"✗ Error testing label management: {e}")
        return False

def test_visualization():
    """Test visualization functionality"""
    print("\nTesting visualization...")

    try:
        from app.visualization import SpectralVisualizer
        import numpy as np

        # Create test data
        sample_rate = 1000.0
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        test_signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)

        # Initialize visualizer
        visualizer = SpectralVisualizer()

        # Test time series plot
        time_series_fig = visualizer.create_time_series_plot(test_signal, sample_rate)
        print("✓ Time series plot created successfully")

        # Test FFT plot
        from app.spectral_analysis import SpectralAnalyzer
        analyzer = SpectralAnalyzer(sample_rate)
        fft_result = analyzer.compute_fft(test_signal)
        fft_fig = visualizer.create_fft_plot(fft_result)
        print("✓ FFT plot created successfully")

        # Test STFT plot
        stft_result = analyzer.compute_stft(test_signal)
        stft_fig = visualizer.create_spectrogram_plot(stft_result)
        print("✓ Spectrogram plot created successfully")

        # Test envelope plot
        envelope_result = analyzer.compute_envelope_analysis(test_signal)
        envelope_fig = visualizer.create_envelope_plot(envelope_result)
        print("✓ Envelope plot created successfully")

        # Test wavelet plot
        wavelet_result = analyzer.compute_wavelet_analysis(test_signal)
        wavelet_fig = visualizer.create_wavelet_plot(wavelet_result)
        print("✓ Wavelet plot created successfully")

        return True

    except Exception as e:
        print(f"✗ Error testing visualization: {e}")
        return False

def main():
    """Run all tests"""
    print("Data Analysis & Labeling Service - Installation Test")
    print("=" * 60)

    tests = [
        ("Module Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("Spectral Analysis", test_spectral_analysis),
        ("Label Management", test_label_management),
        ("Visualization", test_visualization)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Installation is successful.")
        print("\nTo start the application, run:")
        print("  python main.py")
        print("\nThen open your browser to: http://localhost:8050")
        return 0
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
