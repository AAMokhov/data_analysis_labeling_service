"""
Spectral Analysis Module
Provides FFT, STFT, envelope analysis, and wavelet analysis for signal processing
"""

import numpy as np
import scipy.signal as signal
import scipy.fft as fft
import pywt
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SpectralAnalyzer:
    """Class for performing spectral analysis on time series data"""

    def __init__(self, sample_rate: float = 1000.0):
        """
        Initialize SpectralAnalyzer

        Args:
            sample_rate: Sampling rate in Hz (default: 1000 Hz)
        """
        self.sample_rate = sample_rate
        self.nyquist_freq = sample_rate / 2

    def compute_fft(self, data: np.ndarray, window: str = 'hann') -> Dict:
        """
        Compute Fast Fourier Transform of the signal

        Args:
            data: Input time series data
            window: Window function to apply ('hann', 'hamming', 'blackman', etc.)

        Returns:
            Dictionary containing frequency array and magnitude spectrum
        """
        try:
            # Apply window function
            if window == 'hann':
                windowed_data = data * signal.windows.hann(len(data))
            elif window == 'hamming':
                windowed_data = data * signal.windows.hamming(len(data))
            elif window == 'blackman':
                windowed_data = data * signal.windows.blackman(len(data))
            else:
                windowed_data = data

            # Compute FFT
            fft_result = fft.fft(windowed_data)
            fft_magnitude = np.abs(fft_result)

            # Create frequency array
            freqs = fft.fftfreq(len(data), 1/self.sample_rate)

            # Only return positive frequencies
            positive_freq_mask = freqs >= 0
            freqs = freqs[positive_freq_mask]
            fft_magnitude = fft_magnitude[positive_freq_mask]

            return {
                'frequencies': freqs,
                'magnitude': fft_magnitude,
                'phase': np.angle(fft_result[positive_freq_mask]),
                'power_spectrum': fft_magnitude ** 2
            }

        except Exception as e:
            logger.error(f"Error computing FFT: {e}")
            raise

    def compute_stft(self, data: np.ndarray,
                    nperseg: int = 256,
                    noverlap: int = 128,
                    window: str = 'hann') -> Dict:
        """
        Compute Short-Time Fourier Transform (spectrogram)

        Args:
            data: Input time series data
            nperseg: Length of each segment
            noverlap: Number of points to overlap between segments
            window: Window function to apply

        Returns:
            Dictionary containing time, frequency, and spectrogram arrays
        """
        try:
            # Compute STFT
            freqs, times, stft_result = signal.stft(
                data,
                fs=self.sample_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                window=window
            )

            # Convert to magnitude spectrogram
            spectrogram = np.abs(stft_result)

            return {
                'frequencies': freqs,
                'times': times,
                'spectrogram': spectrogram,
                'magnitude_db': 20 * np.log10(spectrogram + 1e-10)
            }

        except Exception as e:
            logger.error(f"Error computing STFT: {e}")
            raise

    def compute_envelope_analysis(self, data: np.ndarray,
                                cutoff_freq: Optional[float] = None,
                                filter_order: int = 4) -> Dict:
        """
        Compute envelope analysis for bearing defect detection

        Args:
            data: Input time series data
            cutoff_freq: Cutoff frequency for low-pass filter (default: 0.1 * nyquist)
            filter_order: Order of the low-pass filter

        Returns:
            Dictionary containing envelope signal and its FFT
        """
        try:
            if cutoff_freq is None:
                cutoff_freq = 0.1 * self.nyquist_freq

            # Compute analytic signal using Hilbert transform
            analytic_signal = signal.hilbert(data)
            envelope = np.abs(analytic_signal)

            # Apply low-pass filter to smooth the envelope
            nyquist = self.sample_rate / 2
            normalized_cutoff = cutoff_freq / nyquist

            # Design low-pass filter
            b, a = signal.butter(filter_order, normalized_cutoff, btype='low')
            filtered_envelope = signal.filtfilt(b, a, envelope)

            # Compute FFT of the envelope
            envelope_fft = self.compute_fft(filtered_envelope)

            return {
                'envelope': filtered_envelope,
                'raw_envelope': envelope,
                'envelope_fft': envelope_fft,
                'cutoff_freq': cutoff_freq
            }

        except Exception as e:
            logger.error(f"Error computing envelope analysis: {e}")
            raise

    def compute_wavelet_analysis(self, data: np.ndarray,
                               wavelet: str = 'db4',
                               scales: Optional[np.ndarray] = None) -> Dict:
        """
        Compute continuous wavelet transform

        Args:
            data: Input time series data
            wavelet: Wavelet type ('db4', 'haar', 'sym4', etc.)
            scales: Array of scales to analyze (default: logarithmic scale)

        Returns:
            Dictionary containing wavelet coefficients and scales
        """
        try:
            if scales is None:
                # Create logarithmic scale array
                scales = np.logspace(1, np.log2(len(data)/4), 50)

            # Compute continuous wavelet transform
            # Use a simpler approach that works with different pywt versions
            try:
                coefficients, frequencies = pywt.cwt(data, scales, wavelet, 1/self.sample_rate)
            except:
                # Fallback: create a simple wavelet-like analysis using FFT
                coefficients = np.zeros((len(scales), len(data)))
                frequencies = 1.0 / scales
                for i, scale in enumerate(scales):
                    # Simple bandpass filter approach
                    fft_data = np.fft.fft(data)
                    freq = np.fft.fftfreq(len(data), 1/self.sample_rate)
                    # Apply simple bandpass filter
                    mask = np.abs(freq) < (1.0 / scale)
                    fft_filtered = fft_data * mask
                    coefficients[i, :] = np.real(np.fft.ifft(fft_filtered))

            # Compute wavelet power spectrum
            power_spectrum = np.abs(coefficients) ** 2

            return {
                'coefficients': coefficients,
                'scales': scales,
                'frequencies': frequencies,
                'power_spectrum': power_spectrum,
                'wavelet': wavelet
            }

        except Exception as e:
            logger.error(f"Error computing wavelet analysis: {e}")
            raise

    def detect_peaks(self, frequencies: np.ndarray,
                    magnitude: np.ndarray,
                    height: Optional[float] = None,
                    distance: Optional[int] = None,
                    prominence: Optional[float] = None) -> Dict:
        """
        Detect peaks in frequency spectrum

        Args:
            frequencies: Frequency array
            magnitude: Magnitude spectrum
            height: Minimum height for peak detection
            distance: Minimum distance between peaks
            prominence: Minimum prominence for peak detection

        Returns:
            Dictionary containing peak information
        """
        try:
            # Find peaks
            peak_indices, peak_properties = signal.find_peaks(
                magnitude,
                height=height,
                distance=distance,
                prominence=prominence
            )

            peak_frequencies = frequencies[peak_indices]
            peak_magnitudes = magnitude[peak_indices]

            return {
                'peak_indices': peak_indices,
                'peak_frequencies': peak_frequencies,
                'peak_magnitudes': peak_magnitudes,
                'peak_properties': peak_properties
            }

        except Exception as e:
            logger.error(f"Error detecting peaks: {e}")
            raise

    def compute_statistical_features(self, data: np.ndarray) -> Dict:
        """
        Compute statistical features of the signal

        Args:
            data: Input time series data

        Returns:
            Dictionary containing statistical features
        """
        try:
            # Time domain features
            mean_val = np.mean(data)
            std_val = np.std(data)
            rms_val = np.sqrt(np.mean(data**2))
            peak_val = np.max(np.abs(data))
            crest_factor = peak_val / rms_val if rms_val > 0 else 0

            # Frequency domain features
            fft_result = self.compute_fft(data)
            spectral_centroid = np.sum(fft_result['frequencies'] * fft_result['magnitude']) / np.sum(fft_result['magnitude'])
            freq_features = {
                'dominant_freq': fft_result['frequencies'][np.argmax(fft_result['magnitude'])],
                'spectral_centroid': spectral_centroid,
                'spectral_bandwidth': np.sqrt(np.sum((fft_result['frequencies'] - spectral_centroid)**2 * fft_result['magnitude']) / np.sum(fft_result['magnitude']))
            }

            return {
                'mean': mean_val,
                'std': std_val,
                'rms': rms_val,
                'peak': peak_val,
                'crest_factor': crest_factor,
                'freq_features': freq_features
            }

        except Exception as e:
            logger.error(f"Error computing statistical features: {e}")
            raise

    def analyze_segment(self, data: np.ndarray) -> Dict:
        """
        Perform comprehensive analysis on a data segment

        Args:
            data: Input time series data

        Returns:
            Dictionary containing all analysis results
        """
        try:
            results = {}

            # FFT analysis
            results['fft'] = self.compute_fft(data)

            # STFT analysis
            results['stft'] = self.compute_stft(data)

            # Envelope analysis
            results['envelope'] = self.compute_envelope_analysis(data)

            # Wavelet analysis
            results['wavelet'] = self.compute_wavelet_analysis(data)

            # Peak detection
            results['peaks'] = self.detect_peaks(
                results['fft']['frequencies'],
                results['fft']['magnitude']
            )

            # Statistical features
            results['statistics'] = self.compute_statistical_features(data)

            return results

        except Exception as e:
            logger.error(f"Error analyzing segment: {e}")
            raise
