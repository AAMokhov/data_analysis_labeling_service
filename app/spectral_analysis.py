"""
Модуль спектрального анализа
Предоставляет FFT, STFT, анализ огибающей и вейвлет-анализ для обработки сигналов
"""

import numpy as np
import scipy.signal as signal
import scipy.fft as fft
import pywt
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SpectralAnalyzer:
    """Класс для выполнения спектрального анализа временных рядов"""

    def __init__(self, sample_rate: float = 25600.0):
        """
        Инициализация SpectralAnalyzer

        Args:
            sample_rate: Частота дискретизации в Гц (по умолчанию: 25600 Гц)
        """
        self.sample_rate = sample_rate
        self.nyquist_freq = sample_rate / 2

    def _ensure_json_serializable(self, obj):
        """
        Обеспечивает JSON-сериализуемость всех объектов

        Args:
            obj: Объект для преобразования

        Returns:
            JSON-сериализуемая версия объекта
        """
        if isinstance(obj, dict):
            return {key: self._ensure_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._ensure_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            if np.iscomplexobj(obj):
                # Для комплексных массивов берем модуль
                return np.abs(obj).tolist()
            else:
                return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, complex):
            # Для комплексных чисел берем модуль
            return abs(obj)
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # Для всех остальных типов пытаемся преобразовать в строку
            try:
                return str(obj)
            except:
                return None

    def compute_fft(self, data: np.ndarray, window: str = 'hann') -> Dict:
        """
        Вычисление быстрого преобразования Фурье сигнала

        Args:
            data: Входные данные временного ряда
            window: Функция окна для применения ('hann', 'hamming', 'blackman', и т.д.)

        Returns:
            Словарь, содержащий массив частот и спектр амплитуд
        """
        try:
            # Применение функции окна
            if window == 'hann':
                windowed_data = data * signal.windows.hann(len(data))
            elif window == 'hamming':
                windowed_data = data * signal.windows.hamming(len(data))
            elif window == 'blackman':
                windowed_data = data * signal.windows.blackman(len(data))
            else:
                windowed_data = data

            # Вычисление FFT
            fft_result = fft.fft(windowed_data)
            fft_magnitude = np.abs(fft_result)

            # Создание массива частот
            freqs = fft.fftfreq(len(data), 1/self.sample_rate)

            # Возврат только положительных частот
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
            logger.error(f"Ошибка вычисления FFT: {e}")
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
            logger.info(f"Вычисление STFT: данные = {len(data)}, частота = {self.sample_rate}, nperseg = {nperseg}")

            # Проверяем размер данных
            if len(data) < nperseg:
                logger.warning(f"Данные слишком короткие ({len(data)} < {nperseg}), используем адаптированные параметры")
                nperseg = min(len(data), 8)  # Минимальный размер окна
                noverlap = max(0, nperseg // 2 - 1)  # Безопасное перекрытие
                logger.info(f"Адаптированные параметры: nperseg = {nperseg}, noverlap = {noverlap}")

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

            logger.info(f"STFT результат: частоты = {len(freqs)}, времена = {len(times)}, спектрограмма = {spectrogram.shape}")

            return {
                'frequencies': freqs,
                'times': times,
                'spectrogram': spectrogram,
                'magnitude_db': 20 * np.log10(spectrogram + 1e-10)
            }

        except Exception as e:
            logger.error(f"Ошибка вычисления STFT: {e}")
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
            logger.error(f"Ошибка вычисления анализа огибающей: {e}")
            raise

    def compute_wavelet_analysis(self, data: np.ndarray,
                               wavelet: str = 'morlet',
                               scales: Optional[np.ndarray] = None) -> Dict:
        """
        Compute continuous wavelet transform using improved implementation

        Args:
            data: Input time series data
            wavelet: Wavelet type ('morlet', 'db4', 'haar', 'sym4', etc.)
            scales: Array of scales to analyze (default: optimized for signal analysis)

        Returns:
            Dictionary containing wavelet coefficients and scales
        """
        try:
            if scales is None:
                # Create more reasonable scale array for electrical signal analysis
                # Focus on frequencies from 0.1 Hz to 1000 Hz
                min_freq = 0.1  # Hz
                max_freq = min(1000.0, self.sample_rate / 4)  # Hz, limited by Nyquist/4

                # Create logarithmic frequency array
                freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), 50)

                # Convert frequencies to scales for Morlet wavelet
                # For Morlet wavelet: f = fc / (scale * dt), where fc ≈ 1 for Morlet
                dt = 1.0 / self.sample_rate
                scales = 1.0 / (freqs * dt)

            # Compute continuous wavelet transform
            try:
                # Try to use pywt.cwt with proper parameters
                coefficients, frequencies = pywt.cwt(data, scales, wavelet, dt)
                logger.info(f"Использован pywt.cwt для вейвлет-анализа")
            except Exception as e:
                logger.warning(f"pywt.cwt не работает ({e}), используем улучшенный fallback")
                # Improved fallback using proper frequency-domain filtering
                coefficients = np.zeros((len(scales), len(data)), dtype=complex)
                frequencies = 1.0 / (scales * dt)  # Convert scales to frequencies

                # Get FFT of the signal
                fft_data = np.fft.fft(data)
                freq_axis = np.fft.fftfreq(len(data), dt)

                for i, target_freq in enumerate(frequencies):
                    # Create a Gaussian filter centered at target frequency
                    sigma = target_freq * 0.1  # Bandwidth control
                    gaussian_filter = np.exp(-0.5 * ((freq_axis - target_freq) / sigma) ** 2)
                    gaussian_filter += np.exp(-0.5 * ((freq_axis + target_freq) / sigma) ** 2)

                    # Apply filter and inverse FFT
                    filtered_fft = fft_data * gaussian_filter
                    coefficients[i, :] = np.fft.ifft(filtered_fft)

                logger.info(f"Использован улучшенный fallback для вейвлет-анализа")

            # Sort by frequency (ascending order)
            freq_sort_idx = np.argsort(frequencies)
            frequencies = frequencies[freq_sort_idx]
            coefficients = coefficients[freq_sort_idx, :]
            scales = scales[freq_sort_idx]

            # Compute wavelet power spectrum
            power_spectrum = np.abs(coefficients) ** 2

            logger.info(f"Вейвлет-анализ: частоты {frequencies[0]:.3f}-{frequencies[-1]:.3f} Гц, "
                       f"power spectrum {np.min(power_spectrum):.6f}-{np.max(power_spectrum):.6f}")

            return {
                'coefficients': coefficients,
                'scales': scales,
                'frequencies': frequencies,
                'power_spectrum': power_spectrum,
                'wavelet': wavelet
            }

        except Exception as e:
            logger.error(f"Ошибка вычисления вейвлет-анализа: {e}")
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
            logger.error(f"Ошибка обнаружения пиков: {e}")
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
            logger.error(f"Ошибка вычисления статистических характеристик: {e}")
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
            # Проверяем валидность данных
            if len(data) < 8:
                logger.error(f"Данные слишком короткие для анализа: {len(data)} точек (минимум 8)")
                raise ValueError(f"Недостаточно данных для анализа: {len(data)} точек")

            logger.info(f"Начало анализа сегмента: {len(data)} точек")
            results = {}

            # Add original data to results for visualization
            results['data'] = data.tolist()

            # FFT analysis
            fft_result = self.compute_fft(data)
            results['fft'] = {
                'frequencies': fft_result['frequencies'].tolist(),
                'magnitude': fft_result['magnitude'].tolist(),
                'phase': fft_result['phase'].tolist(),
                'power_spectrum': fft_result['power_spectrum'].tolist()
            }

            # STFT analysis
            stft_result = self.compute_stft(data)
            results['stft'] = {
                'frequencies': stft_result['frequencies'].tolist(),
                'times': stft_result['times'].tolist(),
                'spectrogram': stft_result['spectrogram'].tolist(),
                'magnitude_db': stft_result['magnitude_db'].tolist()
            }

            # Envelope analysis
            envelope_result = self.compute_envelope_analysis(data)
            results['envelope'] = {
                'envelope': envelope_result['envelope'].tolist(),
                'raw_envelope': envelope_result['raw_envelope'].tolist(),
                'envelope_fft': {
                    'frequencies': envelope_result['envelope_fft']['frequencies'].tolist(),
                    'magnitude': envelope_result['envelope_fft']['magnitude'].tolist(),
                    'phase': envelope_result['envelope_fft']['phase'].tolist(),
                    'power_spectrum': envelope_result['envelope_fft']['power_spectrum'].tolist()
                },
                'cutoff_freq': envelope_result['cutoff_freq']
            }

            # Wavelet analysis
            wavelet_result = self.compute_wavelet_analysis(data)

            # Обработка комплексных коэффициентов для сериализации
            coefficients = wavelet_result['coefficients']
            if np.iscomplexobj(coefficients):
                # Сохраняем только модуль (амплитуду) комплексных коэффициентов
                coefficients_serializable = np.abs(coefficients).tolist()
                logger.info("Вейвлет коэффициенты преобразованы из комплексных в модуль для сериализации")
            else:
                coefficients_serializable = coefficients.tolist()

            results['wavelet'] = {
                'coefficients': coefficients_serializable,
                'scales': wavelet_result['scales'].tolist(),
                'frequencies': wavelet_result['frequencies'].tolist(),
                'power_spectrum': wavelet_result['power_spectrum'].tolist(),
                'wavelet': wavelet_result['wavelet']
            }

            # Peak detection
            peaks_result = self.detect_peaks(
                fft_result['frequencies'],
                fft_result['magnitude']
            )
            results['peaks'] = {
                'peak_indices': peaks_result['peak_indices'].tolist(),
                'peak_frequencies': peaks_result['peak_frequencies'].tolist(),
                'peak_magnitudes': peaks_result['peak_magnitudes'].tolist(),
                'peak_properties': peaks_result['peak_properties']
            }

            # Statistical features
            results['statistics'] = self.compute_statistical_features(data)

            # Обеспечиваем JSON-сериализуемость всех результатов
            logger.info("Проверка и обеспечение JSON-сериализуемости результатов...")
            results = self._ensure_json_serializable(results)

            # Финальная проверка сериализации
            try:
                import json
                json.dumps(results)
                logger.info("✅ Все результаты анализа JSON-сериализуемы")
            except Exception as json_error:
                logger.error(f"❌ Ошибка финальной сериализации: {json_error}")
                raise ValueError(f"Результаты анализа не могут быть сериализованы: {json_error}")

            return results

        except Exception as e:
            logger.error(f"Ошибка анализа сегмента: {e}")
            raise
