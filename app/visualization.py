"""
Модуль визуализации
Предоставляет интерактивные визуализации на основе Plotly для спектрального анализа
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SpectralVisualizer:
    """Класс для создания интерактивных визуализаций спектрального анализа"""

    def __init__(self):
        """Инициализация SpectralVisualizer"""
        self.colors = px.colors.qualitative.Set1
        self.layout_template = 'plotly_white'

    def create_time_series_plot(self, data: np.ndarray,
                               sample_rate: float = 25600.0,
                               title: str = "Временной ряд",
                               segment_id: str = "") -> go.Figure:
        """
        Создание графика временного ряда

        Args:
            data: Данные временного ряда
            sample_rate: Частота дискретизации в Гц
            title: Заголовок графика
            segment_id: ID сегмента для отображения

        Returns:
            Объект фигуры Plotly
        """
        try:
            logger.info(f"Создание графика временного ряда: данные = {len(data)}, частота = {sample_rate}")
            time_axis = np.arange(len(data)) / sample_rate

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=time_axis,
                y=data,
                mode='lines',
                name='Signal',
                line=dict(color=self.colors[0], width=1),
                hovertemplate='Time: %{x:.3f}s<br>Amplitude: %{y:.3f}<extra></extra>'
            ))

            fig.update_layout(
                title=f"{title} - Segment {segment_id}" if segment_id else title,
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
                template=self.layout_template,
                height=400,
                showlegend=False,
                hovermode='x unified'
            )

            return fig

        except Exception as e:
            logger.error(f"Ошибка создания графика временного ряда: {e}")
            # Return empty figure with error message instead of raising
            fig = go.Figure()
            fig.add_annotation(
                text=f"Ошибка создания временного ряда: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title=f"{title} - Segment {segment_id}" if segment_id else title,
                template=self.layout_template,
                height=400
            )
            return fig

    def create_fft_plot(self, fft_result: Dict,
                       title: str = "Спектр Фурье",
                       segment_id: str = "",
                       show_peaks: bool = True) -> go.Figure:
        """
        Create FFT spectrum plot

        Args:
            fft_result: FFT analysis results
            title: Plot title
            segment_id: Segment ID for display
            show_peaks: Whether to highlight detected peaks

        Returns:
            Plotly figure object
        """
        try:
            frequencies = np.array(fft_result['frequencies'])
            magnitude = np.array(fft_result['magnitude'])

            fig = go.Figure()

            # Main spectrum
            fig.add_trace(go.Scatter(
                x=frequencies,
                y=magnitude,
                mode='lines',
                name='Spectrum',
                line=dict(color=self.colors[0], width=1),
                hovertemplate='Frequency: %{x:.1f} Hz<br>Magnitude: %{y:.3f}<extra></extra>'
            ))

            # Highlight peaks if available
            if show_peaks and 'peaks' in fft_result:
                peak_freqs = fft_result['peaks']['peak_frequencies']
                peak_mags = fft_result['peaks']['peak_magnitudes']

                fig.add_trace(go.Scatter(
                    x=peak_freqs,
                    y=peak_mags,
                    mode='markers',
                    name='Peaks',
                    marker=dict(color='red', size=8, symbol='diamond'),
                    hovertemplate='Peak: %{x:.1f} Hz<br>Magnitude: %{y:.3f}<extra></extra>'
                ))

            fig.update_layout(
                title=f"{title} - Segment {segment_id}" if segment_id else title,
                xaxis_title="Frequency (Hz)",
                yaxis_title="Magnitude",
                template=self.layout_template,
                height=400,
                hovermode='x unified'
            )

            return fig

        except Exception as e:
            logger.error(f"Ошибка создания графика FFT: {e}")
            # Return empty figure with error message instead of raising
            fig = go.Figure()
            fig.add_annotation(
                text=f"Ошибка создания FFT: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title=f"{title} - Segment {segment_id}" if segment_id else title,
                template=self.layout_template,
                height=400
            )
            return fig

    def create_spectrogram_plot(self, stft_result: Dict,
                               title: str = "Спектрограмма",
                               segment_id: str = "",
                               max_freq: float = None,
                               use_log_scale: bool = True) -> go.Figure:
        """
        Create spectrogram plot with adaptive frequency scaling

        Args:
            stft_result: STFT analysis results
            title: Plot title
            segment_id: Segment ID for display
            max_freq: Maximum frequency to display (auto-detected if None)
            use_log_scale: Use logarithmic scale for magnitude

        Returns:
            Plotly figure object
        """
        try:
            logger.info(f"Создание спектрограммы: stft_result = {type(stft_result)}, keys = {list(stft_result.keys()) if stft_result else None}")

            if not stft_result:
                logger.error("STFT результаты пустые")
                raise ValueError("STFT результаты пустые")

            # Convert lists to numpy arrays if needed
            frequencies = stft_result.get('frequencies')
            times = stft_result.get('times')
            spectrogram = stft_result.get('spectrogram')

            if frequencies is None or times is None or spectrogram is None:
                logger.error("Отсутствуют необходимые данные STFT")
                raise ValueError("Отсутствуют необходимые данные STFT")

            # Ensure arrays are numpy arrays
            frequencies = np.array(frequencies)
            times = np.array(times)
            spectrogram = np.array(spectrogram)

            logger.info(f"Спектрограмма: частоты = {len(frequencies)}, времена = {len(times)}, спектрограмма = {spectrogram.shape}")

            # Adaptive frequency range detection
            if max_freq is None:
                # Calculate energy per frequency
                energy_per_freq = np.mean(spectrogram, axis=1)

                # Find frequency range containing 95% of energy
                cumulative_energy = np.cumsum(energy_per_freq) / np.sum(energy_per_freq)
                freq_95_percent = frequencies[np.argmax(cumulative_energy >= 0.95)]

                # Set reasonable limits based on data
                if freq_95_percent < 500:
                    max_freq = min(1000, frequencies[-1])  # At least 1 kHz for visibility
                elif freq_95_percent < 2000:
                    max_freq = min(2000, frequencies[-1])
                else:
                    max_freq = min(5000, frequencies[-1])

                logger.info(f"Автоматически выбран частотный диапазон: 0 - {max_freq:.0f} Гц")

            # Filter data to frequency range of interest
            freq_mask = frequencies <= max_freq
            frequencies_filtered = frequencies[freq_mask]
            spectrogram_filtered = spectrogram[freq_mask, :]

            # Use logarithmic scale for better visibility
            if use_log_scale:
                # Use magnitude in dB if available, otherwise calculate
                if 'magnitude_db' in stft_result:
                    magnitude_db = np.array(stft_result['magnitude_db'])
                    spectrogram_display = magnitude_db[freq_mask, :]
                    colorbar_title = "Magnitude (dB)"
                else:
                    spectrogram_display = 20 * np.log10(spectrogram_filtered + 1e-10)
                    colorbar_title = "Magnitude (dB)"
            else:
                spectrogram_display = spectrogram_filtered
                colorbar_title = "Magnitude"

            fig = go.Figure()

            fig.add_trace(go.Heatmap(
                z=spectrogram_display,
                x=times,
                y=frequencies_filtered,
                colorscale='Viridis',
                colorbar=dict(title=colorbar_title),
                hovertemplate='Time: %{x:.3f}s<br>Frequency: %{y:.1f} Hz<br>' +
                             ('Magnitude: %{z:.1f} dB<extra></extra>' if use_log_scale else 'Magnitude: %{z:.3f}<extra></extra>')
            ))

            fig.update_layout(
                title=f"{title} - Segment {segment_id} (0-{max_freq:.0f} Hz)" if segment_id else f"{title} (0-{max_freq:.0f} Hz)",
                xaxis_title="Time (s)",
                yaxis_title="Frequency (Hz)",
                template=self.layout_template,
                height=500  # Увеличена высота для лучшей видимости
            )

            # Set frequency axis range
            fig.update_yaxes(range=[0, max_freq])

            return fig

        except Exception as e:
            logger.error(f"Ошибка создания графика спектрограммы: {e}")
            # Return empty figure with error message instead of raising
            fig = go.Figure()
            fig.add_annotation(
                text=f"Ошибка создания спектрограммы: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title=f"{title} - Segment {segment_id}" if segment_id else title,
                template=self.layout_template,
                height=500
            )
            return fig

    def create_envelope_plot(self, envelope_result: Dict,
                           title: str = "Анализ огибающей",
                           segment_id: str = "") -> go.Figure:
        """
        Create envelope analysis plot

        Args:
            envelope_result: Envelope analysis results
            title: Plot title
            segment_id: Segment ID for display

        Returns:
            Plotly figure object
        """
        try:
            # Create subplots for envelope signal and its FFT
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Envelope Signal', 'Envelope FFT'),
                vertical_spacing=0.1
            )

            # Envelope signal
            envelope_data = envelope_result.get('envelope')
            if envelope_data is None:
                fig.add_annotation(
                    text="Данные огибающей недоступны",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return fig

            envelope = np.array(envelope_data)
            time_axis = np.arange(len(envelope)) / 1000.0  # Assuming 1kHz sample rate

            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=envelope,
                    mode='lines',
                    name='Envelope',
                    line=dict(color=self.colors[0], width=1)
                ),
                row=1, col=1
            )

            # Envelope FFT
            if 'envelope_fft' in envelope_result:
                env_fft = envelope_result['envelope_fft']
                frequencies = env_fft.get('frequencies')
                magnitude = env_fft.get('magnitude')

                if frequencies is not None and magnitude is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=np.array(frequencies),
                            y=np.array(magnitude),
                            mode='lines',
                            name='Envelope FFT',
                            line=dict(color=self.colors[1], width=1)
                        ),
                        row=2, col=1
                    )

            fig.update_layout(
                title=f"{title} - Segment {segment_id}" if segment_id else title,
                template=self.layout_template,
                height=600,
                showlegend=False
            )

            fig.update_xaxes(title_text="Time (s)", row=1, col=1)
            fig.update_yaxes(title_text="Amplitude", row=1, col=1)
            fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
            fig.update_yaxes(title_text="Magnitude", row=2, col=1)

            return fig

        except Exception as e:
            logger.error(f"Ошибка создания графика огибающей: {e}")
            # Return empty figure with error message instead of raising
            fig = go.Figure()
            fig.add_annotation(
                text=f"Ошибка создания графика огибающей: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title=f"{title} - Segment {segment_id}" if segment_id else title,
                template=self.layout_template,
                height=600
            )
            return fig

    def create_wavelet_plot(self, wavelet_result: Dict,
                          title: str = "Вейвлет-анализ",
                          segment_id: str = "",
                          max_freq: float = None,
                          use_log_scale: bool = True,
                          sample_rate: float = 25600.0) -> go.Figure:
        """
        Create improved wavelet analysis plot with adaptive scaling

        Args:
            wavelet_result: Wavelet analysis results
            title: Plot title
            segment_id: Segment ID for display
            max_freq: Maximum frequency to display (auto-detected if None)
            use_log_scale: Use logarithmic scale for power
            sample_rate: Sampling rate for time axis

        Returns:
            Plotly figure object
        """
        try:
            # Check if wavelet_result is valid
            if not wavelet_result or not isinstance(wavelet_result, dict):
                fig = go.Figure()
                fig.add_annotation(
                    text="Данные вейвлет-анализа недоступны",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                fig.update_layout(
                    title=f"{title} - Segment {segment_id}" if segment_id else title,
                    template=self.layout_template,
                    height=400
                )
                return fig

            # Extract data with proper type checking
            power_spectrum = wavelet_result.get('power_spectrum')
            frequencies = wavelet_result.get('frequencies')
            scales = wavelet_result.get('scales')

            # Validate data first
            if power_spectrum is None or frequencies is None:
                fig = go.Figure()
                fig.add_annotation(
                    text="Неверные данные вейвлет-анализа",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                fig.update_layout(
                    title=f"{title} - Segment {segment_id}" if segment_id else title,
                    template=self.layout_template,
                    height=400
                )
                return fig

            # Convert to numpy arrays if they're lists
            if isinstance(power_spectrum, list):
                power_spectrum = np.array(power_spectrum)
            if isinstance(frequencies, list):
                frequencies = np.array(frequencies)
            if isinstance(scales, list):
                scales = np.array(scales)

            # Ensure power_spectrum is 2D
            if power_spectrum.ndim == 1:
                power_spectrum = power_spectrum.reshape(1, -1)

            logger.info(f"Вейвлет график: частоты = {len(frequencies)}, power spectrum = {power_spectrum.shape}")
            logger.info(f"Частотный диапазон: {frequencies[0]:.3f} - {frequencies[-1]:.3f} Гц")
            logger.info(f"Power spectrum диапазон: {np.min(power_spectrum):.6f} - {np.max(power_spectrum):.6f}")

            # Adaptive frequency range detection
            if max_freq is None:
                # Calculate energy per frequency
                energy_per_freq = np.mean(power_spectrum, axis=1)

                # Find frequency range containing 90% of energy
                cumulative_energy = np.cumsum(energy_per_freq) / np.sum(energy_per_freq)
                freq_90_percent = frequencies[np.argmax(cumulative_energy >= 0.90)]

                # Set reasonable limits based on data
                if freq_90_percent < 10:
                    max_freq = min(50, frequencies[-1])
                elif freq_90_percent < 100:
                    max_freq = min(200, frequencies[-1])
                else:
                    max_freq = min(500, frequencies[-1])

                logger.info(f"Автоматически выбран частотный диапазон: {frequencies[0]:.3f} - {max_freq:.0f} Гц")

            # Filter data to frequency range of interest
            freq_mask = frequencies <= max_freq
            frequencies_filtered = frequencies[freq_mask]
            power_spectrum_filtered = power_spectrum[freq_mask, :]

            # Create proper time axis
            time_axis = np.arange(power_spectrum_filtered.shape[1]) / sample_rate * power_spectrum_filtered.shape[1] / len(frequencies_filtered)

            # Apply logarithmic scaling for better visibility
            if use_log_scale and np.max(power_spectrum_filtered) > 0:
                # Use percentile-based scaling to avoid extreme values
                p1 = np.percentile(power_spectrum_filtered[power_spectrum_filtered > 0], 1)
                p99 = np.percentile(power_spectrum_filtered, 99)

                # Clip values to avoid log of zero
                power_spectrum_display = np.clip(power_spectrum_filtered, p1, p99)
                power_spectrum_display = 10 * np.log10(power_spectrum_display / p1)
                colorbar_title = "Power (dB)"
                z_min, z_max = 0, 10 * np.log10(p99 / p1)
            else:
                power_spectrum_display = power_spectrum_filtered
                colorbar_title = "Power"
                z_min, z_max = np.min(power_spectrum_display), np.max(power_spectrum_display)

            fig = go.Figure()

            fig.add_trace(go.Heatmap(
                z=power_spectrum_display,
                x=time_axis,
                y=frequencies_filtered,
                colorscale='Viridis',
                colorbar=dict(title=colorbar_title),
                zmin=z_min,
                zmax=z_max,
                hovertemplate='Time: %{x:.4f}s<br>Frequency: %{y:.3f} Hz<br>' +
                             ('Power: %{z:.1f} dB<extra></extra>' if use_log_scale else 'Power: %{z:.6f}<extra></extra>')
            ))

            fig.update_layout(
                title=f"{title} - Segment {segment_id} ({frequencies_filtered[0]:.3f}-{max_freq:.0f} Hz)" if segment_id else f"{title} ({frequencies_filtered[0]:.3f}-{max_freq:.0f} Hz)",
                xaxis_title="Time (s)",
                yaxis_title="Frequency (Hz)",
                template=self.layout_template,
                height=500  # Увеличена высота для лучшей видимости
            )

            # Set frequency axis range
            fig.update_yaxes(range=[frequencies_filtered[0], max_freq])

            return fig

        except Exception as e:
            logger.error(f"Ошибка создания графика вейвлет-анализа: {e}")
            # Return empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Ошибка создания графика вейвлет-анализа: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title=f"{title} - Segment {segment_id}" if segment_id else title,
                template=self.layout_template,
                height=400
            )
            return fig

    def create_comprehensive_analysis_plot(self, analysis_result: Dict,
                                         segment_id: str = "",
                                         sample_rate: float = 1000.0) -> go.Figure:
        """
        Create comprehensive analysis plot with all visualizations

        Args:
            analysis_result: Complete analysis results
            segment_id: Segment ID for display
            sample_rate: Sampling rate in Hz

        Returns:
            Plotly figure object with subplots
        """
        try:
            # Check if analysis_result is valid
            if not analysis_result or not isinstance(analysis_result, dict):
                fig = go.Figure()
                fig.add_annotation(
                    text="Данные анализа недоступны",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                fig.update_layout(
                    title=f"Comprehensive Analysis - Segment {segment_id}" if segment_id else "Comprehensive Analysis",
                    template=self.layout_template,
                    height=900
                )
                return fig

            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Time Series', 'FFT Spectrum',
                    'Spectrogram', 'Envelope Signal',
                    'Envelope FFT', 'Wavelet Analysis'
                ),
                specs=[
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}]
                ],
                vertical_spacing=0.08,
                horizontal_spacing=0.1
            )

            # Time series
            if 'data' in analysis_result:
                data = analysis_result['data']
                time_axis = np.arange(len(data)) / sample_rate
                fig.add_trace(
                    go.Scatter(x=time_axis, y=data, mode='lines', name='Signal',
                              line=dict(color=self.colors[0], width=1)),
                    row=1, col=1
                )

            # FFT
            if 'fft' in analysis_result:
                fft_result = analysis_result['fft']
                fig.add_trace(
                    go.Scatter(x=fft_result['frequencies'], y=fft_result['magnitude'],
                              mode='lines', name='FFT',
                              line=dict(color=self.colors[1], width=1)),
                    row=1, col=2
                )

            # Spectrogram with adaptive frequency scaling
            if 'stft' in analysis_result:
                stft_result = analysis_result['stft']
                # Convert to numpy arrays if needed
                spectrogram = np.array(stft_result['spectrogram'])
                times = np.array(stft_result['times'])
                frequencies = np.array(stft_result['frequencies'])

                # Apply adaptive frequency scaling similar to standalone spectrogram
                energy_per_freq = np.mean(spectrogram, axis=1)
                cumulative_energy = np.cumsum(energy_per_freq) / np.sum(energy_per_freq)
                freq_95_percent = frequencies[np.argmax(cumulative_energy >= 0.95)]

                # Set reasonable frequency limit for comprehensive view
                if freq_95_percent < 500:
                    max_freq = min(1000, frequencies[-1])
                elif freq_95_percent < 2000:
                    max_freq = min(2000, frequencies[-1])
                else:
                    max_freq = min(3000, frequencies[-1])  # Lower limit for comprehensive view

                # Filter data
                freq_mask = frequencies <= max_freq
                frequencies_filtered = frequencies[freq_mask]
                spectrogram_filtered = spectrogram[freq_mask, :]

                # Use log scale for better visibility
                spectrogram_db = 20 * np.log10(spectrogram_filtered + 1e-10)

                fig.add_trace(
                    go.Heatmap(z=spectrogram_db,
                              x=times,
                              y=frequencies_filtered,
                              colorscale='Viridis',
                              showscale=False),  # Hide colorbar in comprehensive view
                    row=2, col=1
                )

            # Envelope signal
            if 'envelope' in analysis_result:
                envelope = analysis_result['envelope']['envelope']
                time_axis = np.arange(len(envelope)) / sample_rate
                fig.add_trace(
                    go.Scatter(x=time_axis, y=envelope, mode='lines', name='Envelope',
                              line=dict(color=self.colors[2], width=1)),
                    row=2, col=2
                )

            # Envelope FFT
            if 'envelope' in analysis_result and 'envelope_fft' in analysis_result['envelope']:
                env_fft = analysis_result['envelope']['envelope_fft']
                fig.add_trace(
                    go.Scatter(x=env_fft['frequencies'], y=env_fft['magnitude'],
                              mode='lines', name='Envelope FFT',
                              line=dict(color=self.colors[3], width=1)),
                    row=3, col=1
                )

            # Wavelet with improved scaling
            if 'wavelet' in analysis_result:
                wavelet_result = analysis_result['wavelet']
                power_spectrum = wavelet_result.get('power_spectrum')
                frequencies = wavelet_result.get('frequencies')

                # Convert to numpy arrays if they're lists
                if isinstance(power_spectrum, list):
                    power_spectrum = np.array(power_spectrum)
                if isinstance(frequencies, list):
                    frequencies = np.array(frequencies)

                # Ensure power_spectrum is 2D
                if power_spectrum is not None and power_spectrum.ndim == 1:
                    power_spectrum = power_spectrum.reshape(1, -1)

                if power_spectrum is not None and frequencies is not None:
                    # Ensure we have valid arrays for processing
                    try:
                        # Apply adaptive frequency scaling similar to standalone wavelet plot
                        energy_per_freq = np.mean(power_spectrum, axis=1)
                        cumulative_energy = np.cumsum(energy_per_freq) / np.sum(energy_per_freq)
                        freq_90_percent = frequencies[np.argmax(cumulative_energy >= 0.90)]

                        # Set frequency limit for comprehensive view (more conservative)
                        max_freq_limit = frequencies[-1] if len(frequencies) > 0 else 200
                        if freq_90_percent < 10:
                            max_freq = min(30, max_freq_limit)
                        elif freq_90_percent < 50:
                            max_freq = min(100, max_freq_limit)
                        else:
                            max_freq = min(200, max_freq_limit)

                        # Filter data
                        freq_mask = frequencies <= max_freq
                        frequencies_filtered = frequencies[freq_mask]
                        power_spectrum_filtered = power_spectrum[freq_mask, :]

                        # Apply log scale for better visibility
                        if np.max(power_spectrum_filtered) > 0:
                            p1 = np.percentile(power_spectrum_filtered[power_spectrum_filtered > 0], 1)
                            p99 = np.percentile(power_spectrum_filtered, 99)
                            power_spectrum_display = np.clip(power_spectrum_filtered, p1, p99)
                            power_spectrum_display = 10 * np.log10(power_spectrum_display / p1)
                        else:
                            power_spectrum_display = power_spectrum_filtered

                        # Create proper time axis
                        time_axis = np.arange(power_spectrum_display.shape[1]) / sample_rate * power_spectrum_display.shape[1] / len(frequencies_filtered)

                        fig.add_trace(
                            go.Heatmap(z=power_spectrum_display,
                                      x=time_axis,
                                      y=frequencies_filtered,
                                      colorscale='Viridis',
                                      showscale=False),  # Hide colorbar in comprehensive view
                            row=3, col=2
                        )
                    except Exception as e:
                        logger.error(f"Ошибка обработки вейвлет-данных в comprehensive plot: {e}")
                        # Fallback to simple visualization
                        fig.add_trace(
                            go.Heatmap(z=power_spectrum,
                                      x=np.arange(power_spectrum.shape[1]),
                                      y=frequencies,
                                      colorscale='Viridis',
                                      showscale=False),
                            row=3, col=2
                        )

            # Update layout
            fig.update_layout(
                title=f"Комплексный анализ - Сегмент {segment_id}" if segment_id else "Комплексный анализ",
                template=self.layout_template,
                height=900,
                showlegend=False
            )

            # Update axis labels
            fig.update_xaxes(title_text="Время (с)", row=1, col=1)
            fig.update_yaxes(title_text="Амплитуда", row=1, col=1)
            fig.update_xaxes(title_text="Частота (Гц)", row=1, col=2)
            fig.update_yaxes(title_text="Величина", row=1, col=2)
            fig.update_xaxes(title_text="Время (с)", row=2, col=1)
            fig.update_yaxes(title_text="Частота (Гц)", row=2, col=1)
            fig.update_xaxes(title_text="Время (с)", row=2, col=2)
            fig.update_yaxes(title_text="Амплитуда", row=2, col=2)
            fig.update_xaxes(title_text="Частота (Гц)", row=3, col=1)
            fig.update_yaxes(title_text="Величина", row=3, col=1)
            fig.update_xaxes(title_text="Время", row=3, col=2)
            fig.update_yaxes(title_text="Частота (Гц)", row=3, col=2)

            return fig

        except Exception as e:
            logger.error(f"Ошибка создания графика комплексного анализа: {e}")
            # Return empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Ошибка создания комплексного анализа: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title=f"Comprehensive Analysis - Segment {segment_id}" if segment_id else "Comprehensive Analysis",
                template=self.layout_template,
                height=900
            )
            return fig

    def create_statistics_dashboard(self, statistics: Dict) -> go.Figure:
        """
        Create statistics dashboard

        Args:
            statistics: Statistical features

        Returns:
            Plotly figure object
        """
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Time Domain Features', 'Frequency Domain Features',
                              'Statistical Distribution', 'Feature Summary'),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "histogram"}, {"type": "table"}]],
                vertical_spacing=0.1
            )

            # Time domain features
            time_features = ['mean', 'std', 'rms', 'peak', 'crest_factor']
            time_values = [statistics.get(feature, 0) for feature in time_features]

            fig.add_trace(
                go.Bar(x=time_features, y=time_values, name='Time Domain',
                      marker_color=self.colors[0]),
                row=1, col=1
            )

            # Frequency domain features
            if 'freq_features' in statistics:
                freq_features = list(statistics['freq_features'].keys())
                freq_values = list(statistics['freq_features'].values())

                fig.add_trace(
                    go.Bar(x=freq_features, y=freq_values, name='Frequency Domain',
                          marker_color=self.colors[1]),
                    row=1, col=2
                )

            # Update layout
            fig.update_layout(
                title="Statistical Features Dashboard",
                template=self.layout_template,
                height=600,
                showlegend=False
            )

            return fig

        except Exception as e:
            logger.error(f"Ошибка создания панели статистики: {e}")
            raise

    def create_label_statistics_plot(self, label_stats: Dict) -> go.Figure:
        """
        Create label statistics visualization

        Args:
            label_stats: Label statistics from LabelManager

        Returns:
            Plotly figure object
        """
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Defect Categories', 'Severity Levels',
                              'Analysts', 'Progress Overview'),
                specs=[[{"type": "pie"}, {"type": "pie"}],
                       [{"type": "bar"}, {"type": "indicator"}]],
                vertical_spacing=0.1
            )

            # Defect categories pie chart
            if label_stats['categories']:
                categories = list(label_stats['categories'].keys())
                values = list(label_stats['categories'].values())

                fig.add_trace(
                    go.Pie(labels=categories, values=values, name="Categories"),
                    row=1, col=1
                )

            # Severity levels pie chart
            if label_stats['severities']:
                severities = list(label_stats['severities'].keys())
                values = list(label_stats['severities'].values())

                fig.add_trace(
                    go.Pie(labels=severities, values=values, name="Severities"),
                    row=1, col=2
                )

            # Analysts bar chart
            if label_stats['analysts']:
                analysts = list(label_stats['analysts'].keys())
                values = list(label_stats['analysts'].values())

                fig.add_trace(
                    go.Bar(x=analysts, y=values, name="Analysts",
                          marker_color=self.colors[0]),
                    row=2, col=1
                )

            # Progress indicator
            total = label_stats['total_labels']
            max_value = max(total * 1.2, 100)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=total,
                    title={'text': "Total Labels"},
                    gauge={'axis': {'range': [None, max_value]}},
                    delta={'reference': total * 0.8}
                ),
                row=2, col=2
            )

            fig.update_layout(
                title="Labeling Statistics Dashboard",
                template=self.layout_template,
                height=600,
                showlegend=False
            )

            return fig

        except Exception as e:
            logger.error(f"Ошибка создания графика статистики меток: {e}")
            raise
