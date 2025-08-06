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
                               sample_rate: float = 1000.0,
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
            raise

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
            frequencies = fft_result['frequencies']
            magnitude = fft_result['magnitude']

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
            raise

    def create_spectrogram_plot(self, stft_result: Dict,
                               title: str = "Спектрограмма",
                               segment_id: str = "") -> go.Figure:
        """
        Create spectrogram plot

        Args:
            stft_result: STFT analysis results
            title: Plot title
            segment_id: Segment ID for display

        Returns:
            Plotly figure object
        """
        try:
            frequencies = stft_result['frequencies']
            times = stft_result['times']
            spectrogram = stft_result['spectrogram']

            fig = go.Figure()

            fig.add_trace(go.Heatmap(
                z=spectrogram,
                x=times,
                y=frequencies,
                colorscale='Viridis',
                hovertemplate='Time: %{x:.3f}s<br>Frequency: %{y:.1f} Hz<br>Magnitude: %{z:.3f}<extra></extra>'
            ))

            fig.update_layout(
                title=f"{title} - Segment {segment_id}" if segment_id else title,
                xaxis_title="Time (s)",
                yaxis_title="Frequency (Hz)",
                template=self.layout_template,
                height=400
            )

            return fig

        except Exception as e:
            logger.error(f"Ошибка создания графика спектрограммы: {e}")
            raise

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
            envelope = envelope_result['envelope']
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
                fig.add_trace(
                    go.Scatter(
                        x=env_fft['frequencies'],
                        y=env_fft['magnitude'],
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
            raise

    def create_wavelet_plot(self, wavelet_result: Dict,
                          title: str = "Вейвлет-анализ",
                          segment_id: str = "") -> go.Figure:
        """
        Create wavelet analysis plot

        Args:
            wavelet_result: Wavelet analysis results
            title: Plot title
            segment_id: Segment ID for display

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

            # Convert to numpy arrays if they're lists
            if isinstance(power_spectrum, list):
                power_spectrum = np.array(power_spectrum)
            if isinstance(frequencies, list):
                frequencies = np.array(frequencies)
            if isinstance(scales, list):
                scales = np.array(scales)

            # Validate data
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

            # Ensure power_spectrum is 2D
            if power_spectrum.ndim == 1:
                power_spectrum = power_spectrum.reshape(1, -1)

            fig = go.Figure()

            fig.add_trace(go.Heatmap(
                z=power_spectrum,
                x=np.arange(power_spectrum.shape[1]),
                y=frequencies,
                colorscale='Viridis',
                hovertemplate='Time: %{x}<br>Frequency: %{y:.1f} Hz<br>Power: %{z:.3f}<extra></extra>'
            ))

            fig.update_layout(
                title=f"{title} - Segment {segment_id}" if segment_id else title,
                xaxis_title="Time",
                yaxis_title="Frequency (Hz)",
                template=self.layout_template,
                height=400
            )

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

            # Spectrogram
            if 'stft' in analysis_result:
                stft_result = analysis_result['stft']
                fig.add_trace(
                    go.Heatmap(z=stft_result['spectrogram'],
                              x=stft_result['times'],
                              y=stft_result['frequencies'],
                              colorscale='Viridis'),
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

            # Wavelet
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
                    fig.add_trace(
                        go.Heatmap(z=power_spectrum,
                                  x=np.arange(power_spectrum.shape[1]),
                                  y=frequencies,
                                  colorscale='Viridis'),
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
