"""
Main Dash Web Application
Provides interactive web interface for data analysis and labeling
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import os
import logging
from typing import Dict, List, Optional

from data_loader import DataLoader, MultiFileDataLoader
from spectral_analysis import SpectralAnalyzer
from label_manager import LabelManager
from visualization import SpectralVisualizer

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize components
data_loader = None
spectral_analyzer = SpectralAnalyzer(sample_rate=25600.0)  # Частота дискретизации 25.6 кГц
label_manager = LabelManager(output_file="app/data/labeled_data.h5")
visualizer = SpectralVisualizer()

# Инициализация Dash приложения
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Сервис анализа и маркировки данных"

# Макет приложения
app.layout = dbc.Container([
    # Заголовок
    dbc.Row([
        dbc.Col([
            html.H1("Сервис анализа и маркировки данных", className="text-center mb-4"),
            html.Hr()
        ])
    ]),

    # Выбор файла и управление
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Выбор файла данных"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Выберите файл данных:"),
                            dcc.Dropdown(
                                id='file-dropdown',
                                options=[
                                    {'label': 'processed_current_1.h5', 'value': 'app/data/processed_current_1.h5'}
									# ,
                                    # {'label': 'processed_current_2.h5', 'value': 'app/data/processed_current_2.h5'},
                                    # {'label': 'processed_current_3.h5', 'value': 'app/data/processed_current_3.h5'},
                                    # {'label': 'processed_data.h5', 'value': 'app/data/processed_data.h5'}
                                ],
                                value='app/data/processed_current_1.h5',
                                placeholder="Выберите файл данных..."
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("ID сегмента:"),
                            dcc.Dropdown(
                                id='segment-dropdown',
                                placeholder="Выберите сегмент..."
                            )
                        ], width=6)
                    ])
                ])
            ])
        ])
    ], className="mb-4"),

    # Управление анализом
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Управление анализом"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Загрузить сегмент", id="load-btn", color="primary", className="me-2"),
                            dbc.Button("Анализировать сегмент", id="analyze-btn", color="success", className="me-2"),
                            dbc.Button("Предыдущий сегмент", id="prev-btn", color="info", className="me-2"),
                            dbc.Button("Следующий сегмент", id="next-btn", color="info", className="me-2")
                        ])
                    ])
                ])
            ])
        ])
    ], className="mb-4"),

    # Вкладки визуализации
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Визуализация анализа"),
                dbc.CardBody([
                    dcc.Tabs([
                        dcc.Tab(label="Временной ряд", children=[
                            dcc.Graph(id="time-series-plot")
                        ]),
                        dcc.Tab(label="Спектр Фурье", children=[
                            dcc.Graph(id="fft-plot")
                        ]),
                        dcc.Tab(label="Спектрограмма", children=[
                            dcc.Graph(id="spectrogram-plot")
                        ]),
                        dcc.Tab(label="Анализ огибающей", children=[
                            dcc.Graph(id="envelope-plot")
                        ]),
                        dcc.Tab(label="Вейвлет-анализ", children=[
                            dcc.Graph(id="wavelet-plot")
                        ]),
                        dcc.Tab(label="Комплексный вид", children=[
                            dcc.Graph(id="comprehensive-plot")
                        ])
                    ])
                ])
            ])
        ])
    ]),

    # Labeling Interface
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Маркировка сегмента"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Категория дефекта:"),
                            dcc.Dropdown(
                                id='defect-category-dropdown',
                                options=[{'label': cat, 'value': cat} for cat in LabelManager.DEFECT_CATEGORIES],
                                placeholder="Выберите категорию дефекта..."
                            )
                        ], width=3),
                        dbc.Col([
                            html.Label("Уровень серьезности:"),
                            dcc.Dropdown(
                                id='severity-dropdown',
                                options=[{'label': sev, 'value': sev} for sev in LabelManager.SEVERITY_LEVELS],
                                placeholder="Выберите уровень серьезности..."
                            )
                        ], width=3),
                        dbc.Col([
                            html.Label("Уверенность (0-1):"),
                            dcc.Slider(
                                id='confidence-slider',
                                min=0, max=1, step=0.1, value=1.0,
                                marks={i/10: str(i/10) for i in range(0, 11, 2)}
                            )
                        ], width=3),
                        dbc.Col([
                            html.Label("Имя аналитика:"),
                            dcc.Input(
                                id='analyst-input',
                                type='text',
                                placeholder='Введите имя аналитика...',
                                value=''
                            )
                        ], width=3)
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Комментарии:"),
                            dcc.Textarea(
                                id='comments-textarea',
                                placeholder='Введите дополнительные комментарии...',
                                rows=3
                            )
                        ])
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Сохранить метку", id="save-label-btn", color="success", className="me-2"),
                            dbc.Button("Очистить метку", id="clear-label-btn", color="warning", className="me-2"),
                            dbc.Button("Экспорт меток", id="export-btn", color="info", className="me-2")
                        ])
                    ])
                ])
            ])
        ])
    ], className="mb-4"),

    # Current Label Display
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Текущая метка"),
                dbc.CardBody(id="current-label-display")
            ])
        ])
    ], className="mb-4"),

    # Progress and Statistics
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Прогресс маркировки"),
                dbc.CardBody(id="progress-display")
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Статистика меток"),
                dbc.CardBody(id="statistics-display")
            ])
        ], width=6)
    ], className="mb-4"),

    # Скрытые элементы для хранения данных
    dcc.Store(id='current-data-store'),
    dcc.Store(id='analysis-results-store'),
    dcc.Store(id='current-segment-id-store'),

    # Интервальный компонент для автообновления
    dcc.Interval(
        id='interval-component',
        interval=30*1000,  # 30 секунд
        n_intervals=0
    )

], fluid=True)

# Обратные вызовы
@app.callback(
    [Output('segment-dropdown', 'options'),
     Output('segment-dropdown', 'value')],
    [Input('file-dropdown', 'value')]
)
def update_segment_dropdown(selected_file):
    """Обновление выпадающего списка сегментов при выборе файла"""
    if not selected_file or not os.path.exists(selected_file):
        logger.info(f"Файл не выбран или не существует: {selected_file}")
        return [], None

    try:
        global data_loader
        logger.info(f"Инициализация DataLoader для файла: {selected_file}")
        data_loader = DataLoader(selected_file)
        segment_ids = data_loader.get_all_segment_ids()
        logger.info(f"Загружено сегментов: {len(segment_ids)}")

        options = [{'label': seg_id, 'value': seg_id} for seg_id in segment_ids]
        return options, segment_ids[0] if segment_ids else None

    except Exception as e:
        logger.error(f"Ошибка загрузки сегментов: {e}")
        return [], None

@app.callback(
    [Output('current-data-store', 'data'),
     Output('current-segment-id-store', 'data')],
    [Input('load-btn', 'n_clicks'),
     Input('segment-dropdown', 'value')],
    [State('file-dropdown', 'value')]
)
def load_segment_data(n_clicks, segment_id, file_path):
    """Загрузка данных сегмента при нажатии кнопки загрузки или выборе сегмента"""
    if not segment_id or not file_path or not data_loader:
        logger.info(f"Загрузка сегмента: segment_id={segment_id}, file_path={file_path}, data_loader={data_loader is not None}")
        return None, None

    try:
        data = data_loader.get_segment_data(segment_id)
        logger.info(f"Загружены данные сегмента {segment_id}: размер = {len(data)}")
        return data.tolist(), segment_id

    except Exception as e:
        logger.error(f"Ошибка загрузки данных сегмента: {e}")
        return None, None

@app.callback(
    Output('analysis-results-store', 'data'),
    [Input('analyze-btn', 'n_clicks')],
    [State('current-data-store', 'data')]
)
def analyze_segment(n_clicks, data):
    """Выполнение спектрального анализа текущего сегмента"""
    logger.info(f"Callback analyze_segment вызван: n_clicks={n_clicks}, data={len(data) if data else 0}")

    if not n_clicks or not data:
        logger.info("Анализ не выполнен: нет кликов или данных")
        return None

    try:
        data_array = np.array(data)
        logger.info(f"Анализ сегмента: размер данных = {len(data_array)}, тип = {type(data_array)}")
        analysis_results = spectral_analyzer.analyze_segment(data_array)
        logger.info(f"Анализ завершен: получено {len(analysis_results)} результатов")
        logger.info(f"Ключи результатов: {list(analysis_results.keys())}")

        # Проверим основные компоненты
        for key in ['fft', 'stft', 'envelope', 'wavelet']:
            if key in analysis_results:
                logger.info(f"  {key}: найден, ключи = {list(analysis_results[key].keys())}")
            else:
                logger.warning(f"  {key}: отсутствует!")

        # Проверим размер данных для Store
        import sys
        data_size = sys.getsizeof(str(analysis_results))
        logger.info(f"Размер данных для Store: {data_size} байт")
        logger.info(f"Возвращаем результаты анализа с {len(analysis_results)} компонентами")
        return analysis_results

    except Exception as e:
        logger.error(f"Ошибка анализа сегмента: {e}")
        import traceback
        logger.error(f"Трассировка: {traceback.format_exc()}")
        return None

@app.callback(
    [Output('time-series-plot', 'figure'),
     Output('fft-plot', 'figure'),
     Output('spectrogram-plot', 'figure'),
     Output('envelope-plot', 'figure'),
     Output('wavelet-plot', 'figure'),
     Output('comprehensive-plot', 'figure')],
    [Input('analysis-results-store', 'data'),
     Input('current-segment-id-store', 'data'),
     Input('current-data-store', 'data')]
)
def update_plots(analysis_results, segment_id, current_data):
    """Обновление всех графиков визуализации"""
    logger.info(f"update_plots вызван: analysis_results={analysis_results is not None}, segment_id={segment_id}, current_data={len(current_data) if current_data else 0}")
    logger.info(f"Тип analysis_results: {type(analysis_results)}, значение: {analysis_results}")

    if not analysis_results:
        logger.info("Обновление графиков: нет результатов анализа")
        # Возврат пустых графиков
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="Нажмите 'Анализировать сегмент' для отображения данных",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return [empty_fig] * 6

    try:
        logger.info(f"Обновление графиков: сегмент {segment_id}, данные = {len(current_data) if current_data else 0}")

        # Добавление данных к результатам анализа для комплексного графика
        if current_data and 'data' not in analysis_results:
            analysis_results['data'] = current_data

        # Создание отдельных графиков с правильной обработкой ошибок
        try:
            time_series_fig = visualizer.create_time_series_plot(
                np.array(current_data) if current_data else np.array([]), segment_id=segment_id
            )
            logger.info("График временного ряда создан успешно")
        except Exception as e:
            logger.error(f"Ошибка создания графика временного ряда: {e}")
            time_series_fig = go.Figure().add_annotation(text="Ошибка загрузки временного ряда", xref="paper", yref="paper")

        try:
            fft_fig = visualizer.create_fft_plot(
                analysis_results.get('fft'), segment_id=segment_id
            )
        except Exception as e:
            logger.error(f"Ошибка создания графика FFT: {e}")
            fft_fig = go.Figure().add_annotation(text="Ошибка загрузки спектра Фурье", xref="paper", yref="paper")

        try:
            stft_data = analysis_results.get('stft')
            logger.info(f"Создание спектрограммы: STFT данные = {type(stft_data)}, ключи = {list(stft_data.keys()) if stft_data else None}")
            spectrogram_fig = visualizer.create_spectrogram_plot(
                stft_data, segment_id=segment_id
            )
            logger.info("График спектрограммы создан успешно")
        except Exception as e:
            logger.error(f"Ошибка создания графика спектрограммы: {e}")
            spectrogram_fig = go.Figure().add_annotation(text="Ошибка загрузки спектрограммы", xref="paper", yref="paper")

        try:
            envelope_fig = visualizer.create_envelope_plot(
                analysis_results.get('envelope'), segment_id=segment_id
            )
        except Exception as e:
            logger.error(f"Ошибка создания графика огибающей: {e}")
            envelope_fig = go.Figure().add_annotation(text="Ошибка загрузки огибающей", xref="paper", yref="paper")

        try:
            wavelet_fig = visualizer.create_wavelet_plot(
                analysis_results.get('wavelet'), segment_id=segment_id
            )
        except Exception as e:
            logger.error(f"Ошибка создания графика вейвлет-анализа: {e}")
            wavelet_fig = go.Figure().add_annotation(text="Ошибка загрузки вейвлет-анализа", xref="paper", yref="paper")

        try:
            comprehensive_fig = visualizer.create_comprehensive_analysis_plot(
                analysis_results, segment_id=segment_id
            )
        except Exception as e:
            logger.error(f"Ошибка создания комплексного графика: {e}")
            comprehensive_fig = go.Figure().add_annotation(text="Ошибка загрузки комплексного анализа", xref="paper", yref="paper")

        return time_series_fig, fft_fig, spectrogram_fig, envelope_fig, wavelet_fig, comprehensive_fig

    except Exception as e:
        logger.error(f"Ошибка обновления графиков: {e}")
        empty_fig = go.Figure().add_annotation(text="Ошибка загрузки графиков", xref="paper", yref="paper")
        return [empty_fig] * 6

@app.callback(
    Output('current-label-display', 'children'),
    [Input('current-segment-id-store', 'data'),
     Input('interval-component', 'n_intervals')]
)
def update_current_label_display(segment_id, n_intervals):
    """Обновление отображения текущей метки"""
    if not segment_id:
        return html.P("Сегмент не выбран")

    try:
        label = label_manager.get_label(segment_id)
        if label:
            return dbc.Table([
                html.Tr([html.Th("Категория дефекта"), html.Td(label['defect_category'])]),
                html.Tr([html.Th("Серьезность"), html.Td(label['severity'])]),
                html.Tr([html.Th("Уверенность"), html.Td(f"{label['confidence']:.2f}")]),
                html.Tr([html.Th("Аналитик"), html.Td(label['analyst'])]),
                html.Tr([html.Th("Комментарии"), html.Td(label['comments'])]),
                html.Tr([html.Th("Время создания"), html.Td(label['timestamp'])])
            ], bordered=True, size="sm")
        else:
            return html.P("Метка для этого сегмента не назначена")

    except Exception as e:
        logger.error(f"Ошибка обновления отображения метки: {e}")
        return html.P("Ошибка загрузки метки")

@app.callback(
    [Output('progress-display', 'children'),
     Output('statistics-display', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_progress_and_statistics(n_intervals):
    """Обновление отображения прогресса и статистики"""
    try:
        # Получение прогресса
        if data_loader:
            total_segments = len(data_loader.get_all_segment_ids())
            progress = label_manager.get_labeling_progress(total_segments)

            progress_content = dbc.Progress(
                value=progress['progress_percentage'],
                label=f"{progress['labeled_count']}/{progress['total_segments']} сегментов промаркировано",
                color="success" if progress['completion_status'] == 'Complete' else "info"
            )
        else:
            progress_content = html.P("Данные не загружены")

        # Получение статистики
        stats = label_manager.get_label_statistics()
        if stats['total_labels'] > 0:
            stats_content = dbc.Table([
                html.Tr([html.Th("Всего меток"), html.Td(stats['total_labels'])]),
                html.Tr([html.Th("Категорий"), html.Td(len(stats['categories']))]),
                html.Tr([html.Th("Аналитиков"), html.Td(len(stats['analysts']))])
            ], bordered=True, size="sm")
        else:
            stats_content = html.P("Метки недоступны")

        return progress_content, stats_content

    except Exception as e:
        logger.error(f"Ошибка обновления прогресса и статистики: {e}")
        return html.P("Ошибка загрузки прогресса"), html.P("Ошибка загрузки статистики")

@app.callback(
    Output('defect-category-dropdown', 'value'),
    [Input('current-segment-id-store', 'data')]
)
def update_label_form(segment_id):
    """Update label form with current segment's label"""
    if not segment_id:
        return None

    try:
        label = label_manager.get_label(segment_id)
        return label['defect_category'] if label else None
    except:
        return None

@app.callback(
    Output('severity-dropdown', 'value'),
    [Input('current-segment-id-store', 'data')]
)
def update_severity_form(segment_id):
    """Update severity form with current segment's label"""
    if not segment_id:
        return None

    try:
        label = label_manager.get_label(segment_id)
        return label['severity'] if label else None
    except:
        return None

@app.callback(
    Output('confidence-slider', 'value'),
    [Input('current-segment-id-store', 'data')]
)
def update_confidence_form(segment_id):
    """Update confidence form with current segment's label"""
    if not segment_id:
        return 1.0

    try:
        label = label_manager.get_label(segment_id)
        return label['confidence'] if label else 1.0
    except:
        return 1.0

@app.callback(
    Output('analyst-input', 'value'),
    [Input('current-segment-id-store', 'data')]
)
def update_analyst_form(segment_id):
    """Update analyst form with current segment's label"""
    if not segment_id:
        return ""

    try:
        label = label_manager.get_label(segment_id)
        return label['analyst'] if label else ""
    except:
        return ""

@app.callback(
    Output('comments-textarea', 'value'),
    [Input('current-segment-id-store', 'data')]
)
def update_comments_form(segment_id):
    """Update comments form with current segment's label"""
    if not segment_id:
        return ""

    try:
        label = label_manager.get_label(segment_id)
        return label['comments'] if label else ""
    except:
        return ""

@app.callback(
    Output('segment-dropdown', 'value', allow_duplicate=True),
    [Input('prev-btn', 'n_clicks'),
     Input('next-btn', 'n_clicks')],
    [State('segment-dropdown', 'options'),
     State('segment-dropdown', 'value')],
    prevent_initial_call=True
)
def navigate_segments(prev_clicks, next_clicks, options, current_value):
    """Navigate between segments"""
    if not options:
        return current_value

    current_index = next((i for i, opt in enumerate(options) if opt['value'] == current_value), 0)

    if callback_context.triggered_id == 'prev-btn':
        new_index = max(0, current_index - 1)
    elif callback_context.triggered_id == 'next-btn':
        new_index = min(len(options) - 1, current_index + 1)
    else:
        return current_value

    return options[new_index]['value']

@app.callback(
    Output('defect-category-dropdown', 'value', allow_duplicate=True),
    [Input('save-label-btn', 'n_clicks')],
    [State('current-segment-id-store', 'data'),
     State('defect-category-dropdown', 'value'),
     State('severity-dropdown', 'value'),
     State('confidence-slider', 'value'),
     State('analyst-input', 'value'),
     State('comments-textarea', 'value')],
    prevent_initial_call=True
)
def save_label(n_clicks, segment_id, defect_category, severity, confidence, analyst, comments):
    """Сохранение метки для текущего сегмента"""
    if not n_clicks or not segment_id or not defect_category or not severity:
        return defect_category

    try:
        success = label_manager.add_label(
            segment_id=segment_id,
            defect_category=defect_category,
            severity=severity,
            confidence=confidence,
            analyst=analyst,
            comments=comments
        )

        if success:
            return defect_category
        else:
            return None

    except Exception as e:
        logger.error(f"Ошибка сохранения метки: {e}")
        return None

@app.callback(
    [Output('defect-category-dropdown', 'value', allow_duplicate=True),
     Output('severity-dropdown', 'value', allow_duplicate=True),
     Output('confidence-slider', 'value', allow_duplicate=True),
     Output('analyst-input', 'value', allow_duplicate=True),
     Output('comments-textarea', 'value', allow_duplicate=True)],
    [Input('clear-label-btn', 'n_clicks')],
    prevent_initial_call=True
)
def clear_label_form(n_clicks):
    """Очистка формы метки"""
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    return None, None, 1.0, "", ""

@app.callback(
    Output('export-btn', 'children'),
    [Input('export-btn', 'n_clicks')],
    prevent_initial_call=True
)
def export_labels(n_clicks):
    """Экспорт меток в CSV"""
    if not n_clicks:
        return "Экспорт меток"

    try:
        success = label_manager.export_to_csv("app/data/labels_export.csv")
        if success:
            return "Экспорт выполнен успешно!"
        else:
            return "Ошибка экспорта"
    except Exception as e:
        logger.error(f"Ошибка экспорта меток: {e}")
        return "Ошибка экспорта"

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
