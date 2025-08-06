#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы анализа данных
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

import numpy as np
import logging
from app.data_loader import DataLoader
from app.spectral_analysis import SpectralAnalyzer
from app.visualization import SpectralVisualizer

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_analysis():
    """Тестирование анализа данных"""
    try:
        # Инициализация компонентов
        data_loader = DataLoader('app/data/processed_current_1.h5')
        spectral_analyzer = SpectralAnalyzer(sample_rate=25600.0)  # 25.6 кГц
        visualizer = SpectralVisualizer()

        # Получение списка сегментов
        segment_ids = data_loader.get_all_segment_ids()
        logger.info(f"Найдено сегментов: {len(segment_ids)}")

        if len(segment_ids) == 0:
            logger.error("Сегменты не найдены")
            return

        # Тестирование первого сегмента
        test_segment_id = segment_ids[0]
        logger.info(f"Тестирование сегмента: {test_segment_id}")

        # Загрузка данных
        data = data_loader.get_segment_data(test_segment_id)
        logger.info(f"Загружены данные: размер = {len(data)}, тип = {type(data)}")
        logger.info(f"Данные: min = {np.min(data)}, max = {np.max(data)}, mean = {np.mean(data)}")

        # Анализ данных
        analysis_results = spectral_analyzer.analyze_segment(data)
        logger.info(f"Анализ завершен: получено {len(analysis_results)} результатов")

        # Проверка результатов анализа
        for key, value in analysis_results.items():
            if isinstance(value, dict):
                logger.info(f"  {key}: {len(value)} элементов")
            else:
                logger.info(f"  {key}: {type(value)}")

        # Тестирование создания графиков
        logger.info("Тестирование создания графиков...")

        # Временной ряд
        time_series_fig = visualizer.create_time_series_plot(data, sample_rate=25600.0, segment_id=test_segment_id)
        logger.info("График временного ряда создан успешно")

        # FFT
        if 'fft' in analysis_results:
            fft_fig = visualizer.create_fft_plot(analysis_results['fft'], segment_id=test_segment_id)
            logger.info("График FFT создан успешно")

        # Спектрограмма
        if 'stft' in analysis_results:
            spectrogram_fig = visualizer.create_spectrogram_plot(analysis_results['stft'], segment_id=test_segment_id)
            logger.info("График спектрограммы создан успешно")

        logger.info("Все тесты пройдены успешно!")

    except Exception as e:
        logger.error(f"Ошибка в тестировании: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_analysis()
