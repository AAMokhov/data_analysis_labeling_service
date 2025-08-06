"""
Модуль загрузки данных для HDF5 файлов
Обрабатывает загрузку и управление сегментированными данными из HDF5 файлов
"""

import h5py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Класс для загрузки и управления сегментированными данными из HDF5 файлов"""

    def __init__(self, file_path: str):
        """
        Инициализация DataLoader с путем к HDF5 файлу

        Args:
            file_path: Путь к HDF5 файлу, содержащему сегментированные данные
        """
        self.file_path = file_path
        self.segments = {}
        self.segment_ids = []
        self._load_segments()

    def _load_segments(self):
        """Загрузка всех сегментов из HDF5 файла"""
        try:
            with h5py.File(self.file_path, 'r') as f:
                # Переход к структуре сегментов
                if 'segments' in f:
                    segments_group = f['segments']

                                        # Поиск групп фаз (например, phase_current_T, phase_current_R, phase_current_S)
                    for phase_name in segments_group.keys():
                        phase_group = segments_group[phase_name]

                        # Получение всех ID сегментов для этой фазы
                        for segment_id in phase_group.keys():
                            segment_path = f"segments/{phase_name}/{segment_id}/data"
                            if segment_path in f:
                                self.segment_ids.append(segment_id)
                                self.segments[segment_id] = {
                                    'phase': phase_name,
                                    'path': segment_path
                                }

                logger.info(f"Загружено {len(self.segment_ids)} сегментов из {self.file_path}")

        except Exception as e:
            logger.error(f"Ошибка загрузки сегментов из {self.file_path}: {e}")
            raise

    def get_segment_data(self, segment_id: str) -> np.ndarray:
        """
        Получение данных для конкретного сегмента

        Args:
            segment_id: ID сегмента для извлечения

        Returns:
            numpy массив, содержащий данные сегмента
        """
        if segment_id not in self.segments:
            raise ValueError(f"Segment {segment_id} not found")

        try:
            with h5py.File(self.file_path, 'r') as f:
                data = f[self.segments[segment_id]['path']][:]
                return data
        except Exception as e:
            logger.error(f"Ошибка загрузки сегмента {segment_id}: {e}")
            raise

    def get_segment_info(self, segment_id: str) -> Dict:
        """
        Получение информации о конкретном сегменте

        Args:
            segment_id: ID сегмента

        Returns:
            Словарь, содержащий информацию о сегменте
        """
        if segment_id not in self.segments:
            raise ValueError(f"Segment {segment_id} not found")

        data = self.get_segment_data(segment_id)
        return {
            'id': segment_id,
            'phase': self.segments[segment_id]['phase'],
            'length': len(data),
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data)
        }

    def get_all_segment_ids(self) -> List[str]:
        """Get list of all available segment IDs"""
        return sorted(self.segment_ids)

    def get_segments_by_phase(self, phase: str) -> List[str]:
        """
        Get all segment IDs for a specific phase

        Args:
            phase: Phase name (e.g., 'phase_current_T')

        Returns:
            List of segment IDs for the specified phase
        """
        return [seg_id for seg_id, info in self.segments.items() if info['phase'] == phase]

    def get_sample_rate(self) -> Optional[float]:
        """
        Try to infer sample rate from the data structure
        This is a placeholder - actual implementation would depend on metadata

        Returns:
            Estimated sample rate in Hz
        """
        # Default assumption: 1kHz sample rate
        # In a real implementation, this would be extracted from metadata
        return 1000.0


class MultiFileDataLoader:
    """Class for loading data from multiple HDF5 files"""

    def __init__(self, file_paths: List[str]):
        """
        Initialize MultiFileDataLoader with multiple HDF5 file paths

        Args:
            file_paths: List of paths to HDF5 files
        """
        self.file_paths = file_paths
        self.loaders = {}
        self._initialize_loaders()

    def _initialize_loaders(self):
        """Initialize DataLoader for each file"""
        for file_path in self.file_paths:
            try:
                self.loaders[file_path] = DataLoader(file_path)
                logger.info(f"Инициализирован загрузчик для {file_path}")
            except Exception as e:
                logger.error(f"Не удалось инициализировать загрузчик для {file_path}: {e}")

    def get_all_segment_ids(self) -> Dict[str, List[str]]:
        """
        Get all segment IDs from all files

        Returns:
            Dictionary mapping file paths to lists of segment IDs
        """
        return {file_path: loader.get_all_segment_ids()
                for file_path, loader in self.loaders.items()}

    def get_segment_data(self, file_path: str, segment_id: str) -> np.ndarray:
        """
        Get segment data from a specific file

        Args:
            file_path: Path to the HDF5 file
            segment_id: ID of the segment

        Returns:
            numpy array containing the segment data
        """
        if file_path not in self.loaders:
            raise ValueError(f"No loader found for {file_path}")

        return self.loaders[file_path].get_segment_data(segment_id)

    def get_segment_info(self, file_path: str, segment_id: str) -> Dict:
        """
        Get segment information from a specific file

        Args:
            file_path: Path to the HDF5 file
            segment_id: ID of the segment

        Returns:
            Dictionary containing segment information
        """
        if file_path not in self.loaders:
            raise ValueError(f"No loader found for {file_path}")

        return self.loaders[file_path].get_segment_info(segment_id)
