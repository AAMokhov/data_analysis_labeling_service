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

    def get_all_suffixes(self) -> List[str]:
        """Get sorted list of unique suffixes across all segments."""
        suffixes = set()
        for sid in self.segment_ids:
            sfx = self._extract_suffix(sid)
            if sfx is not None:
                suffixes.add(sfx)
        return sorted(suffixes)

    def get_related_segment_ids_by_suffix(self, suffix: str) -> Dict[str, str]:
        """Find segments for all phases by provided suffix."""
        related: Dict[str, str] = {}
        for sid in self.segment_ids:
            if sid.endswith(suffix):
                letter = self._extract_phase_letter(sid)
                if letter:
                    related[letter] = sid
        return related

    def get_multi_phase_data_by_suffix(self, suffix: str) -> Dict[str, np.ndarray]:
        """Load arrays for all phases matching given suffix."""
        out: Dict[str, np.ndarray] = {}
        related = self.get_related_segment_ids_by_suffix(suffix)
        for letter, sid in related.items():
            try:
                out[letter] = self.get_segment_data(sid)
            except Exception:
                continue
        return out

    def _extract_suffix(self, segment_id: str) -> Optional[str]:
        """
        Extract numeric suffix from segment_id, e.g. 'current_R_000123' -> '000123'.
        Fallback to part after last underscore if no digits.
        """
        import re
        m = re.search(r"_(\d+)$", segment_id)
        if m:
            return m.group(1)
        parts = segment_id.split('_')
        return parts[-1] if len(parts) > 1 else None

    def _extract_phase_letter(self, segment_id: str) -> Optional[str]:
        """
        Try to get phase letter from known group name or from segment id.
        Returns one of ['R','S','T'] if detected, else None.
        """
        try:
            phase_group = self.segments[segment_id]['phase']  # e.g. 'phase_current_R'
            for letter in ['R', 'S', 'T']:
                if phase_group.endswith(letter):
                    return letter
        except Exception:
            pass
        # Fallback parse from segment_id like 'current_R_000000'
        for letter in ['_R_', '_S_', '_T_']:
            if letter in segment_id:
                return letter.strip('_')
        return None

    def get_related_segment_ids(self, segment_id: str) -> Dict[str, str]:
        """
        Find segments across phases (R,S,T) that share the same index suffix as given segment.

        Returns:
            Dict phase_letter -> segment_id
        """
        related: Dict[str, str] = {}
        suffix = self._extract_suffix(segment_id)
        if suffix is None:
            return related
        for sid in self.segment_ids:
            if sid.endswith(suffix):
                letter = self._extract_phase_letter(sid)
                if letter:
                    related[letter] = sid
        return related

    def get_multi_phase_data(self, segment_id: str) -> Dict[str, np.ndarray]:
        """
        Load data arrays for all phases related to given segment id (same suffix).

        Returns:
            Dict phase_letter -> numpy array
        """
        out: Dict[str, np.ndarray] = {}
        related = self.get_related_segment_ids(segment_id)
        for letter, sid in related.items():
            try:
                out[letter] = self.get_segment_data(sid)
            except Exception:
                continue
        return out

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
