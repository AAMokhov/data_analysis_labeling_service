"""
Label Manager Module
Handles saving and loading of segment labels and annotations
"""

import h5py
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)


class LabelManager:
    """Class for managing segment labels and annotations"""

    # Predefined defect categories
    DEFECT_CATEGORIES = [
        "Нормальное состояние",
        "Дефект наружного кольца",
        "Дефект внутреннего кольца",
        "Дефект тел качения",
        "Дефект сепаратора",
        "Дисбаланс",
        "Перекос",
        "Другое"
    ]

    # Defect severity levels
    SEVERITY_LEVELS = [
        "Начальная",
        "Средняя",
        "Высокая",
        "Критическая"
    ]

    def __init__(self, output_file: str = "labeled_data.h5"):
        """
        Initialize LabelManager

        Args:
            output_file: Path to the HDF5 file for saving labels
        """
        self.output_file = output_file
        self.labels = {}
        self.metadata = {}
        self._load_existing_labels()

    def _load_existing_labels(self):
        """Load existing labels from the output file if it exists"""
        if os.path.exists(self.output_file):
            try:
                with h5py.File(self.output_file, 'r') as f:
                    # Load labels
                    if 'labels' in f:
                        for segment_id in f['labels'].keys():
                            self.labels[segment_id] = dict(f['labels'][segment_id].attrs)

                    # Load metadata
                    if 'metadata' in f:
                        self.metadata = dict(f['metadata'].attrs)

                logger.info(f"Loaded {len(self.labels)} existing labels from {self.output_file}")

            except Exception as e:
                logger.error(f"Error loading existing labels: {e}")
                # Create new file if loading fails
                self._create_new_file()
        else:
            self._create_new_file()

    def _create_new_file(self):
        """Create a new HDF5 file for labels"""
        try:
            with h5py.File(self.output_file, 'w') as f:
                # Create groups
                f.create_group('labels')
                f.create_group('metadata')

                # Set metadata
                f['metadata'].attrs['created'] = datetime.now().isoformat()
                f['metadata'].attrs['version'] = '1.0'
                f['metadata'].attrs['defect_categories'] = json.dumps(self.DEFECT_CATEGORIES)
                f['metadata'].attrs['severity_levels'] = json.dumps(self.SEVERITY_LEVELS)

            logger.info(f"Created new label file: {self.output_file}")

        except Exception as e:
            logger.error(f"Error creating new label file: {e}")
            raise

    def add_label(self, segment_id: str,
                  defect_category: str,
                  severity: str,
                  confidence: float = 1.0,
                  comments: str = "",
                  analyst: str = "",
                  timestamp: Optional[str] = None) -> bool:
        """
        Add or update a label for a segment

        Args:
            segment_id: ID of the segment
            defect_category: Category of defect (must be in DEFECT_CATEGORIES)
            severity: Severity level (must be in SEVERITY_LEVELS)
            confidence: Confidence level (0.0 to 1.0)
            comments: Additional comments
            analyst: Name of the analyst
            timestamp: Timestamp of labeling (default: current time)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate inputs
            if defect_category not in self.DEFECT_CATEGORIES:
                raise ValueError(f"Invalid defect category: {defect_category}")

            if severity not in self.SEVERITY_LEVELS:
                raise ValueError(f"Invalid severity level: {severity}")

            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"Confidence must be between 0.0 and 1.0")

            if timestamp is None:
                timestamp = datetime.now().isoformat()

            # Create label data
            label_data = {
                'defect_category': defect_category,
                'severity': severity,
                'confidence': confidence,
                'comments': comments,
                'analyst': analyst,
                'timestamp': timestamp,
                'last_modified': datetime.now().isoformat()
            }

            # Update in-memory storage
            self.labels[segment_id] = label_data

            # Save to file
            self._save_labels()

            logger.info(f"Added/updated label for segment {segment_id}: {defect_category} - {severity}")
            return True

        except Exception as e:
            logger.error(f"Error adding label for segment {segment_id}: {e}")
            return False

    def get_label(self, segment_id: str) -> Optional[Dict]:
        """
        Get label for a specific segment

        Args:
            segment_id: ID of the segment

        Returns:
            Label data dictionary or None if not found
        """
        return self.labels.get(segment_id)

    def get_all_labels(self) -> Dict[str, Dict]:
        """Get all labels"""
        return self.labels.copy()

    def remove_label(self, segment_id: str) -> bool:
        """
        Remove label for a specific segment

        Args:
            segment_id: ID of the segment

        Returns:
            True if successful, False otherwise
        """
        try:
            if segment_id in self.labels:
                del self.labels[segment_id]
                self._save_labels()
                logger.info(f"Removed label for segment {segment_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Error removing label for segment {segment_id}: {e}")
            return False

    def get_labels_by_category(self, defect_category: str) -> Dict[str, Dict]:
        """
        Get all labels for a specific defect category

        Args:
            defect_category: Category to filter by

        Returns:
            Dictionary of segment IDs and their labels
        """
        return {seg_id: label for seg_id, label in self.labels.items()
                if label['defect_category'] == defect_category}

    def get_labels_by_severity(self, severity: str) -> Dict[str, Dict]:
        """
        Get all labels for a specific severity level

        Args:
            severity: Severity level to filter by

        Returns:
            Dictionary of segment IDs and their labels
        """
        return {seg_id: label for seg_id, label in self.labels.items()
                if label['severity'] == severity}

    def get_label_statistics(self) -> Dict:
        """
        Get statistics about the labels

        Returns:
            Dictionary containing label statistics
        """
        if not self.labels:
            return {
                'total_labels': 0,
                'categories': {},
                'severities': {},
                'analysts': {}
            }

        stats = {
            'total_labels': len(self.labels),
            'categories': {},
            'severities': {},
            'analysts': {}
        }

        # Count by category
        for label in self.labels.values():
            category = label['defect_category']
            severity = label['severity']
            analyst = label['analyst']

            stats['categories'][category] = stats['categories'].get(category, 0) + 1
            stats['severities'][severity] = stats['severities'].get(severity, 0) + 1
            if analyst:
                stats['analysts'][analyst] = stats['analysts'].get(analyst, 0) + 1

        return stats

    def export_to_csv(self, output_path: str) -> bool:
        """
        Export labels to CSV file

        Args:
            output_path: Path to the output CSV file

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.labels:
                logger.warning("No labels to export")
                return False

            # Convert to DataFrame
            data = []
            for segment_id, label in self.labels.items():
                row = {
                    'segment_id': segment_id,
                    **label
                }
                data.append(row)

            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)

            logger.info(f"Exported {len(self.labels)} labels to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting labels to CSV: {e}")
            return False

    def import_from_csv(self, input_path: str, overwrite: bool = False) -> bool:
        """
        Import labels from CSV file

        Args:
            input_path: Path to the input CSV file
            overwrite: Whether to overwrite existing labels

        Returns:
            True if successful, False otherwise
        """
        try:
            df = pd.read_csv(input_path)

            imported_count = 0
            for _, row in df.iterrows():
                segment_id = row['segment_id']

                # Skip if label exists and overwrite is False
                if segment_id in self.labels and not overwrite:
                    continue

                # Add label
                success = self.add_label(
                    segment_id=segment_id,
                    defect_category=row['defect_category'],
                    severity=row['severity'],
                    confidence=row.get('confidence', 1.0),
                    comments=row.get('comments', ''),
                    analyst=row.get('analyst', ''),
                    timestamp=row.get('timestamp', None)
                )

                if success:
                    imported_count += 1

            logger.info(f"Imported {imported_count} labels from {input_path}")
            return True

        except Exception as e:
            logger.error(f"Error importing labels from CSV: {e}")
            return False

    def _save_labels(self):
        """Save labels to HDF5 file"""
        try:
            with h5py.File(self.output_file, 'a') as f:
                # Clear existing labels group
                if 'labels' in f:
                    del f['labels']

                # Create new labels group
                labels_group = f.create_group('labels')

                # Save each label
                for segment_id, label_data in self.labels.items():
                    segment_group = labels_group.create_group(segment_id)
                    for key, value in label_data.items():
                        segment_group.attrs[key] = value

                # Update metadata
                f['metadata'].attrs['last_modified'] = datetime.now().isoformat()
                f['metadata'].attrs['total_labels'] = len(self.labels)

        except Exception as e:
            logger.error(f"Error saving labels: {e}")
            raise

    def backup_labels(self, backup_path: str) -> bool:
        """
        Create a backup of the labels file

        Args:
            backup_path: Path for the backup file

        Returns:
            True if successful, False otherwise
        """
        try:
            import shutil
            shutil.copy2(self.output_file, backup_path)
            logger.info(f"Created backup at {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False

    def get_labeling_progress(self, total_segments: int) -> Dict:
        """
        Get labeling progress statistics

        Args:
            total_segments: Total number of segments available

        Returns:
            Dictionary containing progress statistics
        """
        labeled_count = len(self.labels)
        unlabeled_count = total_segments - labeled_count
        progress_percentage = (labeled_count / total_segments * 100) if total_segments > 0 else 0

        return {
            'total_segments': total_segments,
            'labeled_count': labeled_count,
            'unlabeled_count': unlabeled_count,
            'progress_percentage': progress_percentage,
            'completion_status': 'Complete' if labeled_count >= total_segments else 'In Progress'
        }
