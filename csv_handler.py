import pandas as pd
import numpy as np
import chardet
from typing import Dict, Any, Tuple, List
import re


class CSVHandler:
    def __init__(self):
        self.df = None
        self.metadata = {}
        self.analysis_results = {}

    def reset(self):
        """Reset the handler state"""
        self.df = None
        self.metadata = {}
        self.analysis_results = {}

    def detect_encoding(self, file_path: str) -> str:
        """Detect file encoding"""
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        return result['encoding']

    def load_csv(self, file_path: str) -> bool:
        """Load and process CSV file"""
        self.reset()
        try:
            encoding = self.detect_encoding(file_path)
            self.df = pd.read_csv(file_path, encoding=encoding)
            self._analyze_data()
            return True
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False

    def _analyze_data(self):
        """Perform comprehensive data analysis"""
        if self.df is None:
            return

        # Basic metadata
        self.metadata = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.astype(str).to_dict(),
            'encoding': 'UTF-8',  # Will be updated during load
            'separator': ',',  # Default, can be detected if needed
            'has_header': True  # Default assumption
        }

        # Detailed analysis
        self.analysis_results = {
            'missing_values': self._analyze_missing_values(),
            'column_stats': self._analyze_column_stats(),
            'data_quality': self._analyze_data_quality(),
            'potential_ids': self._find_potential_ids(),
            'duplicates': self._check_duplicates()
        }

    def _analyze_column_stats(self) -> Dict[str, Any]:
        """Generate statistics for each column"""
        stats = {}
        for col in self.df.columns:
            col_stats = {
                'unique_count': self.df[col].nunique(),
                'is_numeric': pd.api.types.is_numeric_dtype(self.df[col]),
                'is_categorical': self.df[col].nunique() < 20,
                'is_text': pd.api.types.is_string_dtype(self.df[col]),
                'is_datetime': pd.api.types.is_datetime64_any_dtype(self.df[col]),
                'max_length': self._get_max_length(col) if pd.api.types.is_string_dtype(self.df[col]) else None,
                'most_frequent': self.df[col].mode().iloc[0] if len(self.df[col]) > 0 else None,
                'is_constant': self.df[col].nunique() == 1
            }
            stats[col] = col_stats
        return stats

    def _get_max_length(self, column: str) -> int:
        """Get maximum length of string values in a column"""
        return self.df[column].astype(str).str.len().max()

    def _analyze_missing_values(self) -> Dict[str, Any]:
        """Analyze missing values in the dataset"""
        total_cells = self.df.size
        missing_values = self.df.isnull().sum().to_dict()
        missing_percentage = {k: (v / len(self.df)) * 100 for k, v in missing_values.items()}

        return {
            'count_by_column': missing_values,
            'percentage_by_column': missing_percentage,
            'total_missing': sum(missing_values.values()),
            'total_percentage': (sum(missing_values.values()) / total_cells) * 100
        }

    def _analyze_data_quality(self) -> Dict[str, Any]:
        """Check for data quality issues"""
        issues = {}
        for col in self.df.columns:
            col_issues = []
            if pd.api.types.is_string_dtype(self.df[col]):
                # Check for mixed formats
                if self.df[col].apply(lambda x: isinstance(x, str) and bool(re.search(r'[^a-zA-Z0-9\s]', x))).any():
                    col_issues.append('special_characters')
                # Check for numeric strings
                if self.df[col].str.match(r'^\d+$').any():
                    col_issues.append('numeric_strings')

            if pd.api.types.is_numeric_dtype(self.df[col]):
                # Check for outliers
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3 - q1
                if ((self.df[col] < (q1 - 1.5 * iqr)) | (self.df[col] > (q3 + 1.5 * iqr))).any():
                    col_issues.append('potential_outliers')

            if col_issues:
                issues[col] = col_issues
        return issues

    def _find_potential_ids(self) -> List[str]:
        """Identify columns that might be IDs"""
        potential_ids = []
        for col in self.df.columns:
            if (self.df[col].nunique() == len(self.df) and
                    pd.api.types.is_numeric_dtype(self.df[col])):
                potential_ids.append(col)
            elif (col.lower().endswith('id') or
                  col.lower().endswith('_id') or
                  col.lower().startswith('id')):
                potential_ids.append(col)
        return potential_ids

    def _check_duplicates(self) -> Dict[str, Any]:
        """Check for duplicate rows"""
        duplicates = self.df.duplicated()
        return {
            'count': duplicates.sum(),
            'percentage': (duplicates.sum() / len(self.df)) * 100,
            'rows': self.df[duplicates].index.tolist()
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of the data"""
        return {
            'metadata': self.metadata,
            'analysis': self.analysis_results
        }