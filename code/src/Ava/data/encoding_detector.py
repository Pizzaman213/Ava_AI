"""
Encoding detector for robust file reading
Handles various text encodings and Unicode errors
"""

from pathlib import Path
from typing import Iterator
import chardet


class EncodingDetector:
    """Detector for file encodings with robust fallback mechanisms"""

    @staticmethod
    def detect_encoding(file_path: Path, sample_size: int = 10000) -> str:
        """
        Detect the encoding of a file

        Args:
            file_path: Path to the file
            sample_size: Number of bytes to sample for detection

        Returns:
            Detected encoding name
        """
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(sample_size)
                result = chardet.detect(raw_data)
                encoding = result.get('encoding', 'utf-8')

                # Default to utf-8 if detection is uncertain
                if encoding is None or result.get('confidence', 0) < 0.7:
                    encoding = 'utf-8'

                return encoding
        except Exception:
            return 'utf-8'

    @staticmethod
    def read_file_robust(file_path: Path) -> Iterator[str]:
        """
        Read a file with robust encoding handling

        Args:
            file_path: Path to the file to read

        Yields:
            Lines from the file
        """
        # OPTIMIZATION: .jsonl files are almost always UTF-8, skip slow detection
        # This speeds up reading by 10-100x for large files
        if str(file_path).endswith('.jsonl') or str(file_path).endswith('.json'):
            # Fast path for JSON files - just use UTF-8
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    for line in f:
                        line = line.strip()
                        if line:  # Skip empty lines
                            yield line
                return
            except Exception:
                pass  # Fall back to robust detection

        # Try multiple encodings in order of likelihood
        encodings = [
            'utf-8',
            'utf-8-sig',  # UTF-8 with BOM
            'latin-1',
            'cp1252',  # Windows encoding
            'iso-8859-1',
        ]

        # First try to detect encoding (skip for performance on JSON files)
        try:
            detected_encoding = EncodingDetector.detect_encoding(file_path)
            if detected_encoding not in encodings:
                encodings.insert(0, detected_encoding)
        except Exception:
            pass

        # Try each encoding
        last_error = None
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    for line in f:
                        line = line.strip()
                        if line:  # Skip empty lines
                            yield line
                return  # Successfully read file
            except Exception as e:
                last_error = e
                continue

        # If all encodings fail, raise the last error
        if last_error:
            raise IOError(f"Failed to read file {file_path} with any encoding: {last_error}")


# Global instance for easy importing
encoding_detector = EncodingDetector()