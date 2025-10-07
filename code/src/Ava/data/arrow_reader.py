"""
Arrow file reader for efficient data loading
Handles reading Apache Arrow files with proper error handling
"""

from pathlib import Path
from typing import Iterator
import pyarrow as pa


class ArrowReader:
    """Reader for Apache Arrow files"""

    @staticmethod
    def read_arrow_file(file_path: Path) -> Iterator[str]:
        """
        Read an Arrow file and yield text content

        Args:
            file_path: Path to the Arrow file

        Yields:
            Text strings from the Arrow file
        """
        try:
            # Open memory-mapped Arrow file for efficient reading
            with pa.memory_map(str(file_path), 'r') as source:
                # Read as a stream of record batches
                reader = pa.ipc.open_stream(source)

                for batch in reader:
                    # Convert to pandas for easier text extraction
                    df = batch.to_pandas()

                    # Try common text field names
                    text_fields = ['text', 'content', 'document', 'passage', 'input', 'question', 'instruction']

                    for field in text_fields:
                        if field in df.columns:
                            for text in df[field]:
                                if text and isinstance(text, str):
                                    yield text
                            break
                    else:
                        # If no standard field found, try first string column
                        for col in df.columns:
                            if df[col].dtype == 'object':  # String columns are 'object' dtype
                                for text in df[col]:
                                    if text and isinstance(text, str):
                                        yield text
                                break

        except Exception as e:
            # Try alternative reading method
            try:
                # Read as RecordBatchFileReader
                with pa.OSFile(str(file_path), 'r') as source:
                    reader = pa.ipc.RecordBatchFileReader(source)

                    for i in range(reader.num_record_batches):
                        batch = reader.get_batch(i)
                        df = batch.to_pandas()

                        # Try common text field names
                        text_fields = ['text', 'content', 'document', 'passage', 'input', 'question', 'instruction']

                        for field in text_fields:
                            if field in df.columns:
                                for text in df[field]:
                                    if text and isinstance(text, str):
                                        yield text
                                break
                        else:
                            # If no standard field found, try first string column
                            for col in df.columns:
                                if df[col].dtype == 'object':
                                    for text in df[col]:
                                        if text and isinstance(text, str):
                                            yield text
                                    break

            except Exception as inner_e:
                raise IOError(f"Failed to read Arrow file {file_path}: {e}, {inner_e}")


# Global instance for easy importing
arrow_reader = ArrowReader()