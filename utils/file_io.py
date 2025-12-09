import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

class AsyncFileWriter:
    """Async wrapper for file I/O operations."""

    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def write_excel_async(self, df: pd.DataFrame, path: Path) -> None:
        """Non-blocking Excel write."""
        self.executor.submit(df.to_excel, path, index=False)

    def wait_all(self):
        """Wait for all pending writes."""
        self.executor.shutdown(wait=True)
