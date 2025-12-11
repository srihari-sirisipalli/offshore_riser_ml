import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

def save_dataframe(df: pd.DataFrame, path: Path, *, excel_copy: bool = False, index: bool = False) -> Path:
    """
    Save a DataFrame to Parquet for fast I/O with an optional Excel copy for human readability.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=index)

    if excel_copy:
        excel_path = path.with_suffix(".xlsx")
        excel_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(excel_path, index=index)

    return path


def read_dataframe(path: Path) -> pd.DataFrame:
    """
    Load a DataFrame from Parquet/Excel/CSV based on file extension.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file extension for reading: {suffix}")

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
