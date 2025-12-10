import pandas as pd
import numpy as np
import logging
import json
import os
import shutil
import hashlib
from pathlib import Path
from typing import Tuple, List, Dict, Any

class DataPersistence:
    """
    Handles memory-optimized data storage and retrieval using Memory Mapping.
    
    Why Memmap?
    Grid Search and LOFO require training thousands of models on slightly different
    subsets of features. Copying the full dataset for every process exhausts RAM.
    Memmap allows multiple processes to read the same data file on disk as if it
    were in memory, without duplication (OS handles paging).
    
    Responsibilities:
    1. Convert Pandas DataFrames -> .npy memmap files.
    2. Provide an interface to load memmaps in 'read-only' mode.
    3. Calculate and verify hashes for data integrity.
    """

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Use a designated temp/cache directory for memmaps
        # Default to a folder inside 'results' to keep it self-contained
        base_dir = Path(config.get('outputs', {}).get('base_results_dir', 'results'))
        self.storage_dir = base_dir / "00_CACHE_MEMMAP"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def persist_as_memmap(self, 
                          df: pd.DataFrame, 
                          split_name: str, 
                          feature_cols: List[str], 
                          target_cols: List[str]) -> Dict[str, Any]:
        """
        Saves a DataFrame as a memory-mapped dictionary of arrays.
        
        Args:
            df: The dataframe to save.
            split_name: 'train', 'val', or 'test'.
            feature_cols: List of active feature names.
            target_cols: List of target columns [sin, cos].
            
        Returns:
            Metadata dictionary needed to reconstruct the memmap later.
        """
        self.logger.info(f"Persisting {split_name} dataset to memmap...")
        
        # 1. Prepare Data
        # Ensure float32 for efficiency as per spec
        X = df[feature_cols].values.astype(np.float32)
        y = df[target_cols].values.astype(np.float32)
        
        # 2. Define Paths
        x_path = self.storage_dir / f"X_{split_name}.dat"
        y_path = self.storage_dir / f"y_{split_name}.dat"
        
        # 3. Create Memmaps
        # We write to file immediately
        x_mmap = np.memmap(x_path, dtype='float32', mode='w+', shape=X.shape)
        x_mmap[:] = X[:]
        x_mmap.flush() # Ensure written to disk
        
        y_mmap = np.memmap(y_path, dtype='float32', mode='w+', shape=y.shape)
        y_mmap[:] = y[:]
        y_mmap.flush()
        
        # 4. Generate Metadata
        # We explicitly convert shapes to lists to match JSON serialization behavior
        metadata = {
            'split_name': split_name,
            'x_path': str(x_path),
            'y_path': str(y_path),
            'x_shape': list(X.shape),
            'y_shape': list(y.shape),
            'dtype': 'float32',
            'feature_names': feature_cols,
            'target_names': target_cols,
            'x_hash': self._compute_file_hash(x_path),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Save metadata to JSON for easy reload
        meta_path = self.storage_dir / f"metadata_{split_name}.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return metadata

    def load_memmap(self, metadata: Dict[str, Any], mode='r') -> Tuple[np.memmap, np.memmap]:
        """
        Loads the X and y arrays as memory-mapped objects.
        
        Args:
            metadata: The dict returned by persist_as_memmap (or loaded from JSON).
            mode: 'r' (read-only) recommended for training to prevent corruption.
            
        Returns:
            (X_memmap, y_memmap)
        """
        x_path = Path(metadata['x_path'])
        y_path = Path(metadata['y_path'])
        
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Memmap files missing for {metadata['split_name']}")
            
        X = np.memmap(
            x_path, 
            dtype=metadata['dtype'], 
            mode=mode, 
            # Ensure shape is a tuple for np.memmap
            shape=tuple(metadata['x_shape'])
        )
        
        y = np.memmap(
            y_path, 
            dtype=metadata['dtype'], 
            mode=mode, 
            shape=tuple(metadata['y_shape'])
        )
        
        return X, y

    def cleanup(self):
        """Removes the temp memmap directory to free disk space."""
        try:
            if self.storage_dir.exists():
                shutil.rmtree(self.storage_dir)
                self.logger.info("Memmap cache cleared.")
        except Exception as e:
            self.logger.warning(f"Failed to clear memmap cache: {e}")

    def _compute_file_hash(self, filepath: Path) -> str:
        """Computes MD5 hash of the binary file for integrity checking."""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            # Read in chunks to avoid memory spike
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()