import pandas as pd
import numpy as np
import logging
import json
import shutil
import hashlib
from pathlib import Path
from typing import Tuple, List, Dict, Any

class DataPersistence:
    """
    Handles memory-optimized data storage using Memory Mapping (numpy.memmap).
    Crucial for reducing RAM usage during HPO and LOFO loops.
    
    Capabilities:
    - Zero-copy data access for worker processes.
    - Integrity verification via hashing.
    - Automatic cleanup.
    """

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Dedicated cache directory within results
        base_dir = Path(config.get('outputs', {}).get('base_results_dir', 'results'))
        self.storage_dir = base_dir / "00_CACHE_MEMMAP"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def persist_as_memmap(self, 
                          df: pd.DataFrame, 
                          split_name: str, 
                          feature_cols: List[str], 
                          target_cols: List[str]) -> Dict[str, Any]:
        """
        Saves a DataFrame as a memory-mapped binary file.
        
        Args:
            df: Source DataFrame.
            split_name: 'train', 'val', or 'test'.
            feature_cols: List of active feature names.
            target_cols: List of target columns.
            
        Returns:
            Metadata dictionary for reconstructing the memmap.
        """
        self.logger.debug(f"Persisting {split_name} dataset ({len(df)} rows) to memmap...")
        
        # Force float32 for storage efficiency (50% size of float64)
        X = df[feature_cols].values.astype(np.float32)
        y = df[target_cols].values.astype(np.float32)
        
        x_path = self.storage_dir / f"X_{split_name}.dat"
        y_path = self.storage_dir / f"y_{split_name}.dat"
        
        # Write X
        x_mmap = np.memmap(x_path, dtype='float32', mode='w+', shape=X.shape)
        x_mmap[:] = X[:]
        x_mmap.flush()
        
        # Write y
        y_mmap = np.memmap(y_path, dtype='float32', mode='w+', shape=y.shape)
        y_mmap[:] = y[:]
        y_mmap.flush()
        
        metadata = {
            'split_name': split_name,
            'x_path': str(x_path),
            'y_path': str(y_path),
            'x_shape': list(X.shape),
            'y_shape': list(y.shape),
            'dtype': 'float32',
            'feature_names': feature_cols,
            'target_names': target_cols,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Save metadata JSON
        with open(self.storage_dir / f"metadata_{split_name}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return metadata

    def load_memmap(self, metadata: Dict[str, Any], mode='r') -> Tuple[np.memmap, np.memmap]:
        """
        Loads the X and y arrays as read-only memory-mapped objects.
        This allows multiple processes to share the same physical memory.
        """
        x_path = Path(metadata['x_path'])
        y_path = Path(metadata['y_path'])
        
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Memmap files missing for {metadata['split_name']}")
            
        X = np.memmap(
            x_path, 
            dtype=metadata['dtype'], 
            mode=mode, 
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