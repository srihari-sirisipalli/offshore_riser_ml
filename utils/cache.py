"""
Lightweight caching utilities.

- Provides a stable fingerprint helper (lru_cache) for repeated keys.
- Centralizes on-disk cache location helpers for joblib or manual caches.
"""

from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Union


@lru_cache(maxsize=256)
def fingerprint(key: str) -> str:
    """
    Deterministically hash a string key (cached to avoid repeated hashing).
    """
    return sha256(key.encode("utf-8")).hexdigest()


def ensure_cache_dir(base_dir: Union[str, Path], namespace: str) -> Path:
    """
    Create/return a cache directory rooted under base_dir/.cache/{namespace}.
    """
    base = Path(base_dir)
    cache_dir = base / ".cache" / namespace
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
