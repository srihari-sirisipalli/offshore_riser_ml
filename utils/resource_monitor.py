"""
Resource monitoring helpers for lightweight dashboards.

Collects CPU/memory/disk/process stats using psutil when available and
persists a small dashboard-friendly payload.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List


def _safe_import_psutil():
    try:
        import psutil  # type: ignore
        return psutil
    except Exception:
        return None


def capture_resource_snapshot() -> Dict[str, Any]:
    """
    Capture a single-point snapshot of system utilization.

    Returns a dict that is JSON/Parquet friendly. Falls back to
    minimal info if psutil is unavailable.
    """
    snapshot: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cpu_count": os.cpu_count(),
    }
    psutil = _safe_import_psutil()
    if psutil is None:
        snapshot["psutil_available"] = False
        return snapshot

    try:
        snapshot["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        vm = psutil.virtual_memory()
        snapshot["memory"] = {
            "total_mb": round(vm.total / (1024 ** 2), 2),
            "available_mb": round(vm.available / (1024 ** 2), 2),
            "used_mb": round(vm.used / (1024 ** 2), 2),
            "percent": vm.percent,
        }
        disks: List[Dict[str, Any]] = []
        for part in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(part.mountpoint)
                disks.append(
                    {
                        "device": part.device,
                        "mountpoint": part.mountpoint,
                        "total_mb": round(usage.total / (1024 ** 2), 2),
                        "used_mb": round(usage.used / (1024 ** 2), 2),
                        "free_mb": round(usage.free / (1024 ** 2), 2),
                        "percent": usage.percent,
                    }
                )
            except Exception:
                continue
        snapshot["disks"] = disks

        proc = psutil.Process()
        with proc.oneshot():
            mem_info = proc.memory_info()
            snapshot["process"] = {
                "rss_mb": round(mem_info.rss / (1024 ** 2), 2),
                "vms_mb": round(mem_info.vms / (1024 ** 2), 2),
                "num_threads": proc.num_threads(),
                "open_files": len(proc.open_files() or []),
            }
    except Exception:
        snapshot["psutil_available"] = False

    return snapshot


def write_resource_dashboard(base_dir: Path, excel_copy: bool = False) -> Path:
    """
    Persist a small resource dashboard (JSON + Parquet friendly dict list)
    under DATA_INTEGRITY_DIR.
    """
    from utils import constants

    base_dir = Path(base_dir)
    out_dir = base_dir / constants.DATA_INTEGRITY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot = capture_resource_snapshot()

    # Write JSON for human/debug use
    json_path = out_dir / "resource_utilization_dashboard.json"
    json_path.write_text(json.dumps(snapshot, indent=2))

    try:
        import pandas as pd  # type: ignore

        rows: List[Dict[str, Any]] = []
        rows.append(
            {
                "metric": "cpu_percent",
                "value": snapshot.get("cpu_percent"),
                "timestamp": snapshot.get("timestamp"),
            }
        )
        mem = snapshot.get("memory") or {}
        for key, val in mem.items():
            rows.append(
                {
                    "metric": f"memory_{key}",
                    "value": val,
                    "timestamp": snapshot.get("timestamp"),
                }
            )
        for disk in snapshot.get("disks") or []:
            rows.append(
                {
                    "metric": "disk_percent",
                    "value": disk.get("percent"),
                    "mountpoint": disk.get("mountpoint"),
                    "timestamp": snapshot.get("timestamp"),
                }
            )
        proc = snapshot.get("process") or {}
        for key, val in proc.items():
            rows.append(
                {
                    "metric": f"process_{key}",
                    "value": val,
                    "timestamp": snapshot.get("timestamp"),
                }
            )

        df = pd.DataFrame(rows)
        parquet_path = out_dir / "resource_utilization_dashboard.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)
        if excel_copy:
            df.to_excel(parquet_path.with_suffix(".xlsx"), index=False)
    except Exception:
        # Leave JSON-only if pandas/pyarrow not available
        parquet_path = json_path

    return json_path
