"""Provenance records attached to every analysis result / export (traceability for the dissertation)."""
from __future__ import annotations

import datetime as _dt
import hashlib
import os
import platform
from functools import lru_cache
from typing import Any

from .. import __version__


@lru_cache(maxsize=256)
def _sha256_cached(path: str, mtime: float, size: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_sha256(path: str | None) -> str | None:
    """SHA-256 of a source file (cached by path+mtime+size); None if unavailable."""
    if not path or not os.path.isfile(path):
        return None
    st = os.stat(path)
    return _sha256_cached(os.path.abspath(path), st.st_mtime, st.st_size)


def _jsonable(v: Any) -> Any:
    try:
        import numpy as np
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
    except Exception:  # pragma: no cover
        pass
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def provenance(analysis: str, params: dict | None = None, ds=None, *, source_path: str | None = None,
               extra: dict | None = None) -> dict:
    """Build a JSON-serialisable provenance record.

    analysis: short name, e.g. "DSC", "Tensile", "Cyclic", "Integrate", "Grain size (E112)".
    params:   every user/auto parameter that affects the numbers.
    ds:       the Dataset analysed (name, kind, metadata, source path are recorded).
    """
    path = source_path or (getattr(ds, "source_path", None) if ds is not None else None)
    rec = {
        "analysis": analysis,
        "labstudio_version": __version__,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "parameters": _jsonable(params or {}),
    }
    if ds is not None:
        md = {k: v for k, v in (getattr(ds, "metadata", {}) or {}).items() if k != "source_meta"}
        rec["dataset"] = {"name": ds.name, "kind": ds.kind, "metadata": _jsonable(md)}
    if path:
        rec["source_file"] = os.path.basename(path)
        rec["source_sha256"] = file_sha256(path)
    if extra:
        rec.update(_jsonable(extra))
    return rec


def provenance_lines(rec: dict) -> list[str]:
    """'# key: value' comment lines for CSV headers."""
    import json
    out = [f"# LabStudio {rec.get('labstudio_version')} · {rec.get('analysis')} · {rec.get('timestamp')}"]
    if rec.get("source_file"):
        out.append(f"# source: {rec['source_file']} sha256={rec.get('source_sha256')}")
    out.append("# parameters: " + json.dumps(rec.get("parameters", {}), ensure_ascii=False))
    if rec.get("dataset"):
        out.append("# dataset: " + json.dumps(rec["dataset"], ensure_ascii=False))
    for k in ("warnings",):
        if rec.get(k):
            out.append(f"# {k}: " + "; ".join(map(str, rec[k])))
    return out
