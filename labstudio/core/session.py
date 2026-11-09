"""Session save / load (.lstudio = zip of datasets + JSON state of every panel).

Format (zip):
  manifest.json   {"format": "labstudio-session", "version": 1, "labstudio_version": ..., "datasets": [...],
                   "panels": {panel_key: state_dict}}
  datasets/<id>.csv   values (canonical units), one per dataset
Each dataset entry: id, name, kind, source_path, metadata, default_x, default_y,
channels: {col: {dimension, unit, original_unit}}.

Panels implement   get_state() -> dict   (JSON-serialisable; reference datasets by their *id*)
               and set_state(state: dict, id_map: dict[int, Dataset]) -> None
id_map maps the ids written in the file to the freshly loaded Dataset objects.
Images: the micrograph workspace stores its images itself as PNG/NPY members under images/
via the optional hooks  get_state_files() -> dict[name, bytes]  and  set_state_files(files: dict[name, bytes]).
"""
from __future__ import annotations

import io
import json
import zipfile
from typing import Any

import numpy as np
import pandas as pd

from .. import __version__
from .dataset import Channel, Dataset, DatasetStore
from .provenance import _jsonable

FORMAT = "labstudio-session"
VERSION = 1
EXTENSION = ".lstudio"


def _ds_entry(ds: Dataset) -> dict:
    return {
        "id": ds.id, "name": ds.name, "kind": ds.kind, "source_path": ds.source_path,
        "metadata": _jsonable(ds.metadata), "default_x": ds.default_x, "default_y": ds.default_y,
        "channels": {c: {"dimension": ch.dimension, "unit": ch.unit, "original_unit": ch.original_unit}
                     for c, ch in ds.channels.items()},
        "columns": list(ds.data.columns),
    }


def save_session(path: str, store: DatasetStore, panels: dict[str, Any] | None = None) -> None:
    panels = panels or {}
    manifest = {"format": FORMAT, "version": VERSION, "labstudio_version": __version__,
                "datasets": [], "panels": {}}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for ds in store:
            manifest["datasets"].append(_ds_entry(ds))
            buf = io.StringIO()
            ds.data.to_csv(buf, index=False, float_format="%.17g")
            z.writestr(f"datasets/{ds.id}.csv", buf.getvalue())
        for key, panel in panels.items():
            if panel is None or not hasattr(panel, "get_state"):
                continue
            manifest["panels"][key] = _jsonable(panel.get_state())
            if hasattr(panel, "get_state_files"):
                for name, data in (panel.get_state_files() or {}).items():
                    z.writestr(f"files/{key}/{name}", data)
        z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=1))


def load_session(path: str, store: DatasetStore, panels: dict[str, Any] | None = None,
                 *, clear: bool = True) -> dict[int, Dataset]:
    """Load into ``store`` (cleared first by default) and restore panel states. Returns id_map."""
    panels = panels or {}
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("manifest.json").decode("utf-8"))
        if manifest.get("format") != FORMAT:
            raise ValueError("Not a LabStudio session file")
        if clear:
            for ds in list(store):
                store.remove(ds)
        id_map: dict[int, Dataset] = {}
        for e in manifest["datasets"]:
            df = pd.read_csv(io.BytesIO(z.read(f"datasets/{e['id']}.csv")), float_precision="round_trip").astype(float)
            df.columns = e.get("columns", list(df.columns))
            chans = {c: Channel(c, v["dimension"], v["unit"], v.get("original_unit"))
                     for c, v in e["channels"].items()}
            ds = Dataset(name=e["name"], data=df, channels=chans, kind=e["kind"],
                         source_path=e.get("source_path"), metadata=e.get("metadata") or {},
                         default_x=e.get("default_x"), default_y=e.get("default_y"))
            id_map[int(e["id"])] = ds
            store.add(ds)
        files_by_panel: dict[str, dict[str, bytes]] = {}
        for name in z.namelist():
            if name.startswith("files/"):
                _, key, fname = name.split("/", 2)
                files_by_panel.setdefault(key, {})[fname] = z.read(name)
        for key, state in manifest.get("panels", {}).items():
            panel = panels.get(key)
            if panel is None:
                continue
            if hasattr(panel, "set_state_files") and key in files_by_panel:
                panel.set_state_files(files_by_panel[key])
            if hasattr(panel, "set_state"):
                panel.set_state(state, id_map)
    return id_map
