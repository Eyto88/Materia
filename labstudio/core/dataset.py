"""Core data model shared by every LabStudio module.

A ``Dataset`` is one imported file (or a derived curve). Its DataFrame holds
values ALREADY CONVERTED to canonical units (see ``units.CANONICAL``), and
``channels`` records each column's dimension/unit so plots and analysis can label
axes and convert for display.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import units as U

DATASET_KINDS = ("generic", "dsc", "tensile", "compression", "cyclic")

_id_counter = itertools.count(1)


@dataclass
class Channel:
    name: str                 # column name in Dataset.data
    dimension: str            # key in units.CANONICAL
    unit: str                 # canonical unit the stored values are in
    original_unit: str | None = None  # unit as written in the source file

    @property
    def label(self) -> str:
        return U.axis_label(self.name, self.unit)


@dataclass
class Dataset:
    name: str
    data: pd.DataFrame
    channels: dict[str, Channel]
    kind: str = "generic"
    source_path: str | None = None
    # Sample/test metadata in canonical units, e.g.
    #   tensile/compression/cyclic: area_mm2, gauge_length_mm, diameter_mm, thickness_mm, width_mm
    #   dsc: mass_mg, heating_rate_K_per_min
    metadata: dict[str, Any] = field(default_factory=dict)
    default_x: str | None = None
    default_y: str | None = None
    id: int = field(default_factory=lambda: next(_id_counter))

    # ---- access -----------------------------------------------------------
    def columns(self) -> list[str]:
        return list(self.data.columns)

    def values(self, column: str, unit: str | None = None) -> np.ndarray:
        """Column as float array, optionally converted to ``unit``."""
        arr = pd.to_numeric(self.data[column], errors="coerce").to_numpy(dtype=float)
        ch = self.channels.get(column)
        if unit and ch and unit != ch.unit:
            arr = U.convert(arr, ch.unit, unit, ch.dimension)
        return arr

    def channel(self, column: str) -> Channel:
        if column not in self.channels:
            self.channels[column] = Channel(column, "dimensionless", "")
        return self.channels[column]

    def label(self, column: str) -> str:
        return self.channel(column).label

    def columns_of(self, dimension: str) -> list[str]:
        return [c for c, ch in self.channels.items() if ch.dimension == dimension]

    def first_of(self, *dimensions: str) -> str | None:
        for d in dimensions:
            cols = self.columns_of(d)
            if cols:
                return cols[0]
        return None

    # ---- mutation ---------------------------------------------------------
    def add_column(self, name: str, values, dimension: str, unit: str | None = None) -> str:
        """Add a derived column; ``unit`` defaults to the canonical unit (values converted if not)."""
        unit = unit or U.CANONICAL[dimension]
        vals = np.asarray(values, dtype=float)
        if unit != U.CANONICAL[dimension]:
            vals = U.to_canonical(vals, unit, dimension)
            unit = U.CANONICAL[dimension]
        self.data[name] = vals
        self.channels[name] = Channel(name, dimension, unit)
        return name

    def subset(self, mask: np.ndarray, name: str | None = None) -> "Dataset":
        return Dataset(
            name=name or f"{self.name} (subset)",
            data=self.data.loc[np.asarray(mask, bool)].reset_index(drop=True),
            channels={k: Channel(**vars(v)) for k, v in self.channels.items()},
            kind=self.kind, source_path=self.source_path, metadata=dict(self.metadata),
            default_x=self.default_x, default_y=self.default_y,
        )

    @classmethod
    def from_arrays(cls, name: str, columns: dict[str, tuple[np.ndarray, str, str]],
                    kind: str = "generic", **kw) -> "Dataset":
        """columns: {name: (values, dimension, unit)}; values converted to canonical."""
        df = pd.DataFrame()
        chans: dict[str, Channel] = {}
        for col, (vals, dim, unit) in columns.items():
            df[col] = U.to_canonical(np.asarray(vals, float), unit, dim)
            chans[col] = Channel(col, dim, U.CANONICAL[dim], original_unit=unit)
        cols = list(columns)
        kw.setdefault("default_x", cols[0] if cols else None)
        kw.setdefault("default_y", cols[1] if len(cols) > 1 else None)
        return cls(name=name, data=df, channels=chans, kind=kind, **kw)


class DatasetStore:
    """Session-wide list of datasets with simple change callbacks (GUI uses a Qt signal wrapper)."""

    def __init__(self) -> None:
        self._items: list[Dataset] = []
        self._listeners: list = []

    def subscribe(self, fn) -> None:
        self._listeners.append(fn)

    def _emit(self) -> None:
        for fn in self._listeners:
            fn()

    def add(self, ds: Dataset) -> Dataset:
        self._items.append(ds)
        self._emit()
        return ds

    def remove(self, ds: Dataset) -> None:
        self._items = [d for d in self._items if d.id != ds.id]
        self._emit()

    def rename(self, ds: Dataset, name: str) -> None:
        ds.name = name
        self._emit()

    def move(self, ds: Dataset, index: int) -> None:
        items = [d for d in self._items if d.id != ds.id]
        index = max(0, min(index, len(items)))
        items.insert(index, ds)
        self._items = items
        self._emit()

    def index_of(self, ds: Dataset) -> int:
        return next((i for i, d in enumerate(self._items) if d.id == ds.id), -1)

    def get(self, ds_id: int) -> Dataset | None:
        return next((d for d in self._items if d.id == ds_id), None)

    def notify_changed(self) -> None:
        self._emit()

    def __iter__(self):
        return iter(list(self._items))

    def __len__(self) -> int:
        return len(self._items)
