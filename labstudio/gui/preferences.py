"""Preferences dialog: number formatting, CSV precision, default figure style template."""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel,
                               QSpinBox, QVBoxLayout)

from .app_settings import format_value, settings


def _style_template_names() -> list[str]:
    try:  # provided by the plotting module (v1.1)
        from .plot_templates import list_templates
        return [t if isinstance(t, str) else getattr(t, "name", str(t)) for t in list_templates()]
    except Exception:
        return []


class PreferencesDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        s = settings()
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.sig = QSpinBox()
        self.sig.setRange(1, 15)
        self.sig.setValue(s.sig_figs)
        self.sig.setToolTip("Significant figures used in results tables, copy to clipboard and CSV export.")
        self.example = QLabel()
        self.sig.valueChanged.connect(self._update_example)
        form.addRow("Significant figures", self.sig)
        form.addRow("Example", self.example)
        self.full = QCheckBox("Write full precision to CSV (ignore significant figures)")
        self.full.setChecked(bool(s.get("csv_full_precision")))
        form.addRow("", self.full)
        self.tmpl = QComboBox()
        self.tmpl.addItem("(none)", "")
        for n in _style_template_names():
            self.tmpl.addItem(n, n)
        cur = s.get("default_style_template") or ""
        i = self.tmpl.findData(cur)
        self.tmpl.setCurrentIndex(max(0, i))
        self.tmpl.setToolTip("Style template applied automatically to new figures.")
        form.addRow("Default figure style", self.tmpl)
        lay.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._update_example()

    def _update_example(self) -> None:
        self.example.setText(f"20.069457240309 → {format_value(20.069457240309, self.sig.value())}   ·   "
                             f"0.000123456 → {format_value(0.000123456, self.sig.value())}")

    def apply(self) -> None:
        s = settings()
        s.set("sig_figs", self.sig.value())
        s.set("csv_full_precision", self.full.isChecked())
        s.set("default_style_template", self.tmpl.currentData() or "")

    def accept(self) -> None:
        self.apply()
        super().accept()
