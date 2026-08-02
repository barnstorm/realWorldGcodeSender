"""Dataclass-driven editor for every application setting."""

from dataclasses import fields

from PySide6.QtCore import Signal
from PySide6.QtGui import QDoubleValidator, QIntValidator
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QScrollArea, QVBoxLayout, QWidget,
)

from app_config import Configuration
from qt_ui.widgets import button, heading, hline


class SettingsView(QWidget):
    applied = Signal()

    def __init__(self, context):
        super().__init__()
        self.ctx = context
        self.editors = {}
        self.dirty = False
        root = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_host = QWidget()
        sections = QVBoxLayout(form_host)
        sections.setContentsMargins(28, 22, 28, 22)
        for group_name in ("physical_setup", "cutting_parameters", "vision_settings",
                           "communication_settings", "probing_settings"):
            group = getattr(context.config, group_name)
            sections.addWidget(heading(group_name.replace("_", " ").title()))
            form = QFormLayout()
            for field in fields(group):
                value = getattr(group, field.name)
                if isinstance(value, bool):
                    editor = QCheckBox()
                    editor.setChecked(value)
                    editor.toggled.connect(self._changed)
                else:
                    editor = QLineEdit(str(value))
                    if isinstance(value, int):
                        editor.setValidator(QIntValidator(editor))
                    elif isinstance(value, float):
                        editor.setValidator(QDoubleValidator(editor))
                    editor.textEdited.connect(self._changed)
                self.editors[(group_name, field.name)] = (editor, type(value))
                form.addRow(field.name.replace("_", " ").title(), editor)
            form_widget = QWidget()
            form_widget.setLayout(form)
            sections.addWidget(form_widget)
            sections.addWidget(hline())
        sections.addStretch(1)
        scroll.setWidget(form_host)
        root.addWidget(scroll, 1)

        bar = QHBoxLayout()
        defaults = button("Load default")
        load = button("Load from file")
        apply = button("Apply")
        save_as = button("Save as")
        save = button("Save", "primary")
        self.dirty_label = QLabel("")
        bar.addWidget(defaults)
        bar.addWidget(load)
        bar.addStretch(1)
        bar.addWidget(self.dirty_label)
        bar.addWidget(apply)
        bar.addWidget(save_as)
        bar.addWidget(save)
        root.addLayout(bar)
        defaults.clicked.connect(lambda: self._load_config(Configuration.get_default()))
        load.clicked.connect(self._load_file)
        apply.clicked.connect(self.apply)
        save_as.clicked.connect(self._save_as)
        save.clicked.connect(self._save)

    def _changed(self, *_args):
        self.dirty = True
        self.dirty_label.setText("Unsaved changes")

    def _load_config(self, config):
        for (group_name, field_name), (editor, _value_type) in self.editors.items():
            value = getattr(getattr(config, group_name), field_name)
            if isinstance(editor, QCheckBox):
                editor.setChecked(value)
            else:
                editor.setText(str(value))
        self._changed()

    def _load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load settings", "", "JSON (*.json)")
        if path:
            self._load_config(Configuration.load_from_file(path))

    def apply(self):
        for (group_name, field_name), (editor, value_type) in self.editors.items():
            raw = editor.isChecked() if isinstance(editor, QCheckBox) else editor.text()
            setattr(getattr(self.ctx.config, group_name), field_name, value_type(raw))
        from realWorldGcodeSender import refresh_config_globals
        refresh_config_globals()
        config = self.ctx.config
        self.ctx.transform.bed_view_pixels = float(config.vision_settings.bed_view_size_pixels)
        self.ctx.transform.bed_size_y = float(config.get_bed_size().Y)
        self.ctx.transform.left_box_x = float(config.get_left_box_ref().X)
        self.ctx.transform.right_box_x = float(config.get_right_box_ref().X)
        self.dirty = False
        self.dirty_label.setText("Applied")
        self.applied.emit()

    def _save(self):
        self.apply()
        self.ctx.config.save_to_file()
        self.dirty_label.setText("Saved")

    def _save_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save settings", "config.json", "JSON (*.json)")
        if path:
            self.apply()
            self.ctx.config.save_to_file(path)
            self.dirty_label.setText("Saved")

