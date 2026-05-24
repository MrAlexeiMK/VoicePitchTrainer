from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextCharFormat, QTextCursor, QColor
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QPushButton, QTextEdit, QVBoxLayout

from core.app_logger import LogRecord, app_logger


class LogsDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Логи")
        self.setMinimumSize(900, 520)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.text_edit.setStyleSheet(
            "QTextEdit {"
            "font-family: Consolas, 'Courier New', monospace;"
            "font-size: 12px;"
            "background: #111111;"
            "color: #dddddd;"
            "}"
        )

        self.clear_button = QPushButton("Очистить")
        self.clear_button.clicked.connect(self._clear)

        self.close_button = QPushButton("Закрыть")
        self.close_button.clicked.connect(self.close)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.clear_button)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.close_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.text_edit)
        layout.addLayout(buttons_layout)

        for record in app_logger.records():
            self.append_record(record)

        app_logger.add_listener(self.append_record)

    def closeEvent(self, event) -> None:
        app_logger.remove_listener(self.append_record)
        event.accept()

    def append_record(self, record: LogRecord) -> None:
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        fmt = QTextCharFormat()
        fmt.setForeground(QColor(self._color_for(record.level)))

        cursor.insertText(record.formatted() + "\n", fmt)

        self.text_edit.setTextCursor(cursor)
        self.text_edit.ensureCursorVisible()

    def _clear(self) -> None:
        self.text_edit.clear()

    def _color_for(self, level: str) -> str:
        if level == "ERROR":
            return "#ff5555"
        if level == "WARNING":
            return "#ffaa00"
        if level == "INFO":
            return "#66ccff"
        if level == "DEBUG":
            return "#aaaaaa"
        return "#dddddd"
