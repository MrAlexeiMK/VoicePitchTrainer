from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QPen
from PyQt6.QtWidgets import QWidget

from core.constants import GRAPH_RECENT_SECONDS


class FrequencyChart(QWidget):
    seekRequested = pyqtSignal(float)

    def __init__(self) -> None:
        super().__init__()
        self.voice_history: list[tuple[float, float]] = []
        self.song_melody: list[tuple[float, float]] = []
        self.target_frequency = 220.0
        self.allowed_error_hz = 30.0
        self.playback_position = 0.0
        self.song_duration: Optional[float] = None
        self.visible_start = 0.0
        self.visible_end = 1.0
        self.recent_seconds = 45
        # Окно можно сжимать: график остаётся полезным даже при меньшей высоте.
        self.setMinimumHeight(160)

    def set_pitch_data(self, history: list[tuple[float, float]], target_frequency: float, allowed_error_hz: float) -> None:
        self.voice_history = history
        self.song_melody = []
        self.target_frequency = target_frequency
        self.allowed_error_hz = allowed_error_hz
        self.song_duration = None
        self.update()

    def set_singing_data(self, voice_history: list[tuple[float, float]], song_melody: list[tuple[float, float]], playback_position: float, allowed_error_hz: float, song_duration: Optional[float] = None, recent_seconds: int = 45) -> None:
        self.voice_history = voice_history
        self.song_melody = song_melody
        self.playback_position = playback_position
        self.allowed_error_hz = allowed_error_hz
        self.song_duration = song_duration
        self.recent_seconds = recent_seconds
        self.update()

    def mousePressEvent(self, event) -> None:
        if not self.song_melody:
            return
        left, right = 55, 15
        plot_width = max(1, self.width() - left - right)
        x = event.position().x()
        if x < left or x > self.width() - right:
            return
        ratio = (x - left) / plot_width
        timestamp = self.visible_start + ratio * max(1.0, self.visible_end - self.visible_start)
        if self.song_duration is not None:
            timestamp = max(0.0, min(self.song_duration, timestamp))
        self.seekRequested.emit(timestamp)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width, height = self.width(), self.height()
        left, right, top, bottom = 55, 15, 15, 30
        plot_width = max(1, width - left - right)
        plot_height = max(1, height - top - bottom)
        painter.fillRect(0, 0, width, height, self.palette().window())
        all_points = self.voice_history + self.song_melody
        if not all_points:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "График появится после начала тренировки")
            return
        if self.song_melody:
            window = float(self.recent_seconds)
            self.visible_start = max(0.0, self.playback_position - window * 0.25)
            self.visible_end = self.visible_start + window
            if self.song_duration is not None and self.visible_end > self.song_duration:
                self.visible_end = self.song_duration
                self.visible_start = max(0.0, self.visible_end - window)
            visible_points = [(t, f) for t, f in all_points if self.visible_start <= t <= self.visible_end]
            title = "зелёная — вокал песни после Demucs/анализа, синяя — твой голос; клик перематывает"
        else:
            first_time = self.voice_history[0][0]
            end_time = self.voice_history[-1][0]
            self.visible_start = first_time
            self.visible_end = max(end_time, first_time + 1.0)
            visible_points = all_points
            title = "история частоты"
        if not visible_points:
            visible_points = all_points
        frequencies = [frequency for _, frequency in visible_points]
        min_frequency = max(0, min(frequencies) - 30)
        max_frequency = max(frequencies) + 30

        def x_of(timestamp: float) -> int:
            if not self.song_melody:
                first_time = self.voice_history[0][0]
                end_time = max(self.voice_history[-1][0], first_time + 1.0)
                recent_start = max(first_time, end_time - GRAPH_RECENT_SECONDS)
                if timestamp >= recent_start:
                    recent_width = plot_width * 0.78
                    recent_duration = max(1.0, end_time - recent_start)
                    return int(left + plot_width * 0.22 + ((timestamp - recent_start) / recent_duration) * recent_width)
                old_duration = max(1.0, recent_start - first_time)
                return int(left + ((timestamp - first_time) / old_duration) * plot_width * 0.20)
            span = max(1.0, self.visible_end - self.visible_start)
            return int(left + ((timestamp - self.visible_start) / span) * plot_width)

        def y_of(frequency: float) -> int:
            span = max(1e-6, max_frequency - min_frequency)
            return int(top + (max_frequency - frequency) / span * plot_height)

        painter.setPen(QPen(Qt.GlobalColor.lightGray, 1))
        for i in range(5):
            y = top + int(plot_height * i / 4)
            value = max_frequency - (max_frequency - min_frequency) * i / 4
            painter.drawLine(left, y, width - right, y)
            painter.drawText(5, y + 5, f"{value:.0f} Гц")
        painter.setPen(QPen(Qt.GlobalColor.darkGray, 2))
        painter.drawRect(left, top, plot_width, plot_height)
        if not self.song_melody:
            painter.setPen(QPen(Qt.GlobalColor.darkGreen, 1))
            painter.drawLine(left, y_of(self.target_frequency), width - right, y_of(self.target_frequency))
            painter.setPen(QPen(Qt.GlobalColor.darkYellow, 1))
            painter.drawLine(left, y_of(self.target_frequency + self.allowed_error_hz), width - right, y_of(self.target_frequency + self.allowed_error_hz))
            painter.drawLine(left, y_of(self.target_frequency - self.allowed_error_hz), width - right, y_of(self.target_frequency - self.allowed_error_hz))
        painter.save()
        painter.setClipRect(left, top, plot_width, plot_height)
        self._draw_line(painter, self.song_melody, x_of, y_of, Qt.GlobalColor.darkGreen, 2)
        self._draw_line(painter, self.voice_history, x_of, y_of, Qt.GlobalColor.blue, 2)
        if self.song_melody:
            cursor_x = x_of(self.playback_position)
            painter.setPen(QPen(Qt.GlobalColor.red, 2))
            painter.drawLine(cursor_x, top, cursor_x, top + plot_height)
        painter.restore()
        painter.setPen(QPen(Qt.GlobalColor.black, 1))
        painter.drawText(left + 5, height - 8, title)

    def _draw_line(self, painter, points, x_of, y_of, color, width) -> None:
        if len(points) < 2:
            return
        painter.setPen(QPen(color, width))
        previous = None
        for timestamp, frequency in points:
            if self.song_melody and (timestamp < self.visible_start or timestamp > self.visible_end):
                previous = None
                continue
            x = x_of(timestamp)
            if x < 55 or x > self.width() - 15:
                previous = None
                continue
            point = x, y_of(frequency)
            if previous is not None:
                painter.drawLine(previous[0], previous[1], point[0], point[1])
            previous = point
