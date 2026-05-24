import shutil
import time
from datetime import datetime
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
from PyQt6.QtCore import QObject, QSettings, QSize, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from audio.devices import list_input_devices
from audio.pitch_detector import AudioPitchDetector
from core.app_logger import app_logger
from core.constants import TARGET_PRESETS
from core.models import AudioDevice, SingingTrainingSettings, VoiceTrainingSettings
from core.training_storage import (
    CachedSongData,
    TrainingHistoryEntry,
    clear_training_history,
    delete_training_history_entry,
    history_for_song,
    list_cached_songs,
    load_cached_song,
    load_training_history,
    save_cached_song,
    save_training_history_entry,
)
from core.music import classify_voice_by_frequency, note_to_frequency
from song.audio_loader import load_audio_file
from song.melody import analyze_song_melody
from song.scoring import MelodyLookup, SingingAccuracySummary, evaluate_singing_accuracy_summary, estimate_singing_latency
from song.vocal_separator import separate_vocals_with_demucs
from song.youtube_importer import download_youtube_audio_to_wav
from ui.chart_widget import FrequencyChart
from ui.logs_dialog import LogsDialog
from ui.settings_dialog import SettingsDialog


ICON_COLOR = "#202124"
ICON_SIZE = QSize(22, 22)


def _create_icons() -> dict[str, QIcon]:
    icons: dict[str, QIcon] = {}

    def icon_from_svg(svg: str) -> QIcon:
        pixmap = QPixmap(24, 24)
        pixmap.fill(Qt.GlobalColor.transparent)
        renderer = QSvgRenderer(svg.encode("utf-8"))
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        return QIcon(pixmap)

    color = ICON_COLOR

    icons["play"] = icon_from_svg(f"""
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path fill="{color}" d="M8 5v14l11-7z"/>
        </svg>
    """)

    icons["stop"] = icon_from_svg(f"""
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <rect fill="{color}" x="7" y="7" width="10" height="10" rx="1.2"/>
        </svg>
    """)

    icons["pause"] = icon_from_svg(f"""
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <rect fill="{color}" x="7" y="5" width="3.5" height="14" rx="1"/>
            <rect fill="{color}" x="13.5" y="5" width="3.5" height="14" rx="1"/>
        </svg>
    """)

    icons["restart"] = icon_from_svg(f"""
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path fill="{color}" d="M12 5a7 7 0 1 1-6.32 4H3l4-4 4 4H8.1A5 5 0 1 0 12 7z"/>
        </svg>
    """)

    icons["refresh"] = icon_from_svg(f"""
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path fill="{color}" d="M17.65 6.35A7.95 7.95 0 0 0 12 4a8 8 0 1 0 7.45 10.9h-2.13A6 6 0 1 1 16.22 7.78L13 11h8V3z"/>
        </svg>
    """)

    icons["volume"] = icon_from_svg(f"""
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path fill="{color}" d="M4 9v6h4l5 4V5L8 9H4z"/>
            <path fill="{color}" d="M16.2 8.2a5 5 0 0 1 0 7.6l1.4 1.4a7 7 0 0 0 0-10.4z"/>
            <path fill="{color}" d="M18.8 5.6a8.5 8.5 0 0 1 0 12.8l1.4 1.4a10.5 10.5 0 0 0 0-15.6z"/>
        </svg>
    """)

    icons["volume_on"] = icons["volume"]

    icons["volume_off"] = icon_from_svg(f"""
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 28 24">
            <path fill="{color}" d="M3.5 9v6h4l5 4V5l-5 4h-4z"/>
            <path fill="{color}" d="M18.2 8.8l1.4-1.4 2.7 2.7 2.7-2.7 1.4 1.4-2.7 2.7 2.7 2.7-1.4 1.4-2.7-2.7-2.7 2.7-1.4-1.4 2.7-2.7z"/>
        </svg>
    """)


    icons["chart"] = icon_from_svg(f"""
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path fill="{color}" d="M4 19h16v2H2V3h2z"/>
            <path fill="{color}" d="M6 16l4-5 3 3 5-8 1.8 1.1-6.4 10.2-3.2-3.2-3.6 4.4z"/>
        </svg>
    """)

    icons["history_play"] = icon_from_svg("""
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <circle fill="#2e7d32" cx="12" cy="12" r="10"/>
            <path fill="#ffffff" d="M9 7.5v9l7-4.5z"/>
        </svg>
    """)

    icons["delete"] = icon_from_svg("""
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path fill="#d32f2f" d="M6.4 5l5.6 5.6L17.6 5 19 6.4 13.4 12 19 17.6 17.6 19 12 13.4 6.4 19 5 17.6 10.6 12 5 6.4z"/>
        </svg>
    """)

    return icons



def _setup_icon_button(button: QPushButton, icon: QIcon, tooltip: str) -> None:
    button.setText("")
    button.setIcon(icon)
    button.setIconSize(ICON_SIZE)
    button.setToolTip(tooltip)
    button.setMinimumHeight(34)


class PlaybackController:
    """
    Потоковое воспроизведение через sounddevice.OutputStream.

    sd.play() получает весь массив сразу и на длинных песнях/больших numpy-массивах
    иногда даёт лаги UI и старт с ощущением замедления. Здесь audio отдаётся
    аудио-драйверу маленькими блоками из callback, без копирования всего хвоста песни.
    """

    def __init__(self) -> None:
        self.current_kind: str | None = None
        self._stream: sd.OutputStream | None = None
        self._audio: np.ndarray | None = None
        self._position = 0
        self._sample_rate = 44100
        self._finished_callback: Callable[[str], None] | None = None
        self._finishing = False

    def play(
        self,
        kind: str,
        audio: np.ndarray,
        sample_rate: int,
        finished_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.stop()
        values = np.asarray(audio, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        self.current_kind = kind
        self._audio = values
        self._position = 0
        self._sample_rate = int(sample_rate)
        self._finished_callback = finished_callback
        self._finishing = False
        self._stream = sd.OutputStream(
            samplerate=self._sample_rate,
            channels=int(values.shape[1]),
            dtype="float32",
            blocksize=2048,
            callback=self._audio_callback,
            finished_callback=self._stream_finished,
        )
        self._stream.start()

    def stop(self, kind: str | None = None) -> None:
        if kind is not None and self.current_kind != kind:
            return
        if self.current_kind is None:
            return

        stream = self._stream
        self._stream = None
        self.current_kind = None
        self._audio = None
        self._position = 0
        self._finished_callback = None
        self._finishing = False

        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def stop_all(self) -> None:
        self.stop()

    def _audio_callback(self, outdata, frames, time_info, status) -> None:
        audio = self._audio
        if audio is None:
            outdata.fill(0)
            raise sd.CallbackStop

        end = min(self._position + frames, len(audio))
        chunk = audio[self._position:end]
        written = len(chunk)

        if written:
            outdata[:written] = chunk

        if written < frames:
            outdata[written:] = 0
            self._position = end
            self._finishing = True
            raise sd.CallbackStop

        self._position = end

    def _stream_finished(self) -> None:
        kind = self.current_kind
        callback = self._finished_callback

        if kind is None or not self._finishing:
            return

        self._stream = None
        self.current_kind = None
        self._audio = None
        self._position = 0
        self._finished_callback = None
        self._finishing = False

        if callback is not None:
            QTimer.singleShot(0, lambda: callback(kind))


class YouTubeImportWorker(QObject):
    statusChanged = pyqtSignal(str)
    finished = pyqtSignal(object, str)
    failed = pyqtSignal(str)

    def __init__(self, query_or_url: str) -> None:
        super().__init__()
        self.query_or_url = query_or_url
        self.cancel_requested = False

    def cancel(self) -> None:
        self.cancel_requested = True

    def run(self) -> None:
        try:
            app_logger.info("YouTube import started")
            self.statusChanged.emit("Ищу видео в YouTube...")
            path, title = download_youtube_audio_to_wav(
                self.query_or_url,
                progress_callback=self.statusChanged.emit,
                cancel_callback=lambda: self.cancel_requested,
            )
            app_logger.info(f"YouTube import finished: {title}")
            self.finished.emit(path, title)
        except Exception as exc:
            app_logger.error(f"YouTube import failed: {exc}")
            self.failed.emit(str(exc))


class SongLoadWorker(QObject):
    statusChanged = pyqtSignal(str)
    finished = pyqtSignal(object, object, object, int, list, str, bool, object, object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        path: Path,
        title: str,
        voice_settings: VoiceTrainingSettings,
        singing_settings: SingingTrainingSettings,
    ) -> None:
        super().__init__()
        self.path = path
        self.title = title
        self.voice_settings = voice_settings
        self.singing_settings = singing_settings
        self.cancel_requested = False

    def cancel(self) -> None:
        self.cancel_requested = True

    def _raise_if_cancelled(self) -> None:
        if self.cancel_requested:
            raise RuntimeError("Импорт песни отменён")

    def run(self) -> None:
        try:
            app_logger.info(f"Song loading started: {self.title}")
            self.statusChanged.emit("Загружаю аудиофайл...")
            playback_audio, melody_audio, sample_rate, actual_path = load_audio_file(self.path)
            self._raise_if_cancelled()
            app_logger.info(f"Audio loaded: sample_rate={sample_rate}, samples={len(playback_audio)}")

            vocals_audio = None
            instrumental_audio = None
            vocals_path = None
            instrumental_path = None
            demucs_stems_available = False

            use_demucs = bool(getattr(self.singing_settings, "use_demucs", False))
            demucs_model = str(getattr(self.singing_settings, "demucs_model", "htdemucs"))

            if use_demucs:
                self.statusChanged.emit("Подготавливаю AI-отделение вокала через Demucs...")
                app_logger.info(f"Demucs enabled, model={demucs_model}")

                self._raise_if_cancelled()
                separated = separate_vocals_with_demucs(
                    actual_path,
                    demucs_model,
                    progress_callback=self.statusChanged.emit,
                    log_callback=app_logger.info,
                    cancel_callback=lambda: self.cancel_requested,
                )

                vocals_path = separated.vocals_path
                instrumental_path = separated.instrumental_path

                self._raise_if_cancelled()
                self.statusChanged.emit("Загружаю выделенный вокал...")
                vocals_audio, melody_audio, sample_rate, _ = load_audio_file(vocals_path)

                self.statusChanged.emit("Загружаю инструментал...")
                instrumental_audio, _, _, _ = load_audio_file(instrumental_path)

                demucs_stems_available = True
                app_logger.info("Separated vocals and instrumental loaded")
            else:
                self.statusChanged.emit(
                    "AI-отделение вокала выключено. Использую лёгкое выделение центральной мелодии."
                )
                app_logger.warning("Demucs disabled; using lightweight center extraction")

            self.statusChanged.emit("Анализирую мелодию песни...")
            self._raise_if_cancelled()
            melody = analyze_song_melody(
                melody_audio,
                sample_rate,
                self.voice_settings,
                self.singing_settings,
                progress_callback=self.statusChanged.emit,
            )

            self._raise_if_cancelled()
            self.statusChanged.emit("Завершаю импорт...")
            app_logger.info(f"Song melody analysis finished: {len(melody)} points")
            self.finished.emit(
                playback_audio,
                vocals_audio,
                instrumental_audio,
                sample_rate,
                melody,
                self.title,
                demucs_stems_available,
                vocals_path,
                instrumental_path,
            )
        except Exception as exc:
            app_logger.error(f"Song loading failed: {exc}")
            self.failed.emit(str(exc))


class Beeper:
    def __init__(self) -> None:
        self._last_beep_time = 0.0

    def beep(self) -> None:
        now = time.monotonic()
        if now - self._last_beep_time < 0.35:
            return
        self._last_beep_time = now
        QApplication.beep()




def _rank_for_score(score_percent: float) -> tuple[str, str]:
    if score_percent > 80.0:
        return "S+", "#e6002d"
    if score_percent > 70.0:
        return "S", "#ff6d00"
    if score_percent > 50.0:
        return "A", "#7b1fa2"
    if score_percent > 40.0:
        return "B", "#1976d2"
    if score_percent > 30.0:
        return "C", "#00897b"
    if score_percent > 20.0:
        return "D", "#7cb342"
    if score_percent > 10.0:
        return "F", "#8bc34a"
    return "E", "#777777"


def _format_datetime(timestamp: float) -> str:
    if timestamp <= 0:
        return "—"
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


class SongProgressChart(QWidget):
    def __init__(self, entries: list[TrainingHistoryEntry], parent=None) -> None:
        super().__init__(parent)
        self.entries = entries
        self.setMinimumSize(640, 320)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width, height = self.width(), self.height()
        left, right, top, bottom = 54, 20, 24, 44
        plot_width = max(1, width - left - right)
        plot_height = max(1, height - top - bottom)
        painter.fillRect(0, 0, width, height, self.palette().window())

        painter.drawText(0, 0, width, 22, Qt.AlignmentFlag.AlignCenter, "Прогресс точности по попыткам")
        painter.drawLine(left, top, left, top + plot_height)
        painter.drawLine(left, top + plot_height, left + plot_width, top + plot_height)

        for value in [0, 25, 50, 75, 100]:
            y = top + plot_height - int(value / 100.0 * plot_height)
            painter.drawText(6, y - 8, 42, 16, Qt.AlignmentFlag.AlignRight, f"{value}%")
            painter.drawLine(left - 4, y, left + plot_width, y)

        if not self.entries:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Пока нет истории по этой песне")
            return

        if len(self.entries) == 1:
            x_values = [left + plot_width // 2]
        else:
            x_values = [left + int(index / (len(self.entries) - 1) * plot_width) for index in range(len(self.entries))]

        points: list[tuple[int, int]] = []
        for x, entry in zip(x_values, self.entries):
            score = max(0.0, min(100.0, entry.score_percent))
            y = top + plot_height - int(score / 100.0 * plot_height)
            points.append((x, y))

        for first, second in zip(points, points[1:]):
            painter.drawLine(first[0], first[1], second[0], second[1])

        for point, entry in zip(points, self.entries):
            rank, color = _rank_for_score(entry.score_percent)
            painter.setBrush(Qt.GlobalColor.white)
            painter.drawEllipse(point[0] - 4, point[1] - 4, 8, 8)
            painter.drawText(point[0] - 26, point[1] - 24, 52, 18, Qt.AlignmentFlag.AlignCenter, rank)
            painter.drawText(point[0] - 32, top + plot_height + 8, 64, 18, Qt.AlignmentFlag.AlignCenter, f"{entry.score_percent:.0f}%")


class SongProgressDialog(QDialog):
    def __init__(self, song_title: str, entries: list[TrainingHistoryEntry], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Прогресс: {song_title}")
        self.setMinimumSize(700, 420)
        layout = QVBoxLayout(self)
        layout.addWidget(SongProgressChart(entries, self))


class TrainingHistoryDialog(QDialog):
    def __init__(
        self,
        entries: list[TrainingHistoryEntry],
        chart_icon: QIcon,
        play_icon: QIcon,
        stop_icon: QIcon,
        delete_icon: QIcon,
        show_progress: Callable[[str, str], None],
        play_recording: Callable[[TrainingHistoryEntry], bool],
        stop_recording: Callable[[], None],
        delete_entry: Callable[[TrainingHistoryEntry], bool],
        clear_history: Callable[[], bool],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.entries = entries
        self.chart_icon = chart_icon
        self.play_icon = play_icon
        self.stop_icon = stop_icon
        self.delete_icon = delete_icon
        self.show_progress = show_progress
        self.play_recording = play_recording
        self.stop_recording = stop_recording
        self.delete_entry = delete_entry
        self.clear_history = clear_history
        self.play_buttons: dict[tuple[str, float], QPushButton] = {}
        self.active_recording_key: tuple[str, float] | None = None
        self.setWindowTitle("История тренировки пения")
        self.setMinimumSize(820, 520)

        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setSpacing(8)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.content)

        self.clear_button = QPushButton("Очистить весь прогресс")
        self.clear_button.clicked.connect(self._clear_all)

        layout = QVBoxLayout(self)
        layout.addWidget(self.scroll)
        layout.addWidget(self.clear_button)

        self._rebuild()

    def _rebuild(self) -> None:
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.clear_button.setEnabled(bool(self.entries))

        if not self.entries:
            empty_label = QLabel("История пока пустая. Допой песню до конца, чтобы появился результат.")
            empty_label.setWordWrap(True)
            self.content_layout.addWidget(empty_label)
            self.content_layout.addStretch()
            return

        for entry in self.entries:
            self.content_layout.addWidget(self._create_row(entry))
        self.content_layout.addStretch()

    def _create_row(self, entry: TrainingHistoryEntry) -> QWidget:
        row = QFrame()
        row.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QHBoxLayout(row)

        rank, color = _rank_for_score(entry.score_percent)
        label = QLabel(
            f"<b>{entry.title}</b><br>"
            f"{_format_datetime(entry.timestamp)} · "
            f"Точность {entry.score_percent:.0f}% · "
            f"<span style='color:{color}; font-weight:700;'>({rank})</span>"
        )
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setWordWrap(True)
        layout.addWidget(label, 1)

        if entry.recording_path is not None and entry.recording_path.exists():
            play_button = QPushButton("Прослушать попытку")
            play_button.setIcon(self.play_icon)
            play_button.setIconSize(ICON_SIZE)
            play_button.clicked.connect(lambda: self._toggle_recording(entry))
            self.play_buttons[self._entry_key(entry)] = play_button
            layout.addWidget(play_button)

        progress_button = QPushButton("Отобразить прогресс успеха песни")
        progress_button.setIcon(self.chart_icon)
        progress_button.setIconSize(ICON_SIZE)
        progress_button.clicked.connect(lambda: self.show_progress(entry.song_key, entry.title))
        layout.addWidget(progress_button)

        delete_button = QPushButton()
        delete_button.setIcon(self.delete_icon)
        delete_button.setIconSize(ICON_SIZE)
        delete_button.setToolTip("Удалить эту запись из истории")
        delete_button.setFixedWidth(42)
        delete_button.clicked.connect(lambda: self._delete_one(entry))
        layout.addWidget(delete_button)
        return row

    def _toggle_recording(self, entry: TrainingHistoryEntry) -> None:
        started = self.play_recording(entry)
        self._set_active_recording(self._entry_key(entry) if started else None)

    def _set_active_recording(self, active_key: tuple[str, float] | None) -> None:
        self.active_recording_key = active_key
        for key, button in self.play_buttons.items():
            if key == active_key:
                button.setText("Стоп")
                button.setIcon(self.stop_icon)
                button.setToolTip("Остановить прослушивание попытки")
            else:
                button.setText("Прослушать попытку")
                button.setIcon(self.play_icon)
                button.setToolTip("Прослушать попытку")

    def reset_play_buttons(self) -> None:
        self._set_active_recording(None)

    def closeEvent(self, event) -> None:
        self.stop_recording()
        event.accept()

    def _entry_key(self, entry: TrainingHistoryEntry) -> tuple[str, float]:
        return entry.song_key, entry.timestamp

    def _delete_one(self, entry: TrainingHistoryEntry) -> None:
        if self._entry_key(entry) == self.active_recording_key:
            self.stop_recording()
            self.reset_play_buttons()
        if not self.delete_entry(entry):
            return
        self.entries = [
            item for item in self.entries
            if not (item.song_key == entry.song_key and item.timestamp == entry.timestamp)
        ]
        self._rebuild()

    def _clear_all(self) -> None:
        if not self.clear_history():
            return
        self.entries = []
        self._rebuild()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        app_logger.info("Application started")

        self.voice_settings = VoiceTrainingSettings()
        self.singing_settings = SingingTrainingSettings()
        self.persistent_settings = QSettings("AlekseyTools", "VoicePitchTrainer")
        self._restoring_settings = True
        self._load_app_settings()

        self.detector = AudioPitchDetector(self.voice_settings)
        self.beeper = Beeper()
        self.playback = PlaybackController()
        self.current_history_recording_key: tuple[str, float] | None = None
        self.training_history_dialog: TrainingHistoryDialog | None = None

        self.is_running = False
        self.is_playing_voice = False
        self.is_playing_performance = False
        self.song_loaded = False
        self.song_playing = False
        self.song_audio: Optional[np.ndarray] = None
        self.song_vocals_audio: Optional[np.ndarray] = None
        self.song_instrumental_audio: Optional[np.ndarray] = None
        self.song_vocals_path: Optional[Path] = None
        self.song_instrumental_path: Optional[Path] = None
        self.demucs_stems_available = False
        self.song_playback_mode = "full"
        self.song_sample_rate = 44100
        self.song_title = ""
        self.song_started_at = 0.0
        self.song_pause_position = 0.0
        self.song_melody: list[tuple[float, float]] = []
        self.melody_lookup: MelodyLookup | None = None
        self.last_score: SingingAccuracySummary | None = None
        self.last_score_update_time = 0.0
        self.last_latency_update_time = 0.0
        self.cached_latency_ms = 0
        self.last_singing_pitch_time = 0.0
        self.smoothed_volume_percent = 0.0
        self.current_cached_song_key: Optional[str] = None
        self.last_saved_training_voice_count = 0
        self.singing_attempt_started_from_beginning = False

        self.youtube_thread: Optional[QThread] = None
        self.youtube_worker: Optional[YouTubeImportWorker] = None
        self.song_load_thread: Optional[QThread] = None
        self.song_load_worker: Optional[SongLoadWorker] = None
        self.progress_dialog: Optional[QProgressDialog] = None
        self.logs_dialog: Optional[LogsDialog] = None

        self.input_devices: list[AudioDevice] = []
        self.frequency_history: deque[tuple[float, float]] = deque(maxlen=3600 * 30)

        self.setWindowTitle("Тренажёр голоса и пения")
        # Не фиксируем окно слишком большим: пользователь должен иметь возможность
        # уменьшить его на ноутбуке, но основные элементы всё ещё остаются читаемыми.
        self.setMinimumSize(680, 500)
        self.resize(1040, 780)

        self._create_menu()
        self._create_widgets()
        self._create_layout()
        self._create_timers()

        self._load_microphones()
        self._restore_ui_settings()
        self._restoring_settings = False
        self._update_target_label()
        self._refresh_menu_texts()
        self._update_singing_ui_visibility()

    def _create_menu(self) -> None:
        self.settings_action = QAction("Настройки", self)
        self.settings_action.triggered.connect(self._open_settings)
        self.menuBar().addAction(self.settings_action)

        self.logs_action = QAction("Логи", self)
        self.logs_action.triggered.connect(self._show_logs)
        self.menuBar().addAction(self.logs_action)

        self.play_voice_action = None

        self.play_performance_action = QAction("Прослушать мой голос", self)
        self.play_performance_action.triggered.connect(self._toggle_performance_playback)
        self.menuBar().addAction(self.play_performance_action)

        song_menu = self.menuBar().addMenu("Песня")

        self.import_song_file_action = QAction("Импортировать файл", self)
        self.import_song_file_action.triggered.connect(self._import_song_file)
        song_menu.addAction(self.import_song_file_action)

        self.import_youtube_action = QAction("Найти и импортировать из YouTube (может потребоваться VPN)", self)
        self.import_youtube_action.triggered.connect(self._import_song_from_youtube)
        song_menu.addAction(self.import_youtube_action)

        self.recent_songs_menu = song_menu.addMenu("Недавние песни")
        self.training_history_action = QAction("История тренировки пения", self)
        self.training_history_action.triggered.connect(self._show_training_history)
        song_menu.addAction(self.training_history_action)

        song_menu.addSeparator()

        self.play_song_action = QAction("Запустить песню", self)
        self.play_song_action.triggered.connect(self._toggle_song_playback)
        song_menu.addAction(self.play_song_action)

        song_menu.addSeparator()

        self.export_vocals_action = QAction("Сохранить вокал Demucs...", self)
        self.export_vocals_action.triggered.connect(self._export_demucs_vocals)
        song_menu.addAction(self.export_vocals_action)

        self.export_instrumental_action = QAction("Сохранить инструментал Demucs...", self)
        self.export_instrumental_action.triggered.connect(self._export_demucs_instrumental)
        song_menu.addAction(self.export_instrumental_action)

        self.clear_song_action = QAction("Выйти из режима пения", self)
        self.clear_song_action.triggered.connect(self._clear_song)
        song_menu.addAction(self.clear_song_action)

    def _create_widgets(self) -> None:
        self.microphone_combo = QComboBox()
        self.microphone_combo.currentIndexChanged.connect(lambda: self._save_app_settings())

        self.icons = _create_icons()

        self.refresh_microphones_button = QPushButton()
        _setup_icon_button(self.refresh_microphones_button, self.icons["refresh"], "Обновить микрофоны")
        self.refresh_microphones_button.clicked.connect(self._load_microphones)

        self.show_all_devices_checkbox = QCheckBox("Показать все устройства")
        self.show_all_devices_checkbox.setChecked(False)
        self.show_all_devices_checkbox.stateChanged.connect(self._load_microphones)

        self.target_combo = QComboBox()
        for note_name, octave, label, voice_description in TARGET_PRESETS:
            frequency = note_to_frequency(note_name, octave)
            self.target_combo.addItem(
                f"{label} ({voice_description}) — {frequency:.2f} Гц",
                (note_name, octave, label),
            )
        self.target_combo.setCurrentIndex(10)
        self.target_combo.currentIndexChanged.connect(self._update_target_label)
        self.target_combo.currentIndexChanged.connect(lambda: self._save_app_settings())

        self.allowed_error_spin = QDoubleSpinBox()
        self.allowed_error_spin.setRange(1.0, 200.0)
        self.allowed_error_spin.setSingleStep(1.0)
        self.allowed_error_spin.setValue(30.0)
        self.allowed_error_spin.setSuffix(" Гц")
        self.allowed_error_spin.valueChanged.connect(lambda: self._save_app_settings())

        self.beep_button = QPushButton()
        _setup_icon_button(self.beep_button, self.icons["volume_off"], "Сигнал выключен")
        self.beep_button.setCheckable(True)
        self.beep_button.setChecked(False)
        self.beep_button.clicked.connect(self._toggle_beep)
        self.beep_button.clicked.connect(lambda: self._save_app_settings())

        self.start_button = QPushButton()
        _setup_icon_button(self.start_button, self.icons["play"], "Старт")
        self.start_button.clicked.connect(self._toggle_start)

        self.song_button = QPushButton()
        _setup_icon_button(self.song_button, self.icons["play"], "Песня не загружена")
        self.song_button.clicked.connect(self._toggle_song_playback)
        self.song_button.setEnabled(False)

        self.restart_song_button = QPushButton()
        _setup_icon_button(self.restart_song_button, self.icons["restart"], "Начать песню заново")
        self.restart_song_button.clicked.connect(self._restart_song_training)
        self.restart_song_button.setEnabled(False)
        self.restart_song_button.setVisible(False)

        self.performance_button = QPushButton()
        _setup_icon_button(self.performance_button, self.icons["volume"], "Прослушать мой голос")
        self.performance_button.clicked.connect(self._toggle_performance_playback)

        self.playback_mode_combo = QComboBox()
        self.playback_mode_combo.addItem("Всё вместе", "full")
        self.playback_mode_combo.addItem("Только мелодия / инструментал", "instrumental")
        self.playback_mode_combo.addItem("Только вокал", "vocals")
        self.playback_mode_combo.currentIndexChanged.connect(self._playback_mode_changed)


        self.current_note_label = QLabel("—")
        self.current_note_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_note_label.setStyleSheet("font-size: 72px; font-weight: bold;")

        self.voice_type_label = QLabel("Тип голоса: —")
        self.voice_type_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.voice_type_label.setStyleSheet("font-size: 20px;")

        self.frequency_label = QLabel("Частота: — Гц")
        self.frequency_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.target_label = QLabel("Цель: A3")
        self.target_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.song_label = QLabel("Режим: тренинг голоса")
        self.song_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.song_label.setStyleSheet("font-size: 16px;")

        self.score_big_label = QLabel("Точность пения")
        self.score_big_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.score_big_label.setMinimumHeight(36)
        self.score_big_label.setStyleSheet("font-size: 24px; font-weight: bold;")

        self.score_3s_label = self._create_accuracy_card("3 сек", "—")
        self.score_10s_label = self._create_accuracy_card("10 сек", "—")
        self.score_all_label = self._create_accuracy_card("Всё время", "—")

        self.score_detail_label = QLabel("Начни петь — оценка появится автоматически")
        self.score_detail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.score_detail_label.setWordWrap(True)
        self.score_detail_label.setMaximumHeight(42)
        self.score_detail_label.setStyleSheet("font-size: 14px; color: #555555;")

        self.tendency_label = QLabel("")
        self.tendency_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.advice_label = QLabel("")
        self.advice_label.setWordWrap(True)
        self.advice_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.advice_label.setVisible(False)

        self.score_label = QLabel("Точность пения: —")
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.status_label = QLabel("Нажми Старт")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 18px;")

        self.error_label = QLabel("Отклонение: —")
        self.error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.confidence_label = QLabel("Уверенность распознавания: —")
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.error_bar = QProgressBar()
        self.error_bar.setRange(-200, 200)
        self.error_bar.setValue(0)
        self.error_bar.setFormat("0 Гц")

        self.volume_bar = QProgressBar()
        self.volume_bar.setRange(0, 100)
        self.volume_bar.setValue(0)
        self.volume_bar.setFormat("Громкость")

        self.chart = FrequencyChart()
        self.chart.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.chart.seekRequested.connect(self._seek_song)

    def _create_accuracy_card(self, title: str, value: str) -> QLabel:
        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumHeight(72)
        label.setStyleSheet(
            "QLabel {"
            "border: 1px solid #c8c8c8;"
            "border-radius: 8px;"
            "background: #f7f7f7;"
            "padding: 6px;"
            "}"
        )
        self._set_accuracy_card(label, title, value)
        return label

    def _set_accuracy_card(self, label: QLabel, title: str, value: str) -> None:
        label.setText(
            f"<div style='font-size: 13px; color: #555555;'>{title}</div>"
            f"<div style='font-size: 26px; font-weight: 700;'>{value}</div>"
        )

    def _create_layout(self) -> None:
        microphone_layout = QHBoxLayout()
        microphone_layout.addWidget(self.microphone_combo)
        microphone_layout.addWidget(self.refresh_microphones_button)

        controls_layout = QFormLayout()
        controls_layout.addRow("Микрофон", microphone_layout)
        controls_layout.addRow("Список", self.show_all_devices_checkbox)
        controls_layout.addRow("Целевая нота", self.target_combo)
        controls_layout.addRow("Допуск по частоте", self.allowed_error_spin)
        self.target_combo_caption = controls_layout.labelForField(self.target_combo)
        self.allowed_error_caption = controls_layout.labelForField(self.allowed_error_spin)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.beep_button)
        buttons_layout.addWidget(self.song_button)
        buttons_layout.addWidget(self.restart_song_button)
        buttons_layout.addWidget(self.performance_button)

        self.singing_controls_layout = QFormLayout()
        self.singing_controls_layout.addRow("Воспроизведение песни", self.playback_mode_combo)
        self.playback_mode_caption = self.singing_controls_layout.labelForField(self.playback_mode_combo)

        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)

        card_layout = QVBoxLayout(card)
        card_layout.addWidget(self.current_note_label)
        card_layout.addWidget(self.voice_type_label)
        card_layout.addWidget(self.frequency_label)
        card_layout.addWidget(self.target_label)
        card_layout.addWidget(self.song_label)

        self.singing_stats_panel = QFrame()
        self.singing_stats_panel.setFrameShape(QFrame.Shape.StyledPanel)
        self.singing_stats_panel.setMinimumHeight(155)
        self.singing_stats_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        stats_layout = QVBoxLayout(self.singing_stats_panel)
        stats_layout.setContentsMargins(10, 8, 10, 8)
        stats_layout.setSpacing(6)
        stats_layout.addWidget(self.score_big_label)
        stats_layout.addWidget(self.score_all_label)
        accuracy_layout = QHBoxLayout()
        accuracy_layout.setSpacing(10)
        accuracy_layout.addWidget(self.score_3s_label)
        accuracy_layout.addWidget(self.score_10s_label)
        stats_layout.addLayout(accuracy_layout)
        stats_layout.addWidget(self.score_detail_label)
        card_layout.addWidget(self.singing_stats_panel)
        card_layout.addWidget(self.score_label)
        card_layout.addWidget(self.error_label)
        card_layout.addWidget(self.confidence_label)
        card_layout.addWidget(self.status_label)
        card_layout.addWidget(self.error_bar)
        card_layout.addWidget(self.volume_bar)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.addLayout(controls_layout)
        root_layout.addLayout(buttons_layout)
        root_layout.addLayout(self.singing_controls_layout)
        root_layout.addWidget(card)
        root_layout.addWidget(QLabel("График частоты"))
        root_layout.addWidget(self.chart, 1)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setWidget(root)
        self.setCentralWidget(scroll_area)

    def _create_timers(self) -> None:
        self.timer = QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._update_pitch)

        self.chart_timer = QTimer(self)
        self.chart_timer.setInterval(250)
        self.chart_timer.timeout.connect(self._refresh_chart)
        self.chart_timer.start()

        self.song_timer = QTimer(self)
        self.song_timer.setInterval(250)
        self.song_timer.timeout.connect(self._check_song_end)
        self.song_timer.start()

    def _show_logs(self) -> None:
        if self.logs_dialog is None or not self.logs_dialog.isVisible():
            self.logs_dialog = LogsDialog(self)
        self.logs_dialog.show()
        self.logs_dialog.raise_()
        self.logs_dialog.activateWindow()

    def _load_app_settings(self) -> None:
        value = self.persistent_settings.value
        self.voice_settings.recording_seconds = int(value("recording_seconds", self.voice_settings.recording_seconds))
        self.voice_settings.min_confidence = float(value("min_confidence", self.voice_settings.min_confidence))
        self.voice_settings.impulse_threshold = float(value("impulse_threshold", self.voice_settings.impulse_threshold))
        self.voice_settings.stable_frames = int(value("stable_frames", self.voice_settings.stable_frames))
        self.voice_settings.stable_spread_hz = float(value("stable_spread_hz", self.voice_settings.stable_spread_hz))
        self.voice_settings.stable_spread_percent = float(value("stable_spread_percent", self.voice_settings.stable_spread_percent))
        self.voice_settings.noise_gate_strength = float(value("noise_gate_strength", self.voice_settings.noise_gate_strength))
        self.voice_settings.min_noise_gate = float(value("min_noise_gate", self.voice_settings.min_noise_gate))
        self.voice_settings.beep_only_on_stable_voice = value(
            "beep_only_on_stable_voice",
            self.voice_settings.beep_only_on_stable_voice,
            type=bool,
        )
        self.voice_settings.voice_min_frequency = float(value("voice_min_frequency", self.voice_settings.voice_min_frequency))
        self.voice_settings.voice_max_frequency = float(value("voice_max_frequency", self.voice_settings.voice_max_frequency))

        self.singing_settings.allowed_error_cents = float(value("allowed_error_cents", self.singing_settings.allowed_error_cents))
        self.singing_settings.scoring_window_seconds = int(value("scoring_window_seconds", self.singing_settings.scoring_window_seconds))
        self.singing_settings.scoring_method = str(value("scoring_method", self.singing_settings.scoring_method))
        self.singing_settings.voice_latency_ms = 0
        self.singing_settings.auto_detect_latency = True
        self.singing_settings.auto_latency_max_ms = int(value("auto_latency_max_ms", self.singing_settings.auto_latency_max_ms))
        self.singing_settings.auto_latency_step_ms = int(value("auto_latency_step_ms", self.singing_settings.auto_latency_step_ms))
        self.singing_settings.singing_min_confidence = float(value("singing_min_confidence", self.singing_settings.singing_min_confidence))
        self.singing_settings.singing_min_volume = float(value("singing_min_volume", self.singing_settings.singing_min_volume))
        self.singing_settings.singing_min_frequency = float(value("singing_min_frequency", self.singing_settings.singing_min_frequency))
        self.singing_settings.singing_max_frequency = float(value("singing_max_frequency", self.singing_settings.singing_max_frequency))
        self.singing_settings.singing_use_noise_gate = value("singing_use_noise_gate", self.singing_settings.singing_use_noise_gate, type=bool)
        self.singing_settings.melody_analysis_step_ms = int(value("melody_analysis_step_ms", self.singing_settings.melody_analysis_step_ms))
        self.singing_settings.melody_min_confidence = float(value("melody_min_confidence", self.singing_settings.melody_min_confidence))
        self.singing_settings.melody_min_frequency = float(value("melody_min_frequency", self.singing_settings.melody_min_frequency))
        self.singing_settings.melody_max_frequency = float(value("melody_max_frequency", self.singing_settings.melody_max_frequency))
        self.singing_settings.melody_jump_limit_cents = float(value("melody_jump_limit_cents", self.singing_settings.melody_jump_limit_cents))
        self.singing_settings.show_only_recent_seconds = int(value("show_only_recent_seconds", self.singing_settings.show_only_recent_seconds))

        if hasattr(self.singing_settings, "use_demucs"):
            self.singing_settings.use_demucs = value("use_demucs", self.singing_settings.use_demucs, type=bool)
        if hasattr(self.singing_settings, "demucs_model"):
            self.singing_settings.demucs_model = str(value("demucs_model", self.singing_settings.demucs_model))

    def _save_app_settings(self) -> None:
        if getattr(self, "_restoring_settings", False):
            return

        self.persistent_settings.setValue("recording_seconds", self.voice_settings.recording_seconds)
        self.persistent_settings.setValue("min_confidence", self.voice_settings.min_confidence)
        self.persistent_settings.setValue("impulse_threshold", self.voice_settings.impulse_threshold)
        self.persistent_settings.setValue("stable_frames", self.voice_settings.stable_frames)
        self.persistent_settings.setValue("stable_spread_hz", self.voice_settings.stable_spread_hz)
        self.persistent_settings.setValue("stable_spread_percent", self.voice_settings.stable_spread_percent)
        self.persistent_settings.setValue("noise_gate_strength", self.voice_settings.noise_gate_strength)
        self.persistent_settings.setValue("min_noise_gate", self.voice_settings.min_noise_gate)
        self.persistent_settings.setValue("beep_only_on_stable_voice", self.voice_settings.beep_only_on_stable_voice)
        self.persistent_settings.setValue("voice_min_frequency", self.voice_settings.voice_min_frequency)
        self.persistent_settings.setValue("voice_max_frequency", self.voice_settings.voice_max_frequency)

        self.persistent_settings.setValue("allowed_error_cents", self.singing_settings.allowed_error_cents)
        self.persistent_settings.setValue("scoring_window_seconds", self.singing_settings.scoring_window_seconds)
        self.persistent_settings.setValue("scoring_method", self.singing_settings.scoring_method)
        self.persistent_settings.setValue("auto_latency_max_ms", self.singing_settings.auto_latency_max_ms)
        self.persistent_settings.setValue("auto_latency_step_ms", self.singing_settings.auto_latency_step_ms)
        self.persistent_settings.setValue("singing_min_confidence", self.singing_settings.singing_min_confidence)
        self.persistent_settings.setValue("singing_min_volume", self.singing_settings.singing_min_volume)
        self.persistent_settings.setValue("singing_min_frequency", self.singing_settings.singing_min_frequency)
        self.persistent_settings.setValue("singing_max_frequency", self.singing_settings.singing_max_frequency)
        self.persistent_settings.setValue("singing_use_noise_gate", self.singing_settings.singing_use_noise_gate)
        self.persistent_settings.setValue("melody_analysis_step_ms", self.singing_settings.melody_analysis_step_ms)
        self.persistent_settings.setValue("melody_min_confidence", self.singing_settings.melody_min_confidence)
        self.persistent_settings.setValue("melody_min_frequency", self.singing_settings.melody_min_frequency)
        self.persistent_settings.setValue("melody_max_frequency", self.singing_settings.melody_max_frequency)
        self.persistent_settings.setValue("melody_jump_limit_cents", self.singing_settings.melody_jump_limit_cents)
        self.persistent_settings.setValue("show_only_recent_seconds", self.singing_settings.show_only_recent_seconds)

        if hasattr(self.singing_settings, "use_demucs"):
            self.persistent_settings.setValue("use_demucs", self.singing_settings.use_demucs)
        if hasattr(self.singing_settings, "demucs_model"):
            self.persistent_settings.setValue("demucs_model", self.singing_settings.demucs_model)

        self.persistent_settings.setValue("show_all_devices", self.show_all_devices_checkbox.isChecked())
        self.persistent_settings.setValue("selected_microphone_name", self.microphone_combo.currentText())
        self.persistent_settings.setValue("target_index", self.target_combo.currentIndex())
        self.persistent_settings.setValue("allowed_error_hz", self.allowed_error_spin.value())
        self.persistent_settings.setValue("beep_enabled", self.beep_button.isChecked())
        self.persistent_settings.sync()

    def _restore_ui_settings(self) -> None:
        self.show_all_devices_checkbox.setChecked(self.persistent_settings.value("show_all_devices", False, type=bool))

        saved_microphone_name = str(self.persistent_settings.value("selected_microphone_name", ""))
        if saved_microphone_name:
            for index in range(self.microphone_combo.count()):
                if self.microphone_combo.itemText(index) == saved_microphone_name:
                    self.microphone_combo.setCurrentIndex(index)
                    break

        target_index = int(self.persistent_settings.value("target_index", self.target_combo.currentIndex()))
        if 0 <= target_index < self.target_combo.count():
            self.target_combo.setCurrentIndex(target_index)

        self.allowed_error_spin.setValue(
            float(self.persistent_settings.value("allowed_error_hz", self.allowed_error_spin.value()))
        )
        self.beep_button.setChecked(self.persistent_settings.value("beep_enabled", False, type=bool))
        self._toggle_beep()

    def closeEvent(self, event) -> None:
        app_logger.info("Application closing")
        self._save_app_settings()
        self.detector.stop()
        self.playback.stop_all()
        if self.song_load_worker is not None:
            self.song_load_worker.cancel()
        self._stop_worker_thread(self.youtube_thread)
        self._stop_worker_thread(self.song_load_thread)
        event.accept()

    def _stop_worker_thread(self, thread: Optional[QThread]) -> None:
        if thread is None or not thread.isRunning():
            return
        thread.quit()
        if not thread.wait(1500):
            app_logger.warning("Worker thread did not stop in time; terminating")
            thread.terminate()
            thread.wait(1500)

    def _load_microphones(self) -> None:
        current_device_index = self._selected_microphone_index()
        show_all = self.show_all_devices_checkbox.isChecked()
        self.input_devices = list_input_devices(show_all)
        self.microphone_combo.clear()
        app_logger.info(f"Loaded input devices: {len(self.input_devices)}")

        if not self.input_devices:
            self.microphone_combo.addItem("Микрофоны не найдены", None)
            self.status_label.setText("Микрофоны не найдены")
            app_logger.warning("No microphones found")
            return

        selected_combo_index = 0
        for combo_index, device in enumerate(self.input_devices):
            label = f"[{device.index}] {device.name} / {device.sample_rate} Гц / {device.channels} каналов"
            self.microphone_combo.addItem(label, device.index)
            if current_device_index == device.index:
                selected_combo_index = combo_index

        self.microphone_combo.setCurrentIndex(selected_combo_index)

    def _selected_microphone_index(self) -> Optional[int]:
        value = self.microphone_combo.currentData()
        if value is None:
            return None
        return int(value)

    def _selected_audio_device(self) -> Optional[AudioDevice]:
        device_index = self._selected_microphone_index()
        if device_index is None:
            return None

        for device in self.input_devices:
            if device.index == device_index:
                return device

        return None

    def _selected_target(self) -> tuple[str, int, str]:
        value = self.target_combo.currentData()
        if value is None:
            return "A", 3, "A3"

        note_name, octave, label = value
        return str(note_name), int(octave), str(label)

    def _target_frequency(self) -> float:
        note_name, octave, _ = self._selected_target()
        return note_to_frequency(note_name, octave)

    def _refresh_menu_texts(self) -> None:
        if self.play_voice_action is not None:
            if self.is_playing_voice:
                self.play_voice_action.setText("Остановить прослушивание")
            else:
                self.play_voice_action.setText("Прослушать голос")

        performance_text = "Остановить мой голос" if self.is_playing_performance else "Прослушать мой голос"
        self.play_performance_action.setText(performance_text)
        self.performance_button.setText("")
        self.performance_button.setToolTip(performance_text)
        self.performance_button.setIcon(self.icons["stop"] if self.is_playing_performance else self.icons["volume"])

        # В основной панели отдельная кнопка песни больше не нужна:
        # в режиме пения главная кнопка Старт управляет и микрофоном, и песней.
        self.song_button.setVisible(False)
        self.song_button.setEnabled(False)

        self.restart_song_button.setVisible(self.song_loaded)
        self.restart_song_button.setEnabled(self.song_loaded)

        if self.song_loaded:
            start_text = "Стоп тренировки" if self.is_running else "Старт тренировки"
            self.start_button.setToolTip(start_text)
            self.start_button.setIcon(self.icons["stop"] if self.is_running else self.icons["play"])

            song_text = "Пауза песни" if self.song_playing else "Запустить песню"
            self.play_song_action.setText(song_text)
        else:
            start_text = "Стоп" if self.is_running else "Старт"
            self.start_button.setToolTip(start_text)
            self.start_button.setIcon(self.icons["stop"] if self.is_running else self.icons["play"])
            self.play_song_action.setText("Запустить песню")

        stems_export_available = self._demucs_export_available()
        self.export_vocals_action.setEnabled(stems_export_available)
        self.export_instrumental_action.setEnabled(stems_export_available)

        self._refresh_recent_songs_menu()
        self._refresh_playback_mode_combo()

    def _refresh_recent_songs_menu(self) -> None:
        self.recent_songs_menu.clear()
        songs = list_cached_songs()
        if not songs:
            action = QAction("Нет недавних песен", self)
            action.setEnabled(False)
            self.recent_songs_menu.addAction(action)
            return
        for song in songs:
            action = QAction(song.title, self)
            action.setToolTip(_format_datetime(song.updated_at))
            action.triggered.connect(lambda checked=False, key=song.key: self._load_cached_song(key))
            self.recent_songs_menu.addAction(action)

    def _refresh_playback_mode_combo(self) -> None:
        self.playback_mode_combo.setVisible(self.song_loaded)
        self.playback_mode_caption.setVisible(self.song_loaded)
        self.performance_button.setVisible(self.song_loaded)
        self.playback_mode_combo.blockSignals(True)
        for index in range(self.playback_mode_combo.count()):
            mode = self.playback_mode_combo.itemData(index)
            item = self.playback_mode_combo.model().item(index)
            item.setEnabled(bool(self.song_loaded and (mode == "full" or self.demucs_stems_available)))
        current_index = self.playback_mode_combo.findData(self.song_playback_mode)
        if current_index >= 0:
            self.playback_mode_combo.setCurrentIndex(current_index)
        self.playback_mode_combo.blockSignals(False)

    def _playback_mode_changed(self) -> None:
        mode = self.playback_mode_combo.currentData()
        if mode in {"instrumental", "vocals"} and not self.demucs_stems_available:
            QMessageBox.information(self, "Дорожки недоступны", "Для этого режима нужно включить Demucs и заново импортировать песню.")
            self._refresh_playback_mode_combo()
            return
        was_playing = self.song_playing
        position = self._song_position()
        if self.song_playing:
            self.playback.stop("song")
            self.song_playing = False
        self.song_playback_mode = str(mode)
        app_logger.info(f"Song playback mode changed: {mode}")
        if was_playing:
            self._play_song_from_position(position)
        self._refresh_playback_mode_combo()

    def _current_song_audio(self) -> Optional[np.ndarray]:
        if self.song_playback_mode == "vocals" and self.song_vocals_audio is not None:
            return self.song_vocals_audio
        if self.song_playback_mode == "instrumental" and self.song_instrumental_audio is not None:
            return self.song_instrumental_audio
        return self.song_audio

    def _toggle_start(self) -> None:
        if self.is_running:
            if self.song_loaded and self.song_playing:
                self.song_pause_position = self._song_position()
                self.playback.stop("song")
                self.song_playing = False

            self.detector.stop()
            self.timer.stop()
            self.microphone_combo.setEnabled(True)
            self.refresh_microphones_button.setEnabled(True)
            self.show_all_devices_checkbox.setEnabled(True)
            self.is_running = False
            self.status_label.setText("Тренировка остановлена" if self.song_loaded else "Остановлено")
            app_logger.info("Voice capture stopped")
            self._refresh_menu_texts()
            return

        device = self._selected_audio_device()
        if device is None:
            QMessageBox.warning(self, "Ошибка микрофона", "Микрофон не выбран")
            app_logger.warning("Start requested without selected microphone")
            return

        if self.song_loaded and self._current_song_audio() is None:
            QMessageBox.information(self, "Песня", "Сначала импортируй песню")
            return

        self.frequency_history.clear()
        self.detector.clear_session_recording()
        self.detector.set_device(device.index, device.sample_rate)
        app_logger.info(f"Starting voice capture: device={device.name}, sample_rate={device.sample_rate}")

        try:
            self.detector.start(clear_session_recording=False)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка микрофона", str(exc))
            app_logger.error(f"Microphone start failed: {exc}")
            return

        self.timer.start()
        self.microphone_combo.setEnabled(False)
        self.refresh_microphones_button.setEnabled(False)
        self.show_all_devices_checkbox.setEnabled(False)
        self.is_running = True
        self._save_app_settings()

        if self.song_loaded:
            self.status_label.setText("Тренировка началась. Пой поверх песни.")
            self._play_song_from_position(self.song_pause_position)
        else:
            self.status_label.setText("Слушаю...")

        self._refresh_menu_texts()

    def _toggle_beep(self) -> None:
        self.beep_button.setText("")
        self.beep_button.setIcon(self.icons["volume"] if self.beep_button.isChecked() else self.icons["volume_off"])
        self.beep_button.setToolTip("Сигнал включён" if self.beep_button.isChecked() else "Сигнал выключен")

    def _toggle_voice_playback(self) -> None:
        if self.is_playing_voice:
            self.playback.stop("voice")
            self.is_playing_voice = False
            self._refresh_menu_texts()
            self.status_label.setText("Прослушивание остановлено")
            app_logger.info("Voice playback stopped")
            return

        audio = self.detector.get_recent_recording(self.voice_settings.recording_seconds)
        if audio is None or len(audio) == 0:
            QMessageBox.information(self, "Прослушать голос", "Пока нет записанного голоса")
            app_logger.warning("Voice playback requested but buffer is empty")
            return

        self._prepare_playback_start("voice")
        try:
            self.playback.play("voice", audio, self.detector.sample_rate, self._on_playback_finished)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка воспроизведения", str(exc))
            app_logger.error(f"Voice playback failed: {exc}")
            return

        self.is_playing_voice = True
        self._refresh_menu_texts()
        self.status_label.setText("Воспроизвожу сохранённый голос...")
        app_logger.info("Voice playback started")


    def _finish_voice_playback(self) -> None:
        if not self.is_playing_voice:
            return

        self.is_playing_voice = False
        self._refresh_menu_texts()
        self.status_label.setText("Прослушивание завершено")
        app_logger.info("Voice playback finished")

    def _toggle_performance_playback(self) -> None:
        if self.is_playing_performance:
            self.playback.stop("performance")
            self.is_playing_performance = False
            self._refresh_menu_texts()
            self.status_label.setText("Прослушивание моего голоса остановлено")
            return
        audio = self.detector.get_session_recording()
        if audio is None or len(audio) == 0:
            QMessageBox.information(self, "Прослушать мой голос", "Пока нет записи текущего голоса")
            return
        self._prepare_playback_start("performance")
        try:
            self.playback.play("performance", audio, self.detector.sample_rate, self._on_playback_finished)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка воспроизведения", str(exc))
            return
        self.is_playing_performance = True
        self.status_label.setText("Воспроизвожу твоё пение из текущей сессии...")
        self._refresh_menu_texts()


    def _finish_performance_playback(self) -> None:
        if not self.is_playing_performance:
            return
        self.is_playing_performance = False
        self.status_label.setText("Прослушивание моего голоса завершено")
        self._refresh_menu_texts()

    def _prepare_playback_start(self, kind: str) -> None:
        if kind != "song" and self.song_playing:
            self.song_pause_position = self._song_position()
            self.song_playing = False
        if kind != "voice" and self.is_playing_voice:
            self.is_playing_voice = False
        if kind != "performance" and self.is_playing_performance:
            self.is_playing_performance = False
        self._refresh_menu_texts()

    def _on_playback_finished(self, kind: str) -> None:
        if kind == "voice":
            self._finish_voice_playback()
            return
        if kind == "performance":
            self._finish_performance_playback()
            return
        if kind == "song":
            self._finish_song_playback()
            return
        if kind == "history":
            self.current_history_recording_key = None
            if self.training_history_dialog is not None:
                self.training_history_dialog.reset_play_buttons()
            self.status_label.setText("Прослушивание попытки завершено")

    def _finish_song_playback(self) -> None:
        if not self.song_playing:
            return
        self.status_label.setText("Песня закончилась")
        self._record_completed_training_result_if_available()
        self.song_playing = False
        self.song_pause_position = 0.0
        self._refresh_menu_texts()
        app_logger.info("Song playback finished")

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self, self.voice_settings, self.singing_settings)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        dialog.apply_to(self.voice_settings, self.singing_settings)
        self.detector.apply_settings()
        self._save_app_settings()
        self._refresh_menu_texts()
        app_logger.info("Settings updated")

    def _import_song_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Импортировать песню",
            "",
            "Audio/Video (*.wav *.mp3 *.mp4 *.m4a *.flac);;All files (*.*)",
        )
        if not path:
            return

        app_logger.info(f"Import song file requested: {path}")
        self._start_song_load(Path(path), Path(path).stem)

    def _import_song_from_youtube(self) -> None:
        text, ok = QInputDialog.getText(
            self,
            "Найти и импортировать из YouTube",
            "Введи название песни или ссылку:",
        )
        if not ok or not text.strip():
            return

        self._show_progress("YouTube импорт", "Готовлю импорт...")
        self.import_youtube_action.setEnabled(False)
        app_logger.info(f"YouTube import requested: {text.strip()}")

        self.youtube_thread = QThread(self)
        self.youtube_worker = YouTubeImportWorker(text.strip())
        self.youtube_worker.moveToThread(self.youtube_thread)

        self.youtube_thread.started.connect(self.youtube_worker.run)
        self.youtube_worker.statusChanged.connect(self._update_progress_text)
        self.youtube_worker.finished.connect(self._on_youtube_downloaded)
        self.youtube_worker.failed.connect(self._on_youtube_failed)
        self.youtube_worker.finished.connect(self.youtube_thread.quit)
        self.youtube_worker.failed.connect(self.youtube_thread.quit)
        self.youtube_thread.finished.connect(self.youtube_worker.deleteLater)
        self.youtube_thread.finished.connect(self.youtube_thread.deleteLater)
        self.youtube_thread.finished.connect(self._cleanup_youtube_worker)

        self.youtube_thread.start()

    def _on_youtube_downloaded(self, path: object, title: str) -> None:
        self.import_youtube_action.setEnabled(True)
        self._start_song_load(Path(path), title)

    def _on_youtube_failed(self, message: str) -> None:
        self._hide_progress()
        self.import_song_file_action.setEnabled(True)
        self.import_youtube_action.setEnabled(True)
        if "отмен" in message.lower():
            self.status_label.setText("YouTube импорт отменён")
            app_logger.warning(message)
            return
        QMessageBox.critical(self, "Ошибка YouTube импорта", message)

    def _cleanup_youtube_worker(self) -> None:
        self.youtube_thread = None
        self.youtube_worker = None

    def _start_song_load(self, path: Path, title: str) -> None:
        self._show_progress("Анализ песни", "Готовлю анализ...")
        self.import_song_file_action.setEnabled(False)
        self.import_youtube_action.setEnabled(False)

        self.song_load_thread = QThread(self)
        self.song_load_worker = SongLoadWorker(path, title, self.voice_settings, self.singing_settings)
        self.song_load_worker.moveToThread(self.song_load_thread)

        self.song_load_thread.started.connect(self.song_load_worker.run)
        self.song_load_worker.statusChanged.connect(self._update_progress_text)
        self.song_load_worker.finished.connect(self._on_song_loaded)
        self.song_load_worker.failed.connect(self._on_song_load_failed)
        self.song_load_worker.finished.connect(self.song_load_thread.quit)
        self.song_load_worker.failed.connect(self.song_load_thread.quit)
        self.song_load_thread.finished.connect(self.song_load_worker.deleteLater)
        self.song_load_thread.finished.connect(self.song_load_thread.deleteLater)
        self.song_load_thread.finished.connect(self._cleanup_song_load_worker)

        self.song_load_thread.start()

    def _on_song_loaded(
        self,
        audio: object,
        vocals_audio: object,
        instrumental_audio: object,
        sample_rate: int,
        melody: list,
        title: str,
        demucs_stems_available: bool,
        vocals_path: object,
        instrumental_path: object,
    ) -> None:
        self._hide_progress()
        self.import_song_file_action.setEnabled(True)
        self.import_youtube_action.setEnabled(True)

        self.song_audio = audio
        self.song_vocals_audio = vocals_audio
        self.song_instrumental_audio = instrumental_audio
        self.song_vocals_path = Path(vocals_path) if vocals_path is not None else None
        self.song_instrumental_path = Path(instrumental_path) if instrumental_path is not None else None
        self.demucs_stems_available = demucs_stems_available
        self.song_sample_rate = sample_rate
        self.song_melody = melody
        self.melody_lookup = MelodyLookup(self.song_melody)
        self.song_title = title
        self.song_loaded = True
        self.song_playing = False
        self.song_pause_position = 0.0
        self.song_playback_mode = "full"
        self.current_cached_song_key = self._save_current_song_to_cache()
        self.last_saved_training_voice_count = 0

        self.frequency_history.clear()
        self.detector.clear_session_recording()
        self.song_label.setText(f"Режим: тренинг пения / {self.song_title}")

        self.score_label.setText("")
        self.status_label.setText("Песня импортирована. Нажми Старт тренировки.")
        self._refresh_menu_texts()
        self._update_singing_ui_visibility()
        app_logger.info(
            f"Song loaded: {title}, melody_points={len(self.song_melody)}, "
            f"demucs_stems_available={self.demucs_stems_available}"
        )

    def _save_current_song_to_cache(self) -> Optional[str]:
        if self.song_audio is None:
            return None
        try:
            return save_cached_song(
                title=self.song_title,
                playback_audio=self.song_audio,
                sample_rate=self.song_sample_rate,
                melody=self.song_melody,
                vocals_audio=self.song_vocals_audio,
                instrumental_audio=self.song_instrumental_audio,
                demucs_stems_available=self.demucs_stems_available,
            )
        except Exception as exc:
            app_logger.error(f"Failed to cache song: {exc}")
            return None

    def _load_cached_song(self, key: str) -> None:
        try:
            data = load_cached_song(key)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка загрузки песни", str(exc))
            app_logger.error(f"Failed to load cached song: {exc}")
            return
        self._apply_cached_song(data)

    def _apply_cached_song(self, data: CachedSongData) -> None:
        if self.song_playing:
            self.playback.stop("song")
        self.song_audio = data.playback_audio
        self.song_vocals_audio = data.vocals_audio
        self.song_instrumental_audio = data.instrumental_audio
        self.song_vocals_path = data.vocals_path
        self.song_instrumental_path = data.instrumental_path
        self.demucs_stems_available = data.info.demucs_stems_available
        self.song_sample_rate = data.info.sample_rate
        self.song_melody = data.melody
        self.melody_lookup = MelodyLookup(self.song_melody)
        self.song_title = data.info.title
        self.current_cached_song_key = data.info.key
        self.song_loaded = True
        self.song_playing = False
        self.song_pause_position = 0.0
        self.song_started_at = 0.0
        self.song_playback_mode = "full"
        self.last_saved_training_voice_count = 0
        self.singing_attempt_started_from_beginning = False
        self.frequency_history.clear()
        self.detector.clear_session_recording()
        self.last_score = None
        self.last_score_update_time = 0.0
        self.last_latency_update_time = 0.0
        self.cached_latency_ms = 0
        self.song_label.setText(f"Режим: тренинг пения / {self.song_title}")
        self.score_label.setText("")
        self._set_total_accuracy_card("—", None)
        self._set_accuracy_card(self.score_3s_label, "3 сек", "—")
        self._set_accuracy_card(self.score_10s_label, "10 сек", "—")
        self.score_detail_label.setText("Песня загружена из недавних. Нажми Старт тренировки.")
        self.status_label.setText("Песня загружена из недавних")
        self._refresh_menu_texts()
        self._update_singing_ui_visibility()
        self._refresh_chart()
        app_logger.info(f"Cached song loaded: {self.song_title}")

    def _demucs_export_available(self) -> bool:
        return bool(
            self.demucs_stems_available
            and self.song_vocals_path is not None
            and self.song_instrumental_path is not None
            and self.song_vocals_path.exists()
            and self.song_instrumental_path.exists()
        )

    def _export_demucs_vocals(self) -> None:
        self._export_demucs_stem(
            source_path=self.song_vocals_path,
            default_filename=self._default_stem_filename("vocals"),
            title="Сохранить вокал Demucs",
        )

    def _export_demucs_instrumental(self) -> None:
        self._export_demucs_stem(
            source_path=self.song_instrumental_path,
            default_filename=self._default_stem_filename("instrumental"),
            title="Сохранить инструментал Demucs",
        )

    def _export_demucs_stem(self, source_path: Optional[Path], default_filename: str, title: str) -> None:
        if source_path is None or not source_path.exists():
            QMessageBox.information(
                self,
                "Demucs дорожки недоступны",
                "Сначала включи Demucs в настройках и заново импортируй песню.",
            )
            self._refresh_menu_texts()
            return

        destination, _ = QFileDialog.getSaveFileName(
            self,
            title,
            default_filename,
            "WAV audio (*.wav)",
        )
        if not destination:
            return

        destination_path = Path(destination)
        if destination_path.suffix.lower() != ".wav":
            destination_path = destination_path.with_suffix(".wav")

        try:
            shutil.copyfile(source_path, destination_path)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка сохранения", str(exc))
            app_logger.error(f"Failed to export Demucs stem: {exc}")
            return

        self.status_label.setText(f"Файл сохранён: {destination_path.name}")
        app_logger.info(f"Demucs stem exported: {destination_path}")

    def _default_stem_filename(self, suffix: str) -> str:
        cleaned_title = "".join(
            char if char.isalnum() or char in " ._-" else "_"
            for char in self.song_title.strip()
        ).strip(" ._")
        if not cleaned_title:
            cleaned_title = "song"
        return f"{cleaned_title}_{suffix}.wav"

    def _on_song_load_failed(self, message: str) -> None:
        self._hide_progress()
        self.import_song_file_action.setEnabled(True)
        self.import_youtube_action.setEnabled(True)
        if "отмен" in message.lower():
            self.status_label.setText("Импорт песни отменён")
            app_logger.warning(message)
            return
        QMessageBox.critical(self, "Ошибка импорта", message)

    def _cleanup_song_load_worker(self) -> None:
        self.song_load_thread = None
        self.song_load_worker = None

    def _show_progress(self, title: str, text: str) -> None:
        if self.progress_dialog is None:
            self.progress_dialog = QProgressDialog(text, "Отмена", 0, 0, self)
            self.progress_dialog.setWindowTitle(title)
            self.progress_dialog.setWindowModality(Qt.WindowModality.NonModal)
            self.progress_dialog.setMinimumDuration(0)
            self.progress_dialog.canceled.connect(self._cancel_active_import)
            self.progress_dialog.show()
        else:
            self.progress_dialog.setWindowTitle(title)
            self.progress_dialog.setLabelText(text)

        app_logger.info(text)

    def _cancel_active_import(self) -> None:
        if self.youtube_worker is not None:
            self.youtube_worker.cancel()
            self.status_label.setText("Отмена YouTube импорта запрошена")
            app_logger.warning("YouTube import cancellation requested")
            self.import_song_file_action.setEnabled(True)
            self.import_youtube_action.setEnabled(True)
            self._hide_progress()
            return
        if self.song_load_worker is not None:
            self.song_load_worker.cancel()
            self.status_label.setText("Отмена импорта запрошена")
            app_logger.warning("Song import cancellation requested")
            self.import_song_file_action.setEnabled(True)
            self.import_youtube_action.setEnabled(True)
            self._hide_progress()
            return
        self.import_song_file_action.setEnabled(True)
        self.import_youtube_action.setEnabled(True)
        self._hide_progress()

    def _update_progress_text(self, text: str) -> None:
        if self.progress_dialog is not None:
            self.progress_dialog.setLabelText(text)

        app_logger.info(text)

    def _hide_progress(self) -> None:
        if self.progress_dialog is not None:
            self.progress_dialog.blockSignals(True)
            self.progress_dialog.close()
            self.progress_dialog = None

    def _toggle_song_playback(self) -> None:
        if not self.song_loaded or self._current_song_audio() is None:
            QMessageBox.information(self, "Песня", "Сначала импортируй песню")
            return

        if self.song_playing:
            self.playback.stop("song")
            self.song_pause_position = self._song_position()
            self.song_playing = False
            self.status_label.setText("Песня на паузе")
            self._refresh_menu_texts()
            app_logger.info(f"Song paused at {self.song_pause_position:.2f}s")
            return

        self._play_song_from_position(self.song_pause_position)

    def _play_song_from_position(self, position: float) -> None:
        audio_source = self._current_song_audio()
        if audio_source is None:
            return

        duration = len(audio_source) / self.song_sample_rate
        self.song_pause_position = max(0.0, min(position, duration))
        if len(self.frequency_history) == 0:
            self.singing_attempt_started_from_beginning = self.song_pause_position <= 1.0

        start_sample = int(self.song_pause_position * self.song_sample_rate)
        audio = audio_source[start_sample:]

        if len(audio) == 0:
            self.song_pause_position = 0.0
            audio = audio_source

        self._prepare_playback_start("song")
        try:
            self.playback.play("song", audio, self.song_sample_rate, self._on_playback_finished)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка воспроизведения песни", str(exc))
            app_logger.error(f"Song playback failed: {exc}")
            return

        self.song_started_at = time.monotonic() - self.song_pause_position
        self.song_playing = True
        self.status_label.setText("Песня играет. Пой поверх неё.")
        self._refresh_menu_texts()
        app_logger.info(
            f"Song playback started at {self.song_pause_position:.2f}s, mode={self.song_playback_mode}"
        )

    def _restart_song_training(self) -> None:
        if not self.song_loaded:
            return
        if self.song_playing:
            self.playback.stop("song")
            self.song_playing = False
        self.song_pause_position = 0.0
        self.song_started_at = 0.0
        self.frequency_history.clear()
        self.detector.clear_session_recording()
        self.last_score = None
        self.last_score_update_time = 0.0
        self.last_latency_update_time = 0.0
        self.cached_latency_ms = 0
        self.singing_attempt_started_from_beginning = True
        self.current_note_label.setText("—")
        self.frequency_label.setText("Твой голос: —")
        self._set_accuracy_card(self.score_3s_label, "3 сек", "—")
        self._set_accuracy_card(self.score_10s_label, "10 сек", "—")
        self._set_total_accuracy_card("—", None)
        self.score_detail_label.setText("Нажми «Старт тренировки» и пой поверх мелодии")
        self.status_label.setText("Тренировка сброшена")
        self.status_label.setStyleSheet("font-size: 18px; color: gray;")
        self._refresh_menu_texts()
        self._refresh_chart()
        app_logger.info("Singing training restarted")

    def _seek_song(self, position: float) -> None:
        if not self.song_loaded:
            return

        was_playing = self.song_playing

        if self.song_playing:
            self.playback.stop("song")
            self.song_playing = False

        self.song_pause_position = position
        self.frequency_history.clear()
        self.detector.clear_session_recording()
        self.last_score_update_time = 0.0
        self.last_latency_update_time = 0.0
        self.cached_latency_ms = 0
        self.singing_attempt_started_from_beginning = position <= 1.0
        self.status_label.setText(f"Позиция песни: {position:.1f} сек.")
        app_logger.info(f"Song seek requested: {position:.2f}s")

        if was_playing:
            self._play_song_from_position(position)

        self._refresh_menu_texts()
        self._refresh_chart()

    def _check_song_end(self) -> None:
        if not self.song_playing:
            return

        audio_source = self._current_song_audio()
        if audio_source is None:
            return

        duration = len(audio_source) / self.song_sample_rate

        if self._song_position() >= duration:
            self.playback.stop("song")
            self._finish_song_playback()

    def _song_position(self) -> float:
        if not self.song_playing:
            return self.song_pause_position
        return max(0.0, time.monotonic() - self.song_started_at)

    def _song_duration(self) -> Optional[float]:
        audio_source = self._current_song_audio()
        if audio_source is None:
            return None
        return len(audio_source) / self.song_sample_rate

    def _clear_song(self) -> None:
        if self.song_playing:
            self.playback.stop("song")

        app_logger.info("Singing mode cleared")
        self.song_loaded = False
        self.song_playing = False
        self.song_audio = None
        self.song_vocals_audio = None
        self.song_instrumental_audio = None
        self.song_vocals_path = None
        self.song_instrumental_path = None
        self.demucs_stems_available = False
        self.song_playback_mode = "full"
        self.song_melody = []
        self.melody_lookup = None
        self.song_pause_position = 0.0
        self.current_cached_song_key = None
        self.last_saved_training_voice_count = 0
        self.singing_attempt_started_from_beginning = False
        self.song_label.setText("Режим: тренинг голоса")
        self.score_label.setText("")
        self.score_big_label.setText("Точность пения")
        self._set_accuracy_card(self.score_3s_label, "3 сек", "—")
        self._set_accuracy_card(self.score_10s_label, "10 сек", "—")
        self._set_total_accuracy_card("—", None)
        self.score_detail_label.setText("Начни петь — оценка появится автоматически")
        self.tendency_label.setText("Оценка появится, когда будет достаточно распознанных нот")
        self.advice_label.setText("")
        self.frequency_history.clear()
        self._refresh_menu_texts()
        self._update_singing_ui_visibility()

    def _update_singing_ui_visibility(self) -> None:
        pitch_mode = not self.song_loaded
        self.target_combo.setVisible(pitch_mode)
        self.target_combo_caption.setVisible(pitch_mode)
        self.allowed_error_spin.setVisible(pitch_mode)
        self.allowed_error_caption.setVisible(pitch_mode)
        self.target_label.setVisible(pitch_mode)
        self.voice_type_label.setVisible(pitch_mode)
        self.song_button.setVisible(False)
        self.singing_stats_panel.setVisible(self.song_loaded)
        self.score_label.setVisible(False)
        self.confidence_label.setVisible(pitch_mode)
        self.error_label.setVisible(pitch_mode)
        self.error_bar.setVisible(pitch_mode)
        self._refresh_playback_mode_combo()

    def _update_target_label(self) -> None:
        note_name, octave, label = self._selected_target()
        frequency = note_to_frequency(note_name, octave)
        voice_type = classify_voice_by_frequency(frequency)
        self.target_label.setText(f"Цель: {label} / {frequency:.2f} Гц / {voice_type}")

    def _refresh_chart(self) -> None:
        if self.song_loaded:
            self.chart.set_singing_data(
                list(self.frequency_history),
                self.song_melody,
                self._song_position(),
                self.singing_settings.allowed_error_cents / 2.0,
                self._song_duration(),
                self.singing_settings.show_only_recent_seconds,
            )
        else:
            self.chart.set_pitch_data(
                list(self.frequency_history),
                self._target_frequency(),
                self.allowed_error_spin.value(),
            )

    def _expected_song_frequency(self, timestamp: float) -> Optional[float]:
        if self.melody_lookup is None:
            return None
        return self.melody_lookup.expected_frequency_at(timestamp)

    def _update_pitch(self) -> None:
        if self.song_loaded:
            frame = self.detector.read_latest_singing_pitch(self.singing_settings)
        else:
            frame = self.detector.read_latest_pitch()

        if frame is None:
            if self.song_loaded:
                self._handle_missing_singing_pitch()
            else:
                self.status_label.setText("Стабильный голос не найден")
                self.status_label.setStyleSheet("font-size: 18px; color: gray;")
                self.volume_bar.setValue(0)
            return

        timestamp = self._song_position() if self.song_loaded else time.monotonic()

        # В режиме пения стабильность голоса не является фильтром для метрики:
        # скоринг сам оценит качество попадания по кадрам. В режиме тренировки
        # голоса оставляем строгую логику, чтобы не реагировать на шум/случайные звуки.
        if self.song_loaded or frame.stable_voice:
            self.frequency_history.append((timestamp, frame.frequency_hz))

        if self.song_loaded:
            target_frequency = self._expected_song_frequency(timestamp) or frame.frequency_hz
            allowed_error = self.singing_settings.allowed_error_cents
        else:
            target_frequency = self._target_frequency()
            allowed_error = self.allowed_error_spin.value()

        error_hz = frame.frequency_hz - target_frequency
        clipped_error = max(-200, min(200, int(round(error_hz))))
        raw_volume_percent = max(0, min(100, int(frame.volume * 1200)))
        volume_percent = self._smooth_volume_percent(raw_volume_percent)
        voice_type = classify_voice_by_frequency(frame.frequency_hz)

        self.current_note_label.setText(frame.note_name)
        self.voice_type_label.setText(f"Тип голоса: {voice_type}")
        self.frequency_label.setText(f"Частота: {frame.frequency_hz:.2f} Гц")
        self.error_label.setText(f"Отклонение от цели: {error_hz:+.2f} Гц")
        self.confidence_label.setText(f"Уверенность распознавания: {frame.confidence:.2f}")
        self.error_bar.setValue(clipped_error)
        self.error_bar.setFormat(f"{error_hz:+.2f} Гц")
        self.volume_bar.setValue(volume_percent)

        if self.song_loaded:
            self.last_singing_pitch_time = time.monotonic()
            self._update_singing_score()
            self.current_note_label.setText(frame.note_name)
            self.frequency_label.setText(f"Твой голос: {frame.frequency_hz:.2f} Гц")
            self.volume_bar.setValue(volume_percent)
            return

        if not frame.stable_voice:
            self.status_label.setText("Жду устойчивый голос...")
            self.status_label.setStyleSheet("font-size: 18px; color: gray;")
            return

        if abs(error_hz) <= allowed_error:
            self.status_label.setText("Хорошо: попадаешь в диапазон")
            self.status_label.setStyleSheet("font-size: 18px; color: green;")
            return

        direction = "слишком высоко" if error_hz > 0 else "слишком низко"
        self.status_label.setText(f"Мимо диапазона: {direction}")
        self.status_label.setStyleSheet("font-size: 18px; color: red;")

        if self.beep_button.isChecked() and (frame.stable_voice or not self.voice_settings.beep_only_on_stable_voice):
            self.beeper.beep()

    def _handle_missing_singing_pitch(self) -> None:
        # Одиночные пропуски pitch detection нормальны для пения: согласные,
        # вдохи, переходы, атаки нот. Не мигаем статусом и не обнуляем громкость
        # каждый раз, иначе UI выглядит так, будто оценка постоянно ломается.
        time_since_pitch = time.monotonic() - self.last_singing_pitch_time if self.last_singing_pitch_time else 999.0

        if time_since_pitch < 1.25:
            self.volume_bar.setValue(self._smooth_volume_percent(0))
            return

        self.volume_bar.setValue(self._smooth_volume_percent(0))

        if self.last_score is not None:
            total = self.last_score.total_score
            if total.checked_frames >= 3:
                self.status_label.setText("Продолжай петь — оценка обновится при следующей распознанной ноте")
                self.status_label.setStyleSheet("font-size: 18px; color: #666666;")
                return

        self.status_label.setText("Пой в микрофон — точность появится после нескольких нот")
        self.status_label.setStyleSheet("font-size: 18px; color: gray;")

    def _smooth_volume_percent(self, value: int) -> int:
        # Быстрая реакция на появление голоса и мягкое затухание вниз.
        # Это только визуальный индикатор, на точность не влияет.
        alpha = 0.45 if value > self.smoothed_volume_percent else 0.12
        self.smoothed_volume_percent = self.smoothed_volume_percent * (1.0 - alpha) + value * alpha
        if self.smoothed_volume_percent < 1.0:
            self.smoothed_volume_percent = 0.0
        return int(round(self.smoothed_volume_percent))

    def _update_singing_score(self) -> None:
        if self.melody_lookup is None:
            return

        now = time.monotonic()

        # Realtime scoring no longer runs on every audio frame.
        # 4 times per second is enough for UI and prevents song playback/UI lag.
        if now - self.last_score_update_time < 0.25:
            return

        self.last_score_update_time = now
        voice_history = list(self.frequency_history)
        current_position = self._song_position()

        # Задержка больше не является ручной настройкой. Мы периодически оцениваем её
        # по последнему окну, а между пересчётами используем кэш, чтобы UI не лагал.
        if now - self.last_latency_update_time >= 1.5:
            self.cached_latency_ms = estimate_singing_latency(
                voice_history=voice_history,
                melody=self.song_melody,
                current_position=current_position,
                allowed_error_cents=self.singing_settings.allowed_error_cents,
                window_seconds=self.singing_settings.scoring_window_seconds,
                auto_latency_max_ms=self.singing_settings.auto_latency_max_ms,
                auto_latency_step_ms=self.singing_settings.auto_latency_step_ms,
                melody_lookup=self.melody_lookup,
                previous_latency_ms=self.cached_latency_ms,
                scoring_method=self.singing_settings.scoring_method,
            )
            self.last_latency_update_time = now

        summary = evaluate_singing_accuracy_summary(
            voice_history=voice_history,
            melody=self.song_melody,
            current_position=current_position,
            allowed_error_cents=self.singing_settings.allowed_error_cents,
            latency_ms=self.cached_latency_ms,
            melody_lookup=self.melody_lookup,
            scoring_method=self.singing_settings.scoring_method,
        )
        self.last_score = summary

        rank = None if summary.total_score.checked_frames < 3 else _rank_for_score(summary.total_score.score_percent)
        self._set_total_accuracy_card(self._format_score(summary.total_score), rank)
        self._set_accuracy_card(self.score_3s_label, "3 сек", self._format_score(summary.short_score))
        self._set_accuracy_card(self.score_10s_label, "10 сек", self._format_score(summary.medium_score))

        if summary.short_score.checked_frames == 0 and summary.medium_score.checked_frames == 0 and summary.total_score.checked_frames == 0:
            self.score_detail_label.setText("Пой в микрофон — точность появится после нескольких нот")
            self.advice_label.setText("")
            self.status_label.setText("Пой в микрофон, чтобы появилась оценка")
            return

        self.score_detail_label.setText(self._build_singing_status(summary))
        self.advice_label.setText("")

        score_for_status = summary.short_score if summary.short_score.checked_frames > 0 else summary.medium_score
        if score_for_status.score_percent >= 80:
            self.status_label.setText("Хорошо: форма мелодии похожа")
            self.status_label.setStyleSheet("font-size: 18px; color: green;")
        elif score_for_status.too_high_percent > score_for_status.too_low_percent + 20:
            self.status_label.setText("Часто выше нужной формы")
            self.status_label.setStyleSheet("font-size: 18px; color: red;")
        elif score_for_status.too_low_percent > score_for_status.too_high_percent + 20:
            self.status_label.setText("Часто ниже нужной формы")
            self.status_label.setStyleSheet("font-size: 18px; color: red;")
        else:
            self.status_label.setText("Неровно: работай над стабильностью и переходами")
            self.status_label.setStyleSheet("font-size: 18px; color: orange;")

    def _set_total_accuracy_card(self, value: str, rank: Optional[tuple[str, str]]) -> None:
        if rank is None:
            rank_html = ""
        else:
            rank_label, rank_color = rank
            rank_html = f" <span style='color:{rank_color}; font-weight:900;'>({rank_label})</span>"
        self.score_all_label.setText(
            "<div style='font-size: 14px; color: #555555;'>Всё время</div>"
            f"<div style='font-size: 32px; font-weight: 800;'>{value}{rank_html}</div>"
        )

    def _record_completed_training_result_if_available(self) -> None:
        if not self.singing_attempt_started_from_beginning:
            return
        if not self.song_loaded or self.current_cached_song_key is None or self.last_score is None:
            return
        duration = self._song_duration()
        if duration is None or self._song_position() < duration - 0.75:
            return
        total = self.last_score.total_score
        if total.checked_frames < 3:
            return
        voice_count = len(self.frequency_history)
        if voice_count == self.last_saved_training_voice_count:
            return
        rank, _ = _rank_for_score(total.score_percent)
        try:
            save_training_history_entry(
                song_key=self.current_cached_song_key,
                title=self.song_title or "Песня",
                score_percent=total.score_percent,
                rank=rank,
                recording_audio=self.detector.get_session_recording(),
                recording_sample_rate=self.detector.sample_rate,
            )
            self.last_saved_training_voice_count = voice_count
            self.singing_attempt_started_from_beginning = False
            self._refresh_recent_songs_menu()
            app_logger.info(f"Training result saved: song={self.song_title}, score={total.score_percent:.1f}, rank={rank}")
        except Exception as exc:
            app_logger.error(f"Failed to save training result: {exc}")

    def _show_training_history(self) -> None:
        dialog = TrainingHistoryDialog(
            load_training_history(),
            self.icons["chart"],
            self.icons["history_play"],
            self.icons["stop"],
            self.icons["delete"],
            self._show_song_progress,
            self._toggle_training_history_recording,
            self._stop_training_history_recording,
            self._delete_training_history_entry,
            self._clear_training_history,
            self,
        )
        self.training_history_dialog = dialog
        try:
            dialog.exec()
        finally:
            self._stop_training_history_recording()
            self.training_history_dialog = None

    def _toggle_training_history_recording(self, entry: TrainingHistoryEntry) -> bool:
        entry_key = (entry.song_key, entry.timestamp)

        if self.current_history_recording_key == entry_key and self.playback.current_kind == "history":
            self._stop_training_history_recording()
            return False

        if entry.recording_path is None or not entry.recording_path.exists():
            QMessageBox.information(self, "Запись недоступна", "Звуковой файл этой попытки не найден.")
            return False

        try:
            from core.training_storage import read_attempt_recording
            audio, sample_rate = read_attempt_recording(entry.recording_path)
            self._prepare_playback_start("history")
            self.playback.play("history", audio, sample_rate, self._on_playback_finished)
            self.current_history_recording_key = entry_key
            self.status_label.setText(f"Воспроизвожу попытку: {entry.title}")
            app_logger.info(f"Training history recording playback started: {entry.recording_path}")
            return True
        except Exception as exc:
            self.current_history_recording_key = None
            QMessageBox.critical(self, "Ошибка воспроизведения", str(exc))
            app_logger.error(f"Failed to play training history recording: {exc}")
            return False

    def _stop_training_history_recording(self) -> None:
        if self.playback.current_kind == "history":
            self.playback.stop("history")
            self.status_label.setText("Прослушивание попытки остановлено")
        self.current_history_recording_key = None
        if self.training_history_dialog is not None:
            self.training_history_dialog.reset_play_buttons()

    def _delete_training_history_entry(self, entry: TrainingHistoryEntry) -> bool:
        answer = QMessageBox.question(
            self,
            "Удалить запись",
            f"Удалить результат по песне «{entry.title}» от {_format_datetime(entry.timestamp)}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return False
        try:
            delete_training_history_entry(entry.song_key, entry.timestamp)
            app_logger.info(f"Training history entry deleted: song={entry.title}, timestamp={entry.timestamp}")
            return True
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка удаления", str(exc))
            app_logger.error(f"Failed to delete training history entry: {exc}")
            return False

    def _clear_training_history(self) -> bool:
        answer = QMessageBox.question(
            self,
            "Очистить весь прогресс",
            "Удалить всю историю тренировок пения? Это действие нельзя отменить.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return False
        try:
            clear_training_history()
            app_logger.warning("Training history cleared")
            return True
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка очистки", str(exc))
            app_logger.error(f"Failed to clear training history: {exc}")
            return False

    def _show_song_progress(self, song_key: str, title: str) -> None:
        dialog = SongProgressDialog(title, history_for_song(song_key), self)
        dialog.exec()

    def _format_score(self, score) -> str:
        if score.checked_frames < 3:
            return "—"
        return f"{score.score_percent:.0f}%"

    def _build_singing_status(self, summary: SingingAccuracySummary) -> str:
        short = summary.short_score
        medium = summary.medium_score
        if short.checked_frames >= 3:
            if short.score_percent >= 80:
                return "Сейчас хорошо попадаешь в мелодию"
            if short.score_percent >= 55:
                return "Сейчас близко, но есть заметные промахи"
            return "Сейчас далеко от мелодии. Пой медленнее и ближе к контуру"
        if medium.checked_frames >= 3:
            return "Продолжай петь — оценка за 3 секунды скоро появится"
        return "Начни петь — оценка появится автоматически"
