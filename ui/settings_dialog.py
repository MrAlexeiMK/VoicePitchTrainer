from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QFormLayout,
    QFrame,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.models import SingingTrainingSettings, VoiceTrainingSettings


class SettingsDialog(QDialog):
    def __init__(
        self,
        parent,
        voice_settings: VoiceTrainingSettings,
        singing_settings: SingingTrainingSettings,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Настройки")
        self.setMinimumSize(640, 460)
        self.resize(760, 620)

        tabs = QTabWidget()
        tabs.addTab(self._scrollable(self._create_voice_settings_tab(voice_settings)), "Тренинг голоса")
        tabs.addTab(self._scrollable(self._create_singing_settings_tab(singing_settings)), "Тренинг пения")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        self.reset_button = QPushButton("Сбросить настройки")
        self.reset_button.clicked.connect(self._reset_to_defaults)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.reset_button)
        buttons_layout.addStretch()
        buttons_layout.addWidget(buttons)

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addLayout(buttons_layout)

    def _scrollable(self, widget: QWidget) -> QScrollArea:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setWidget(widget)
        return scroll_area

    def _create_voice_settings_tab(self, settings: VoiceTrainingSettings) -> QWidget:
        widget = QWidget()

        self.min_confidence_spin = QDoubleSpinBox()
        self.min_confidence_spin.setRange(0.01, 0.90)
        self.min_confidence_spin.setSingleStep(0.01)
        self.min_confidence_spin.setValue(settings.min_confidence)

        self.voice_min_frequency_spin = QDoubleSpinBox()
        self.voice_min_frequency_spin.setRange(40.0, 500.0)
        self.voice_min_frequency_spin.setSingleStep(5.0)
        self.voice_min_frequency_spin.setValue(settings.voice_min_frequency)
        self.voice_min_frequency_spin.setSuffix(" Гц")

        self.voice_max_frequency_spin = QDoubleSpinBox()
        self.voice_max_frequency_spin.setRange(80.0, 1200.0)
        self.voice_max_frequency_spin.setSingleStep(5.0)
        self.voice_max_frequency_spin.setValue(settings.voice_max_frequency)
        self.voice_max_frequency_spin.setSuffix(" Гц")

        self.impulse_threshold_spin = QDoubleSpinBox()
        self.impulse_threshold_spin.setRange(6.0, 30.0)
        self.impulse_threshold_spin.setSingleStep(0.5)
        self.impulse_threshold_spin.setValue(settings.impulse_threshold)

        self.stable_frames_spin = QSpinBox()
        self.stable_frames_spin.setRange(1, 8)
        self.stable_frames_spin.setValue(settings.stable_frames)

        self.stable_spread_hz_spin = QDoubleSpinBox()
        self.stable_spread_hz_spin.setRange(5.0, 120.0)
        self.stable_spread_hz_spin.setSingleStep(5.0)
        self.stable_spread_hz_spin.setValue(settings.stable_spread_hz)
        self.stable_spread_hz_spin.setSuffix(" Гц")

        self.noise_gate_strength_spin = QDoubleSpinBox()
        self.noise_gate_strength_spin.setRange(0.5, 5.0)
        self.noise_gate_strength_spin.setSingleStep(0.1)
        self.noise_gate_strength_spin.setValue(settings.noise_gate_strength)

        self.beep_only_stable_checkbox = QCheckBox("Пищать только на устойчивый голос")
        self.beep_only_stable_checkbox.setChecked(settings.beep_only_on_stable_voice)

        form = QFormLayout(widget)
        form.addRow("Минимальная частота голоса", self.voice_min_frequency_spin)
        form.addRow("Максимальная частота голоса", self.voice_max_frequency_spin)
        form.addRow("Минимальная уверенность", self.min_confidence_spin)
        form.addRow("Отсечение кликов", self.impulse_threshold_spin)
        form.addRow("Кадров для устойчивого голоса", self.stable_frames_spin)
        form.addRow("Допустимый разброс устойчивости", self.stable_spread_hz_spin)
        form.addRow("Сила noise gate", self.noise_gate_strength_spin)
        form.addRow("Сигнал", self.beep_only_stable_checkbox)

        return widget

    def _create_singing_settings_tab(self, settings: SingingTrainingSettings) -> QWidget:
        widget = QWidget()

        self.use_demucs_checkbox = QCheckBox("Использовать Demucs для отделения вокала")
        self.use_demucs_checkbox.setChecked(settings.use_demucs)

        self.demucs_model_combo = QComboBox()
        self.demucs_model_combo.addItems(["htdemucs", "htdemucs_ft", "mdx_extra", "mdx_extra_q"])
        current_index = self.demucs_model_combo.findText(settings.demucs_model)
        self.demucs_model_combo.setCurrentIndex(current_index if current_index >= 0 else 0)

        self.singing_allowed_error_spin = QDoubleSpinBox()
        self.singing_allowed_error_spin.setRange(20.0, 300.0)
        self.singing_allowed_error_spin.setSingleStep(5.0)
        self.singing_allowed_error_spin.setValue(settings.allowed_error_cents)
        self.singing_allowed_error_spin.setSuffix(" cents")

        self.scoring_method_combo = QComboBox()
        self.scoring_method_combo.addItem("Сбалансированная", "balanced")
        self.scoring_method_combo.addItem("Строгая", "strict")
        self.scoring_method_combo.addItem("Мягкая", "soft")
        scoring_index = self.scoring_method_combo.findData(settings.scoring_method)
        self.scoring_method_combo.setCurrentIndex(scoring_index if scoring_index >= 0 else 0)

        self.scoring_window_spin = QSpinBox()
        self.scoring_window_spin.setRange(5, 180)
        self.scoring_window_spin.setValue(settings.scoring_window_seconds)
        self.scoring_window_spin.setSuffix(" секунд")

        self.auto_latency_max_spin = QSpinBox()
        self.auto_latency_max_spin.setRange(50, 2000)
        self.auto_latency_max_spin.setSingleStep(50)
        self.auto_latency_max_spin.setValue(settings.auto_latency_max_ms)
        self.auto_latency_max_spin.setSuffix(" мс")

        self.auto_latency_step_spin = QSpinBox()
        self.auto_latency_step_spin.setRange(5, 100)
        self.auto_latency_step_spin.setSingleStep(5)
        self.auto_latency_step_spin.setValue(settings.auto_latency_step_ms)
        self.auto_latency_step_spin.setSuffix(" мс")

        self.singing_min_confidence_spin = QDoubleSpinBox()
        self.singing_min_confidence_spin.setRange(0.01, 0.90)
        self.singing_min_confidence_spin.setSingleStep(0.01)
        self.singing_min_confidence_spin.setValue(settings.singing_min_confidence)

        self.singing_min_volume_spin = QDoubleSpinBox()
        self.singing_min_volume_spin.setRange(0.001, 0.100)
        self.singing_min_volume_spin.setSingleStep(0.001)
        self.singing_min_volume_spin.setDecimals(3)
        self.singing_min_volume_spin.setValue(settings.singing_min_volume)

        self.singing_min_frequency_spin = QDoubleSpinBox()
        self.singing_min_frequency_spin.setRange(40.0, 500.0)
        self.singing_min_frequency_spin.setSingleStep(5.0)
        self.singing_min_frequency_spin.setValue(settings.singing_min_frequency)
        self.singing_min_frequency_spin.setSuffix(" Гц")

        self.singing_max_frequency_spin = QDoubleSpinBox()
        self.singing_max_frequency_spin.setRange(80.0, 2000.0)
        self.singing_max_frequency_spin.setSingleStep(5.0)
        self.singing_max_frequency_spin.setValue(settings.singing_max_frequency)
        self.singing_max_frequency_spin.setSuffix(" Гц")

        self.singing_noise_gate_checkbox = QCheckBox("Использовать noise gate для микрофона в режиме пения")
        self.singing_noise_gate_checkbox.setChecked(settings.singing_use_noise_gate)

        self.melody_step_spin = QSpinBox()
        self.melody_step_spin.setRange(20, 500)
        self.melody_step_spin.setSingleStep(10)
        self.melody_step_spin.setValue(settings.melody_analysis_step_ms)
        self.melody_step_spin.setSuffix(" мс")

        self.melody_min_confidence_spin = QDoubleSpinBox()
        self.melody_min_confidence_spin.setRange(0.01, 0.90)
        self.melody_min_confidence_spin.setSingleStep(0.01)
        self.melody_min_confidence_spin.setValue(settings.melody_min_confidence)

        self.melody_min_frequency_spin = QDoubleSpinBox()
        self.melody_min_frequency_spin.setRange(40.0, 500.0)
        self.melody_min_frequency_spin.setSingleStep(5.0)
        self.melody_min_frequency_spin.setValue(settings.melody_min_frequency)
        self.melody_min_frequency_spin.setSuffix(" Гц")

        self.melody_max_frequency_spin = QDoubleSpinBox()
        self.melody_max_frequency_spin.setRange(80.0, 2000.0)
        self.melody_max_frequency_spin.setSingleStep(5.0)
        self.melody_max_frequency_spin.setValue(settings.melody_max_frequency)
        self.melody_max_frequency_spin.setSuffix(" Гц")

        self.jump_limit_spin = QDoubleSpinBox()
        self.jump_limit_spin.setRange(100.0, 2400.0)
        self.jump_limit_spin.setSingleStep(50.0)
        self.jump_limit_spin.setValue(settings.melody_jump_limit_cents)
        self.jump_limit_spin.setSuffix(" cents")

        self.recent_seconds_spin = QSpinBox()
        self.recent_seconds_spin.setRange(10, 180)
        self.recent_seconds_spin.setValue(settings.show_only_recent_seconds)
        self.recent_seconds_spin.setSuffix(" секунд")

        hint = QLabel(
            "Задержка голоса рассчитывается автоматически: программа подбирает сдвиг между услышанной мелодией "
            "и твоим голосом. Ручной константы больше нет. В режиме пения стабильный голос не обязателен."
        )
        hint.setWordWrap(True)

        form = QFormLayout(widget)
        form.addRow("AI вокал", self.use_demucs_checkbox)
        form.addRow("Модель Demucs", self.demucs_model_combo)
        form.addRow("Допуск оценки", self.singing_allowed_error_spin)
        form.addRow("Метрика точности", self.scoring_method_combo)
        form.addRow("Окно авто-задержки", self.scoring_window_spin)
        form.addRow("Максимум авто-задержки", self.auto_latency_max_spin)
        form.addRow("Шаг авто-задержки", self.auto_latency_step_spin)
        form.addRow("Мин. уверенность микрофона", self.singing_min_confidence_spin)
        form.addRow("Мин. громкость микрофона", self.singing_min_volume_spin)
        form.addRow("Мин. частота микрофона", self.singing_min_frequency_spin)
        form.addRow("Макс. частота микрофона", self.singing_max_frequency_spin)
        form.addRow("Фильтр микрофона", self.singing_noise_gate_checkbox)
        form.addRow("Шаг анализа мелодии", self.melody_step_spin)
        form.addRow("Уверенность мелодии", self.melody_min_confidence_spin)
        form.addRow("Мин. частота мелодии", self.melody_min_frequency_spin)
        form.addRow("Макс. частота мелодии", self.melody_max_frequency_spin)
        form.addRow("Лимит скачка мелодии", self.jump_limit_spin)
        form.addRow("Окно графика", self.recent_seconds_spin)
        form.addRow("", hint)

        return widget


    def _reset_to_defaults(self) -> None:
        self._apply_voice_defaults(VoiceTrainingSettings())
        self._apply_singing_defaults(SingingTrainingSettings())

    def _apply_voice_defaults(self, settings: VoiceTrainingSettings) -> None:
        self.min_confidence_spin.setValue(settings.min_confidence)
        self.voice_min_frequency_spin.setValue(settings.voice_min_frequency)
        self.voice_max_frequency_spin.setValue(settings.voice_max_frequency)
        self.impulse_threshold_spin.setValue(settings.impulse_threshold)
        self.stable_frames_spin.setValue(settings.stable_frames)
        self.stable_spread_hz_spin.setValue(settings.stable_spread_hz)
        self.noise_gate_strength_spin.setValue(settings.noise_gate_strength)
        self.beep_only_stable_checkbox.setChecked(settings.beep_only_on_stable_voice)

    def _apply_singing_defaults(self, settings: SingingTrainingSettings) -> None:
        self.use_demucs_checkbox.setChecked(settings.use_demucs)
        current_index = self.demucs_model_combo.findText(settings.demucs_model)
        self.demucs_model_combo.setCurrentIndex(current_index if current_index >= 0 else 0)
        self.singing_allowed_error_spin.setValue(settings.allowed_error_cents)
        scoring_index = self.scoring_method_combo.findData(settings.scoring_method)
        self.scoring_method_combo.setCurrentIndex(scoring_index if scoring_index >= 0 else 0)
        self.scoring_window_spin.setValue(settings.scoring_window_seconds)
        self.auto_latency_max_spin.setValue(settings.auto_latency_max_ms)
        self.auto_latency_step_spin.setValue(settings.auto_latency_step_ms)
        self.singing_min_confidence_spin.setValue(settings.singing_min_confidence)
        self.singing_min_volume_spin.setValue(settings.singing_min_volume)
        self.singing_min_frequency_spin.setValue(settings.singing_min_frequency)
        self.singing_max_frequency_spin.setValue(settings.singing_max_frequency)
        self.singing_noise_gate_checkbox.setChecked(settings.singing_use_noise_gate)
        self.melody_step_spin.setValue(settings.melody_analysis_step_ms)
        self.melody_min_confidence_spin.setValue(settings.melody_min_confidence)
        self.melody_min_frequency_spin.setValue(settings.melody_min_frequency)
        self.melody_max_frequency_spin.setValue(settings.melody_max_frequency)
        self.jump_limit_spin.setValue(settings.melody_jump_limit_cents)
        self.recent_seconds_spin.setValue(settings.show_only_recent_seconds)

    def apply_to(
        self,
        voice_settings: VoiceTrainingSettings,
        singing_settings: SingingTrainingSettings,
    ) -> None:
        voice_settings.min_confidence = self.min_confidence_spin.value()
        voice_settings.voice_min_frequency = self.voice_min_frequency_spin.value()
        voice_settings.voice_max_frequency = self.voice_max_frequency_spin.value()
        voice_settings.impulse_threshold = self.impulse_threshold_spin.value()
        voice_settings.stable_frames = self.stable_frames_spin.value()
        voice_settings.stable_spread_hz = self.stable_spread_hz_spin.value()
        voice_settings.noise_gate_strength = self.noise_gate_strength_spin.value()
        voice_settings.beep_only_on_stable_voice = self.beep_only_stable_checkbox.isChecked()

        singing_settings.use_demucs = self.use_demucs_checkbox.isChecked()
        singing_settings.demucs_model = self.demucs_model_combo.currentText()
        singing_settings.allowed_error_cents = self.singing_allowed_error_spin.value()
        singing_settings.scoring_method = self.scoring_method_combo.currentData()
        singing_settings.scoring_window_seconds = self.scoring_window_spin.value()
        singing_settings.voice_latency_ms = 0
        singing_settings.auto_detect_latency = True
        singing_settings.auto_latency_max_ms = self.auto_latency_max_spin.value()
        singing_settings.auto_latency_step_ms = self.auto_latency_step_spin.value()
        singing_settings.singing_min_confidence = self.singing_min_confidence_spin.value()
        singing_settings.singing_min_volume = self.singing_min_volume_spin.value()
        singing_settings.singing_min_frequency = self.singing_min_frequency_spin.value()
        singing_settings.singing_max_frequency = self.singing_max_frequency_spin.value()
        singing_settings.singing_use_noise_gate = self.singing_noise_gate_checkbox.isChecked()
        singing_settings.melody_analysis_step_ms = self.melody_step_spin.value()
        singing_settings.melody_min_confidence = self.melody_min_confidence_spin.value()
        singing_settings.melody_min_frequency = self.melody_min_frequency_spin.value()
        singing_settings.melody_max_frequency = self.melody_max_frequency_spin.value()
        singing_settings.melody_jump_limit_cents = self.jump_limit_spin.value()
        singing_settings.show_only_recent_seconds = self.recent_seconds_spin.value()
