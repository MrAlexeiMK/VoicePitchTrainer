from dataclasses import dataclass

from core.constants import DEFAULT_RECORDING_SECONDS


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    channels: int
    sample_rate: int


@dataclass(frozen=True)
class PitchFrame:
    frequency_hz: float
    note_name: str
    confidence: float
    volume: float
    stable_voice: bool


@dataclass
class VoiceTrainingSettings:
    # Оставлено внутренне для circular buffer старой функции get_recent_recording().
    # В UI больше не настраивается: "Прослушать мой голос" играет запись с момента Старт до Стоп.
    recording_seconds: int = DEFAULT_RECORDING_SECONDS
    min_confidence: float = 0.10
    impulse_threshold: float = 10.0
    stable_frames: int = 2
    stable_spread_hz: float = 55.0
    stable_spread_percent: float = 22.0
    noise_gate_strength: float = 1.6
    min_noise_gate: float = 0.0008
    beep_only_on_stable_voice: bool = True
    voice_min_frequency: float = 70.0
    voice_max_frequency: float = 450.0


@dataclass
class SingingTrainingSettings:
    # Оценка пения идёт в cents и с автоматической транспозицией.
    allowed_error_cents: float = 80.0
    scoring_window_seconds: int = 45
    scoring_method: str = "balanced"

    # Задержка реакции неизвестна заранее: программа оценивает её автоматически
    # по последнему устойчивому окну и применяет найденное значение к 3 сек / 10 сек / всему времени.
    voice_latency_ms: int = 0
    auto_detect_latency: bool = True
    auto_latency_max_ms: int = 700
    auto_latency_step_ms: int = 20

    # В режиме пения детекция мягче, чем в режиме тренировки голоса:
    # стабильность голоса не обязательна, а noise gate можно выключить.
    singing_min_confidence: float = 0.06
    # Мягкий режим пения всё равно должен отсеивать тишину: без этого
    # autocorrelation может находить случайную "ноту" в шуме комнаты.
    singing_min_volume: float = 0.010
    singing_min_frequency: float = 60.0
    singing_max_frequency: float = 900.0
    singing_use_noise_gate: bool = False

    melody_analysis_step_ms: int = 80
    melody_min_confidence: float = 0.08
    melody_min_frequency: float = 70.0
    melody_max_frequency: float = 650.0
    melody_jump_limit_cents: float = 900.0
    show_only_recent_seconds: int = 45

    use_demucs: bool = True
    demucs_model: str = "htdemucs"
