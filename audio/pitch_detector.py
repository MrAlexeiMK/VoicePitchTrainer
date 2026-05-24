import queue
from collections import deque
from typing import Optional

import numpy as np
import sounddevice as sd

from audio.processing import detect_pitch_autocorrelation, normalize_for_playback
from core.constants import BLOCK_SIZE, DEFAULT_SAMPLE_RATE, MAX_SESSION_RECORDING_SECONDS
from core.models import PitchFrame, SingingTrainingSettings, VoiceTrainingSettings
from core.music import frequency_to_note


class AudioPitchDetector:
    def __init__(self, settings: VoiceTrainingSettings) -> None:
        self.settings = settings
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=32)
        self._stream: Optional[sd.InputStream] = None
        self._device_index: Optional[int] = None
        self._sample_rate = DEFAULT_SAMPLE_RATE
        self._noise_floor = 0.003
        self._recent_frequencies: deque[float] = deque(maxlen=settings.stable_frames)
        self._recorded_chunks: deque[np.ndarray] = deque()
        self._recorded_samples_count = 0
        self._session_chunks: deque[np.ndarray] = deque()
        self._session_samples_count = 0

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def apply_settings(self) -> None:
        self._recent_frequencies = deque(self._recent_frequencies, maxlen=self.settings.stable_frames)
        self._trim_recording_buffer()
        self._trim_session_recording_buffer()

    def set_device(self, device_index: int, sample_rate: int) -> None:
        self._device_index = device_index
        self._sample_rate = sample_rate
        self._trim_recording_buffer()
        self._trim_session_recording_buffer()

    def start(self, clear_session_recording: bool = True) -> None:
        if self._stream is not None:
            return
        if clear_session_recording:
            self.clear_session_recording()
        self._noise_floor = 0.003
        self._recent_frequencies.clear()
        self._stream = sd.InputStream(
            device=self._device_index,
            channels=1,
            samplerate=self._sample_rate,
            blocksize=BLOCK_SIZE,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None
        self._recent_frequencies.clear()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def clear_session_recording(self) -> None:
        self._session_chunks.clear()
        self._session_samples_count = 0

    def get_session_recording(self) -> Optional[np.ndarray]:
        if not self._session_chunks:
            return None
        samples = np.concatenate(self._session_chunks).astype(np.float32)
        return normalize_for_playback(samples)

    def get_recent_recording(self, seconds: int) -> Optional[np.ndarray]:
        chunks = list(self._recorded_chunks)
        if not chunks:
            return None
        samples = np.concatenate(chunks).astype(np.float32)
        needed = max(1, int(seconds * self._sample_rate))
        if len(samples) > needed:
            samples = samples[-needed:]
        return normalize_for_playback(samples)

    def read_latest_pitch(self) -> Optional[PitchFrame]:
        """
        Строгий режим тренировки голоса.

        Важно: здесь намеренно оставлена прежняя логика детекции почти один-в-один.
        Мягкая детекция без сильного noise gate используется только в read_latest_singing_pitch().
        """
        latest_audio: Optional[np.ndarray] = None
        while True:
            try:
                latest_audio = self._queue.get_nowait()
            except queue.Empty:
                break
        if latest_audio is None:
            return None

        samples = latest_audio.astype(np.float32).flatten()
        volume = float(np.sqrt(np.mean(samples * samples)))
        if volume < self._noise_floor * 2.0:
            self._noise_floor = self._noise_floor * 0.96 + volume * 0.04
            self._recent_frequencies.clear()
            return None

        detected = detect_pitch_autocorrelation(samples, self._sample_rate, self.settings)
        if detected is None:
            self._recent_frequencies.clear()
            return None

        frequency_hz, confidence = detected
        if frequency_hz < self.settings.voice_min_frequency or frequency_hz > self.settings.voice_max_frequency:
            self._recent_frequencies.clear()
            return None

        self._recent_frequencies.append(frequency_hz)
        stable_voice = False
        if len(self._recent_frequencies) >= self.settings.stable_frames:
            average = sum(self._recent_frequencies) / len(self._recent_frequencies)
            spread = max(self._recent_frequencies) - min(self._recent_frequencies)
            allowed_spread = max(self.settings.stable_spread_hz, average * self.settings.stable_spread_percent / 100.0)
            stable_voice = spread <= allowed_spread
            if stable_voice:
                frequency_hz = average

        return PitchFrame(
            frequency_hz=frequency_hz,
            note_name=frequency_to_note(frequency_hz),
            confidence=confidence,
            volume=volume,
            stable_voice=stable_voice,
        )

    def read_latest_singing_pitch(self, settings: SingingTrainingSettings) -> Optional[PitchFrame]:
        return self._read_latest_pitch(
            min_frequency=settings.singing_min_frequency,
            max_frequency=settings.singing_max_frequency,
            min_confidence=settings.singing_min_confidence,
            use_noise_gate=settings.singing_use_noise_gate,
            use_noise_floor=False,
            smooth_octave=True,
            min_volume=settings.singing_min_volume,
        )

    def _read_latest_pitch(
        self,
        min_frequency: float,
        max_frequency: float,
        min_confidence: float,
        use_noise_gate: bool,
        use_noise_floor: bool,
        smooth_octave: bool,
        min_volume: float = 0.0,
    ) -> Optional[PitchFrame]:
        latest_audio: Optional[np.ndarray] = None
        while True:
            try:
                latest_audio = self._queue.get_nowait()
            except queue.Empty:
                break
        if latest_audio is None:
            return None

        samples = latest_audio.astype(np.float32).flatten()
        volume = float(np.sqrt(np.mean(samples * samples)))

        # В режиме пения мы не требуем стабильный голос и не применяем жёсткий
        # voice-mode gate, но абсолютную тишину всё равно нужно отсеивать.
        # Иначе autocorrelation иногда находит случайные частоты в шуме комнаты,
        # из-за чего синяя линия графика "поёт" даже когда пользователь молчит.
        if min_volume > 0.0 and volume < min_volume:
            self._noise_floor = self._noise_floor * 0.98 + volume * 0.02
            return None

        # Строгий noise floor нужен для режима тренировки голоса, где важна именно
        # устойчивая фонация. В режиме пения он мешает: метроном/песня/комната
        # могут менять фон, а нам нужны все достаточно уверенные pitch frames.
        if use_noise_floor and volume < self._noise_floor * 2.0:
            self._noise_floor = self._noise_floor * 0.96 + volume * 0.04
            self._recent_frequencies.clear()
            return None

        detected = detect_pitch_autocorrelation(
            samples,
            self._sample_rate,
            self.settings,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            min_confidence=min_confidence,
            use_noise_gate=use_noise_gate,
            reject_impulses=True,
        )
        if detected is None:
            if use_noise_floor:
                self._recent_frequencies.clear()
            return None

        frequency_hz, confidence = detected
        if smooth_octave:
            frequency_hz = self._smooth_octave_jump(frequency_hz, min_frequency, max_frequency)

        self._recent_frequencies.append(frequency_hz)
        stable_voice = False
        if len(self._recent_frequencies) >= self.settings.stable_frames:
            average = sum(self._recent_frequencies) / len(self._recent_frequencies)
            spread = max(self._recent_frequencies) - min(self._recent_frequencies)
            allowed_spread = max(self.settings.stable_spread_hz, average * self.settings.stable_spread_percent / 100.0)
            stable_voice = spread <= allowed_spread
            if stable_voice:
                frequency_hz = average

        return PitchFrame(
            frequency_hz=frequency_hz,
            note_name=frequency_to_note(frequency_hz),
            confidence=confidence,
            volume=volume,
            stable_voice=stable_voice,
        )

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            return
        audio = indata[:, 0].copy().astype(np.float32)
        self._recorded_chunks.append(audio)
        self._session_chunks.append(audio.copy())
        self._recorded_samples_count += len(audio)
        self._session_samples_count += len(audio)
        self._trim_recording_buffer()
        self._trim_session_recording_buffer()
        try:
            self._queue.put_nowait(audio.copy())
        except queue.Full:
            pass

    def _smooth_octave_jump(self, frequency_hz: float, min_frequency: float, max_frequency: float) -> float:
        if not self._recent_frequencies:
            return frequency_hz

        previous = sum(self._recent_frequencies) / len(self._recent_frequencies)
        candidates = [frequency_hz]

        half = frequency_hz / 2.0
        if half >= min_frequency:
            candidates.append(half)

        double = frequency_hz * 2.0
        if double <= max_frequency:
            candidates.append(double)

        return min(candidates, key=lambda candidate: abs(candidate - previous))

    def _trim_session_recording_buffer(self) -> None:
        max_samples = max(1, int(MAX_SESSION_RECORDING_SECONDS * self._sample_rate))
        while self._session_samples_count > max_samples and self._session_chunks:
            removed = self._session_chunks.popleft()
            self._session_samples_count -= len(removed)

    def _trim_recording_buffer(self) -> None:
        max_samples = max(1, int(self.settings.recording_seconds * self._sample_rate))
        while self._recorded_samples_count > max_samples and self._recorded_chunks:
            removed = self._recorded_chunks.popleft()
            self._recorded_samples_count -= len(removed)
