import numpy as np

from audio.processing import detect_pitch_autocorrelation
from core.models import SingingTrainingSettings, VoiceTrainingSettings
from core.music import cents_between


def preprocess_for_melody_analysis(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    result = audio.astype(np.float32).copy()
    result -= float(np.mean(result))
    if len(result) > 1:
        previous = np.concatenate(([result[0]], result[:-1]))
        result = result - 0.97 * previous
    peak = float(np.max(np.abs(result))) if len(result) else 0.0
    if peak > 0:
        result = result / peak * 0.9
    return result.astype(np.float32)


def analyze_song_melody(audio: np.ndarray, sample_rate: int, voice_settings: VoiceTrainingSettings, singing_settings: SingingTrainingSettings, progress_callback=None) -> list[tuple[float, float]]:
    audio = preprocess_for_melody_analysis(audio, sample_rate)
    step_samples = max(512, int(sample_rate * singing_settings.melody_analysis_step_ms / 1000))
    window_samples = max(2048, step_samples * 4)
    total = max(1, len(audio) - window_samples)
    raw_points: list[tuple[float, float]] = []
    for index, start in enumerate(range(0, total, step_samples)):
        if progress_callback is not None and index % 25 == 0:
            progress_callback(f"Анализирую вокальную мелодию: {int(start / total * 100)}%")
        chunk = audio[start:start + window_samples]
        detected = detect_pitch_autocorrelation(chunk, sample_rate, voice_settings, min_frequency=singing_settings.melody_min_frequency, max_frequency=singing_settings.melody_max_frequency, min_confidence=singing_settings.melody_min_confidence)
        if detected is None:
            continue
        frequency_hz, _ = detected
        raw_points.append((start / sample_rate, frequency_hz))
    if progress_callback is not None:
        progress_callback("Чищу вокальную линию от гармоник и скачков...")
    return clean_and_smooth_melody(raw_points, singing_settings)


def clean_and_smooth_melody(points: list[tuple[float, float]], settings: SingingTrainingSettings) -> list[tuple[float, float]]:
    if not points:
        return []
    corrected: list[tuple[float, float]] = []
    previous_frequency: float | None = None
    for timestamp, frequency in points:
        while frequency > settings.melody_max_frequency and frequency / 2 >= settings.melody_min_frequency:
            frequency /= 2.0
        if previous_frequency is not None:
            candidates = [frequency]
            if frequency / 2 >= settings.melody_min_frequency:
                candidates.append(frequency / 2)
            if frequency * 2 <= settings.melody_max_frequency:
                candidates.append(frequency * 2)
            frequency = min(candidates, key=lambda candidate: abs(cents_between(candidate, previous_frequency)))
            if abs(cents_between(frequency, previous_frequency)) > settings.melody_jump_limit_cents:
                continue
        corrected.append((timestamp, frequency))
        previous_frequency = frequency
    if len(corrected) < 3:
        return corrected
    smoothed: list[tuple[float, float]] = []
    for index, (timestamp, frequency) in enumerate(corrected):
        left = max(0, index - 2)
        right = min(len(corrected), index + 3)
        values = [item[1] for item in corrected[left:right]]
        smoothed.append((timestamp, float(np.median(values))))
    return smoothed
