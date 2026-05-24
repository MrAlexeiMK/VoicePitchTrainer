from typing import Optional

import numpy as np

from core.constants import MAX_FREQUENCY, MIN_FREQUENCY
from core.models import VoiceTrainingSettings


def light_noise_gate(samples: np.ndarray, settings: VoiceTrainingSettings) -> np.ndarray:
    result = samples.astype(np.float64)
    result -= np.mean(result)
    abs_samples = np.abs(result)
    noise_level = np.percentile(abs_samples, 25)
    gate = max(noise_level * settings.noise_gate_strength, settings.min_noise_gate)
    result[abs_samples < gate] = 0.0
    return result


def normalize_for_playback(samples: np.ndarray) -> np.ndarray:
    result = samples.astype(np.float32).copy()
    result -= float(np.mean(result))
    peak = float(np.max(np.abs(result))) if len(result) else 0.0
    if peak > 0.98:
        result *= 0.98 / peak
    return result


def is_impulse_like_sound(samples: np.ndarray, settings: VoiceTrainingSettings) -> bool:
    values = samples.astype(np.float64)
    abs_values = np.abs(values)
    rms = float(np.sqrt(np.mean(values * values)))
    if rms <= 1e-8:
        return True

    peak = float(np.max(abs_values))
    crest_factor = peak / rms
    active_ratio = float(np.mean(abs_values > rms * 0.6))
    zero_crossings = float(np.mean(np.abs(np.diff(np.signbit(values)))))
    return crest_factor > settings.impulse_threshold and active_ratio < 0.08 and zero_crossings > 0.12


def detect_pitch_autocorrelation(
    samples: np.ndarray,
    sample_rate: int,
    settings: VoiceTrainingSettings,
    min_frequency: float = MIN_FREQUENCY,
    max_frequency: float = MAX_FREQUENCY,
    min_confidence: Optional[float] = None,
    use_noise_gate: bool = True,
    reject_impulses: bool = True,
) -> Optional[tuple[float, float]]:
    if reject_impulses and is_impulse_like_sound(samples, settings):
        return None

    # В режиме тренировки пения noise gate может съедать тихие/фоновые участки
    # и делать метрику точности слишком зависимой от фильтра микрофона.
    values = light_noise_gate(samples, settings) if use_noise_gate else samples.astype(np.float64)
    values -= np.mean(values)
    if np.max(np.abs(values)) < 1e-6:
        return None

    values *= np.hanning(len(values))
    correlation = np.correlate(values, values, mode="full")
    correlation = correlation[len(correlation) // 2:]

    min_lag = int(sample_rate / max_frequency)
    max_lag = int(sample_rate / min_frequency)
    max_lag = min(max_lag, len(correlation) - 1)

    search_area = correlation[min_lag:max_lag]
    if len(search_area) == 0:
        return None

    peak_index = int(np.argmax(search_area)) + min_lag
    if peak_index <= 0:
        return None

    refined_peak = float(peak_index)
    if 1 <= peak_index < len(correlation) - 1:
        left = correlation[peak_index - 1]
        center = correlation[peak_index]
        right = correlation[peak_index + 1]
        denominator = left - 2.0 * center + right
        if abs(denominator) > 1e-12:
            refined_peak += 0.5 * (left - right) / denominator

    if refined_peak <= 0:
        return None

    frequency_hz = sample_rate / refined_peak
    if frequency_hz < min_frequency or frequency_hz > max_frequency:
        return None

    confidence = correlation[peak_index] / correlation[0] if correlation[0] != 0 else 0.0
    required_confidence = settings.min_confidence if min_confidence is None else min_confidence
    if confidence < required_confidence:
        return None

    return float(frequency_hz), float(confidence)
