from bisect import bisect_left
from dataclasses import dataclass
from statistics import median

from core.music import cents_between


@dataclass(frozen=True)
class SingingScore:
    score_percent: float
    checked_frames: int
    good_frames: int
    median_shift_cents: float
    average_abs_error_cents: float
    too_high_percent: float
    too_low_percent: float
    detected_latency_ms: int
    advice: str


@dataclass(frozen=True)
class SingingAccuracySummary:
    short_score: SingingScore
    medium_score: SingingScore
    total_score: SingingScore
    detected_latency_ms: int


class MelodyLookup:
    """
    Быстрый поиск частоты мелодии по timestamp.

    Раньше expected_frequency_at() каждый раз делал min(melody, key=...),
    то есть O(N) на один voice frame. В realtime это резко грузит UI.
    Здесь поиск O(log N).
    """

    def __init__(self, melody: list[tuple[float, float]]) -> None:
        self.melody = melody
        self.times = [timestamp for timestamp, _ in melody]

    def expected_frequency_at(self, timestamp: float, max_distance_seconds: float = 0.30) -> float | None:
        if not self.melody or timestamp < 0:
            return None

        index = bisect_left(self.times, timestamp)
        candidates = []

        if index < len(self.melody):
            candidates.append(self.melody[index])
        if index > 0:
            candidates.append(self.melody[index - 1])

        if not candidates:
            return None

        nearest_timestamp, nearest_frequency = min(candidates, key=lambda item: abs(item[0] - timestamp))

        if abs(nearest_timestamp - timestamp) > max_distance_seconds:
            return None

        return nearest_frequency


def evaluate_singing(
    voice_history: list[tuple[float, float]],
    melody: list[tuple[float, float]],
    current_position: float,
    allowed_error_cents: float,
    window_seconds: int,
    voice_latency_ms: int = 0,
    auto_detect_latency: bool = True,
    auto_latency_max_ms: int = 500,
    auto_latency_step_ms: int = 20,
    melody_lookup: MelodyLookup | None = None,
    scoring_method: str = "balanced",
) -> SingingScore:
    lookup = melody_lookup if melody_lookup is not None else MelodyLookup(melody)

    if not auto_detect_latency:
        return evaluate_singing_with_fixed_latency(
            voice_history=voice_history,
            melody=melody,
            current_position=current_position,
            allowed_error_cents=allowed_error_cents,
            window_seconds=window_seconds,
            latency_ms=voice_latency_ms,
            melody_lookup=lookup,
            scoring_method=scoring_method,
        )

    latency_ms = estimate_singing_latency(
        voice_history=voice_history,
        melody=melody,
        current_position=current_position,
        allowed_error_cents=allowed_error_cents,
        window_seconds=window_seconds,
        auto_latency_max_ms=auto_latency_max_ms,
        auto_latency_step_ms=auto_latency_step_ms,
        melody_lookup=lookup,
        previous_latency_ms=voice_latency_ms,
        scoring_method=scoring_method,
    )

    return evaluate_singing_with_fixed_latency(
        voice_history=voice_history,
        melody=melody,
        current_position=current_position,
        allowed_error_cents=allowed_error_cents,
        window_seconds=window_seconds,
        latency_ms=latency_ms,
        melody_lookup=lookup,
        scoring_method=scoring_method,
    )


def estimate_singing_latency(
    voice_history: list[tuple[float, float]],
    melody: list[tuple[float, float]],
    current_position: float,
    allowed_error_cents: float,
    window_seconds: int,
    auto_latency_max_ms: int = 500,
    auto_latency_step_ms: int = 20,
    melody_lookup: MelodyLookup | None = None,
    previous_latency_ms: int | None = None,
    scoring_method: str = "balanced",
) -> int:
    if not voice_history or not melody:
        return max(0, int(previous_latency_ms or 0))

    lookup = melody_lookup if melody_lookup is not None else MelodyLookup(melody)
    relevant_voice = _select_relevant_voice(voice_history, current_position, window_seconds)
    if not relevant_voice:
        return max(0, int(previous_latency_ms or 0))

    step = max(5, int(auto_latency_step_ms))
    maximum = max(0, int(auto_latency_max_ms))
    candidates = list(range(0, maximum + 1, step))

    previous = None if previous_latency_ms is None else max(0, int(previous_latency_ms))
    if previous is not None and previous not in candidates:
        candidates.append(previous)
        candidates.sort()

    best_latency = previous if previous is not None else 0
    best_quality: tuple[float, float, float, float] | None = None

    for latency_ms in candidates:
        # Latency ищем по реальным voice frames. Если здесь штрафовать молчание,
        # оценщик задержки будет хуже работать в паузах и на коротких фразах.
        score = _evaluate_voice_driven_with_fixed_latency(
            relevant_voice=relevant_voice,
            lookup=lookup,
            allowed_error_cents=allowed_error_cents,
            latency_ms=latency_ms,
            scoring_method=scoring_method,
        )
        if score.checked_frames < 4:
            continue

        coverage = score.checked_frames / max(1, len(relevant_voice))
        smoothness_penalty = 0.0
        if previous is not None:
            # Задержка не должна прыгать из-за одного случайно удачного фрагмента.
            smoothness_penalty = abs(latency_ms - previous) / max(1, maximum or 1)

        effective_score = score.score_percent * min(1.0, coverage / 0.50)
        quality = (
            effective_score,
            coverage * 100.0,
            -score.average_abs_error_cents,
            -smoothness_penalty,
        )

        if best_quality is None or quality > best_quality:
            best_quality = quality
            best_latency = latency_ms

    return max(0, int(best_latency))


def evaluate_singing_accuracy_summary(
    voice_history: list[tuple[float, float]],
    melody: list[tuple[float, float]],
    current_position: float,
    allowed_error_cents: float,
    latency_ms: int,
    melody_lookup: MelodyLookup | None = None,
    scoring_method: str = "balanced",
) -> SingingAccuracySummary:
    lookup = melody_lookup if melody_lookup is not None else MelodyLookup(melody)
    short_score = evaluate_singing_with_fixed_latency(
        voice_history=voice_history,
        melody=melody,
        current_position=current_position,
        allowed_error_cents=allowed_error_cents,
        window_seconds=3,
        latency_ms=latency_ms,
        melody_lookup=lookup,
        scoring_method=scoring_method,
    )
    medium_score = evaluate_singing_with_fixed_latency(
        voice_history=voice_history,
        melody=melody,
        current_position=current_position,
        allowed_error_cents=allowed_error_cents,
        window_seconds=10,
        latency_ms=latency_ms,
        melody_lookup=lookup,
        scoring_method=scoring_method,
    )
    total_score = evaluate_singing_with_fixed_latency(
        voice_history=voice_history,
        melody=melody,
        current_position=current_position,
        allowed_error_cents=allowed_error_cents,
        window_seconds=None,
        latency_ms=latency_ms,
        melody_lookup=lookup,
        scoring_method=scoring_method,
    )
    return SingingAccuracySummary(
        short_score=short_score,
        medium_score=medium_score,
        total_score=total_score,
        detected_latency_ms=latency_ms,
    )


def evaluate_singing_with_fixed_latency(
    voice_history: list[tuple[float, float]],
    melody: list[tuple[float, float]],
    current_position: float,
    allowed_error_cents: float,
    window_seconds: int | None,
    latency_ms: int,
    melody_lookup: MelodyLookup | None = None,
    scoring_method: str = "balanced",
) -> SingingScore:
    if not melody:
        return _empty_score("Пока мало данных: запусти песню, нажми Старт и пой в микрофон.", latency_ms)

    relevant_melody = _select_relevant_melody(melody, current_position, window_seconds)
    if not relevant_melody:
        return _empty_score("Пока мало ожидаемых нот для оценки.", latency_ms)

    relevant_voice = _select_relevant_voice(voice_history, current_position, window_seconds)
    return _evaluate_melody_driven_with_fixed_latency(
        relevant_voice=relevant_voice,
        relevant_melody=relevant_melody,
        allowed_error_cents=allowed_error_cents,
        latency_ms=latency_ms,
        scoring_method=scoring_method,
    )


def _select_relevant_voice(
    voice_history: list[tuple[float, float]],
    current_position: float,
    window_seconds: int | None,
) -> list[tuple[float, float]]:
    if window_seconds is None:
        start = 0.0
    else:
        start = max(0.0, current_position - window_seconds)

    return [
        (timestamp, frequency)
        for timestamp, frequency in voice_history
        if timestamp >= start and timestamp <= current_position + 0.20
    ]


def _select_relevant_melody(
    melody: list[tuple[float, float]],
    current_position: float,
    window_seconds: int | None,
) -> list[tuple[float, float]]:
    if window_seconds is None:
        start = 0.0
    else:
        start = max(0.0, current_position - window_seconds)

    # Если пользователь уже дошёл до конца песни, current_position может быть 0 после
    # завершения playback. Для total-score всё равно берём фактически пройденный участок.
    end = max(0.0, current_position)
    points = [(timestamp, frequency) for timestamp, frequency in melody if start <= timestamp <= end]

    # Realtime UI не должен перемалывать тысячи точек на каждом кадре.
    if window_seconds is None:
        return points[-900:]
    return points[-360:]


def _evaluate_melody_driven_with_fixed_latency(
    relevant_voice: list[tuple[float, float]],
    relevant_melody: list[tuple[float, float]],
    allowed_error_cents: float,
    latency_ms: int,
    scoring_method: str = "balanced",
) -> SingingScore:
    expected_count = len(relevant_melody)
    if expected_count == 0:
        return _empty_score("Пока мало ожидаемых нот для оценки.", latency_ms)

    if not relevant_voice:
        return _zero_score_for_expected_notes(expected_count, "В нужных местах пока не найден голос.", latency_ms)

    voice_times = [timestamp for timestamp, _ in relevant_voice]
    latency_seconds = latency_ms / 1000.0
    pairs: list[tuple[float, float]] = []
    misses = 0

    for song_timestamp, expected_frequency in relevant_melody:
        voice_timestamp = song_timestamp + latency_seconds
        nearest = _nearest_voice_frame(relevant_voice, voice_times, voice_timestamp, max_distance_seconds=0.32)
        if nearest is None:
            misses += 1
            continue
        _, voice_frequency = nearest
        pairs.append((voice_frequency, expected_frequency))

    if not pairs:
        return _zero_score_for_expected_notes(expected_count, "В нужных местах пока не найден голос.", latency_ms)

    raw_errors = [cents_between(voice, expected) for voice, expected in pairs]

    # Позволяем петь в другом регистре: оцениваем форму мелодии после вычитания
    # типичного сдвига, а не абсолютную октаву/тональность каждого кадра.
    median_shift = median(raw_errors)
    adjusted_errors = [error - median_shift for error in raw_errors]

    good = sum(1 for error in adjusted_errors if abs(error) <= allowed_error_cents)
    average_abs = sum(abs(error) for error in adjusted_errors) / len(adjusted_errors)

    high = sum(1 for error in adjusted_errors if error > allowed_error_cents)
    low = sum(1 for error in adjusted_errors if error < -allowed_error_cents)

    # Главное отличие от старой метрики: если в ожидаемой точке мелодии пользователь
    # молчал, этот кадр получает 0 баллов. Молчание больше не исчезает из знаменателя.
    score = _score_errors(adjusted_errors, allowed_error_cents, scoring_method, misses=misses)
    matched_count = len(adjusted_errors)
    high_percent = high / matched_count * 100.0 if matched_count else 0.0
    low_percent = low / matched_count * 100.0 if matched_count else 0.0

    advice = build_advice(score, average_abs, high_percent, low_percent, median_shift)

    return SingingScore(
        score_percent=score,
        checked_frames=expected_count,
        good_frames=good,
        median_shift_cents=median_shift,
        average_abs_error_cents=average_abs,
        too_high_percent=high_percent,
        too_low_percent=low_percent,
        detected_latency_ms=latency_ms,
        advice=advice,
    )


def _evaluate_voice_driven_with_fixed_latency(
    relevant_voice: list[tuple[float, float]],
    lookup: MelodyLookup,
    allowed_error_cents: float,
    latency_ms: int,
    scoring_method: str = "balanced",
) -> SingingScore:
    latency_seconds = latency_ms / 1000.0
    pairs: list[tuple[float, float]] = []

    for voice_timestamp, voice_frequency in relevant_voice:
        song_timestamp = voice_timestamp - latency_seconds
        expected = lookup.expected_frequency_at(song_timestamp)
        if expected is None:
            continue
        pairs.append((voice_frequency, expected))

    if not pairs:
        return _empty_score("Пока мало совпадающих участков: попробуй петь ближе к вокальной линии песни.", latency_ms)

    raw_errors = [cents_between(voice, expected) for voice, expected in pairs]
    median_shift = median(raw_errors)
    adjusted_errors = [error - median_shift for error in raw_errors]

    good = sum(1 for error in adjusted_errors if abs(error) <= allowed_error_cents)
    average_abs = sum(abs(error) for error in adjusted_errors) / len(adjusted_errors)
    high = sum(1 for error in adjusted_errors if error > allowed_error_cents)
    low = sum(1 for error in adjusted_errors if error < -allowed_error_cents)

    score = _score_errors(adjusted_errors, allowed_error_cents, scoring_method)
    high_percent = high / len(adjusted_errors) * 100.0
    low_percent = low / len(adjusted_errors) * 100.0

    return SingingScore(
        score_percent=score,
        checked_frames=len(adjusted_errors),
        good_frames=good,
        median_shift_cents=median_shift,
        average_abs_error_cents=average_abs,
        too_high_percent=high_percent,
        too_low_percent=low_percent,
        detected_latency_ms=latency_ms,
        advice=build_advice(score, average_abs, high_percent, low_percent, median_shift),
    )


def _nearest_voice_frame(
    voice_frames: list[tuple[float, float]],
    voice_times: list[float],
    timestamp: float,
    max_distance_seconds: float,
) -> tuple[float, float] | None:
    index = bisect_left(voice_times, timestamp)
    candidates = []
    if index < len(voice_frames):
        candidates.append(voice_frames[index])
    if index > 0:
        candidates.append(voice_frames[index - 1])
    if not candidates:
        return None
    nearest = min(candidates, key=lambda item: abs(item[0] - timestamp))
    if abs(nearest[0] - timestamp) > max_distance_seconds:
        return None
    return nearest


def _score_errors(
    errors: list[float],
    allowed_error_cents: float,
    scoring_method: str,
    misses: int = 0,
) -> float:
    total = len(errors) + max(0, misses)
    if total == 0:
        return 0.0

    allowed = max(1.0, allowed_error_cents)

    if scoring_method == "strict":
        frame_scores = [1.0 if abs(error) <= allowed else 0.0 for error in errors]
    elif scoring_method == "soft":
        # Мягкий режим полезен для новичков: небольшие и средние промахи не обнуляются резко.
        limit = allowed * 3.0
        frame_scores = [max(0.0, 1.0 - abs(error) / limit) for error in errors]
    else:
        # Balanced по умолчанию: внутри допуска кадр почти идеален, дальше оценка плавно падает.
        # Так процент ощущается честнее, чем binary hit/miss, но большие промахи всё равно караются.
        limit = allowed * 2.5
        frame_scores = []
        for error in errors:
            distance = abs(error)
            if distance <= allowed:
                frame_scores.append(1.0)
            elif distance >= limit:
                frame_scores.append(0.0)
            else:
                frame_scores.append(1.0 - (distance - allowed) / (limit - allowed))

    # misses intentionally add zeroes to denominator.
    return sum(frame_scores) / total * 100.0


def _zero_score_for_expected_notes(expected_count: int, advice: str, latency_ms: int) -> SingingScore:
    return SingingScore(
        score_percent=0.0,
        checked_frames=expected_count,
        good_frames=0,
        median_shift_cents=0.0,
        average_abs_error_cents=0.0,
        too_high_percent=0.0,
        too_low_percent=0.0,
        detected_latency_ms=latency_ms,
        advice=advice,
    )


def _empty_score(advice: str, latency_ms: int = 0) -> SingingScore:
    return SingingScore(
        score_percent=0.0,
        checked_frames=0,
        good_frames=0,
        median_shift_cents=0.0,
        average_abs_error_cents=0.0,
        too_high_percent=0.0,
        too_low_percent=0.0,
        detected_latency_ms=latency_ms,
        advice=advice,
    )


def expected_frequency_at(melody: list[tuple[float, float]], timestamp: float) -> float | None:
    return MelodyLookup(melody).expected_frequency_at(timestamp)


def build_advice(
    score: float,
    average_abs_error: float,
    high_percent: float,
    low_percent: float,
    median_shift: float,
) -> str:
    if score >= 85:
        return "Очень хорошо. Теперь работай над ровностью длинных нот."
    if high_percent > low_percent + 20:
        return "Ты часто поёшь выше нужной формы. Попробуй не задирать окончания фраз."
    if low_percent > high_percent + 20:
        return "Ты часто поёшь ниже нужной формы. Добавь опоры дыхания на концах фраз."
    if average_abs_error > 180:
        return "Ошибка большая. Сначала пой медленнее и слушай контур мелодии."
    if abs(median_shift) > 250:
        return "Ты поёшь в другом регистре, это нормально. Оценка учитывает транспозицию."
    return "Неплохо. Главная цель — уменьшить разброс и ровнее держать фразы."
