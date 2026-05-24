import math

from core.constants import NOTE_NAMES


def note_to_frequency(note_name: str, octave: int) -> float:
    note_index = NOTE_NAMES.index(note_name)
    midi_note = 12 * (octave + 1) + note_index
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def frequency_to_midi(frequency_hz: float) -> float:
    return 69.0 + 12.0 * math.log2(frequency_hz / 440.0)


def frequency_to_note(frequency_hz: float) -> str:
    midi_note = int(round(frequency_to_midi(frequency_hz)))
    note_index = midi_note % 12
    octave = midi_note // 12 - 1
    return f"{NOTE_NAMES[note_index]}{octave}"


def classify_voice_by_frequency(frequency_hz: float) -> str:
    if frequency_hz < 87:
        return "очень низкий мужской голос"
    if frequency_hz < 110:
        return "низкий мужской голос"
    if frequency_hz < 147:
        return "средний мужской голос"
    if frequency_hz < 175:
        return "высокий мужской голос"
    if frequency_hz < 220:
        return "андрогинный диапазон"
    if frequency_hz < 277:
        return "низкий женский голос"
    if frequency_hz < 370:
        return "средний женский голос"
    if frequency_hz < 494:
        return "высокий женский голос"
    return "очень высокий женский голос"


def cents_between(frequency_hz: float, reference_hz: float) -> float:
    if frequency_hz <= 0 or reference_hz <= 0:
        return 0.0
    return 1200.0 * math.log2(frequency_hz / reference_hz)
