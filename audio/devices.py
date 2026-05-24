import sounddevice as sd

from core.constants import BAD_DEVICE_WORDS, DEFAULT_SAMPLE_RATE, GOOD_DEVICE_WORDS
from core.models import AudioDevice


def normalized_device_name(name: str) -> str:
    result = name.lower().strip()
    for suffix in ["(wasapi)", "(mme)", "(directsound)", "(windows directsound)", "(windows wasapi)"]:
        result = result.replace(suffix, "")
    return " ".join(result.split())


def is_probably_real_microphone(name: str) -> bool:
    normalized = normalized_device_name(name)
    if any(word in normalized for word in BAD_DEVICE_WORDS):
        return False
    return any(word in normalized for word in GOOD_DEVICE_WORDS)


def can_open_input_device(index: int, channels: int, sample_rate: int) -> bool:
    try:
        sd.check_input_settings(device=index, channels=min(1, channels), samplerate=sample_rate)
        return True
    except Exception:
        return False


def list_input_devices(show_all: bool) -> list[AudioDevice]:
    unique_devices: dict[str, AudioDevice] = {}

    for index, device in enumerate(sd.query_devices()):
        channels = int(device.get("max_input_channels", 0))
        if channels <= 0:
            continue

        name = str(device.get("name", f"Устройство {index}"))
        sample_rate = int(float(device.get("default_samplerate", DEFAULT_SAMPLE_RATE)))

        if not show_all and not is_probably_real_microphone(name):
            continue

        if not can_open_input_device(index, channels, sample_rate):
            continue

        key = normalized_device_name(name)
        candidate = AudioDevice(index=index, name=name, channels=channels, sample_rate=sample_rate)
        current = unique_devices.get(key)

        if current is None or candidate.channels > current.channels:
            unique_devices[key] = candidate

    return sorted(unique_devices.values(), key=lambda item: item.name.lower())
