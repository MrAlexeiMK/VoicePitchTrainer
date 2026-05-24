import subprocess
import tempfile
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if sample_width == 1:
        audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Неподдерживаемая глубина WAV: {sample_width} bytes")

    if channels > 1:
        audio = audio.reshape(-1, channels)

    return audio.astype(np.float32), sample_rate


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return audio.mean(axis=1).astype(np.float32)


def extract_center_vocal_like_signal(audio: np.ndarray) -> np.ndarray:
    """
    Лёгкая локальная попытка выделить центральный вокал/ведущую мелодию.

    Это не Demucs и не настоящая stem separation, но для многих stereo-треков
    помогает уменьшить крайние инструменты и сделать pitch-анализ стабильнее.
    Если трек mono — возвращаем его как есть.
    """
    if audio.ndim == 1 or audio.shape[1] < 2:
        return to_mono(audio)

    left = audio[:, 0].astype(np.float32)
    right = audio[:, 1].astype(np.float32)

    center = (left + right) * 0.5
    side = (left - right) * 0.5

    # Ослабляем широкие stereo-компоненты, которые часто являются инструментами/ревербом.
    vocal_like = center - 0.35 * side

    peak = float(np.max(np.abs(vocal_like))) if len(vocal_like) else 0.0
    if peak > 1.0:
        vocal_like = vocal_like / peak

    return vocal_like.astype(np.float32)


def convert_to_wav_with_embedded_ffmpeg(source_path: str | Path) -> Path:
    source_path = Path(source_path)
    output_path = Path(tempfile.mkstemp(prefix="voice_trainer_", suffix=".wav")[1])
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "2",
        "-ar",
        "44100",
        "-f",
        "wav",
        str(output_path),
    ]

    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr[-2000:])

    return output_path


def load_audio_file(path: str | Path) -> tuple[np.ndarray, np.ndarray, int, Path]:
    path = Path(path)

    if path.suffix.lower() == ".wav":
        raw_audio, sample_rate = read_wav(path)
        playback_audio = to_mono(raw_audio)
        melody_audio = extract_center_vocal_like_signal(raw_audio)
        return playback_audio, melody_audio, sample_rate, path

    wav_path = convert_to_wav_with_embedded_ffmpeg(path)
    raw_audio, sample_rate = read_wav(wav_path)
    playback_audio = to_mono(raw_audio)
    melody_audio = extract_center_vocal_like_signal(raw_audio)
    return playback_audio, melody_audio, sample_rate, wav_path
