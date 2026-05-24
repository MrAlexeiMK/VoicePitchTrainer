import json
import os
import shutil
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CachedSongInfo:
    key: str
    title: str
    sample_rate: int
    demucs_stems_available: bool
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class CachedSongData:
    info: CachedSongInfo
    playback_audio: np.ndarray
    vocals_audio: Optional[np.ndarray]
    instrumental_audio: Optional[np.ndarray]
    melody: list[tuple[float, float]]
    vocals_path: Optional[Path]
    instrumental_path: Optional[Path]


@dataclass(frozen=True)
class TrainingHistoryEntry:
    song_key: str
    title: str
    timestamp: float
    score_percent: float
    rank: str


def storage_root() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "AlekseyTools" / "VoicePitchTrainer"
    return Path.home() / ".voice_pitch_trainer"


def songs_dir() -> Path:
    return storage_root() / "songs"


def history_path() -> Path:
    return storage_root() / "training_history.json"


def save_cached_song(
    title: str,
    playback_audio: np.ndarray,
    sample_rate: int,
    melody: list[tuple[float, float]],
    vocals_audio: Optional[np.ndarray] = None,
    instrumental_audio: Optional[np.ndarray] = None,
    demucs_stems_available: bool = False,
) -> str:
    key = uuid.uuid4().hex
    song_dir = songs_dir() / key
    song_dir.mkdir(parents=True, exist_ok=True)

    _write_wav_float(song_dir / "playback.wav", playback_audio, sample_rate)

    vocals_path = None
    instrumental_path = None
    if vocals_audio is not None:
        vocals_path = song_dir / "vocals.wav"
        _write_wav_float(vocals_path, vocals_audio, sample_rate)
    if instrumental_audio is not None:
        instrumental_path = song_dir / "instrumental.wav"
        _write_wav_float(instrumental_path, instrumental_audio, sample_rate)

    now = time.time()
    metadata = {
        "key": key,
        "title": title,
        "sample_rate": int(sample_rate),
        "demucs_stems_available": bool(demucs_stems_available and vocals_path and instrumental_path),
        "created_at": now,
        "updated_at": now,
        "melody": [[float(timestamp), float(frequency)] for timestamp, frequency in melody],
    }
    _write_json(song_dir / "song.json", metadata)
    return key


def list_cached_songs(limit: int = 20) -> list[CachedSongInfo]:
    result: list[CachedSongInfo] = []
    for metadata_path in songs_dir().glob("*/song.json"):
        try:
            data = _read_json(metadata_path)
            result.append(_song_info_from_json(data))
        except Exception:
            continue
    return sorted(result, key=lambda item: item.updated_at, reverse=True)[:limit]


def load_cached_song(key: str) -> CachedSongData:
    song_dir = songs_dir() / key
    metadata = _read_json(song_dir / "song.json")
    info = _song_info_from_json(metadata)
    playback_audio, sample_rate = _read_wav_float(song_dir / "playback.wav")
    if sample_rate != info.sample_rate:
        info = CachedSongInfo(
            key=info.key,
            title=info.title,
            sample_rate=sample_rate,
            demucs_stems_available=info.demucs_stems_available,
            created_at=info.created_at,
            updated_at=info.updated_at,
        )

    vocals_path = song_dir / "vocals.wav"
    instrumental_path = song_dir / "instrumental.wav"
    vocals_audio = _read_wav_float(vocals_path)[0] if vocals_path.exists() else None
    instrumental_audio = _read_wav_float(instrumental_path)[0] if instrumental_path.exists() else None
    melody = [(float(timestamp), float(frequency)) for timestamp, frequency in metadata.get("melody", [])]

    _touch_cached_song(key)
    return CachedSongData(
        info=info,
        playback_audio=playback_audio,
        vocals_audio=vocals_audio,
        instrumental_audio=instrumental_audio,
        melody=melody,
        vocals_path=vocals_path if vocals_path.exists() else None,
        instrumental_path=instrumental_path if instrumental_path.exists() else None,
    )


def save_training_history_entry(song_key: str, title: str, score_percent: float, rank: str) -> None:
    storage_root().mkdir(parents=True, exist_ok=True)
    entries = [_entry_to_json(entry) for entry in load_training_history()]
    entries.append(
        {
            "song_key": song_key,
            "title": title,
            "timestamp": time.time(),
            "score_percent": float(score_percent),
            "rank": rank,
        }
    )
    _write_json(history_path(), entries[-500:])


def delete_training_history_entry(song_key: str, timestamp: float) -> None:
    entries = [
        _entry_to_json(entry)
        for entry in load_training_history()
        if not (entry.song_key == song_key and abs(entry.timestamp - timestamp) < 0.001)
    ]
    _write_json(history_path(), entries)


def clear_training_history() -> None:
    path = history_path()
    if path.exists():
        path.unlink()


def load_training_history() -> list[TrainingHistoryEntry]:
    path = history_path()
    if not path.exists():
        return []
    try:
        data = _read_json(path)
    except Exception:
        return []
    result: list[TrainingHistoryEntry] = []
    for item in data if isinstance(data, list) else []:
        try:
            result.append(
                TrainingHistoryEntry(
                    song_key=str(item.get("song_key", "")),
                    title=str(item.get("title", "Песня")),
                    timestamp=float(item.get("timestamp", 0.0)),
                    score_percent=float(item.get("score_percent", 0.0)),
                    rank=str(item.get("rank", "E")),
                )
            )
        except Exception:
            continue
    return sorted(result, key=lambda item: item.timestamp, reverse=True)


def history_for_song(song_key: str) -> list[TrainingHistoryEntry]:
    return sorted(
        [entry for entry in load_training_history() if entry.song_key == song_key],
        key=lambda item: item.timestamp,
    )


def _touch_cached_song(key: str) -> None:
    path = songs_dir() / key / "song.json"
    if not path.exists():
        return
    try:
        data = _read_json(path)
        data["updated_at"] = time.time()
        _write_json(path, data)
    except Exception:
        return


def _song_info_from_json(data: dict) -> CachedSongInfo:
    return CachedSongInfo(
        key=str(data.get("key", "")),
        title=str(data.get("title", "Песня")),
        sample_rate=int(data.get("sample_rate", 44100)),
        demucs_stems_available=bool(data.get("demucs_stems_available", False)),
        created_at=float(data.get("created_at", 0.0)),
        updated_at=float(data.get("updated_at", 0.0)),
    )


def _entry_to_json(entry: TrainingHistoryEntry) -> dict:
    return {
        "song_key": entry.song_key,
        "title": entry.title,
        "timestamp": entry.timestamp,
        "score_percent": entry.score_percent,
        "rank": entry.rank,
    }


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.move(str(temp_path), str(path))


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_wav_float(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = audio.astype(np.float32)
    if values.ndim == 1:
        channels = 1
        interleaved = values
    else:
        channels = values.shape[1]
        interleaved = values.reshape(-1)
    interleaved = np.clip(interleaved, -1.0, 1.0)
    pcm = (interleaved * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm.tobytes())


def _read_wav_float(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels)
    return audio.astype(np.float32), sample_rate
