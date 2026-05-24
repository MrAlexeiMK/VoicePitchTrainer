import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model


@dataclass(frozen=True)
class DemucsSeparatedStems:
    vocals_path: Path
    instrumental_path: Path


def kill_active_demucs_processes() -> None:
    """
    Оставлено для совместимости со старым кодом.

    Demucs работает через Python API внутри текущего процесса, поэтому
    отдельные дочерние Python-процессы не создаются.
    """
    return


def separate_vocals_with_demucs(
    source_path: Path,
    model_name: str = "htdemucs",
    progress_callback: Optional[Callable[[str], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
) -> DemucsSeparatedStems:
    """
    Разделяет песню через Demucs без запуска `python -m demucs`.

    Demucs/Torch импортируются на уровне модуля намеренно: так PyInstaller
    видит зависимости при обычной сборке `pyinstaller --onefile --windowed main.py`.

    В PyInstaller onefile Demucs иногда попадает в сборку без package data
    `demucs/remote/files.txt`. Этот файл нужен get_model() для списка моделей.
    Перед загрузкой модели восстанавливаем минимальный files.txt для популярных
    моделей Demucs 4.
    """

    def progress(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    def log(message: str) -> None:
        if log_callback is not None:
            log_callback(message)

    def raise_if_cancelled() -> None:
        if cancel_callback is not None and cancel_callback():
            raise RuntimeError("Импорт песни отменён")

    progress("Проверяю данные Demucs...")
    _ensure_demucs_remote_files_txt()

    progress(f"Загружаю Demucs модель '{model_name}'...")
    log(f"Demucs API mode enabled, model={model_name}")

    raise_if_cancelled()

    model = get_model(name=model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model_sample_rate = int(getattr(model, "samplerate", 44100))
    model_channels = int(getattr(model, "audio_channels", 2))
    sources = list(getattr(model, "sources", []))

    progress(
        f"Demucs модель загружена: device={device}, "
        f"sample_rate={model_sample_rate}, channels={model_channels}"
    )

    raise_if_cancelled()

    progress("Читаю WAV без FFmpeg/torchaudio...")
    audio, sample_rate = _read_wav_float(source_path)

    if sample_rate != model_sample_rate:
        progress(f"Ресемплирую аудио: {sample_rate} -> {model_sample_rate} Гц")
        audio = _resample_linear(audio, sample_rate, model_sample_rate)
        sample_rate = model_sample_rate

    audio = _ensure_channels(audio, model_channels)

    progress("Готовлю аудио для Demucs...")
    wav = torch.from_numpy(audio.T).float()

    ref = wav.mean(dim=0)
    ref_mean = ref.mean()
    ref_std = ref.std()
    if float(ref_std) <= 1e-8:
        ref_std = torch.tensor(1.0)

    wav = (wav - ref_mean) / ref_std
    wav = wav.to(device)

    progress("Demucs: отделяю вокал и инструментал. Это может занять время...")
    raise_if_cancelled()

    with torch.no_grad():
        separated = apply_model(
            model,
            wav[None],
            device=device,
            split=True,
            overlap=0.25,
            progress=False,
        )[0]

    raise_if_cancelled()

    separated = separated.cpu() * ref_std + ref_mean

    if "vocals" not in sources:
        raise RuntimeError(f"В модели Demucs не найден stem 'vocals'. Доступные stems: {sources}")

    vocal_index = sources.index("vocals")
    vocals = separated[vocal_index]

    instrumental_parts = [
        separated[index]
        for index, source_name in enumerate(sources)
        if source_name != "vocals"
    ]

    if not instrumental_parts:
        raise RuntimeError("Demucs не вернул инструментальные stems")

    instrumental = sum(instrumental_parts)

    output_dir = Path(tempfile.mkdtemp(prefix="voice_trainer_demucs_api_"))
    vocals_path = output_dir / "vocals.wav"
    instrumental_path = output_dir / "no_vocals.wav"

    progress("Сохраняю vocals.wav...")
    _write_wav_float(vocals_path, vocals.numpy().T, sample_rate)

    progress("Сохраняю no_vocals.wav...")
    _write_wav_float(instrumental_path, instrumental.numpy().T, sample_rate)

    progress("Demucs готов: вокал и инструментал разделены")
    return DemucsSeparatedStems(vocals_path=vocals_path, instrumental_path=instrumental_path)


def _ensure_demucs_remote_files_txt() -> None:
    """
    Восстанавливает `demucs/remote/files.txt`, если PyInstaller не положил его
    в onefile-сборку.

    Важно: в frozen exe `demucs.remote.__file__` иногда равен None. Поэтому тут
    нельзя делать `Path(demucs.remote.__file__)` без проверки.
    """
    try:
        import demucs
        import demucs.remote
    except Exception:
        return

    remote_dir = _detect_demucs_remote_dir(demucs, demucs.remote)
    if remote_dir is None:
        return

    files_txt = remote_dir / "files.txt"
    if files_txt.exists():
        return

    try:
        remote_dir.mkdir(parents=True, exist_ok=True)
        files_txt.write_text(_demucs_remote_files_txt_content(), encoding="utf-8")
    except Exception:
        # Не падаем здесь своим кодом. Если Demucs всё ещё не сможет загрузить
        # модель, пользователь получит уже реальную ошибку get_model().
        return


def _detect_demucs_remote_dir(demucs_module, remote_module) -> Optional[Path]:
    remote_file = getattr(remote_module, "__file__", None)
    if remote_file:
        return Path(remote_file).resolve().parent

    demucs_file = getattr(demucs_module, "__file__", None)
    if demucs_file:
        return Path(demucs_file).resolve().parent / "remote"

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass) / "demucs" / "remote"

    return None


def _demucs_remote_files_txt_content() -> str:
    # Минимальный список remote weights для моделей, доступных в настройках:
    # htdemucs, htdemucs_ft, mdx_extra, mdx_extra_q.
    return "\n".join(
        [
            "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th",
            "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/f7e0c4bc-ba3fe64a.th",
            "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/d12395a8-e57c48e6.th",
            "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/92cfc3b6-ef3bcb9c.th",
            "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/04573f0d-f3cf25b2.th",
            "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/75fc33f5-1941ce65.th",
            "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th",
            "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/31966d8d-3b6fcf95.th",
            "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5d2d6c55-db83574e.th",
            "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/7ecf8ec1-70f50cc9.th",
            "https://dl.fbaipublicfiles.com/demucs/mdx_final/83fc094f-4a16d450.th",
            "https://dl.fbaipublicfiles.com/demucs/mdx_final/464b36d7-e5a9386e.th",
            "https://dl.fbaipublicfiles.com/demucs/mdx_final/14fc6a69-a89dd0ee.th",
            "https://dl.fbaipublicfiles.com/demucs/mdx_final/7fd6ef75-a905dd85.th",
        ]
    ) + "\n"


def _read_wav_float(path: Path) -> tuple[np.ndarray, int]:
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
    else:
        audio = audio.reshape(-1, 1)

    return audio.astype(np.float32), sample_rate


def _write_wav_float(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim == 1:
        channels = 1
        samples = audio
    else:
        channels = audio.shape[1]
        samples = audio.reshape(-1)

    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def _ensure_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)

    current_channels = audio.shape[1]

    if current_channels == channels:
        return audio.astype(np.float32)

    if channels == 1:
        return audio.mean(axis=1, keepdims=True).astype(np.float32)

    if current_channels == 1:
        return np.repeat(audio, channels, axis=1).astype(np.float32)

    return audio[:, :channels].astype(np.float32)


def _resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or len(audio) == 0:
        return audio.astype(np.float32)

    duration = len(audio) / source_rate
    target_length = max(1, int(duration * target_rate))

    source_positions = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    target_positions = np.linspace(0.0, duration, num=target_length, endpoint=False)

    if audio.ndim == 1:
        return np.interp(target_positions, source_positions, audio).astype(np.float32)

    channels = [
        np.interp(target_positions, source_positions, audio[:, channel])
        for channel in range(audio.shape[1])
    ]

    return np.stack(channels, axis=1).astype(np.float32)
