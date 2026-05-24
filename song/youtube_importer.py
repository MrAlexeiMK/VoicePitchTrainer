import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import imageio_ffmpeg


class YouTubeImportError(RuntimeError):
    pass


def download_youtube_audio_to_wav(
    query_or_url: str,
    progress_callback: Optional[Callable[[str], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
) -> tuple[Path, str]:
    """
    Скачивает аудио с YouTube в WAV.

    В обычном запуске из Python используем subprocess:
    - его можно безопасно убить по timeout;
    - зависания yt-dlp не подвешивают UI.

    В PyInstaller --onefile/--windowed нельзя запускать `sys.executable -m yt_dlp`,
    потому что sys.executable указывает на VoicePitchTrainer.exe. Такой запуск
    открывает второе окно приложения вместо yt-dlp. Поэтому в frozen-режиме
    используем yt-dlp Python API внутри worker thread.
    """
    if getattr(sys, "frozen", False):
        return _download_with_ytdlp_api(query_or_url, progress_callback, cancel_callback)
    return _download_with_subprocess(query_or_url, progress_callback, cancel_callback)


def _download_with_subprocess(
    query_or_url: str,
    progress_callback: Optional[Callable[[str], None]],
    cancel_callback: Optional[Callable[[], bool]],
) -> tuple[Path, str]:
    output_dir = Path(tempfile.mkdtemp(prefix="voice_trainer_youtube_"))
    output_template = str(output_dir / "%(title).180B.%(ext)s")
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    source = _source_from_query(query_or_url)

    if _is_url(query_or_url):
        _progress(progress_callback, "Открываю YouTube ссылку...")
    else:
        _progress(progress_callback, "Ищу подходящий результат на YouTube...")

    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--newline",
        "--no-playlist",
        "--no-warnings",
        "--force-ipv4",
        "--socket-timeout",
        "20",
        "--retries",
        "2",
        "--fragment-retries",
        "2",
        "--extractor-retries",
        "2",
        "--extract-audio",
        "--audio-format",
        "wav",
        "--audio-quality",
        "192K",
        "--ffmpeg-location",
        ffmpeg_path,
        "-o",
        output_template,
        source,
    ]

    _progress(progress_callback, "Запускаю yt-dlp...")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0,
    )

    assert process.stdout is not None

    output_lines: list[str] = []
    last_output_at = time.monotonic()
    started_at = time.monotonic()

    no_output_timeout_seconds = 90
    total_timeout_seconds = 900

    while True:
        if _cancel_requested(cancel_callback):
            _kill_process(process)
            raise YouTubeImportError("YouTube импорт отменён")

        line = process.stdout.readline()

        if line:
            last_output_at = time.monotonic()
            cleaned = line.strip()
            if cleaned:
                output_lines.append(cleaned)
                _handle_ytdlp_line(cleaned, progress_callback)
        elif process.poll() is not None:
            break
        else:
            now = time.monotonic()

            if now - last_output_at > no_output_timeout_seconds:
                _kill_process(process)
                raise YouTubeImportError(
                    "yt-dlp слишком долго не отвечает. "
                    "Проверь интернет/доступность YouTube или обнови yt-dlp: "
                    "python -m pip install -U yt-dlp"
                )

            if now - started_at > total_timeout_seconds:
                _kill_process(process)
                raise YouTubeImportError("YouTube импорт превысил лимит времени 15 минут")

            time.sleep(0.1)

    return_code = process.wait()

    if return_code != 0:
        tail = "\n".join(output_lines[-20:])
        raise YouTubeImportError(
            "yt-dlp завершился с ошибкой.\n\n"
            f"Последние строки:\n{tail}"
        )

    return _finish_download(output_dir, None, progress_callback)


def _download_with_ytdlp_api(
    query_or_url: str,
    progress_callback: Optional[Callable[[str], None]],
    cancel_callback: Optional[Callable[[], bool]],
) -> tuple[Path, str]:
    output_dir = Path(tempfile.mkdtemp(prefix="voice_trainer_youtube_"))
    output_template = str(output_dir / "%(title).180B.%(ext)s")
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    source = _source_from_query(query_or_url)

    if _is_url(query_or_url):
        _progress(progress_callback, "Открываю YouTube ссылку...")
    else:
        _progress(progress_callback, "Ищу подходящий результат на YouTube...")

    _progress(progress_callback, "Загружаю встроенный yt-dlp...")
    try:
        import yt_dlp
    except Exception as exc:
        raise YouTubeImportError(
            "Не удалось импортировать yt-dlp внутри exe. "
            "Собери приложение с зависимостью yt-dlp или установи её в окружение."
        ) from exc

    last_status_at = 0.0

    def progress_hook(info: dict) -> None:
        nonlocal last_status_at

        if _cancel_requested(cancel_callback):
            raise YouTubeImportError("YouTube импорт отменён")

        status = str(info.get("status", ""))
        now = time.monotonic()

        if status == "downloading":
            if now - last_status_at < 0.35:
                return
            last_status_at = now

            percent = info.get("_percent_str")
            speed = info.get("_speed_str")
            eta = info.get("_eta_str")

            parts = []
            if percent:
                parts.append(str(percent).strip())
            if speed:
                parts.append(str(speed).strip())
            if eta:
                parts.append(f"ETA {str(eta).strip()}")

            if parts:
                _progress(progress_callback, "Скачиваю аудио: " + " / ".join(parts))
            else:
                _progress(progress_callback, "Скачиваю аудио...")

        elif status == "finished":
            _progress(progress_callback, "Конвертирую аудио в WAV...")

    ydl_options = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "force_ipv4": True,
        "socket_timeout": 20,
        "retries": 2,
        "fragment_retries": 2,
        "extractor_retries": 2,
        "ffmpeg_location": ffmpeg_path,
        "progress_hooks": [progress_hook],
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }

    _progress(progress_callback, "Запускаю yt-dlp...")

    try:
        with yt_dlp.YoutubeDL(ydl_options) as downloader:
            info = downloader.extract_info(source, download=True)
    except YouTubeImportError:
        raise
    except Exception as exc:
        raise YouTubeImportError(f"yt-dlp завершился с ошибкой:\n\n{exc}") from exc

    return _finish_download(output_dir, info, progress_callback)


def _finish_download(
    output_dir: Path,
    info: object,
    progress_callback: Optional[Callable[[str], None]],
) -> tuple[Path, str]:
    wav_files = sorted(output_dir.glob("*.wav"))

    if not wav_files:
        # Иногда yt-dlp создаёт имя с более длинным title или вложенной директорией.
        wav_files = sorted(output_dir.rglob("*.wav"))

    if not wav_files:
        existing_files = sorted(path.name for path in output_dir.rglob("*") if path.is_file())
        tail = "\n".join(existing_files[-20:])
        raise YouTubeImportError(
            "yt-dlp завершился, но WAV файл не найден.\n\n"
            f"Файлы в папке загрузки:\n{tail}"
        )

    wav_path = wav_files[0]
    title = _guess_title_from_info_or_wav(info, wav_path)

    _progress(progress_callback, f"YouTube импорт завершён: {title}")

    return wav_path, title


def _source_from_query(query_or_url: str) -> str:
    query_or_url = query_or_url.strip()

    if _is_url(query_or_url):
        return query_or_url

    return f"ytsearch1:{query_or_url}"


def _is_url(value: str) -> bool:
    value = value.strip().lower()
    return value.startswith("http://") or value.startswith("https://")


def _progress(callback: Optional[Callable[[str], None]], message: str) -> None:
    if callback is not None:
        callback(message)


def _cancel_requested(cancel_callback: Optional[Callable[[], bool]]) -> bool:
    if cancel_callback is None:
        return False
    try:
        return bool(cancel_callback())
    except Exception:
        return False


def _handle_ytdlp_line(line: str, progress_callback: Optional[Callable[[str], None]]) -> None:
    lower = line.lower()

    if "[youtube]" in lower and "downloading webpage" in lower:
        _progress(progress_callback, "YouTube: открываю страницу видео.")
    elif "[youtube]" in lower and "downloading" in lower:
        _progress(progress_callback, f"YouTube: {line}")
    elif "[download]" in lower and "%" in line:
        _progress(progress_callback, f"Скачиваю аудио: {line}")
    elif "[download]" in lower and "destination" in lower:
        _progress(progress_callback, "Начинаю скачивание аудио.")
    elif "[extractaudio]" in lower or ("destination:" in lower and ".wav" in lower):
        _progress(progress_callback, "Конвертирую аудио в WAV.")
    elif "[ffmpeg]" in lower:
        _progress(progress_callback, "Обрабатываю аудио через FFmpeg.")
    elif "error:" in lower:
        _progress(progress_callback, f"Ошибка yt-dlp: {line}")
    else:
        # Не спамим UI каждой технической строкой, но оставляем ощущение движения.
        if any(marker in lower for marker in ["extracting", "download", "merging", "deleting"]):
            _progress(progress_callback, f"yt-dlp: {line}")


def _kill_process(process: subprocess.Popen) -> None:
    try:
        process.terminate()
    except Exception:
        pass

    try:
        process.wait(timeout=3)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def _guess_title_from_info_or_wav(info: object, path: Path) -> str:
    if isinstance(info, dict):
        if "entries" in info and info["entries"]:
            first_entry = info["entries"][0]
            if isinstance(first_entry, dict):
                title = first_entry.get("title")
                if title:
                    return str(title)

        title = info.get("title")
        if title:
            return str(title)

    return _guess_title_from_wav(path)


def _guess_title_from_wav(path: Path) -> str:
    return path.stem
