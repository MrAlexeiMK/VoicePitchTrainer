import datetime
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class LogRecord:
    timestamp: str
    level: str
    message: str

    def formatted(self) -> str:
        return f"{self.timestamp} [{self.level}] {self.message}"


class AppLogger:
    def __init__(self) -> None:
        self._records: list[LogRecord] = []
        self._listeners: list[Callable[[LogRecord], None]] = []

    def add_listener(self, listener: Callable[[LogRecord], None]) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[LogRecord], None]) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def records(self) -> list[LogRecord]:
        return list(self._records)

    def info(self, message: str) -> None:
        self._append("INFO", message)

    def warning(self, message: str) -> None:
        self._append("WARNING", message)

    def error(self, message: str) -> None:
        self._append("ERROR", message)

    def debug(self, message: str) -> None:
        self._append("DEBUG", message)

    def _append(self, level: str, message: str) -> None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = LogRecord(timestamp=timestamp, level=level, message=message)
        self._records.append(record)

        # Не даём окну логов бесконечно разрастаться.
        if len(self._records) > 5000:
            self._records = self._records[-5000:]

        for listener in list(self._listeners):
            listener(record)


app_logger = AppLogger()
