import sys

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication, QLabel, QProgressBar, QVBoxLayout, QWidget

from ui.main_window import MainWindow


class StartupSplash(QWidget):
    def __init__(self) -> None:
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle("Загрузка")
        self.setFixedSize(360, 120)

        title = QLabel("Тренажёр голоса и пения")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold;")

        message = QLabel("Загружается приложение...")
        message.setAlignment(Qt.AlignmentFlag.AlignCenter)

        progress = QProgressBar()
        progress.setRange(0, 0)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(message)
        layout.addWidget(progress)

    def center_on_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geometry = screen.availableGeometry()
        self.move(
            geometry.center().x() - self.width() // 2,
            geometry.center().y() - self.height() // 2,
        )


def main() -> None:
    app = QApplication(sys.argv)

    splash = StartupSplash()
    splash.center_on_screen()
    splash.show()
    app.processEvents()

    def create_main_window() -> None:
        window = MainWindow()
        window.show()
        splash.close()
        app.main_window = window

    QTimer.singleShot(50, create_main_window)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
