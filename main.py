import sys


def main():
    if len(sys.argv) > 1:
        from app.cli import run_cli

        return run_cli(sys.argv[1:])

    from PySide6.QtWidgets import QApplication

    from app.ui.main_window import DeepfakeDetectorWindow

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = DeepfakeDetectorWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
