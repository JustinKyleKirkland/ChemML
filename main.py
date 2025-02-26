import sys

from PyQt5.QtWidgets import QApplication

from gui.gui import CSVInteractiveApp  # Import your main GUI class
from utils.logging_config import setup_logging  # Import your logging setup function


def main():
    # Set up logging
    setup_logging()

    # Create the application instance
    app = QApplication(sys.argv)

    # Create and show the main GUI window
    window = CSVInteractiveApp()
    window.show()

    # Run the application event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
