import logging
import os


def setup_logging():
	logging.basicConfig(
		level=logging.DEBUG,
		format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
	)
	# Create a logs directory if it doesn't exist
	os.makedirs("logs", exist_ok=True)

	# Define the log file path
	log_file_path = os.path.join("logs", "app.log")

	# Create a custom logger
	logger = logging.getLogger("CSVInteractiveApp")
	# logger.setLevel(log_level)

	# Check if handlers already exist to avoid duplicates
	if not logger.handlers:
		# Define a formatter
		formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

		# Create a file handler
		file_handler = logging.FileHandler(log_file_path)
		# file_handler.setLevel(log_level)
		file_handler.setFormatter(formatter)

		# Create a console handler
		console_handler = logging.StreamHandler()
		# console_handler.setLevel(log_level)
		console_handler.setFormatter(formatter)

		# Add the handlers to the logger
		logger.addHandler(file_handler)
		logger.addHandler(console_handler)

	# Log the application start
	logger.info("Logging is set up and ready to go.")


# Optional: Function to log an error message
def log_error(message):
	logger = logging.getLogger()
	logger.error(message)


# Optional: Function to log a warning message
def log_warning(message):
	logger = logging.getLogger()
	logger.warning(message)


# Optional: Function to log an info message
def log_info(message):
	logger = logging.getLogger()
	logger.info(message)
