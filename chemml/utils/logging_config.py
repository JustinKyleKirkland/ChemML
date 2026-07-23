"""Centralised logging configuration for ChemML.

All application loggers are children of the ``ChemML`` logger. We attach the
file and console handlers to that single logger and disable propagation so that
each record is emitted exactly once (previously ``basicConfig`` on the root
logger plus handlers on ``ChemML`` produced duplicate lines).
"""

import logging
import os

LOG_DIR = "logs"
LOG_FILE = "app.log"
_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
	"""Configure and return the root ``ChemML`` logger.

	Safe to call multiple times; handlers are only added once.

	Args:
		level: Logging level for the ChemML logger (default ``logging.INFO``).

	Returns:
		The configured ``ChemML`` logger.
	"""
	os.makedirs(LOG_DIR, exist_ok=True)
	log_file_path = os.path.join(LOG_DIR, LOG_FILE)

	logger = logging.getLogger("ChemML")
	logger.setLevel(level)
	# Don't hand records to the root logger; we manage our own handlers.
	logger.propagate = False

	if not logger.handlers:
		formatter = logging.Formatter(_FORMAT)

		file_handler = logging.FileHandler(log_file_path)
		file_handler.setFormatter(formatter)

		console_handler = logging.StreamHandler()
		console_handler.setFormatter(formatter)

		logger.addHandler(file_handler)
		logger.addHandler(console_handler)

	logger.debug("Logging is set up and ready to go.")
	return logger
