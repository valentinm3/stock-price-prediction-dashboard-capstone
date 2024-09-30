# logger.py

import logging
import os

class Logger:
    def __init__(self, name: str, log_file: str):
        """
        Initializes the Logger with a name and log file.

        Args:
            name (str): Name of the logger.
            log_file (str): Path to the log file.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Prevent adding multiple handlers if they already exist
        if not self.logger.handlers:
            # Create logs directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # File handler
            fh = logging.FileHandler(log_file, mode='w')  # Ensure overwrite
            fh.setLevel(logging.INFO)

            # Console handler (optional, for errors)
            ch = logging.StreamHandler()
            ch.setLevel(logging.ERROR)

            # Formatter
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)

            # Add handlers to the logger
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

        # Disable propagation to the root logger to prevent writing to main.log
        self.logger.propagate = False

    def __getattr__(self, attr):
        """
        Delegate attribute access to the internal logger.

        Args:
            attr (str): Attribute name.

        Returns:
            Any: Attribute value from the internal logger.
        """
        return getattr(self.logger, attr)

    @staticmethod
    def get_logger_for_module(module_name: str, log_file: str) -> 'Logger':
        """
        Creates a logger for a specific module and returns it.

        Args:
            module_name (str): Name of the module for logging.
            log_file (str): Path to the log file.

        Returns:
            Logger: Configured Logger instance for the module.
        """
        # Ensure the logs directory exists
        log_dir = os.path.dirname(log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Create a Logger instance for the module
        logger_instance = Logger(module_name, log_file)
        return logger_instance
