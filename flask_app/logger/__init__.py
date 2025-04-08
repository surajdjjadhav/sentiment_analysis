import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
from from_root import from_root


# Constants
log_dir = "log"
log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
max_log_size = 5 * 1024 * 1024  # 5 MB
backup_count = 3

# Construct log dir file
log_dir_path = os.path.join(from_root(), log_dir)  # Call from_root() if it's a function
os.makedirs(log_dir_path, exist_ok=True)
log_file_path = os.path.join(log_dir_path, log_file)

def config_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s")

    # File handler
    file_handler = RotatingFileHandler(log_file_path, maxBytes=max_log_size, backupCount=backup_count)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Configure the logger
config_logger()
