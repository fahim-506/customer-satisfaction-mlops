import logging
import colorlog
from pathlib import Path

# Create logs folder if it doesn't exist
Path("logs").mkdir(exist_ok=True)

# Console handler with colors
console_handler = colorlog.StreamHandler()
console_formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s:%(name)s:%(message)s",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    }
)
console_handler.setFormatter(console_formatter)

# File handler (plain text)
file_handler = logging.FileHandler("logs/pipeline.log")
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
file_handler.setFormatter(file_formatter)

# Logger setup
logger = colorlog.getLogger("mlops_logger")
logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)