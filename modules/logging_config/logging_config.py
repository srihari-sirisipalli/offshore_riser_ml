import logging
import sys
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""
    COLORS = {
        'DEBUG': Fore.WHITE + Style.DIM if COLORAMA_AVAILABLE else '',
        'INFO': Fore.CYAN if COLORAMA_AVAILABLE else '',
        'WARNING': Fore.YELLOW if COLORAMA_AVAILABLE else '',
        'ERROR': Fore.RED if COLORAMA_AVAILABLE else '',
        'CRITICAL': Fore.RED + Style.BRIGHT if COLORAMA_AVAILABLE else ''
    }
    RESET = Style.RESET_ALL if COLORAMA_AVAILABLE else ''

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        return super().format(record)

class LoggingConfigurator:
    """Configures system-wide logging."""
    
    def __init__(self, config: dict):
        self.config = config.get('logging', {})
        self.log_level = getattr(logging, self.config.get('level', 'INFO').upper())
        self.log_dir = Path("logs")
        
    def setup(self) -> None:
        """Setup all loggers and handlers."""
        self.log_dir.mkdir(exist_ok=True)
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        root_logger.handlers = []  # Clear existing
        
        # Console Handler
        if self.config.get('log_to_console', True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            if self.config.get('colorful_console', True):
                formatter = ColoredFormatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            else:
                formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
            
        # File Handler
        if self.config.get('log_to_file', True):
            self._add_file_handler(root_logger, "pipeline.log")
            
    def _add_file_handler(self, logger: logging.Logger, filename: str):
        file_path = self.log_dir / filename
        handler = RotatingFileHandler(file_path, maxBytes=10*1024*1024, backupCount=5)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] [%(funcName)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def get_logger(self, name: str) -> logging.Logger:
        return logging.getLogger(name)