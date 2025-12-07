import pytest
import logging
import os
from pathlib import Path
from modules.logging_config import LoggingConfigurator

def test_logger_creation(tmp_path):
    # Change CWD to tmp_path to avoid creating logs in project root
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        config = {'logging': {'level': 'DEBUG', 'log_to_file': True}}
        lc = LoggingConfigurator(config)
        lc.setup()
        
        logger = lc.get_logger('test_mod')
        logger.info("Test message")
        
        assert Path("logs/pipeline.log").exists()
        with open("logs/pipeline.log", 'r') as f:
            assert "Test message" in f.read()
    finally:
        logging.shutdown()
        os.chdir(old_cwd)