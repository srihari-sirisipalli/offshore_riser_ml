import functools
import logging
from utils.exceptions import RiserMLException

def handle_engine_errors(operation_name: str):
    """Decorator for consistent error handling in engines."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RiserMLException:
                # Re-raise our custom exceptions
                raise
            except Exception as e:
                # Wrap unexpected errors
                logger = args[0].logger if hasattr(args[0], 'logger') else logging.getLogger()
                logger.error(f"{operation_name} failed: {e}", exc_info=True)
                raise RiserMLException(f"{operation_name} failed: {str(e)}") from e
        return wrapper
    return decorator