"""
Logging configuration for CO2 Reduction AI Agent
"""
import logging
import sys
from pathlib import Path
from typing import Optional
import config


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_to_console: bool = True,
    log_to_file: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging with file and console handlers
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        format_string: Custom format string for log messages
        
    Returns:
        Configured logger instance
    """
    # Get configuration values
    log_level = log_level or config.LOG_LEVEL
    log_file = log_file or config.LOG_FILE
    
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger("co2_agent")
    logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Define format string with timestamps and context
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
        )
    
    formatter = logging.Formatter(
        format_string,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    logger.info(f"Logging initialized - Level: {log_level}, File: {log_file if log_to_file else 'None'}")
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance
    
    Args:
        name: Logger name (defaults to 'co2_agent')
        
    Returns:
        Logger instance
    """
    logger_name = name or "co2_agent"
    logger = logging.getLogger(logger_name)
    
    # If logger has no handlers, set it up
    if not logger.handlers:
        return setup_logging()
    
    return logger


def set_log_level(level: str) -> None:
    """
    Change the log level for all handlers
    
    Args:
        level: New log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger("co2_agent")
    logger.setLevel(numeric_level)
    
    for handler in logger.handlers:
        handler.setLevel(numeric_level)
    
    logger.info(f"Log level changed to {level.upper()}")


def log_function_call(func):
    """
    Decorator to log function calls with arguments and results
    
    Args:
        func: Function to decorate
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        logger = get_logger()
        func_name = func.__name__
        
        # Log function entry
        logger.debug(f"Entering {func_name}() with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Exiting {func_name}() successfully")
            return result
        except Exception as e:
            logger.error(f"Exception in {func_name}(): {str(e)}")
            raise
    
    return wrapper


class StructuredLogger:
    """Logger with structured logging support"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize structured logger
        
        Args:
            logger: Base logger instance
        """
        self.logger = logger or get_logger()
    
    def log_with_context(
        self,
        level: str,
        message: str,
        **context
    ) -> None:
        """
        Log message with structured context
        
        Args:
            level: Log level (debug, info, warning, error, critical)
            message: Log message
            **context: Additional context as keyword arguments
        """
        # Format context
        context_str = " | ".join([f"{k}={v}" for k, v in context.items()])
        full_message = f"{message} | {context_str}" if context_str else message
        
        # Get logging method
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(full_message)
    
    def debug(self, message: str, **context) -> None:
        """Log debug message with context"""
        self.log_with_context("debug", message, **context)
    
    def info(self, message: str, **context) -> None:
        """Log info message with context"""
        self.log_with_context("info", message, **context)
    
    def warning(self, message: str, **context) -> None:
        """Log warning message with context"""
        self.log_with_context("warning", message, **context)
    
    def error(self, message: str, **context) -> None:
        """Log error message with context"""
        self.log_with_context("error", message, **context)
    
    def critical(self, message: str, **context) -> None:
        """Log critical message with context"""
        self.log_with_context("critical", message, **context)


# Initialize default logger on module import
_default_logger = None


def initialize_default_logger() -> logging.Logger:
    """
    Initialize the default application logger
    
    Returns:
        Configured logger instance
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging()
    return _default_logger


# Convenience functions using default logger
def debug(message: str, **context) -> None:
    """Log debug message"""
    logger = initialize_default_logger()
    structured = StructuredLogger(logger)
    structured.debug(message, **context)


def info(message: str, **context) -> None:
    """Log info message"""
    logger = initialize_default_logger()
    structured = StructuredLogger(logger)
    structured.info(message, **context)


def warning(message: str, **context) -> None:
    """Log warning message"""
    logger = initialize_default_logger()
    structured = StructuredLogger(logger)
    structured.warning(message, **context)


def error(message: str, **context) -> None:
    """Log error message"""
    logger = initialize_default_logger()
    structured = StructuredLogger(logger)
    structured.error(message, **context)


def critical(message: str, **context) -> None:
    """Log critical message"""
    logger = initialize_default_logger()
    structured = StructuredLogger(logger)
    structured.critical(message, **context)
