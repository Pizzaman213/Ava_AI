"""
Colored Logging Utility

Provides colorful, clean console output for training logs with:
- ANSI color support with auto-detection
- Level-based coloring
- Clean timestamp formatting
- Duplicate log prevention
"""

import logging
import sys
from typing import Optional


# ANSI color codes
class Colors:
    """ANSI escape codes for terminal colors."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    # Foreground colors
    BLACK = '\033[30m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'

    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'


def supports_color() -> bool:
    """Check if the terminal supports color output."""
    # Check if stdout is a TTY
    if not hasattr(sys.stdout, 'isatty') or not sys.stdout.isatty():
        return False

    # Check for common environment variables that indicate color support
    import os
    if os.environ.get('NO_COLOR'):
        return False
    if os.environ.get('FORCE_COLOR'):
        return True

    # Check TERM environment variable
    term = os.environ.get('TERM', '')
    if term in ('dumb', ''):
        return False

    return True


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to log output based on level.

    Features:
    - Level-based coloring (DEBUG=gray, INFO=cyan, WARNING=yellow, ERROR=red)
    - Clean timestamp format (HH:MM:SS only)
    - Auto-detection of terminal color support
    - Highlighting for important metrics (loss, accuracy, etc.)
    """

    LEVEL_COLORS = {
        logging.DEBUG: Colors.GRAY,
        logging.INFO: Colors.CYAN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.RED + Colors.BOLD,
    }

    LEVEL_NAMES = {
        logging.DEBUG: 'DEBUG',
        logging.INFO: 'INFO',
        logging.WARNING: 'WARN',
        logging.ERROR: 'ERROR',
        logging.CRITICAL: 'CRIT',
    }

    def __init__(self, use_colors: Optional[bool] = None, show_level: bool = False):
        """
        Initialize the colored formatter.

        Args:
            use_colors: Force colors on/off. None for auto-detection.
            show_level: Whether to show log level in output.
        """
        super().__init__()
        self.use_colors = use_colors if use_colors is not None else supports_color()
        self.show_level = show_level

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors."""
        # Get timestamp (HH:MM:SS only for cleaner output)
        from datetime import datetime
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')

        # Get the message
        message = record.getMessage()

        if self.use_colors:
            # Get color for this level
            level_color = self.LEVEL_COLORS.get(record.levelno, Colors.WHITE)

            # Format timestamp
            colored_time = f"{Colors.DIM}[{timestamp}]{Colors.RESET}"

            # Format level if requested
            level_str = ""
            if self.show_level:
                level_name = self.LEVEL_NAMES.get(record.levelno, 'INFO')
                level_str = f" {level_color}{level_name}{Colors.RESET}"

            # Highlight section headers (lines with === or ---)
            if message.strip().startswith('=' * 10) or message.strip().startswith('-' * 10):
                message = f"{Colors.BOLD}{Colors.CYAN}{message}{Colors.RESET}"
            # Highlight major section titles (all caps with spaces)
            elif message.strip().isupper() and len(message.strip()) > 5:
                message = f"{Colors.BOLD}{Colors.CYAN}{message}{Colors.RESET}"
            # Highlight warnings in messages
            elif record.levelno == logging.WARNING:
                message = f"{Colors.YELLOW}{message}{Colors.RESET}"
            # Highlight errors
            elif record.levelno >= logging.ERROR:
                message = f"{Colors.RED}{message}{Colors.RESET}"
            # Apply metric highlighting for INFO messages
            else:
                message = self._highlight_metrics(message)

            return f"{colored_time}{level_str} {message}"
        else:
            # Plain format without colors
            level_str = ""
            if self.show_level:
                level_name = self.LEVEL_NAMES.get(record.levelno, 'INFO')
                level_str = f" {level_name}"
            return f"[{timestamp}]{level_str} {message}"

    def _highlight_metrics(self, message: str) -> str:
        """Highlight important metrics in the message."""
        import re

        # Highlight loss values (Loss: X.XXXX or loss=X.XXXX)
        message = re.sub(
            r'(Loss[:\s=]+)(\d+\.\d+)',
            f'\\1{Colors.YELLOW}\\2{Colors.RESET}',
            message,
            flags=re.IGNORECASE
        )

        # Highlight learning rate (LR: X.XXe-XX)
        message = re.sub(
            r'(LR[:\s=]+)(\d+\.\d+e[+-]?\d+)',
            f'\\1{Colors.MAGENTA}\\2{Colors.RESET}',
            message,
            flags=re.IGNORECASE
        )

        # Highlight parameter counts (XXX.XM or XXX params)
        message = re.sub(
            r'(\d+[\d,]*\.?\d*[MBK]?\s*params?)',
            f'{Colors.GREEN}\\1{Colors.RESET}',
            message,
            flags=re.IGNORECASE
        )

        # Highlight device info (cuda:X, GPU X)
        message = re.sub(
            r'(cuda:\d+|GPU\s*\d+)',
            f'{Colors.MAGENTA}\\1{Colors.RESET}',
            message,
            flags=re.IGNORECASE
        )

        # Highlight memory info (XX.XX GB, XX MB)
        message = re.sub(
            r'(\d+\.?\d*\s*[GM]B)',
            f'{Colors.MAGENTA}\\1{Colors.RESET}',
            message,
            flags=re.IGNORECASE
        )

        # Highlight percentages
        message = re.sub(
            r'(\d+\.?\d*%)',
            f'{Colors.CYAN}\\1{Colors.RESET}',
            message
        )

        return message


class CleanFormatter(logging.Formatter):
    """
    Simple clean formatter without colors but with improved formatting.
    Used for file logging or when colors are disabled.
    """

    def __init__(self, include_date: bool = True):
        """
        Initialize the clean formatter.

        Args:
            include_date: Whether to include date in timestamp.
        """
        if include_date:
            fmt = '%(asctime)s | %(levelname)s | %(message)s'
            datefmt = '%Y-%m-%d %H:%M:%S'
        else:
            fmt = '[%(asctime)s] %(message)s'
            datefmt = '%H:%M:%S'
        super().__init__(fmt=fmt, datefmt=datefmt)


def setup_colored_logging(
    logger_name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    use_colors: Optional[bool] = None,
    show_level: bool = False,
    propagate: bool = False
) -> logging.Logger:
    """
    Set up a logger with colored console output and optional file logging.

    Args:
        logger_name: Name for the logger.
        level: Logging level (default: INFO).
        log_file: Optional path to log file.
        use_colors: Force colors on/off. None for auto-detection.
        show_level: Whether to show log level in output.
        propagate: Whether to propagate to parent loggers (default: False to avoid duplicates).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(logger_name)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    logger.setLevel(level)
    logger.propagate = propagate  # Prevent duplicate logs

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter(use_colors=use_colors, show_level=show_level))
    logger.addHandler(console_handler)

    # File handler (plain format, no colors)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(CleanFormatter(include_date=True))
        logger.addHandler(file_handler)

    return logger


def configure_root_logger(level: int = logging.WARNING):
    """
    Configure the root logger to reduce noise from other modules.

    Args:
        level: Minimum level for root logger (default: WARNING).
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear any existing handlers on root
    root_logger.handlers.clear()

    # Add a simple handler for root that won't duplicate our custom logs
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
    root_logger.addHandler(handler)


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger by name. Convenience function for module-level logging.

    Args:
        name: Logger name. If None, returns root logger.

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


# Convenience functions for colored output (can be used without logger)
def print_header(text: str, char: str = '=', width: int = 70):
    """Print a colored section header."""
    if supports_color():
        line = char * width
        print(f"{Colors.BOLD}{Colors.CYAN}{line}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{text.center(width)}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{line}{Colors.RESET}")
    else:
        line = char * width
        print(line)
        print(text.center(width))
        print(line)


def print_success(text: str):
    """Print a success message in green."""
    if supports_color():
        print(f"{Colors.GREEN}{text}{Colors.RESET}")
    else:
        print(text)


def print_warning(text: str):
    """Print a warning message in yellow."""
    if supports_color():
        print(f"{Colors.YELLOW}{text}{Colors.RESET}")
    else:
        print(f"WARNING: {text}")


def print_error(text: str):
    """Print an error message in red."""
    if supports_color():
        print(f"{Colors.RED}{text}{Colors.RESET}")
    else:
        print(f"ERROR: {text}")


def print_info(text: str):
    """Print an info message in cyan."""
    if supports_color():
        print(f"{Colors.CYAN}{text}{Colors.RESET}")
    else:
        print(text)
