"""
Colored Logging Utility.

Provides colorful, clean console output for training logs with:
- ANSI color support with auto-detection
- Level-based coloring with icons
- Clean timestamp formatting
- Duplicate log prevention
- Metric highlighting (loss, LR, memory, etc.)
- Beautiful section headers and status indicators

Usage:
    from ava.core.logging import setup_colored_logging, print_header

    logger = setup_colored_logging('training', level=logging.INFO)
    logger.info("Training started")

    print_header("Training Configuration")
    print_success("Model loaded successfully")
"""

import logging
import sys
from typing import Optional


class Colors:
    """ANSI escape codes for terminal colors."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'

    # Foreground colors (bright versions)
    BLACK = '\033[30m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'

    # Standard foreground colors
    DARK_RED = '\033[31m'
    DARK_GREEN = '\033[32m'
    DARK_YELLOW = '\033[33m'
    DARK_BLUE = '\033[34m'
    DARK_MAGENTA = '\033[35m'
    DARK_CYAN = '\033[36m'

    # Orange (256-color mode)
    ORANGE = '\033[38;5;208m'
    LIGHT_BLUE = '\033[38;5;117m'
    PINK = '\033[38;5;213m'
    LIME = '\033[38;5;118m'
    GOLD = '\033[38;5;220m'
    PURPLE = '\033[38;5;141m'

    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'


class Icons:
    """Unicode icons for log messages."""
    # Status icons
    SUCCESS = '✓'
    ERROR = '✗'
    WARNING = '⚠'
    INFO = 'ℹ'
    DEBUG = '◦'

    # Progress icons
    ROCKET = '🚀'
    GEAR = '⚙'
    CLOCK = '⏱'
    CHECK = '✔'
    CROSS = '✘'
    ARROW_RIGHT = '→'
    ARROW_DOWN = '↓'
    BULLET = '•'
    STAR = '★'
    SPARKLE = '✦'

    # Training icons
    BRAIN = '🧠'
    GPU = '🎮'
    MEMORY = '💾'
    CHART = '📊'
    FOLDER = '📁'
    FILE = '📄'
    DATA = '📦'
    LIGHTNING = '⚡'
    FIRE = '🔥'
    TARGET = '🎯'
    HOURGLASS = '⏳'
    SAVE = '💾'
    LOAD = '📥'


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
    - Level-based coloring with icons (DEBUG=gray, INFO=cyan, WARNING=yellow, ERROR=red)
    - Clean timestamp format (HH:MM:SS only)
    - Auto-detection of terminal color support
    - Highlighting for important metrics (loss, accuracy, etc.)
    - Smart detection of message types for appropriate styling
    """

    LEVEL_COLORS = {
        logging.DEBUG: Colors.GRAY,
        logging.INFO: Colors.CYAN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.RED + Colors.BOLD,
    }

    LEVEL_ICONS = {
        logging.DEBUG: Icons.DEBUG,
        logging.INFO: Icons.BULLET,
        logging.WARNING: Icons.WARNING,
        logging.ERROR: Icons.ERROR,
        logging.CRITICAL: Icons.CROSS,
    }

    LEVEL_NAMES = {
        logging.DEBUG: 'DEBUG',
        logging.INFO: 'INFO',
        logging.WARNING: 'WARN',
        logging.ERROR: 'ERROR',
        logging.CRITICAL: 'CRIT',
    }

    def __init__(self, use_colors: Optional[bool] = None, show_level: bool = False, show_icons: bool = True):
        """
        Initialize the colored formatter.

        Args:
            use_colors: Force colors on/off. None for auto-detection.
            show_level: Whether to show log level in output.
            show_icons: Whether to show icons (default True).
        """
        super().__init__()
        self.use_colors = use_colors if use_colors is not None else supports_color()
        self.show_level = show_level
        self.show_icons = show_icons

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
            level_icon = self.LEVEL_ICONS.get(record.levelno, Icons.BULLET) if self.show_icons else ''

            # Format timestamp with subtle styling
            colored_time = f"{Colors.DIM}{Colors.DARK_CYAN}[{timestamp}]{Colors.RESET}"

            # Format level if requested
            level_str = ""
            if self.show_level:
                level_name = self.LEVEL_NAMES.get(record.levelno, 'INFO')
                level_str = f" {level_color}{level_name}{Colors.RESET}"

            # Detect and style different message types
            message = self._style_message(message, record.levelno, level_icon)

            return f"{colored_time}{level_str} {message}"
        else:
            # Plain format without colors
            level_str = ""
            if self.show_level:
                level_name = self.LEVEL_NAMES.get(record.levelno, 'INFO')
                level_str = f" {level_name}"
            return f"[{timestamp}]{level_str} {message}"

    def _style_message(self, message: str, levelno: int, icon: str) -> str:
        """Apply appropriate styling based on message content and level."""
        stripped = message.strip()

        # Section headers (=== or ---)
        if stripped.startswith('=' * 10) or stripped.startswith('-' * 10):
            return f"{Colors.BOLD}{Colors.CYAN}{message}{Colors.RESET}"

        # Major section titles (all caps)
        if stripped.isupper() and len(stripped) > 5 and ' ' in stripped:
            return f"{Colors.BOLD}{Colors.CYAN}{Icons.SPARKLE} {message}{Colors.RESET}"

        # Warning messages
        if levelno == logging.WARNING:
            return f"{Colors.YELLOW}{Icons.WARNING} {message}{Colors.RESET}"

        # Error messages
        if levelno >= logging.ERROR:
            return f"{Colors.RED}{Icons.ERROR} {message}{Colors.RESET}"

        # Success patterns
        if any(word in stripped.lower() for word in ['success', 'complete', 'finished', 'done', 'saved', 'loaded']):
            return f"{Colors.GREEN}{Icons.SUCCESS} {self._highlight_metrics(message)}{Colors.RESET}"

        # Training step patterns (Step XXX, Epoch XXX)
        if stripped.startswith(('Step ', 'Epoch ', '[Step', '[Epoch')):
            return f"{Colors.LIGHT_BLUE}{icon} {self._highlight_metrics(message)}{Colors.RESET}"

        # Configuration/settings patterns
        if any(word in stripped.lower() for word in ['config', 'setting', 'parameter', 'option']):
            return f"{Colors.PURPLE}{Icons.GEAR} {self._highlight_metrics(message)}{Colors.RESET}"

        # Data loading patterns
        if any(word in stripped.lower() for word in ['loading', 'dataset', 'dataloader', 'batch', 'samples']):
            return f"{Colors.ORANGE}{Icons.DATA} {self._highlight_metrics(message)}{Colors.RESET}"

        # GPU/Memory patterns
        if any(word in stripped.lower() for word in ['gpu', 'cuda', 'memory', 'vram']):
            return f"{Colors.MAGENTA}{Icons.GPU} {self._highlight_metrics(message)}{Colors.RESET}"

        # Model patterns
        if any(word in stripped.lower() for word in ['model', 'param', 'layer', 'weight']):
            return f"{Colors.PINK}{Icons.BRAIN} {self._highlight_metrics(message)}{Colors.RESET}"

        # Default: apply metric highlighting with icon
        return f"{icon} {self._highlight_metrics(message)}"

    def _highlight_metrics(self, message: str) -> str:
        """Highlight important metrics in the message."""
        import re

        # Highlight loss values (Loss: X.XXXX or loss=X.XXXX)
        message = re.sub(
            r'(Loss[:\s=]+)(\d+\.\d+)',
            f'\\1{Colors.GOLD}{Colors.BOLD}\\2{Colors.RESET}',
            message,
            flags=re.IGNORECASE
        )

        # Highlight learning rate (LR: X.XXe-XX)
        message = re.sub(
            r'(LR[:\s=]+)(\d+\.\d+e[+-]?\d+)',
            f'\\1{Colors.PURPLE}{Colors.BOLD}\\2{Colors.RESET}',
            message,
            flags=re.IGNORECASE
        )

        # Highlight parameter counts (XXX.XM or XXX params)
        message = re.sub(
            r'(\d+[\d,]*\.?\d*[MBK]?\s*params?)',
            f'{Colors.LIME}{Colors.BOLD}\\1{Colors.RESET}',
            message,
            flags=re.IGNORECASE
        )

        # Highlight device info (cuda:X, GPU X)
        message = re.sub(
            r'(cuda:\d+|GPU\s*\d+)',
            f'{Colors.MAGENTA}{Colors.BOLD}\\1{Colors.RESET}',
            message,
            flags=re.IGNORECASE
        )

        # Highlight memory info (XX.XX GB, XX MB)
        message = re.sub(
            r'(\d+\.?\d*\s*[GM]B)',
            f'{Colors.ORANGE}\\1{Colors.RESET}',
            message,
            flags=re.IGNORECASE
        )

        # Highlight percentages
        message = re.sub(
            r'(\d+\.?\d*%)',
            f'{Colors.CYAN}{Colors.BOLD}\\1{Colors.RESET}',
            message
        )

        # Highlight step/epoch numbers
        message = re.sub(
            r'(step|epoch)\s*(\d+)',
            f'\\1 {Colors.LIGHT_BLUE}{Colors.BOLD}\\2{Colors.RESET}',
            message,
            flags=re.IGNORECASE
        )

        # Highlight throughput (samples/sec, tokens/sec)
        message = re.sub(
            r'(\d+[\d,]*\.?\d*)\s*(samples?|tokens?)/s(ec)?',
            f'{Colors.GREEN}{Colors.BOLD}\\1{Colors.RESET} \\2/s\\3',
            message,
            flags=re.IGNORECASE
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
def print_header(text: str, char: str = '═', width: int = 70, icon: str = None):
    """Print a colored section header with optional icon."""
    if supports_color():
        line = char * width
        icon_str = f"{icon} " if icon else ""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{line}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{icon_str}{text.center(width - len(icon_str) if icon else width)}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{line}{Colors.RESET}")
    else:
        line = char * width
        print(f"\n{line}")
        print(text.center(width))
        print(line)


def print_subheader(text: str, char: str = '─', width: int = 60):
    """Print a colored sub-section header."""
    if supports_color():
        line = char * width
        print(f"\n{Colors.DARK_CYAN}{line}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.RESET}")
        print(f"{Colors.DARK_CYAN}{line}{Colors.RESET}")
    else:
        line = char * width
        print(f"\n{line}")
        print(text)
        print(line)


def print_success(text: str):
    """Print a success message in green with checkmark."""
    if supports_color():
        print(f"{Colors.GREEN}{Icons.SUCCESS} {text}{Colors.RESET}")
    else:
        print(f"[OK] {text}")


def print_warning(text: str):
    """Print a warning message in yellow with icon."""
    if supports_color():
        print(f"{Colors.YELLOW}{Icons.WARNING} {text}{Colors.RESET}")
    else:
        print(f"[WARNING] {text}")


def print_error(text: str):
    """Print an error message in red with icon."""
    if supports_color():
        print(f"{Colors.RED}{Icons.ERROR} {text}{Colors.RESET}")
    else:
        print(f"[ERROR] {text}")


def print_info(text: str):
    """Print an info message in cyan with icon."""
    if supports_color():
        print(f"{Colors.CYAN}{Icons.BULLET} {text}{Colors.RESET}")
    else:
        print(f"[INFO] {text}")


def print_step(text: str):
    """Print a step/progress message."""
    if supports_color():
        print(f"{Colors.LIGHT_BLUE}{Icons.ARROW_RIGHT} {text}{Colors.RESET}")
    else:
        print(f"-> {text}")


def print_config(key: str, value: str, indent: int = 2):
    """Print a configuration key-value pair with nice formatting."""
    spaces = ' ' * indent
    if supports_color():
        print(f"{spaces}{Colors.GRAY}{Icons.BULLET}{Colors.RESET} {Colors.WHITE}{key}:{Colors.RESET} {Colors.LIME}{value}{Colors.RESET}")
    else:
        print(f"{spaces}• {key}: {value}")


def print_metric(name: str, value: str, unit: str = '', good: bool = None):
    """Print a metric with optional status coloring."""
    if supports_color():
        if good is True:
            value_color = Colors.GREEN
        elif good is False:
            value_color = Colors.RED
        else:
            value_color = Colors.GOLD
        unit_str = f" {Colors.GRAY}{unit}{Colors.RESET}" if unit else ""
        print(f"  {Colors.WHITE}{name}:{Colors.RESET} {value_color}{Colors.BOLD}{value}{Colors.RESET}{unit_str}")
    else:
        unit_str = f" {unit}" if unit else ""
        print(f"  {name}: {value}{unit_str}")


def print_progress(current: int, total: int, prefix: str = '', width: int = 40):
    """Print a progress bar."""
    percent = current / total if total > 0 else 0
    filled = int(width * percent)
    bar = '█' * filled + '░' * (width - filled)

    if supports_color():
        color = Colors.GREEN if percent >= 1.0 else Colors.CYAN
        print(f"\r{prefix}{color}{bar}{Colors.RESET} {Colors.BOLD}{percent*100:.1f}%{Colors.RESET} ({current}/{total})", end='', flush=True)
        if percent >= 1.0:
            print()  # New line when complete
    else:
        print(f"\r{prefix}{bar} {percent*100:.1f}% ({current}/{total})", end='', flush=True)
        if percent >= 1.0:
            print()


def print_table_row(columns: list, widths: list = None, header: bool = False):
    """Print a formatted table row."""
    if widths is None:
        widths = [20] * len(columns)

    formatted = []
    for col, width in zip(columns, widths):
        formatted.append(str(col).ljust(width)[:width])

    row = ' │ '.join(formatted)
    if supports_color():
        if header:
            print(f"{Colors.BOLD}{Colors.CYAN}{row}{Colors.RESET}")
            separator = '─┼─'.join('─' * w for w in widths)
            print(f"{Colors.DARK_CYAN}{separator}{Colors.RESET}")
        else:
            print(f"{Colors.WHITE}{row}{Colors.RESET}")
    else:
        print(row)
        if header:
            print('-+-'.join('-' * w for w in widths))


def print_box(text: str, style: str = 'info', width: int = 60):
    """Print text in a styled box."""
    lines = text.split('\n')
    max_len = min(max(len(line) for line in lines), width - 4)

    if supports_color():
        if style == 'success':
            border_color = Colors.GREEN
        elif style == 'warning':
            border_color = Colors.YELLOW
        elif style == 'error':
            border_color = Colors.RED
        else:
            border_color = Colors.CYAN

        top = f"╭{'─' * (max_len + 2)}╮"
        bottom = f"╰{'─' * (max_len + 2)}╯"

        print(f"{border_color}{top}{Colors.RESET}")
        for line in lines:
            padded = line.ljust(max_len)[:max_len]
            print(f"{border_color}│{Colors.RESET} {padded} {border_color}│{Colors.RESET}")
        print(f"{border_color}{bottom}{Colors.RESET}")
    else:
        top = f"+{'-' * (max_len + 2)}+"
        bottom = top
        print(top)
        for line in lines:
            print(f"| {line.ljust(max_len)[:max_len]} |")
        print(bottom)


__all__ = [
    'Colors',
    'Icons',
    'supports_color',
    'ColoredFormatter',
    'CleanFormatter',
    'setup_colored_logging',
    'configure_root_logger',
    'get_logger',
    'print_header',
    'print_subheader',
    'print_success',
    'print_warning',
    'print_error',
    'print_info',
    'print_step',
    'print_config',
    'print_metric',
    'print_progress',
    'print_table_row',
    'print_box',
]
