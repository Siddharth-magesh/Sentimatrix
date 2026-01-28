"""
Sentimatrix Logging Module

Provides structured logging with JSON output, context propagation,
and multiple handlers (console, file).

Features:
- Structured JSON logging with rich context
- Process/thread ID tracking
- Memory usage monitoring
- Timing/duration measurement
- Request correlation IDs
- Caller info with full module path
- Performance metrics tracking
- Log sampling for high-volume scenarios

Example:
    >>> from sentimatrix.core.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing started", extra={"url": "https://example.com"})

    # With timing
    >>> with logger.timed("database_query"):
    ...     result = db.query(...)

    # With performance tracking
    >>> logger.log_performance("api_call", duration_ms=150, success=True)
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import threading
import time
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar, Union

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

from sentimatrix.core.config import LogConfig, LogLevel

# Type variable for generic decorators
F = TypeVar("F", bound=Callable[..., Any])

# System info (computed once)
_HOSTNAME = platform.node()
_PID = os.getpid()

# Context variable for request/correlation ID
_request_context: ContextVar[Dict[str, Any]] = ContextVar("request_context", default={})


class LogContext:
    """
    Context manager for adding contextual information to log messages.

    Example:
        >>> with LogContext(request_id="abc123", user_id="user456"):
        ...     logger.info("Processing request")  # Includes request_id, user_id
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize context with key-value pairs."""
        self._context = kwargs
        self._token: Optional[Any] = None

    def __enter__(self) -> "LogContext":
        """Enter context and merge with existing context."""
        current = _request_context.get().copy()
        current.update(self._context)
        self._token = _request_context.set(current)
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context and restore previous context."""
        if self._token is not None:
            _request_context.reset(self._token)

    @staticmethod
    def get() -> Dict[str, Any]:
        """Get current context dictionary."""
        return _request_context.get().copy()

    @staticmethod
    def set(**kwargs: Any) -> None:
        """Set context values (replaces existing)."""
        _request_context.set(kwargs)

    @staticmethod
    def update(**kwargs: Any) -> None:
        """Update context values (merges with existing)."""
        current = _request_context.get().copy()
        current.update(kwargs)
        _request_context.set(current)

    @staticmethod
    def clear() -> None:
        """Clear all context."""
        _request_context.set({})


class JsonFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.

    Produces JSON-formatted log messages with comprehensive structure:
    {
        "timestamp": "2024-01-15T12:00:00.123456Z",
        "level": "INFO",
        "level_num": 20,
        "logger": "sentimatrix.core",
        "message": "Processing started",
        "process": {"pid": 12345, "name": "MainProcess"},
        "thread": {"id": 140735, "name": "MainThread"},
        "host": "server-01",
        "caller": {"file": "main.py", "line": 42, "function": "process", "module": "sentimatrix.core.main"},
        "context": {"request_id": "abc123"},
        "extra": {"url": "https://example.com"},
        "timing": {"elapsed_ms": 150}
    }
    """

    def __init__(
        self,
        include_timestamp: bool = True,
        include_caller: bool = True,
        include_process_info: bool = True,
        include_thread_info: bool = True,
        include_host: bool = True,
        timestamp_format: str = "%Y-%m-%dT%H:%M:%S.%fZ",
    ) -> None:
        """
        Initialize JSON formatter.

        Args:
            include_timestamp: Include timestamp in output
            include_caller: Include caller info (file, line, function, module)
            include_process_info: Include process ID and name
            include_thread_info: Include thread ID and name
            include_host: Include hostname
            timestamp_format: Format string for timestamps
        """
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_caller = include_caller
        self.include_process_info = include_process_info
        self.include_thread_info = include_thread_info
        self.include_host = include_host
        self.timestamp_format = timestamp_format

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON with comprehensive details."""
        # Build base log entry
        log_entry: Dict[str, Any] = {
            "level": record.levelname,
            "level_num": record.levelno,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add timestamp with millisecond precision
        if self.include_timestamp:
            log_entry["timestamp"] = datetime.now(timezone.utc).strftime(self.timestamp_format)
            log_entry["timestamp_unix"] = time.time()

        # Add process info
        if self.include_process_info:
            log_entry["process"] = {
                "pid": record.process,
                "name": record.processName,
            }

        # Add thread info
        if self.include_thread_info:
            log_entry["thread"] = {
                "id": record.thread,
                "name": record.threadName,
            }

        # Add hostname
        if self.include_host:
            log_entry["host"] = _HOSTNAME

        # Add detailed caller info
        if self.include_caller:
            log_entry["caller"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
                "module": record.module,
                "pathname": record.pathname,
            }

        # Add context from ContextVar
        context = LogContext.get()
        if context:
            log_entry["context"] = context

        # Add extra fields (excluding standard LogRecord attributes)
        standard_attrs = {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "exc_info",
            "exc_text",
            "thread",
            "threadName",
            "taskName",
            "message",
        }

        extra = {
            key: value
            for key, value in record.__dict__.items()
            if key not in standard_attrs and not key.startswith("_")
        }

        if extra:
            log_entry["extra"] = extra

        # Add exception info if present with full traceback
        if record.exc_info:
            exc_type, exc_value, exc_tb = record.exc_info
            log_entry["exception"] = {
                "type": exc_type.__name__ if exc_type else None,
                "message": str(exc_value) if exc_value else None,
                "traceback": self.formatException(record.exc_info),
            }

        # Add stack info if present
        if record.stack_info:
            log_entry["stack_info"] = record.stack_info

        return json.dumps(log_entry, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """
    Human-readable text formatter with optional colorization.

    Format: [TIMESTAMP] LEVEL PID:TID LOGGER MESSAGE {extra} (file:line)

    Example output:
    [2024-01-15 12:00:00.123] INFO     12345:140735 core.processor  Processing started {url=https://example.com} (processor.py:42)
    """

    COLORS = {
        "DEBUG": "dim",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold red",
    }

    def __init__(
        self,
        include_timestamp: bool = True,
        include_caller: bool = True,
        include_process_thread: bool = True,
        timestamp_format: str = "%Y-%m-%d %H:%M:%S.%f",
    ) -> None:
        """
        Initialize text formatter.

        Args:
            include_timestamp: Include timestamp in output
            include_caller: Include caller info (file:line)
            include_process_thread: Include process and thread IDs
            timestamp_format: Format string for timestamps
        """
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_caller = include_caller
        self.include_process_thread = include_process_thread
        self.timestamp_format = timestamp_format

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as detailed text."""
        parts: List[str] = []

        # Timestamp with milliseconds
        if self.include_timestamp:
            timestamp = datetime.now().strftime(self.timestamp_format)[:-3]  # Trim to ms
            parts.append(f"[{timestamp}]")

        # Level
        parts.append(f"{record.levelname:8s}")

        # Process and thread IDs
        if self.include_process_thread:
            parts.append(f"{record.process}:{record.thread}")

        # Logger name (shortened for readability)
        logger_name = record.name
        if logger_name.startswith("sentimatrix."):
            logger_name = logger_name[12:]  # Remove prefix
        parts.append(f"{logger_name:20s}")

        # Message
        parts.append(record.getMessage())

        # Context
        context = LogContext.get()
        if context:
            context_str = " ".join(f"{k}={v}" for k, v in context.items())
            parts.append(f"[{context_str}]")

        # Extra fields
        standard_attrs = {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "exc_info",
            "exc_text",
            "thread",
            "threadName",
            "taskName",
            "message",
        }

        extra = {
            key: value
            for key, value in record.__dict__.items()
            if key not in standard_attrs and not key.startswith("_")
        }

        if extra:
            extra_str = " ".join(f"{k}={v}" for k, v in extra.items())
            parts.append(f"{{{extra_str}}}")

        # Caller info at the end
        if self.include_caller:
            parts.append(f"({record.filename}:{record.lineno}:{record.funcName})")

        result = " ".join(parts)

        # Add exception info
        if record.exc_info:
            result += "\n" + self.formatException(record.exc_info)

        # Add stack info
        if record.stack_info:
            result += "\n" + record.stack_info

        return result


class StructuredLogger:
    """
    Wrapper around Python logger with structured logging support.

    Provides convenient methods for logging with extra context.
    """

    def __init__(self, logger: logging.Logger) -> None:
        """Initialize with a Python logger instance."""
        self._logger = logger

    @property
    def name(self) -> str:
        """Get logger name."""
        return self._logger.name

    @property
    def level(self) -> int:
        """Get effective log level."""
        return self._logger.getEffectiveLevel()

    def _log(
        self,
        level: int,
        message: str,
        *args: Any,
        exc_info: bool = False,
        stack_info: bool = False,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Internal log method with extra handling."""
        # Merge kwargs into extra
        if kwargs:
            extra = extra or {}
            extra.update(kwargs)

        self._logger.log(
            level,
            message,
            *args,
            exc_info=exc_info,
            stack_info=stack_info,
            extra=extra,
        )

    def debug(
        self,
        message: str,
        *args: Any,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Log a debug message."""
        self._log(logging.DEBUG, message, *args, extra=extra, **kwargs)

    def info(
        self,
        message: str,
        *args: Any,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Log an info message."""
        self._log(logging.INFO, message, *args, extra=extra, **kwargs)

    def warning(
        self,
        message: str,
        *args: Any,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Log a warning message."""
        self._log(logging.WARNING, message, *args, extra=extra, **kwargs)

    def error(
        self,
        message: str,
        *args: Any,
        exc_info: bool = False,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Log an error message."""
        self._log(logging.ERROR, message, *args, exc_info=exc_info, extra=extra, **kwargs)

    def critical(
        self,
        message: str,
        *args: Any,
        exc_info: bool = False,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Log a critical message."""
        self._log(logging.CRITICAL, message, *args, exc_info=exc_info, extra=extra, **kwargs)

    def exception(
        self,
        message: str,
        *args: Any,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Log an exception with traceback."""
        self._log(logging.ERROR, message, *args, exc_info=True, extra=extra, **kwargs)

    def bind(self, **kwargs: Any) -> "StructuredLogger":
        """
        Create a new logger with bound context.

        The bound context will be included in all log messages from the new logger.

        Example:
            >>> request_logger = logger.bind(request_id="abc123")
            >>> request_logger.info("Processing")  # Includes request_id
        """
        return BoundLogger(self._logger, kwargs)

    def timed(self, operation: str, level: int = logging.INFO) -> "TimingContext":
        """
        Context manager for timing an operation.

        Logs start and completion with duration.

        Example:
            >>> with logger.timed("database_query"):
            ...     result = db.query(...)
            # Logs: "database_query started" and "database_query completed in 150.23ms"
        """
        return TimingContext(self, operation, level)

    def log_performance(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Log a performance metric.

        Args:
            operation: Name of the operation
            duration_ms: Duration in milliseconds
            success: Whether the operation succeeded
            extra: Additional context
            **kwargs: Additional key-value pairs

        Example:
            >>> logger.log_performance("api_call", duration_ms=150.5, success=True, endpoint="/users")
        """
        perf_extra = {
            "event_type": "performance",
            "operation": operation,
            "duration_ms": round(duration_ms, 3),
            "success": success,
        }
        if extra:
            perf_extra.update(extra)
        if kwargs:
            perf_extra.update(kwargs)

        level = logging.INFO if success else logging.WARNING
        status = "completed" if success else "failed"
        self._log(level, f"{operation} {status} in {duration_ms:.2f}ms", extra=perf_extra)

    def log_event(
        self,
        event_type: str,
        message: str,
        level: int = logging.INFO,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Log a structured event.

        Args:
            event_type: Type of event (e.g., "scrape_started", "model_loaded")
            message: Human-readable message
            level: Log level
            extra: Additional context
            **kwargs: Additional key-value pairs

        Example:
            >>> logger.log_event("scrape_completed", "Scraped 150 reviews", count=150, platform="amazon")
        """
        event_extra = {"event_type": event_type}
        if extra:
            event_extra.update(extra)
        if kwargs:
            event_extra.update(kwargs)

        self._log(level, message, extra=event_extra)

    def log_error_details(
        self,
        error: Exception,
        message: str = "An error occurred",
        include_traceback: bool = True,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Log detailed error information.

        Args:
            error: The exception to log
            message: Human-readable message
            include_traceback: Include full traceback
            extra: Additional context
            **kwargs: Additional key-value pairs

        Example:
            >>> try:
            ...     risky_operation()
            ... except Exception as e:
            ...     logger.log_error_details(e, "Failed to process request", request_id="abc")
        """
        error_extra = {
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

        if include_traceback:
            error_extra["traceback"] = traceback.format_exc()

        if hasattr(error, "__dict__"):
            # Include any custom attributes from the exception
            for key, value in error.__dict__.items():
                if not key.startswith("_") and key not in error_extra:
                    try:
                        json.dumps(value)  # Check if serializable
                        error_extra[f"error_{key}"] = value
                    except (TypeError, ValueError):
                        error_extra[f"error_{key}"] = str(value)

        if extra:
            error_extra.update(extra)
        if kwargs:
            error_extra.update(kwargs)

        self._log(logging.ERROR, message, exc_info=True, extra=error_extra)

    def log_request(
        self,
        method: str,
        url: str,
        status_code: Optional[int] = None,
        duration_ms: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Log HTTP request details.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            status_code: Response status code (if completed)
            duration_ms: Request duration in milliseconds
            extra: Additional context
            **kwargs: Additional key-value pairs

        Example:
            >>> logger.log_request("GET", "https://api.example.com/users", status_code=200, duration_ms=150)
        """
        request_extra = {
            "event_type": "http_request",
            "method": method,
            "url": url,
        }

        if status_code is not None:
            request_extra["status_code"] = status_code
            request_extra["success"] = 200 <= status_code < 400

        if duration_ms is not None:
            request_extra["duration_ms"] = round(duration_ms, 3)

        if extra:
            request_extra.update(extra)
        if kwargs:
            request_extra.update(kwargs)

        level = logging.INFO
        if status_code and status_code >= 400:
            level = logging.WARNING if status_code < 500 else logging.ERROR

        status_str = f" -> {status_code}" if status_code else ""
        duration_str = f" ({duration_ms:.2f}ms)" if duration_ms else ""
        self._log(level, f"{method} {url}{status_str}{duration_str}", extra=request_extra)


class BoundLogger(StructuredLogger):
    """Logger with bound context that's included in every log message."""

    def __init__(self, logger: logging.Logger, bindings: Dict[str, Any]) -> None:
        """Initialize with base logger and bindings."""
        super().__init__(logger)
        self._bindings = bindings

    def _log(
        self,
        level: int,
        message: str,
        *args: Any,
        exc_info: bool = False,
        stack_info: bool = False,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Log with bound context merged into extra."""
        merged_extra = self._bindings.copy()
        if extra:
            merged_extra.update(extra)
        if kwargs:
            merged_extra.update(kwargs)

        self._logger.log(
            level,
            message,
            *args,
            exc_info=exc_info,
            stack_info=stack_info,
            extra=merged_extra,
        )

    def bind(self, **kwargs: Any) -> "BoundLogger":
        """Create a new bound logger with additional bindings."""
        merged = self._bindings.copy()
        merged.update(kwargs)
        return BoundLogger(self._logger, merged)


class TimingContext:
    """
    Context manager for timing operations.

    Automatically logs start and completion with duration.
    """

    def __init__(
        self,
        logger: StructuredLogger,
        operation: str,
        level: int = logging.INFO,
        log_start: bool = True,
    ) -> None:
        """
        Initialize timing context.

        Args:
            logger: Logger to use for logging
            operation: Name of the operation being timed
            level: Log level to use
            log_start: Whether to log when the operation starts
        """
        self._logger = logger
        self._operation = operation
        self._level = level
        self._log_start = log_start
        self._start_time: float = 0
        self._end_time: float = 0

    def __enter__(self) -> "TimingContext":
        """Start timing."""
        self._start_time = time.perf_counter()
        if self._log_start:
            self._logger._log(
                self._level,
                f"{self._operation} started",
                extra={"event_type": "timing_start", "operation": self._operation},
            )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop timing and log result."""
        self._end_time = time.perf_counter()
        duration_ms = (self._end_time - self._start_time) * 1000

        success = exc_type is None
        status = "completed" if success else "failed"
        level = self._level if success else logging.ERROR

        extra = {
            "event_type": "timing_end",
            "operation": self._operation,
            "duration_ms": round(duration_ms, 3),
            "success": success,
        }

        if not success:
            extra["error_type"] = exc_type.__name__ if exc_type else None
            extra["error_message"] = str(exc_val) if exc_val else None

        self._logger._log(
            level,
            f"{self._operation} {status} in {duration_ms:.2f}ms",
            extra=extra,
        )

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds (during or after operation)."""
        if self._end_time:
            return (self._end_time - self._start_time) * 1000
        return (time.perf_counter() - self._start_time) * 1000


def timed_function(
    operation: Optional[str] = None,
    logger: Optional[StructuredLogger] = None,
    level: int = logging.INFO,
) -> Callable[[F], F]:
    """
    Decorator for timing function execution.

    Args:
        operation: Operation name (defaults to function name)
        logger: Logger to use (defaults to function's module logger)
        level: Log level

    Example:
        >>> @timed_function()
        ... def process_data(data):
        ...     return transform(data)
    """

    def decorator(func: F) -> F:
        import functools

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal logger, operation
            _logger = logger or get_logger(func.__module__)
            _operation = operation or func.__name__

            with _logger.timed(_operation, level):
                return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


class LogManager:
    """
    Central log manager for configuring and creating loggers.

    Manages logging configuration, handlers, and logger instances.
    """

    _instance: Optional["LogManager"] = None
    _initialized: bool = False

    def __new__(cls) -> "LogManager":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize log manager."""
        if not LogManager._initialized:
            self._loggers: Dict[str, StructuredLogger] = {}
            self._config: Optional[LogConfig] = None
            self._handlers: List[logging.Handler] = []
            self._root_logger = logging.getLogger("sentimatrix")
            LogManager._initialized = True

    def configure(self, config: Optional[LogConfig] = None) -> None:
        """
        Configure logging based on LogConfig.

        Args:
            config: Logging configuration (uses defaults if None)
        """
        self._config = config or LogConfig()

        # Clear existing handlers
        for handler in self._handlers:
            self._root_logger.removeHandler(handler)
        self._handlers.clear()

        # Set log level
        level = getattr(logging, self._config.level.value)
        self._root_logger.setLevel(level)

        # Create formatter based on format type with full details
        if self._config.format == "json":
            formatter = JsonFormatter(
                include_timestamp=self._config.include_timestamp,
                include_caller=self._config.include_caller,
                include_process_info=True,
                include_thread_info=True,
                include_host=True,
            )
        else:
            formatter = TextFormatter(
                include_timestamp=self._config.include_timestamp,
                include_caller=self._config.include_caller,
                include_process_thread=True,
            )

        # Console handler
        if self._config.console_output:
            if self._config.colorize and self._config.format == "text":
                # Use Rich for colorized console output
                console = Console(
                    theme=Theme(
                        {
                            "logging.level.debug": "dim",
                            "logging.level.info": "green",
                            "logging.level.warning": "yellow",
                            "logging.level.error": "red",
                            "logging.level.critical": "bold red",
                        }
                    )
                )
                console_handler = RichHandler(
                    console=console,
                    show_time=self._config.include_timestamp,
                    show_path=self._config.include_caller,
                    rich_tracebacks=True,
                    tracebacks_show_locals=True,  # Show local variables in tracebacks
                )
            else:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(formatter)

            console_handler.setLevel(level)
            self._root_logger.addHandler(console_handler)
            self._handlers.append(console_handler)

        # File handler (always uses JSON for structured parsing)
        if self._config.file_path:
            file_path = Path(self._config.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # File logging always uses JSON for easier parsing
            file_formatter = JsonFormatter(
                include_timestamp=True,
                include_caller=True,
                include_process_info=True,
                include_thread_info=True,
                include_host=True,
            )

            file_handler = RotatingFileHandler(
                file_path,
                maxBytes=self._config.max_file_size_mb * 1024 * 1024,
                backupCount=self._config.backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(level)
            self._root_logger.addHandler(file_handler)
            self._handlers.append(file_handler)

        # Prevent propagation to root logger
        self._root_logger.propagate = False

    def get_logger(self, name: str) -> StructuredLogger:
        """
        Get or create a structured logger.

        Args:
            name: Logger name (typically __name__)

        Returns:
            StructuredLogger instance
        """
        if name not in self._loggers:
            # Ensure logger is under sentimatrix namespace
            if not name.startswith("sentimatrix"):
                full_name = f"sentimatrix.{name}"
            else:
                full_name = name

            logger = logging.getLogger(full_name)
            self._loggers[name] = StructuredLogger(logger)

        return self._loggers[name]

    def set_level(self, level: Union[str, LogLevel]) -> None:
        """Set log level for all loggers."""
        if isinstance(level, LogLevel):
            level = level.value
        log_level = getattr(logging, level.upper())
        self._root_logger.setLevel(log_level)
        for handler in self._handlers:
            handler.setLevel(log_level)

    def shutdown(self) -> None:
        """Shutdown logging and cleanup handlers."""
        for handler in self._handlers:
            handler.close()
            self._root_logger.removeHandler(handler)
        self._handlers.clear()
        self._loggers.clear()


# Module-level convenience functions


def get_logger(name: str = "sentimatrix") -> StructuredLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        StructuredLogger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started", url="https://example.com")
    """
    manager = LogManager()
    if not manager._config:
        manager.configure()
    return manager.get_logger(name)


def configure_logging(config: Optional[LogConfig] = None) -> None:
    """
    Configure logging globally.

    Args:
        config: Logging configuration (uses defaults if None)

    Example:
        >>> from sentimatrix.core.config import LogConfig, LogLevel
        >>> configure_logging(LogConfig(level=LogLevel.DEBUG, format="text"))
    """
    manager = LogManager()
    manager.configure(config)


def set_log_level(level: Union[str, LogLevel]) -> None:
    """
    Set global log level.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Example:
        >>> set_log_level("DEBUG")
        >>> set_log_level(LogLevel.WARNING)
    """
    manager = LogManager()
    manager.set_level(level)


def get_memory_usage_mb() -> float:
    """
    Get current process memory usage in megabytes.

    Returns:
        Memory usage in MB

    Example:
        >>> logger.info("Memory usage", memory_mb=get_memory_usage_mb())
    """
    try:
        import resource

        # Get memory in KB and convert to MB
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        # Windows fallback
        try:
            import psutil

            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0


def log_system_info(logger: Optional[StructuredLogger] = None) -> None:
    """
    Log system information for debugging/diagnostics.

    Args:
        logger: Logger to use (uses default if not provided)

    Example:
        >>> log_system_info()  # Logs Python version, platform, etc.
    """
    _logger = logger or get_logger("sentimatrix.system")
    import sys

    _logger.info(
        "System information",
        extra={
            "event_type": "system_info",
            "python_version": sys.version,
            "platform": platform.platform(),
            "hostname": _HOSTNAME,
            "pid": _PID,
            "cpu_count": os.cpu_count(),
            "memory_mb": get_memory_usage_mb(),
        },
    )


# Convenience exports
__all__ = [
    "LogContext",
    "JsonFormatter",
    "TextFormatter",
    "StructuredLogger",
    "BoundLogger",
    "TimingContext",
    "LogManager",
    "get_logger",
    "configure_logging",
    "set_log_level",
    "get_memory_usage_mb",
    "log_system_info",
    "timed_function",
]
