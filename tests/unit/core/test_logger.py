"""
Unit Tests for Logger Module

Tests the structured logging system including:
- Logger configuration
- JSON and text formatters
- Log context propagation
- Structured logger functionality
- Bound loggers
"""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sentimatrix.core.config import LogConfig, LogLevel
from sentimatrix.core.logger import (
    BoundLogger,
    JsonFormatter,
    LogContext,
    LogManager,
    StructuredLogger,
    TextFormatter,
    configure_logging,
    get_logger,
    set_log_level,
)


class TestLogContext:
    """Tests for LogContext context manager."""

    def test_context_enter_exit(self):
        """Test basic context manager functionality."""
        LogContext.clear()

        with LogContext(request_id="123"):
            context = LogContext.get()
            assert context["request_id"] == "123"

        # Context should be cleared after exit
        context = LogContext.get()
        assert "request_id" not in context

    def test_nested_context(self):
        """Test nested context managers."""
        LogContext.clear()

        with LogContext(request_id="123"):
            assert LogContext.get()["request_id"] == "123"

            with LogContext(user_id="456"):
                context = LogContext.get()
                assert context["request_id"] == "123"
                assert context["user_id"] == "456"

            # Inner context should be popped
            context = LogContext.get()
            assert context["request_id"] == "123"
            assert "user_id" not in context

    def test_context_set(self):
        """Test static set method."""
        LogContext.clear()
        LogContext.set(key1="value1", key2="value2")

        context = LogContext.get()
        assert context["key1"] == "value1"
        assert context["key2"] == "value2"

        # Set replaces all values
        LogContext.set(key3="value3")
        context = LogContext.get()
        assert "key1" not in context
        assert context["key3"] == "value3"

        LogContext.clear()

    def test_context_update(self):
        """Test static update method."""
        LogContext.clear()
        LogContext.set(key1="value1")
        LogContext.update(key2="value2")

        context = LogContext.get()
        assert context["key1"] == "value1"
        assert context["key2"] == "value2"

        LogContext.clear()

    def test_context_clear(self):
        """Test static clear method."""
        LogContext.set(key1="value1")
        LogContext.clear()

        context = LogContext.get()
        assert context == {}

    def test_get_returns_copy(self):
        """Test that get() returns a copy, not the original."""
        LogContext.clear()
        LogContext.set(key="value")

        context1 = LogContext.get()
        context1["new_key"] = "new_value"

        context2 = LogContext.get()
        assert "new_key" not in context2

        LogContext.clear()


class TestJsonFormatter:
    """Tests for JsonFormatter."""

    def test_format_basic(self):
        """Test basic JSON formatting."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"
        assert parsed["message"] == "Test message"
        assert "timestamp" in parsed
        assert "caller" in parsed
        assert parsed["caller"]["line"] == 42

    def test_format_without_timestamp(self):
        """Test formatting without timestamp."""
        formatter = JsonFormatter(include_timestamp=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert "timestamp" not in parsed

    def test_format_without_caller(self):
        """Test formatting without caller info."""
        formatter = JsonFormatter(include_caller=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert "caller" not in parsed

    def test_format_with_context(self):
        """Test formatting with log context."""
        formatter = JsonFormatter()
        LogContext.clear()

        with LogContext(request_id="req-123"):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg="Test",
                args=(),
                exc_info=None,
            )

            output = formatter.format(record)
            parsed = json.loads(output)

            assert "context" in parsed
            assert parsed["context"]["request_id"] == "req-123"

    def test_format_with_extra(self):
        """Test formatting with extra fields."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.url = "https://example.com"
        record.status = 200

        output = formatter.format(record)
        parsed = json.loads(output)

        assert "extra" in parsed
        assert parsed["extra"]["url"] == "https://example.com"
        assert parsed["extra"]["status"] == 200

    def test_format_with_exception(self):
        """Test formatting with exception info."""
        formatter = JsonFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert "exception" in parsed
        assert parsed["exception"]["type"] == "ValueError"
        assert parsed["exception"]["message"] == "Test error"
        assert "traceback" in parsed["exception"]
        assert "ValueError" in parsed["exception"]["traceback"]


class TestTextFormatter:
    """Tests for TextFormatter."""

    def test_format_basic(self):
        """Test basic text formatting."""
        formatter = TextFormatter()
        record = logging.LogRecord(
            name="sentimatrix.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        assert "INFO" in output
        assert "Test message" in output

    def test_format_without_timestamp(self):
        """Test formatting without timestamp."""
        formatter = TextFormatter(include_timestamp=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        # Should not start with timestamp bracket
        assert not output.strip().startswith("[")

    def test_format_with_caller(self):
        """Test formatting with caller info."""
        formatter = TextFormatter(include_caller=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        assert "test.py:42" in output


class TestStructuredLogger:
    """Tests for StructuredLogger."""

    @pytest.fixture
    def mock_logger(self):
        """Provide a mock Python logger."""
        logger = MagicMock(spec=logging.Logger)
        logger.name = "test.logger"
        logger.getEffectiveLevel.return_value = logging.INFO
        return logger

    @pytest.fixture
    def structured_logger(self, mock_logger):
        """Provide a StructuredLogger instance."""
        return StructuredLogger(mock_logger)

    def test_name_property(self, structured_logger, mock_logger):
        """Test name property."""
        assert structured_logger.name == mock_logger.name

    def test_level_property(self, structured_logger, mock_logger):
        """Test level property."""
        assert structured_logger.level == logging.INFO

    def test_debug(self, structured_logger, mock_logger):
        """Test debug logging."""
        structured_logger.debug("Debug message", url="https://example.com")
        mock_logger.log.assert_called_once()
        args, kwargs = mock_logger.log.call_args
        assert args[0] == logging.DEBUG
        assert args[1] == "Debug message"
        assert kwargs["extra"]["url"] == "https://example.com"

    def test_info(self, structured_logger, mock_logger):
        """Test info logging."""
        structured_logger.info("Info message")
        mock_logger.log.assert_called_once()
        args, _ = mock_logger.log.call_args
        assert args[0] == logging.INFO

    def test_warning(self, structured_logger, mock_logger):
        """Test warning logging."""
        structured_logger.warning("Warning message")
        mock_logger.log.assert_called_once()
        args, _ = mock_logger.log.call_args
        assert args[0] == logging.WARNING

    def test_error(self, structured_logger, mock_logger):
        """Test error logging."""
        structured_logger.error("Error message", exc_info=True)
        mock_logger.log.assert_called_once()
        args, kwargs = mock_logger.log.call_args
        assert args[0] == logging.ERROR
        assert kwargs["exc_info"] is True

    def test_critical(self, structured_logger, mock_logger):
        """Test critical logging."""
        structured_logger.critical("Critical message")
        mock_logger.log.assert_called_once()
        args, _ = mock_logger.log.call_args
        assert args[0] == logging.CRITICAL

    def test_exception(self, structured_logger, mock_logger):
        """Test exception logging."""
        structured_logger.exception("Exception occurred")
        mock_logger.log.assert_called_once()
        args, kwargs = mock_logger.log.call_args
        assert args[0] == logging.ERROR
        assert kwargs["exc_info"] is True

    def test_bind_creates_bound_logger(self, structured_logger):
        """Test bind creates a BoundLogger."""
        bound = structured_logger.bind(request_id="123")
        assert isinstance(bound, BoundLogger)


class TestBoundLogger:
    """Tests for BoundLogger."""

    @pytest.fixture
    def mock_logger(self):
        """Provide a mock Python logger."""
        logger = MagicMock(spec=logging.Logger)
        logger.name = "test.logger"
        return logger

    def test_bound_logger_includes_bindings(self, mock_logger):
        """Test that bound logger includes bindings in logs."""
        bound = BoundLogger(mock_logger, {"request_id": "123"})
        bound.info("Test message")

        mock_logger.log.assert_called_once()
        _, kwargs = mock_logger.log.call_args
        assert kwargs["extra"]["request_id"] == "123"

    def test_bound_logger_merges_extra(self, mock_logger):
        """Test that bound logger merges extra with bindings."""
        bound = BoundLogger(mock_logger, {"request_id": "123"})
        bound.info("Test message", extra={"user_id": "456"})

        _, kwargs = mock_logger.log.call_args
        assert kwargs["extra"]["request_id"] == "123"
        assert kwargs["extra"]["user_id"] == "456"

    def test_bound_logger_chain(self, mock_logger):
        """Test chaining bind calls."""
        bound1 = BoundLogger(mock_logger, {"request_id": "123"})
        bound2 = bound1.bind(user_id="456")

        bound2.info("Test")

        _, kwargs = mock_logger.log.call_args
        assert kwargs["extra"]["request_id"] == "123"
        assert kwargs["extra"]["user_id"] == "456"


class TestLogManager:
    """Tests for LogManager singleton."""

    def test_singleton(self):
        """Test LogManager is a singleton."""
        manager1 = LogManager()
        manager2 = LogManager()
        assert manager1 is manager2

    def test_configure_json(self):
        """Test configuring with JSON format."""
        manager = LogManager()
        config = LogConfig(format="json", console_output=False)
        manager.configure(config)

        # Should not raise
        assert manager._config is not None

    def test_configure_text(self):
        """Test configuring with text format."""
        manager = LogManager()
        config = LogConfig(format="text", console_output=False, colorize=False)
        manager.configure(config)

        assert manager._config is not None

    def test_get_logger(self):
        """Test getting a logger."""
        manager = LogManager()
        manager.configure(LogConfig(console_output=False))

        logger = manager.get_logger("test.module")
        assert isinstance(logger, StructuredLogger)
        assert "sentimatrix" in logger.name or "test" in logger.name

    def test_get_logger_cached(self):
        """Test that loggers are cached."""
        manager = LogManager()
        manager.configure(LogConfig(console_output=False))

        logger1 = manager.get_logger("test.module")
        logger2 = manager.get_logger("test.module")

        assert logger1 is logger2

    def test_set_level(self):
        """Test setting log level."""
        manager = LogManager()
        manager.configure(LogConfig(level=LogLevel.INFO, console_output=False))
        manager.set_level(LogLevel.DEBUG)

        # Should not raise

    def test_set_level_string(self):
        """Test setting log level with string."""
        manager = LogManager()
        manager.configure(LogConfig(console_output=False))
        manager.set_level("DEBUG")

        # Should not raise

    def test_shutdown(self):
        """Test shutdown cleans up resources."""
        manager = LogManager()
        manager.configure(LogConfig(console_output=False))
        manager.get_logger("test")

        manager.shutdown()

        assert len(manager._handlers) == 0


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_logger_function(self):
        """Test get_logger module function."""
        logger = get_logger("test.module")
        assert isinstance(logger, StructuredLogger)

    def test_configure_logging_function(self):
        """Test configure_logging module function."""
        config = LogConfig(level=LogLevel.WARNING, console_output=False)
        configure_logging(config)

        # Should not raise

    def test_set_log_level_function(self):
        """Test set_log_level module function."""
        configure_logging(LogConfig(console_output=False))
        set_log_level("ERROR")

        # Should not raise

    def test_set_log_level_enum(self):
        """Test set_log_level with enum."""
        configure_logging(LogConfig(console_output=False))
        set_log_level(LogLevel.CRITICAL)

        # Should not raise


class TestFileLogging:
    """Tests for file-based logging."""

    def test_file_handler_creation(self, temp_dir: Path):
        """Test that file handler is created."""
        log_file = temp_dir / "test.log"
        config = LogConfig(
            file_path=str(log_file),
            console_output=False,
        )

        manager = LogManager()
        manager.configure(config)

        # Log something
        logger = manager.get_logger("test")
        logger.info("Test message")

        # File should be created
        assert log_file.exists()

        manager.shutdown()

    def test_log_rotation_config(self, temp_dir: Path):
        """Test log rotation configuration."""
        log_file = temp_dir / "test.log"
        config = LogConfig(
            file_path=str(log_file),
            max_file_size_mb=1,
            backup_count=3,
            console_output=False,
        )

        manager = LogManager()
        manager.configure(config)

        # Should have rotating file handler
        assert any(
            "RotatingFileHandler" in type(h).__name__
            for h in manager._handlers
        )

        manager.shutdown()


class TestTimingContext:
    """Tests for TimingContext timing functionality."""

    @pytest.fixture
    def mock_logger(self):
        """Provide a mock Python logger."""
        logger = MagicMock(spec=logging.Logger)
        logger.name = "test.logger"
        return logger

    def test_timed_context_success(self, mock_logger):
        """Test timing context logs start and completion."""
        from sentimatrix.core.logger import StructuredLogger, TimingContext
        import time

        structured = StructuredLogger(mock_logger)

        with structured.timed("test_operation"):
            time.sleep(0.01)  # Small delay

        # Should have logged start and end
        assert mock_logger.log.call_count == 2

        # Check completion message contains duration
        last_call = mock_logger.log.call_args_list[-1]
        assert "completed" in last_call[0][1]
        assert "extra" in last_call[1]
        assert last_call[1]["extra"]["success"] is True

    def test_timed_context_failure(self, mock_logger):
        """Test timing context logs failure on exception."""
        from sentimatrix.core.logger import StructuredLogger

        structured = StructuredLogger(mock_logger)

        try:
            with structured.timed("failing_operation"):
                raise ValueError("Test error")
        except ValueError:
            pass

        # Should have logged start and failure
        last_call = mock_logger.log.call_args_list[-1]
        assert "failed" in last_call[0][1]
        assert last_call[1]["extra"]["success"] is False
        assert last_call[1]["extra"]["error_type"] == "ValueError"

    def test_timing_context_elapsed(self, mock_logger):
        """Test timing context elapsed property."""
        from sentimatrix.core.logger import TimingContext, StructuredLogger
        import time

        structured = StructuredLogger(mock_logger)
        ctx = TimingContext(structured, "test_op", log_start=False)

        with ctx:
            time.sleep(0.01)
            during = ctx.elapsed_ms

        after = ctx.elapsed_ms

        assert during > 0
        assert after >= during


class TestAdvancedLogging:
    """Tests for advanced logging methods."""

    @pytest.fixture
    def mock_logger(self):
        """Provide a mock Python logger."""
        logger = MagicMock(spec=logging.Logger)
        logger.name = "test.logger"
        return logger

    def test_log_performance(self, mock_logger):
        """Test log_performance method."""
        from sentimatrix.core.logger import StructuredLogger

        structured = StructuredLogger(mock_logger)
        structured.log_performance("api_call", duration_ms=150.5, success=True, endpoint="/users")

        mock_logger.log.assert_called_once()
        _, kwargs = mock_logger.log.call_args
        assert kwargs["extra"]["event_type"] == "performance"
        assert kwargs["extra"]["operation"] == "api_call"
        assert kwargs["extra"]["duration_ms"] == 150.5
        assert kwargs["extra"]["success"] is True
        assert kwargs["extra"]["endpoint"] == "/users"

    def test_log_performance_failure(self, mock_logger):
        """Test log_performance with failure."""
        from sentimatrix.core.logger import StructuredLogger

        structured = StructuredLogger(mock_logger)
        structured.log_performance("db_query", duration_ms=5000, success=False)

        args, kwargs = mock_logger.log.call_args
        # Should use WARNING level for failures
        assert args[0] == logging.WARNING
        assert kwargs["extra"]["success"] is False

    def test_log_event(self, mock_logger):
        """Test log_event method."""
        from sentimatrix.core.logger import StructuredLogger

        structured = StructuredLogger(mock_logger)
        structured.log_event("scrape_started", "Started scraping Amazon", platform="amazon")

        _, kwargs = mock_logger.log.call_args
        assert kwargs["extra"]["event_type"] == "scrape_started"
        assert kwargs["extra"]["platform"] == "amazon"

    def test_log_request(self, mock_logger):
        """Test log_request method."""
        from sentimatrix.core.logger import StructuredLogger

        structured = StructuredLogger(mock_logger)
        structured.log_request("GET", "https://api.example.com/users", status_code=200, duration_ms=150)

        args, kwargs = mock_logger.log.call_args
        assert args[0] == logging.INFO  # 200 status = INFO
        assert kwargs["extra"]["method"] == "GET"
        assert kwargs["extra"]["status_code"] == 200
        assert kwargs["extra"]["success"] is True

    def test_log_request_error(self, mock_logger):
        """Test log_request with error status."""
        from sentimatrix.core.logger import StructuredLogger

        structured = StructuredLogger(mock_logger)
        structured.log_request("POST", "https://api.example.com/data", status_code=500, duration_ms=1000)

        args, _ = mock_logger.log.call_args
        assert args[0] == logging.ERROR  # 500 status = ERROR

    def test_log_error_details(self, mock_logger):
        """Test log_error_details method."""
        from sentimatrix.core.logger import StructuredLogger

        structured = StructuredLogger(mock_logger)

        try:
            raise ValueError("Test error message")
        except ValueError as e:
            structured.log_error_details(e, "Failed to process", request_id="123")

        _, kwargs = mock_logger.log.call_args
        assert kwargs["extra"]["error_type"] == "ValueError"
        assert kwargs["extra"]["error_message"] == "Test error message"
        assert kwargs["extra"]["request_id"] == "123"
        assert "traceback" in kwargs["extra"]


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_memory_usage(self):
        """Test get_memory_usage_mb returns a value."""
        from sentimatrix.core.logger import get_memory_usage_mb

        memory = get_memory_usage_mb()
        # Should return some positive value (or 0 if psutil not available)
        assert memory >= 0

    def test_timed_function_decorator(self, mock_logger=None):
        """Test timed_function decorator."""
        from sentimatrix.core.logger import timed_function, get_logger
        import time

        @timed_function(operation="test_func")
        def slow_function():
            time.sleep(0.01)
            return "result"

        result = slow_function()
        assert result == "result"
