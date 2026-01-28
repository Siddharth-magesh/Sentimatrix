"""
Unit Tests for Exception Module

Tests the exception hierarchy including:
- Base exception behavior
- Error codes and messages
- Exception serialization
- Specific exception types
"""

import pytest

from sentimatrix.core.exceptions import (
    AuthenticationError,
    CacheConnectionError,
    CacheError,
    CacheReadError,
    CacheSerializationError,
    CacheWriteError,
    CaptchaDetectedError,
    ConfigNotFoundError,
    ConfigurationError,
    ConfigValidationError,
    ConnectionTimeoutError,
    ContentFilteredError,
    DeviceError,
    ErrorCode,
    InvalidInputError,
    InvalidModelError,
    InvalidResponseError,
    LLMProviderError,
    MissingRequiredFieldError,
    ModelError,
    ModelInferenceError,
    ModelLoadError,
    ModelNotFoundError,
    PipelineError,
    PipelineStateError,
    PipelineStepError,
    PlaywrightError,
    ProviderError,
    ProviderInitializationError,
    ProviderNotFoundError,
    QuotaExceededError,
    RateLimitError,
    ReadTimeoutError,
    ScraperBlockedError,
    ScraperConnectionError,
    ScraperError,
    ScraperParseError,
    ScraperTimeoutError,
    SeleniumError,
    SentimatrixError,
    TimeoutError,
    TokenLimitExceededError,
    ValidationError,
)


class TestErrorCode:
    """Tests for ErrorCode enum."""

    def test_error_code_values(self):
        """Test error codes have expected values."""
        assert ErrorCode.UNKNOWN == 1000
        assert ErrorCode.CONFIG_NOT_FOUND == 1100
        assert ErrorCode.LLM_API_ERROR == 1400
        assert ErrorCode.SCRAPER_CONNECTION_ERROR == 1500
        assert ErrorCode.RATE_LIMIT_EXCEEDED == 1700

    def test_error_code_names(self):
        """Test error code names are accessible."""
        assert ErrorCode.UNKNOWN.name == "UNKNOWN"
        assert ErrorCode.CONFIG_NOT_FOUND.name == "CONFIG_NOT_FOUND"


class TestSentimatrixError:
    """Tests for base SentimatrixError."""

    def test_basic_initialization(self):
        """Test basic error initialization."""
        error = SentimatrixError("Something went wrong")
        assert error.message == "Something went wrong"
        assert error.code == ErrorCode.UNKNOWN
        assert error.details == {}
        assert error.original_error is None

    def test_initialization_with_all_params(self):
        """Test error initialization with all parameters."""
        original = ValueError("Original error")
        error = SentimatrixError(
            message="Test error",
            code=ErrorCode.INTERNAL,
            details={"key": "value"},
            original_error=original,
        )
        assert error.message == "Test error"
        assert error.code == ErrorCode.INTERNAL
        assert error.details == {"key": "value"}
        assert error.original_error is original

    def test_str_representation(self):
        """Test string representation of error."""
        error = SentimatrixError(
            "Test error",
            code=ErrorCode.CONFIG_PARSE_ERROR,
            details={"file": "config.yaml"},
        )
        str_repr = str(error)
        assert "CONFIG_PARSE_ERROR" in str_repr
        assert "1101" in str_repr
        assert "Test error" in str_repr
        assert "config.yaml" in str_repr

    def test_str_with_original_error(self):
        """Test string representation with original error."""
        original = ValueError("Original cause")
        error = SentimatrixError("Wrapped error", original_error=original)
        str_repr = str(error)
        assert "Caused by" in str_repr
        assert "ValueError" in str_repr

    def test_repr(self):
        """Test detailed representation."""
        error = SentimatrixError("Test", code=ErrorCode.INTERNAL)
        repr_str = repr(error)
        assert "SentimatrixError" in repr_str
        assert "message=" in repr_str

    def test_to_dict(self):
        """Test error serialization to dictionary."""
        error = SentimatrixError(
            "Test error",
            code=ErrorCode.VALIDATION_FAILED,
            details={"field": "name"},
        )
        error_dict = error.to_dict()

        assert error_dict["error_type"] == "SentimatrixError"
        assert error_dict["message"] == "Test error"
        assert error_dict["code"] == 1200
        assert error_dict["code_name"] == "VALIDATION_FAILED"
        assert error_dict["details"]["field"] == "name"

    def test_exception_inheritance(self):
        """Test that SentimatrixError inherits from Exception."""
        error = SentimatrixError("Test")
        assert isinstance(error, Exception)

        # Can be raised and caught
        with pytest.raises(SentimatrixError):
            raise error


class TestConfigurationErrors:
    """Tests for configuration-related exceptions."""

    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("Invalid config")
        assert isinstance(error, SentimatrixError)
        assert error.code == ErrorCode.CONFIG_PARSE_ERROR

    def test_config_not_found_error(self):
        """Test ConfigNotFoundError."""
        error = ConfigNotFoundError("/path/to/config.yaml")
        assert "not found" in error.message
        assert "/path/to/config.yaml" in error.message
        assert error.details["path"] == "/path/to/config.yaml"
        assert error.code == ErrorCode.CONFIG_NOT_FOUND

    def test_config_validation_error(self):
        """Test ConfigValidationError."""
        error = ConfigValidationError("temperature", "must be between 0 and 1", 1.5)
        assert "temperature" in error.message
        assert error.details["field"] == "temperature"
        assert error.details["value"] == 1.5
        assert error.code == ErrorCode.CONFIG_VALIDATION_ERROR


class TestValidationErrors:
    """Tests for validation-related exceptions."""

    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("Invalid input")
        assert isinstance(error, SentimatrixError)
        assert error.code == ErrorCode.VALIDATION_FAILED

    def test_invalid_input_error(self):
        """Test InvalidInputError."""
        error = InvalidInputError("text", "cannot be empty", "")
        assert "text" in error.message
        assert "cannot be empty" in error.message
        assert error.code == ErrorCode.INVALID_INPUT

    def test_missing_required_field_error(self):
        """Test MissingRequiredFieldError."""
        error = MissingRequiredFieldError("api_key")
        assert "api_key" in error.message
        assert error.details["field"] == "api_key"
        assert error.code == ErrorCode.MISSING_REQUIRED_FIELD


class TestProviderErrors:
    """Tests for provider-related exceptions."""

    def test_provider_error(self):
        """Test ProviderError."""
        error = ProviderError("Connection failed", provider="openai")
        assert error.provider == "openai"
        assert error.details["provider"] == "openai"

    def test_provider_not_found_error(self):
        """Test ProviderNotFoundError."""
        error = ProviderNotFoundError("custom_provider")
        assert "not found" in error.message.lower()
        assert "custom_provider" in error.message
        assert error.code == ErrorCode.PROVIDER_NOT_FOUND

    def test_provider_initialization_error(self):
        """Test ProviderInitializationError."""
        error = ProviderInitializationError("openai", "Missing API key")
        assert "openai" in error.message
        assert "Missing API key" in error.message
        assert error.code == ErrorCode.PROVIDER_INITIALIZATION_FAILED


class TestLLMProviderErrors:
    """Tests for LLM provider exceptions."""

    def test_llm_provider_error(self):
        """Test LLMProviderError."""
        error = LLMProviderError("API error", provider="openai", model="gpt-4")
        assert error.provider == "openai"
        assert error.model == "gpt-4"
        assert error.details["model"] == "gpt-4"

    def test_authentication_error(self):
        """Test AuthenticationError."""
        error = AuthenticationError("openai", "Invalid API key")
        assert error.code == ErrorCode.LLM_AUTHENTICATION_FAILED
        assert "openai" in str(error)

    def test_invalid_model_error(self):
        """Test InvalidModelError."""
        error = InvalidModelError("openai", "gpt-5-turbo")
        assert "gpt-5-turbo" in error.message
        assert error.model == "gpt-5-turbo"
        assert error.code == ErrorCode.LLM_INVALID_MODEL

    def test_content_filtered_error(self):
        """Test ContentFilteredError."""
        error = ContentFilteredError("openai", "Harmful content detected")
        assert "filtered" in error.message.lower()
        assert error.code == ErrorCode.LLM_CONTENT_FILTERED

    def test_token_limit_exceeded_error(self):
        """Test TokenLimitExceededError."""
        error = TokenLimitExceededError("openai", "gpt-4", requested=10000, limit=8192)
        assert error.details["requested_tokens"] == 10000
        assert error.details["token_limit"] == 8192
        assert error.code == ErrorCode.LLM_TOKEN_LIMIT_EXCEEDED

    def test_invalid_response_error(self):
        """Test InvalidResponseError."""
        error = InvalidResponseError("openai", "Empty response")
        assert "Invalid response" in error.message
        assert error.code == ErrorCode.LLM_INVALID_RESPONSE


class TestProviderSpecificErrors:
    """Tests for provider-specific LLM errors."""

    def test_openai_error(self):
        """Test OpenAI-specific error."""
        from sentimatrix.core.exceptions import OpenAIError
        error = OpenAIError("Rate limited", model="gpt-4")
        assert error.provider == "openai"
        assert error.model == "gpt-4"

    def test_anthropic_error(self):
        """Test Anthropic-specific error."""
        from sentimatrix.core.exceptions import AnthropicError
        error = AnthropicError("Overloaded")
        assert error.provider == "anthropic"

    def test_groq_error(self):
        """Test Groq-specific error."""
        from sentimatrix.core.exceptions import GroqError
        error = GroqError("Service unavailable")
        assert error.provider == "groq"


class TestScraperErrors:
    """Tests for scraper-related exceptions."""

    def test_scraper_error(self):
        """Test ScraperError."""
        error = ScraperError("Failed to scrape", provider="playwright", url="https://example.com")
        assert error.url == "https://example.com"
        assert error.details["url"] == "https://example.com"

    def test_scraper_connection_error(self):
        """Test ScraperConnectionError."""
        error = ScraperConnectionError("playwright", "https://example.com", "DNS resolution failed")
        assert "connect" in error.message.lower()
        assert error.code == ErrorCode.SCRAPER_CONNECTION_ERROR

    def test_scraper_timeout_error(self):
        """Test ScraperTimeoutError."""
        error = ScraperTimeoutError("playwright", "https://slow.com", 30)
        assert "30" in error.message
        assert error.details["timeout_seconds"] == 30
        assert error.code == ErrorCode.SCRAPER_TIMEOUT

    def test_scraper_blocked_error(self):
        """Test ScraperBlockedError."""
        error = ScraperBlockedError("playwright", "https://protected.com", "Access denied")
        assert "blocked" in error.message.lower()
        assert error.code == ErrorCode.SCRAPER_BLOCKED

    def test_captcha_detected_error(self):
        """Test CaptchaDetectedError."""
        error = CaptchaDetectedError("selenium", "https://captcha.com")
        assert "CAPTCHA" in error.message
        assert error.code == ErrorCode.SCRAPER_CAPTCHA_DETECTED

    def test_scraper_parse_error(self):
        """Test ScraperParseError."""
        error = ScraperParseError("playwright", "https://example.com", "Invalid HTML")
        assert "parse" in error.message.lower()
        assert error.code == ErrorCode.SCRAPER_PARSE_ERROR

    def test_playwright_error(self):
        """Test Playwright-specific error."""
        error = PlaywrightError(action="navigation", message="Browser crashed")
        assert error.provider == "playwright"
        assert "navigation" in error.message
        assert "Browser crashed" in error.message

    def test_selenium_error(self):
        """Test Selenium-specific error."""
        error = SeleniumError("WebDriver not found")
        assert error.provider == "selenium"


class TestModelErrors:
    """Tests for ML model exceptions."""

    def test_model_error(self):
        """Test ModelError."""
        error = ModelError("Inference failed", model_name="bert-base")
        assert error.model_name == "bert-base"
        assert error.details["model_name"] == "bert-base"

    def test_model_not_found_error(self):
        """Test ModelNotFoundError."""
        error = ModelNotFoundError("nonexistent-model")
        assert "not found" in error.message.lower()
        assert error.code == ErrorCode.MODEL_NOT_FOUND

    def test_model_load_error(self):
        """Test ModelLoadError."""
        error = ModelLoadError("bert-base", "Out of memory")
        assert "load" in error.message.lower()
        assert error.code == ErrorCode.MODEL_LOAD_ERROR

    def test_model_inference_error(self):
        """Test ModelInferenceError."""
        error = ModelInferenceError("bert-base", "Input too long")
        assert "inference" in error.message.lower()
        assert error.code == ErrorCode.MODEL_INFERENCE_ERROR

    def test_device_error(self):
        """Test DeviceError."""
        error = DeviceError("bert-base", "cuda", "CUDA out of memory")
        assert "cuda" in error.message
        assert error.details["device"] == "cuda"
        assert error.code == ErrorCode.MODEL_DEVICE_ERROR


class TestRateLimitErrors:
    """Tests for rate limit exceptions."""

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        error = RateLimitError("Too many requests", provider="openai", retry_after=60)
        assert error.provider == "openai"
        assert error.retry_after == 60
        assert error.details["retry_after_seconds"] == 60

    def test_quota_exceeded_error(self):
        """Test QuotaExceededError."""
        error = QuotaExceededError("openai", "tokens")
        assert "quota" in error.message.lower()
        assert error.details["quota_type"] == "tokens"
        assert error.code == ErrorCode.QUOTA_EXCEEDED


class TestTimeoutErrors:
    """Tests for timeout exceptions."""

    def test_timeout_error(self):
        """Test TimeoutError."""
        error = TimeoutError("Request timed out", timeout=30, operation="api_call")
        assert error.timeout == 30
        assert error.operation == "api_call"

    def test_connection_timeout_error(self):
        """Test ConnectionTimeoutError."""
        error = ConnectionTimeoutError("api.openai.com", 10)
        assert "api.openai.com" in str(error)
        assert error.timeout == 10
        assert error.code == ErrorCode.CONNECTION_TIMEOUT

    def test_read_timeout_error(self):
        """Test ReadTimeoutError."""
        error = ReadTimeoutError("https://slow.api.com/endpoint", 60)
        assert error.timeout == 60
        assert error.code == ErrorCode.READ_TIMEOUT


class TestCacheErrors:
    """Tests for cache exceptions."""

    def test_cache_error(self):
        """Test CacheError."""
        error = CacheError("Cache operation failed", backend="memory")
        assert error.backend == "memory"
        assert error.details["backend"] == "memory"

    def test_cache_connection_error(self):
        """Test CacheConnectionError."""
        error = CacheConnectionError("redis", "Connection refused")
        assert "redis" in error.message
        assert error.code == ErrorCode.CACHE_CONNECTION_ERROR

    def test_cache_read_error(self):
        """Test CacheReadError."""
        error = CacheReadError("memory", "user:123", "Key corrupted")
        assert error.details["key"] == "user:123"
        assert error.code == ErrorCode.CACHE_READ_ERROR

    def test_cache_write_error(self):
        """Test CacheWriteError."""
        error = CacheWriteError("redis", "session:456", "Serialization failed")
        assert error.details["key"] == "session:456"
        assert error.code == ErrorCode.CACHE_WRITE_ERROR

    def test_cache_serialization_error(self):
        """Test CacheSerializationError."""
        error = CacheSerializationError("memory", "serialize", "Cannot pickle lambda")
        assert error.details["operation"] == "serialize"
        assert error.code == ErrorCode.CACHE_SERIALIZATION_ERROR


class TestPipelineErrors:
    """Tests for pipeline exceptions."""

    def test_pipeline_error(self):
        """Test PipelineError."""
        error = PipelineError("Pipeline failed", pipeline_name="sentiment", step_name="analyze")
        assert error.pipeline_name == "sentiment"
        assert error.step_name == "analyze"

    def test_pipeline_step_error(self):
        """Test PipelineStepError."""
        original = ValueError("Division by zero")
        error = PipelineStepError("normalize", "Invalid input data", original_error=original)
        assert error.step_name == "normalize"
        assert error.original_error is original
        assert error.code == ErrorCode.PIPELINE_STEP_FAILED

    def test_pipeline_state_error(self):
        """Test PipelineStateError."""
        error = PipelineStateError("main", "stopped", "running")
        assert "stopped" in error.message
        assert "running" in error.message
        assert error.code == ErrorCode.PIPELINE_INVALID_STATE
