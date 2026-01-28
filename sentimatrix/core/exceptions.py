"""
Sentimatrix Exception Hierarchy

Defines a comprehensive exception hierarchy for error handling throughout
the application. All exceptions inherit from SentimatrixError.

Exception Hierarchy:
    SentimatrixError (base)
    ├── ConfigurationError
    ├── ValidationError
    ├── ProviderError
    │   ├── LLMProviderError
    │   │   ├── OpenAIError
    │   │   ├── AnthropicError
    │   │   ├── GroqError
    │   │   └── ...
    │   ├── ScraperError
    │   │   ├── PlaywrightError
    │   │   ├── SeleniumError
    │   │   └── ...
    │   └── ModelError
    ├── RateLimitError
    ├── TimeoutError
    ├── CacheError
    └── PipelineError
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, Optional


class ErrorCode(IntEnum):
    """Standardized error codes for Sentimatrix errors."""

    # General errors (1000-1099)
    UNKNOWN = 1000
    INTERNAL = 1001
    NOT_IMPLEMENTED = 1002

    # Configuration errors (1100-1199)
    CONFIG_NOT_FOUND = 1100
    CONFIG_PARSE_ERROR = 1101
    CONFIG_VALIDATION_ERROR = 1102
    CONFIG_MISSING_REQUIRED = 1103

    # Validation errors (1200-1299)
    VALIDATION_FAILED = 1200
    INVALID_INPUT = 1201
    INVALID_FORMAT = 1202
    MISSING_REQUIRED_FIELD = 1203

    # Provider errors (1300-1399)
    PROVIDER_NOT_FOUND = 1300
    PROVIDER_NOT_AVAILABLE = 1301
    PROVIDER_INITIALIZATION_FAILED = 1302

    # LLM errors (1400-1499)
    LLM_API_ERROR = 1400
    LLM_AUTHENTICATION_FAILED = 1401
    LLM_INVALID_MODEL = 1402
    LLM_CONTENT_FILTERED = 1403
    LLM_TOKEN_LIMIT_EXCEEDED = 1404
    LLM_INVALID_RESPONSE = 1405

    # Scraper errors (1500-1599)
    SCRAPER_CONNECTION_ERROR = 1500
    SCRAPER_TIMEOUT = 1501
    SCRAPER_BLOCKED = 1502
    SCRAPER_PARSE_ERROR = 1503
    SCRAPER_CAPTCHA_DETECTED = 1504
    SCRAPER_RATE_LIMITED = 1505
    SCRAPER_HTTP_ERROR = 1506
    SCRAPER_PROXY_ERROR = 1507

    # Model errors (1600-1699)
    MODEL_NOT_FOUND = 1600
    MODEL_LOAD_ERROR = 1601
    MODEL_INFERENCE_ERROR = 1602
    MODEL_DEVICE_ERROR = 1603

    # Rate limit errors (1700-1799)
    RATE_LIMIT_EXCEEDED = 1700
    QUOTA_EXCEEDED = 1701
    CONCURRENT_LIMIT_EXCEEDED = 1702

    # Timeout errors (1800-1899)
    REQUEST_TIMEOUT = 1800
    CONNECTION_TIMEOUT = 1801
    READ_TIMEOUT = 1802

    # Cache errors (1900-1999)
    CACHE_CONNECTION_ERROR = 1900
    CACHE_READ_ERROR = 1901
    CACHE_WRITE_ERROR = 1902
    CACHE_SERIALIZATION_ERROR = 1903

    # Pipeline errors (2000-2099)
    PIPELINE_EXECUTION_ERROR = 2000
    PIPELINE_STEP_FAILED = 2001
    PIPELINE_INVALID_STATE = 2002


class SentimatrixError(Exception):
    """
    Base exception for all Sentimatrix errors.

    Attributes:
        message: Human-readable error message
        code: Standardized error code
        details: Additional error details/context
        original_error: Original exception that caused this error
    """

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        """
        Initialize a SentimatrixError.

        Args:
            message: Human-readable error message
            code: Error code from ErrorCode enum
            details: Additional error context
            original_error: Original exception if wrapping another error
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.original_error = original_error

    def __str__(self) -> str:
        """Return string representation of the error."""
        base = f"[{self.code.name}({self.code.value})] {self.message}"
        if self.details:
            base += f" | Details: {self.details}"
        if self.original_error:
            base += f" | Caused by: {type(self.original_error).__name__}: {self.original_error}"
        return base

    def __repr__(self) -> str:
        """Return detailed representation of the error."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"code={self.code}, "
            f"details={self.details!r})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "code": self.code.value,
            "code_name": self.code.name,
            "details": self.details,
            "original_error": str(self.original_error) if self.original_error else None,
        }


# Configuration Errors


class ConfigurationError(SentimatrixError):
    """Raised when there's an error in configuration."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.CONFIG_PARSE_ERROR,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, code, details, original_error)


class ConfigNotFoundError(ConfigurationError):
    """Raised when configuration file is not found."""

    def __init__(self, path: str) -> None:
        super().__init__(
            f"Configuration file not found: {path}",
            code=ErrorCode.CONFIG_NOT_FOUND,
            details={"path": path},
        )


class ConfigValidationError(ConfigurationError):
    """Raised when configuration validation fails."""

    def __init__(self, field: str, message: str, value: Any = None) -> None:
        super().__init__(
            f"Configuration validation error for '{field}': {message}",
            code=ErrorCode.CONFIG_VALIDATION_ERROR,
            details={"field": field, "value": value, "validation_message": message},
        )


# Validation Errors


class ValidationError(SentimatrixError):
    """Raised when input validation fails."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.VALIDATION_FAILED,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, code, details, original_error)


class InvalidInputError(ValidationError):
    """Raised when input data is invalid."""

    def __init__(self, field: str, message: str, value: Any = None) -> None:
        super().__init__(
            f"Invalid input for '{field}': {message}",
            code=ErrorCode.INVALID_INPUT,
            details={"field": field, "value": repr(value)[:100] if value else None},
        )


class MissingRequiredFieldError(ValidationError):
    """Raised when a required field is missing."""

    def __init__(self, field: str) -> None:
        super().__init__(
            f"Missing required field: {field}",
            code=ErrorCode.MISSING_REQUIRED_FIELD,
            details={"field": field},
        )


# Provider Errors


class ProviderError(SentimatrixError):
    """Base class for provider-related errors."""

    def __init__(
        self,
        message: str,
        provider: str,
        code: ErrorCode = ErrorCode.PROVIDER_NOT_AVAILABLE,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        details = details or {}
        details["provider"] = provider
        super().__init__(message, code, details, original_error)
        self.provider = provider


class ProviderNotFoundError(ProviderError):
    """Raised when requested provider is not found."""

    def __init__(self, provider: str) -> None:
        super().__init__(
            f"Provider not found: {provider}",
            provider=provider,
            code=ErrorCode.PROVIDER_NOT_FOUND,
        )


class ProviderInitializationError(ProviderError):
    """Raised when provider initialization fails."""

    def __init__(self, provider: str, reason: str) -> None:
        super().__init__(
            f"Failed to initialize provider '{provider}': {reason}",
            provider=provider,
            code=ErrorCode.PROVIDER_INITIALIZATION_FAILED,
            details={"reason": reason},
        )


# LLM Provider Errors


class LLMProviderError(ProviderError):
    """Base class for LLM provider errors."""

    def __init__(
        self,
        message: str,
        provider: str,
        model: Optional[str] = None,
        code: ErrorCode = ErrorCode.LLM_API_ERROR,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        details = details or {}
        if model:
            details["model"] = model
        super().__init__(message, provider, code, details, original_error)
        self.model = model


class AuthenticationError(LLMProviderError):
    """Raised when API authentication fails."""

    def __init__(self, provider: str, message: str = "Authentication failed") -> None:
        super().__init__(
            message,
            provider=provider,
            code=ErrorCode.LLM_AUTHENTICATION_FAILED,
        )


class InvalidModelError(LLMProviderError):
    """Raised when specified model is invalid or unavailable."""

    def __init__(self, provider: str, model: str) -> None:
        super().__init__(
            f"Invalid or unavailable model: {model}",
            provider=provider,
            model=model,
            code=ErrorCode.LLM_INVALID_MODEL,
        )


class ContentFilteredError(LLMProviderError):
    """Raised when content is filtered by provider's safety systems."""

    def __init__(self, provider: str, reason: Optional[str] = None) -> None:
        message = "Content was filtered by safety systems"
        if reason:
            message += f": {reason}"
        super().__init__(
            message,
            provider=provider,
            code=ErrorCode.LLM_CONTENT_FILTERED,
            details={"reason": reason} if reason else None,
        )


class TokenLimitExceededError(LLMProviderError):
    """Raised when token limit is exceeded."""

    def __init__(
        self, provider: str, model: str, requested: int, limit: int
    ) -> None:
        super().__init__(
            f"Token limit exceeded: requested {requested}, limit {limit}",
            provider=provider,
            model=model,
            code=ErrorCode.LLM_TOKEN_LIMIT_EXCEEDED,
            details={"requested_tokens": requested, "token_limit": limit},
        )


class InvalidResponseError(LLMProviderError):
    """Raised when LLM response is invalid or malformed."""

    def __init__(self, provider: str, reason: str) -> None:
        super().__init__(
            f"Invalid response from LLM: {reason}",
            provider=provider,
            code=ErrorCode.LLM_INVALID_RESPONSE,
            details={"reason": reason},
        )


# Provider-specific LLM errors


class OpenAIError(LLMProviderError):
    """OpenAI-specific error."""

    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        code: ErrorCode = ErrorCode.LLM_API_ERROR,
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, "openai", model, code, original_error=original_error)


class AnthropicError(LLMProviderError):
    """Anthropic-specific error."""

    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        code: ErrorCode = ErrorCode.LLM_API_ERROR,
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, "anthropic", model, code, original_error=original_error)


class GroqError(LLMProviderError):
    """Groq-specific error."""

    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        code: ErrorCode = ErrorCode.LLM_API_ERROR,
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, "groq", model, code, original_error=original_error)


class GeminiError(LLMProviderError):
    """Google Gemini-specific error."""

    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        code: ErrorCode = ErrorCode.LLM_API_ERROR,
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, "gemini", model, code, original_error=original_error)


class OllamaError(LLMProviderError):
    """Ollama-specific error."""

    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        code: ErrorCode = ErrorCode.LLM_API_ERROR,
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, "ollama", model, code, original_error=original_error)


# Scraper Errors


class ScraperError(ProviderError):
    """Base class for scraper errors."""

    def __init__(
        self,
        message: str,
        provider: str,
        url: Optional[str] = None,
        code: ErrorCode = ErrorCode.SCRAPER_CONNECTION_ERROR,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        details = details or {}
        if url:
            details["url"] = url
        super().__init__(message, provider, code, details, original_error)
        self.url = url


class ScraperConnectionError(ScraperError):
    """Raised when scraper fails to connect."""

    def __init__(self, provider: str, url: str, reason: str) -> None:
        super().__init__(
            f"Failed to connect to {url}: {reason}",
            provider=provider,
            url=url,
            code=ErrorCode.SCRAPER_CONNECTION_ERROR,
            details={"reason": reason},
        )


class ScraperTimeoutError(ScraperError):
    """Raised when scraper operation times out."""

    def __init__(self, provider: str, url: str, timeout: int) -> None:
        super().__init__(
            f"Scraper timeout after {timeout}s for {url}",
            provider=provider,
            url=url,
            code=ErrorCode.SCRAPER_TIMEOUT,
            details={"timeout_seconds": timeout},
        )


class ScraperBlockedError(ScraperError):
    """Raised when scraper is blocked by the target site."""

    def __init__(self, provider: str, url: str, reason: Optional[str] = None) -> None:
        message = f"Scraper blocked at {url}"
        if reason:
            message += f": {reason}"
        super().__init__(
            message,
            provider=provider,
            url=url,
            code=ErrorCode.SCRAPER_BLOCKED,
            details={"reason": reason} if reason else None,
        )


class CaptchaDetectedError(ScraperError):
    """Raised when CAPTCHA is detected."""

    def __init__(self, provider: str, url: str) -> None:
        super().__init__(
            f"CAPTCHA detected at {url}",
            provider=provider,
            url=url,
            code=ErrorCode.SCRAPER_CAPTCHA_DETECTED,
        )


class ScraperParseError(ScraperError):
    """Raised when content parsing fails."""

    def __init__(self, provider: str, url: str, reason: str) -> None:
        super().__init__(
            f"Failed to parse content from {url}: {reason}",
            provider=provider,
            url=url,
            code=ErrorCode.SCRAPER_PARSE_ERROR,
            details={"reason": reason},
        )


# Provider-specific scraper errors


class HTTPError(ScraperError):
    """HTTP status code error."""

    def __init__(
        self,
        url: str,
        status_code: int,
        message: Optional[str] = None,
    ) -> None:
        msg = message or f"HTTP {status_code} error"
        super().__init__(
            f"{msg} for {url}",
            provider="httpx",
            url=url,
            code=ErrorCode.SCRAPER_HTTP_ERROR,
            details={"status_code": status_code},
        )
        self.status_code = status_code


class ProxyError(ScraperError):
    """Proxy-related error."""

    def __init__(
        self,
        message: str,
        proxy_url: Optional[str] = None,
        url: Optional[str] = None,
    ) -> None:
        super().__init__(
            message,
            provider="proxy",
            url=url,
            code=ErrorCode.SCRAPER_PROXY_ERROR,
            details={"proxy_url": proxy_url} if proxy_url else None,
        )
        self.proxy_url = proxy_url


class PlaywrightError(ScraperError):
    """Playwright-specific error."""

    def __init__(
        self,
        action: str,
        message: str,
        url: Optional[str] = None,
        code: ErrorCode = ErrorCode.SCRAPER_CONNECTION_ERROR,
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            f"Playwright {action} error: {message}",
            "playwright",
            url,
            code,
            details={"action": action},
            original_error=original_error,
        )
        self.action = action


class SeleniumError(ScraperError):
    """Selenium-specific error."""

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        code: ErrorCode = ErrorCode.SCRAPER_CONNECTION_ERROR,
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, "selenium", url, code, original_error=original_error)


# Model Errors


class ModelError(ProviderError):
    """Base class for ML model errors."""

    def __init__(
        self,
        message: str,
        model_name: str,
        code: ErrorCode = ErrorCode.MODEL_INFERENCE_ERROR,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        details = details or {}
        details["model_name"] = model_name
        super().__init__(message, "model", code, details, original_error)
        self.model_name = model_name


class ModelNotFoundError(ModelError):
    """Raised when model cannot be found or loaded."""

    def __init__(self, model_name: str) -> None:
        super().__init__(
            f"Model not found: {model_name}",
            model_name=model_name,
            code=ErrorCode.MODEL_NOT_FOUND,
        )


class ModelLoadError(ModelError):
    """Raised when model fails to load."""

    def __init__(self, model_name: str, reason: str) -> None:
        super().__init__(
            f"Failed to load model '{model_name}': {reason}",
            model_name=model_name,
            code=ErrorCode.MODEL_LOAD_ERROR,
            details={"reason": reason},
        )


class ModelInferenceError(ModelError):
    """Raised when model inference fails."""

    def __init__(self, model_name: str, reason: str) -> None:
        super().__init__(
            f"Inference error in model '{model_name}': {reason}",
            model_name=model_name,
            code=ErrorCode.MODEL_INFERENCE_ERROR,
            details={"reason": reason},
        )


class DeviceError(ModelError):
    """Raised when there's a device-related error."""

    def __init__(self, model_name: str, device: str, reason: str) -> None:
        super().__init__(
            f"Device error for model '{model_name}' on {device}: {reason}",
            model_name=model_name,
            code=ErrorCode.MODEL_DEVICE_ERROR,
            details={"device": device, "reason": reason},
        )


# Rate Limit Errors


class RateLimitError(SentimatrixError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        provider: str,
        retry_after: Optional[int] = None,
        code: ErrorCode = ErrorCode.RATE_LIMIT_EXCEEDED,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        details = details or {}
        details["provider"] = provider
        if retry_after:
            details["retry_after_seconds"] = retry_after
        super().__init__(message, code, details)
        self.provider = provider
        self.retry_after = retry_after


class QuotaExceededError(RateLimitError):
    """Raised when API quota is exceeded."""

    def __init__(self, provider: str, quota_type: str = "requests") -> None:
        super().__init__(
            f"API quota exceeded for {provider}: {quota_type}",
            provider=provider,
            code=ErrorCode.QUOTA_EXCEEDED,
            details={"quota_type": quota_type},
        )


# Timeout Errors


class TimeoutError(SentimatrixError):
    """Raised when an operation times out."""

    def __init__(
        self,
        message: str,
        timeout: int,
        operation: str,
        code: ErrorCode = ErrorCode.REQUEST_TIMEOUT,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        details = details or {}
        details["timeout_seconds"] = timeout
        details["operation"] = operation
        super().__init__(message, code, details)
        self.timeout = timeout
        self.operation = operation


class ConnectionTimeoutError(TimeoutError):
    """Raised when connection times out."""

    def __init__(self, host: str, timeout: int) -> None:
        super().__init__(
            f"Connection timeout to {host} after {timeout}s",
            timeout=timeout,
            operation="connection",
            code=ErrorCode.CONNECTION_TIMEOUT,
            details={"host": host},
        )


class ReadTimeoutError(TimeoutError):
    """Raised when read operation times out."""

    def __init__(self, url: str, timeout: int) -> None:
        super().__init__(
            f"Read timeout from {url} after {timeout}s",
            timeout=timeout,
            operation="read",
            code=ErrorCode.READ_TIMEOUT,
            details={"url": url},
        )


# Cache Errors


class CacheError(SentimatrixError):
    """Base class for cache errors."""

    def __init__(
        self,
        message: str,
        backend: str,
        code: ErrorCode = ErrorCode.CACHE_CONNECTION_ERROR,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        details = details or {}
        details["backend"] = backend
        super().__init__(message, code, details, original_error)
        self.backend = backend


class CacheConnectionError(CacheError):
    """Raised when cache connection fails."""

    def __init__(self, backend: str, reason: str) -> None:
        super().__init__(
            f"Failed to connect to {backend} cache: {reason}",
            backend=backend,
            code=ErrorCode.CACHE_CONNECTION_ERROR,
            details={"reason": reason},
        )


class CacheReadError(CacheError):
    """Raised when cache read fails."""

    def __init__(self, backend: str, key: str, reason: str) -> None:
        super().__init__(
            f"Failed to read from {backend} cache: {reason}",
            backend=backend,
            code=ErrorCode.CACHE_READ_ERROR,
            details={"key": key, "reason": reason},
        )


class CacheWriteError(CacheError):
    """Raised when cache write fails."""

    def __init__(self, backend: str, key: str, reason: str) -> None:
        super().__init__(
            f"Failed to write to {backend} cache: {reason}",
            backend=backend,
            code=ErrorCode.CACHE_WRITE_ERROR,
            details={"key": key, "reason": reason},
        )


class CacheSerializationError(CacheError):
    """Raised when cache serialization fails."""

    def __init__(self, backend: str, operation: str, reason: str) -> None:
        super().__init__(
            f"Cache serialization error during {operation}: {reason}",
            backend=backend,
            code=ErrorCode.CACHE_SERIALIZATION_ERROR,
            details={"operation": operation, "reason": reason},
        )


# Pipeline Errors


class PipelineError(SentimatrixError):
    """Base class for pipeline errors."""

    def __init__(
        self,
        message: str,
        pipeline_name: Optional[str] = None,
        step_name: Optional[str] = None,
        code: ErrorCode = ErrorCode.PIPELINE_EXECUTION_ERROR,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        details = details or {}
        if pipeline_name:
            details["pipeline_name"] = pipeline_name
        if step_name:
            details["step_name"] = step_name
        super().__init__(message, code, details, original_error)
        self.pipeline_name = pipeline_name
        self.step_name = step_name


class PipelineStepError(PipelineError):
    """Raised when a pipeline step fails."""

    def __init__(
        self,
        step_name: str,
        reason: str,
        pipeline_name: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            f"Pipeline step '{step_name}' failed: {reason}",
            pipeline_name=pipeline_name,
            step_name=step_name,
            code=ErrorCode.PIPELINE_STEP_FAILED,
            details={"reason": reason},
            original_error=original_error,
        )


class PipelineStateError(PipelineError):
    """Raised when pipeline is in an invalid state."""

    def __init__(self, pipeline_name: str, state: str, expected: str) -> None:
        super().__init__(
            f"Pipeline '{pipeline_name}' in invalid state: {state} (expected: {expected})",
            pipeline_name=pipeline_name,
            code=ErrorCode.PIPELINE_INVALID_STATE,
            details={"current_state": state, "expected_state": expected},
        )
