"""
Sentimatrix Core Module

Contains foundational components:
- Configuration management
- Logging infrastructure
- Exception hierarchy
- Cache layer
- Pipeline orchestration
"""

from sentimatrix.core.cache import CacheManager, MemoryCache
from sentimatrix.core.config import (
    CacheConfig,
    LLMConfig,
    LogConfig,
    ModelConfig,
    ProxyConfig,
    RateLimitConfig,
    RetryConfig,
    ScraperConfig,
    SentimatrixConfig,
)
from sentimatrix.core.exceptions import (
    CacheError,
    ConfigurationError,
    LLMProviderError,
    ModelError,
    PipelineError,
    PipelineStateError,
    PipelineStepError,
    ProviderError,
    RateLimitError,
    ScraperError,
    SentimatrixError,
    TimeoutError,
    ValidationError,
)
from sentimatrix.core.logger import LogManager, StructuredLogger, get_logger
from sentimatrix.core.pipeline import (
    Pipeline,
    PipelineContext,
    PipelineResult,
    PipelineState,
    PipelineStep,
    FunctionStep,
    ParallelSteps,
    ConditionalStep,
    StepConfig,
    StepResult,
    StepState,
    create_pipeline,
)

__all__ = [
    # Config
    "SentimatrixConfig",
    "LLMConfig",
    "ScraperConfig",
    "ModelConfig",
    "CacheConfig",
    "LogConfig",
    "ProxyConfig",
    "RateLimitConfig",
    "RetryConfig",
    # Exceptions
    "SentimatrixError",
    "ConfigurationError",
    "ProviderError",
    "LLMProviderError",
    "ScraperError",
    "ModelError",
    "ValidationError",
    "CacheError",
    "RateLimitError",
    "TimeoutError",
    "PipelineError",
    "PipelineStateError",
    "PipelineStepError",
    # Logger
    "LogManager",
    "StructuredLogger",
    "get_logger",
    # Cache
    "CacheManager",
    "MemoryCache",
    # Pipeline
    "Pipeline",
    "PipelineContext",
    "PipelineResult",
    "PipelineState",
    "PipelineStep",
    "FunctionStep",
    "ParallelSteps",
    "ConditionalStep",
    "StepConfig",
    "StepResult",
    "StepState",
    "create_pipeline",
]
