"""
Sentimatrix V2 - Advanced Sentiment Analysis Toolkit

A comprehensive sentiment analysis library with multi-provider LLM support,
web scraping capabilities, and emotion detection.

Example:
    >>> from sentimatrix import Sentimatrix
    >>>
    >>> async with Sentimatrix() as sm:
    ...     # Quick sentiment analysis
    ...     result = await sm.analyze_sentiment("This product is amazing!")
    ...     print(result.sentiment)  # "positive"
    ...
    ...     # Full analysis with emotions
    ...     analysis = await sm.analyze("I love this!")
    ...     print(analysis.sentiment.sentiment)  # "positive"
    ...     print(analysis.emotions.primary_emotion.label)  # "joy"
    ...
    ...     # Scrape and analyze reviews
    ...     reviews = await sm.scrape_amazon("B08N5WRWNW", limit=50)
    ...     result = await sm.analyze_reviews(reviews)
    ...     print(f"Positive: {result.positive_ratio:.1%}")
"""

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
    ProviderError,
    RateLimitError,
    ScraperError,
    SentimatrixError,
    ValidationError,
)
from sentimatrix.core.logger import LogManager, get_logger
from sentimatrix.main import (
    Sentimatrix,
    AnalysisResult,
    ReviewAnalysisResult,
    InsightsResult,
    ComparisonResult,
    create_sentimatrix,
)

__version__ = "0.2.0"
__author__ = "Sentimatrix Team"
__all__ = [
    # Version
    "__version__",
    # Main class
    "Sentimatrix",
    "create_sentimatrix",
    # Result types
    "AnalysisResult",
    "ReviewAnalysisResult",
    "InsightsResult",
    "ComparisonResult",
    # Config classes
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
    "ScraperError",
    "ValidationError",
    "CacheError",
    "RateLimitError",
    # Logger
    "LogManager",
    "get_logger",
]
