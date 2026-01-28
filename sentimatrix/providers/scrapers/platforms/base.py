"""
Base Platform Scraper

Abstract base class for platform-specific scrapers with common functionality:
- URL validation and normalization
- Rate limiting integration
- Retry logic with fallback methods
- Review extraction with common interface
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Awaitable, TypeVar

from sentimatrix.core.config import ScraperConfig
from sentimatrix.core.exceptions import (
    ScraperError,
    ScraperParseError,
    ValidationError,
)
from sentimatrix.providers.base import (
    BaseProvider,
    ProviderInfo,
    ProviderType,
    ProviderCapabilities,
    Review,
)
from sentimatrix.providers.scrapers.rate_limiter import RateLimiter
from sentimatrix.providers.scrapers.utils import RetryHandler


class SortOrder(str, Enum):
    """Sort order for reviews."""

    RECENT = "recent"
    HELPFUL = "helpful"
    RATING_HIGH = "rating_high"
    RATING_LOW = "rating_low"
    RELEVANCE = "relevance"


class ReviewFilter(str, Enum):
    """Filter for reviews."""

    ALL = "all"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    CRITICAL = "critical"
    VERIFIED = "verified"


@dataclass
class PlatformConfig:
    """Configuration for platform scrapers."""

    # Rate limiting
    requests_per_second: float = 1.0
    burst_size: int = 5

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0

    # Scraping settings
    timeout: int = 30
    headless: bool = True

    # Platform-specific
    country: str = "us"
    language: str = "en"

    # API keys (optional)
    api_key: Optional[str] = None

    def to_scraper_config(self) -> ScraperConfig:
        """Convert to ScraperConfig."""
        return ScraperConfig(
            timeout=self.timeout,
            headless=self.headless,
        )


@dataclass
class ProductInfo:
    """Information about a product/item."""

    id: str
    name: str
    platform: str
    url: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = None
    rating: Optional[float] = None
    review_count: Optional[int] = None
    image_url: Optional[str] = None
    category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "platform": self.platform,
            "url": self.url,
            "description": self.description,
            "price": self.price,
            "currency": self.currency,
            "rating": self.rating,
            "review_count": self.review_count,
            "image_url": self.image_url,
            "category": self.category,
            "metadata": self.metadata,
        }


T = TypeVar("T")


class BasePlatformScraper(BaseProvider, ABC):
    """
    Abstract base class for platform-specific scrapers.

    Provides common functionality:
    - URL validation and ID extraction
    - Rate limiting
    - Retry logic with fallback
    - Review extraction interface
    """

    def __init__(
        self,
        config: Optional[PlatformConfig] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        """
        Initialize platform scraper.

        Args:
            config: Platform-specific configuration
            rate_limiter: Optional rate limiter (created if not provided)
        """
        self._platform_config = config or PlatformConfig()
        super().__init__(self._platform_config.to_scraper_config())

        self._rate_limiter = rate_limiter or RateLimiter(
            requests_per_second=self._platform_config.requests_per_second,
            burst_size=self._platform_config.burst_size,
        )

        self._retry_handler = RetryHandler(
            max_retries=self._platform_config.max_retries,
            initial_delay=self._platform_config.retry_delay,
        )

        self._scraper = None  # Underlying scraper (HTTPX or Playwright)

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Get the platform name (e.g., 'amazon', 'steam')."""
        pass

    @property
    @abstractmethod
    def platform_domain(self) -> str:
        """Get the platform domain (e.g., 'amazon.com')."""
        pass

    @property
    def rate_limiter(self) -> RateLimiter:
        """Get the rate limiter."""
        return self._rate_limiter

    @abstractmethod
    def validate_url(self, url: str) -> bool:
        """
        Validate if URL belongs to this platform.

        Args:
            url: URL to validate

        Returns:
            True if URL is valid for this platform
        """
        pass

    @abstractmethod
    def extract_id(self, url: str) -> Optional[str]:
        """
        Extract the product/item ID from a URL.

        Args:
            url: URL to extract ID from

        Returns:
            Extracted ID or None if not found
        """
        pass

    @abstractmethod
    async def scrape_reviews(
        self,
        identifier: str,
        limit: int = 100,
        sort_by: SortOrder = SortOrder.RECENT,
        filter_by: ReviewFilter = ReviewFilter.ALL,
        **kwargs: Any,
    ) -> List[Review]:
        """
        Scrape reviews from the platform.

        Args:
            identifier: Product/item ID or URL
            limit: Maximum number of reviews to fetch
            sort_by: Sort order for reviews
            filter_by: Filter for reviews
            **kwargs: Platform-specific parameters

        Returns:
            List of Review objects
        """
        pass

    @abstractmethod
    async def get_product_info(self, identifier: str) -> ProductInfo:
        """
        Get product/item information.

        Args:
            identifier: Product/item ID or URL

        Returns:
            ProductInfo object
        """
        pass

    async def scrape_reviews_from_url(
        self,
        url: str,
        limit: int = 100,
        **kwargs: Any,
    ) -> List[Review]:
        """
        Scrape reviews from a URL.

        Args:
            url: URL to scrape
            limit: Maximum number of reviews
            **kwargs: Additional parameters

        Returns:
            List of Review objects

        Raises:
            ValidationError: If URL is not valid for this platform
        """
        if not self.validate_url(url):
            raise ValidationError(
                f"Invalid URL for {self.platform_name}: {url}"
            )

        identifier = self.extract_id(url)
        if not identifier:
            raise ValidationError(
                f"Could not extract ID from URL: {url}"
            )

        return await self.scrape_reviews(identifier, limit=limit, **kwargs)

    async def scrape_with_fallback(
        self,
        identifier: str,
        methods: List[Callable[..., Awaitable[List[Review]]]],
        limit: int = 100,
        **kwargs: Any,
    ) -> List[Review]:
        """
        Try multiple scraping methods with fallback.

        Args:
            identifier: Product/item ID
            methods: List of async methods to try in order
            limit: Maximum number of reviews
            **kwargs: Additional parameters

        Returns:
            List of Review objects from first successful method

        Raises:
            ScraperError: If all methods fail
        """
        last_error: Optional[Exception] = None

        for method in methods:
            try:
                return await method(identifier, limit=limit, **kwargs)
            except Exception as e:
                last_error = e
                continue

        raise ScraperError(
            f"All scraping methods failed for {self.platform_name}",
            provider=self.platform_name,
            original_error=last_error,
        )

    def generate_review_id(
        self,
        platform: str,
        text: str,
        author: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> str:
        """
        Generate a unique review ID.

        Args:
            platform: Platform name
            text: Review text
            author: Review author
            timestamp: Review timestamp

        Returns:
            Unique hash-based ID
        """
        components = [platform, text[:100]]
        if author:
            components.append(author)
        if timestamp:
            components.append(timestamp.isoformat())

        content = "|".join(components)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def normalize_rating(
        self,
        rating: float,
        min_rating: float,
        max_rating: float,
        target_max: float = 5.0,
    ) -> float:
        """
        Normalize rating to a standard scale.

        Args:
            rating: Original rating
            min_rating: Minimum possible rating
            max_rating: Maximum possible rating
            target_max: Target maximum (default 5.0)

        Returns:
            Normalized rating
        """
        if max_rating == min_rating:
            return target_max / 2

        normalized = (rating - min_rating) / (max_rating - min_rating) * target_max
        return round(normalized, 2)

    def parse_date(
        self,
        date_str: str,
        formats: Optional[List[str]] = None,
    ) -> Optional[datetime]:
        """
        Parse a date string with multiple format attempts.

        Args:
            date_str: Date string to parse
            formats: List of date formats to try

        Returns:
            Parsed datetime or None
        """
        if not date_str:
            return None

        # Clean up the string
        date_str = date_str.strip()

        # Default formats
        if formats is None:
            formats = [
                "%Y-%m-%d",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%B %d, %Y",
                "%b %d, %Y",
                "%d %B %Y",
                "%d %b %Y",
                "%m/%d/%Y",
                "%d/%m/%Y",
            ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        return None

    def clean_text(self, text: str) -> str:
        """
        Clean review text.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    async def _rate_limited_request(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute a function with rate limiting.

        Args:
            func: Async function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        await self._rate_limiter.acquire(domain=self.platform_domain)
        return await func(*args, **kwargs)
