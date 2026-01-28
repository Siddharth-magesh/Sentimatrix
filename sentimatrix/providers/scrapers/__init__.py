"""
Sentimatrix Scraper Providers

Provider implementations for web scraping:
- HTTPXScraper: Async HTTP client for static content
- PlaywrightScraper: Browser automation for JS-rendered content

Platform-specific scrapers:
- AmazonScraper: Amazon product reviews
- SteamScraper: Steam game reviews
- YouTubeScraper: YouTube video comments
- RedditScraper: Reddit posts and comments

Also includes utilities:
- RateLimiter: Token bucket and other rate limiting strategies
- ProxyManager: Proxy rotation and health tracking
- UserAgentRotator: User agent rotation for anti-detection
- RetryHandler: Retry logic with exponential backoff

Usage:
    >>> from sentimatrix.providers.scrapers import HTTPXScraper, PlaywrightScraper
    >>> from sentimatrix.providers.scrapers.platforms import AmazonScraper
    >>> from sentimatrix.core.config import ScraperConfig

    # HTTP scraping (static content)
    >>> config = ScraperConfig(timeout=30)
    >>> async with HTTPXScraper(config) as scraper:
    ...     content = await scraper.scrape("https://example.com")
    ...     print(content.title)

    # Browser scraping (JavaScript content)
    >>> async with PlaywrightScraper(config) as scraper:
    ...     content = await scraper.scrape("https://spa-app.com")
    ...     await scraper.screenshot("https://spa-app.com", "shot.png")

    # Platform scraping
    >>> async with AmazonScraper() as scraper:
    ...     reviews = await scraper.scrape_reviews("B08N5WRWNW", limit=50)
"""

from sentimatrix.providers.scrapers.rate_limiter import (
    RateLimiter,
    RateLimitStrategy,
    RateLimitStats,
    TokenBucketLimiter,
    FixedWindowLimiter,
    SlidingWindowLimiter,
    create_rate_limiter,
)
from sentimatrix.providers.scrapers.utils import (
    ProxyManager,
    ProxyInfo,
    ProxyProtocol,
    RotationStrategy,
    UserAgentRotator,
    DeviceType,
    RetryHandler,
    extract_domain,
    normalize_url,
    parse_cookies,
)
from sentimatrix.providers.scrapers.httpx_scraper import HTTPXScraper
from sentimatrix.providers.scrapers.playwright_scraper import (
    PlaywrightScraper,
    BrowserType,
    WaitStrategy,
    PageAction,
)

__all__ = [
    # Scrapers
    "HTTPXScraper",
    "PlaywrightScraper",
    # Playwright types
    "BrowserType",
    "WaitStrategy",
    "PageAction",
    # Rate limiting
    "RateLimiter",
    "RateLimitStrategy",
    "RateLimitStats",
    "TokenBucketLimiter",
    "FixedWindowLimiter",
    "SlidingWindowLimiter",
    "create_rate_limiter",
    # Proxy management
    "ProxyManager",
    "ProxyInfo",
    "ProxyProtocol",
    "RotationStrategy",
    # User agent
    "UserAgentRotator",
    "DeviceType",
    # Retry
    "RetryHandler",
    # Utilities
    "extract_domain",
    "normalize_url",
    "parse_cookies",
]
