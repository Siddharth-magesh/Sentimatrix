"""
Sentimatrix Platform Scrapers

Specialized scrapers for extracting reviews from specific platforms:
- AmazonScraper: E-commerce product reviews
- SteamScraper: Game reviews via Steam API
- YouTubeScraper: Video comments via YouTube Data API
- RedditScraper: Posts and comments via Reddit API (PRAW)

Usage:
    >>> from sentimatrix.providers.scrapers.platforms import AmazonScraper

    >>> async with AmazonScraper() as scraper:
    ...     reviews = await scraper.scrape_reviews("B08N5WRWNW", limit=100)
    ...     for review in reviews:
    ...         print(f"{review.rating}: {review.text[:50]}...")
"""

from sentimatrix.providers.scrapers.platforms.base import (
    BasePlatformScraper,
    PlatformConfig,
    ProductInfo,
    ReviewFilter,
    SortOrder,
)
from sentimatrix.providers.scrapers.platforms.amazon import AmazonScraper
from sentimatrix.providers.scrapers.platforms.steam import SteamScraper
from sentimatrix.providers.scrapers.platforms.youtube import YouTubeScraper
from sentimatrix.providers.scrapers.platforms.reddit import RedditScraper

__all__ = [
    # Base
    "BasePlatformScraper",
    "PlatformConfig",
    "ProductInfo",
    "ReviewFilter",
    "SortOrder",
    # Platform scrapers
    "AmazonScraper",
    "SteamScraper",
    "YouTubeScraper",
    "RedditScraper",
]
