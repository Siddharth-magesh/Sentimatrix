"""
Sentimatrix Platform Scrapers

Specialized scrapers for extracting reviews from specific platforms:

E-Commerce:
- AmazonScraper: Product reviews via Playwright

Gaming:
- SteamScraper: Game reviews via Steam API

Video/Media:
- YouTubeScraper: Video comments via YouTube Data API
- IMDBScraper: Movie/TV reviews via Playwright/OMDb API

Social Media:
- RedditScraper: Posts and comments via Reddit JSON API

Reviews/Local:
- YelpScraper: Business reviews via Playwright/Fusion API
- TrustpilotScraper: Company reviews via Playwright
- GoogleReviewsScraper: Place reviews via Google Places API

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

# E-Commerce
from sentimatrix.providers.scrapers.platforms.amazon import AmazonScraper

# Gaming
from sentimatrix.providers.scrapers.platforms.steam import SteamScraper

# Video/Media
from sentimatrix.providers.scrapers.platforms.youtube import YouTubeScraper
from sentimatrix.providers.scrapers.platforms.imdb import IMDBScraper

# Social Media
from sentimatrix.providers.scrapers.platforms.reddit import RedditScraper

# Reviews/Local
from sentimatrix.providers.scrapers.platforms.yelp import YelpScraper
from sentimatrix.providers.scrapers.platforms.trustpilot import TrustpilotScraper
from sentimatrix.providers.scrapers.platforms.google_reviews import GoogleReviewsScraper

__all__ = [
    # Base
    "BasePlatformScraper",
    "PlatformConfig",
    "ProductInfo",
    "ReviewFilter",
    "SortOrder",
    # E-Commerce
    "AmazonScraper",
    # Gaming
    "SteamScraper",
    # Video/Media
    "YouTubeScraper",
    "IMDBScraper",
    # Social Media
    "RedditScraper",
    # Reviews/Local
    "YelpScraper",
    "TrustpilotScraper",
    "GoogleReviewsScraper",
]
