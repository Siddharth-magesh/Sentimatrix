"""
Sentimatrix Commercial Scraping APIs

Commercial web scraping API integrations that provide:
- Proxy rotation
- CAPTCHA solving
- JavaScript rendering
- Anti-bot bypass
- Geo-targeting

Supported Services:
- ScraperAPI: Simple API with JS rendering and CAPTCHA handling
- Apify: Actor-based scraping with 2000+ pre-built scrapers
- Bright Data: Enterprise-grade with 72M+ residential proxies
- Oxylabs: Web scraper API with e-commerce specialization
- Zyte: All-in-one API with automatic extraction
- ScrapingBee: Simple API with headless browser support
- ScrapingAnt: Budget-friendly scraping with JS rendering

Usage:
    >>> from sentimatrix.providers.scrapers.commercial import ScraperAPIClient
    >>>
    >>> async with ScraperAPIClient(api_key="your_key") as client:
    ...     content = await client.scrape("https://example.com")
    ...     print(content.html)
"""

from sentimatrix.providers.scrapers.commercial.base import (
    BaseCommercialClient,
    CommercialAPIConfig,
    ScrapeResult,
    OutputFormat,
    DeviceEmulation,
)
from sentimatrix.providers.scrapers.commercial.scraper_api import (
    ScraperAPIClient,
    ScraperAPIConfig,
)
from sentimatrix.providers.scrapers.commercial.apify import (
    ApifyClient,
    ApifyConfig,
    POPULAR_ACTORS,
)
from sentimatrix.providers.scrapers.commercial.bright_data import (
    BrightDataClient,
    BrightDataConfig,
    BrightDataZone,
)
from sentimatrix.providers.scrapers.commercial.oxylabs import (
    OxylabsClient,
    OxylabsConfig,
    OxylabsSource,
)
from sentimatrix.providers.scrapers.commercial.zyte import (
    ZyteClient,
    ZyteConfig,
    ZyteExtractionType,
    ZyteAction,
)
from sentimatrix.providers.scrapers.commercial.scrapingbee import (
    ScrapingBeeClient,
    ScrapingBeeConfig,
)
from sentimatrix.providers.scrapers.commercial.scrapingant import (
    ScrapingAntClient,
    ScrapingAntConfig,
    Cookie,
)

__all__ = [
    # Base
    "BaseCommercialClient",
    "CommercialAPIConfig",
    "ScrapeResult",
    "OutputFormat",
    "DeviceEmulation",
    # ScraperAPI
    "ScraperAPIClient",
    "ScraperAPIConfig",
    # Apify
    "ApifyClient",
    "ApifyConfig",
    "POPULAR_ACTORS",
    # Bright Data
    "BrightDataClient",
    "BrightDataConfig",
    "BrightDataZone",
    # Oxylabs
    "OxylabsClient",
    "OxylabsConfig",
    "OxylabsSource",
    # Zyte
    "ZyteClient",
    "ZyteConfig",
    "ZyteExtractionType",
    "ZyteAction",
    # ScrapingBee
    "ScrapingBeeClient",
    "ScrapingBeeConfig",
    # ScrapingAnt
    "ScrapingAntClient",
    "ScrapingAntConfig",
    "Cookie",
]
