---
title: Base Scraper
description: Scraper base class reference
---

# Base Scraper

Abstract base class for platform scrapers.

## Class Definition

```python
class BaseScraper(ABC):
    """Base class for platform scrapers."""

    @abstractmethod
    async def scrape(
        self,
        url: str,
        max_reviews: int = 50,
        **kwargs
    ) -> list[Review]:
        """Scrape reviews from URL."""

    @abstractmethod
    async def validate_url(self, url: str) -> bool:
        """Validate URL format for platform."""

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Platform identifier."""

    @property
    @abstractmethod
    def supported_url_patterns(self) -> list[str]:
        """URL patterns this scraper handles."""
```

## Implementing a Scraper

```python
class MyPlatformScraper(BaseScraper):
    @property
    def platform_name(self) -> str:
        return "my_platform"

    @property
    def supported_url_patterns(self) -> list[str]:
        return [r"https?://myplatform\.com/.*"]

    async def validate_url(self, url: str) -> bool:
        return "myplatform.com" in url

    async def scrape(
        self,
        url: str,
        max_reviews: int = 50,
        **kwargs
    ) -> list[Review]:
        # Implementation
        reviews = []
        # ... scraping logic
        return reviews
```

## Review Object

```python
@dataclass
class Review:
    text: str
    rating: int | float | bool | None = None
    title: str | None = None
    author: str | None = None
    posted_date: datetime | None = None
    helpful_count: int = 0
    platform: str = ""
    metadata: dict = field(default_factory=dict)
```

## Related

- [API Overview](index.md)
- [Scraper Manager](scraper-manager.md)
- [Scrapers Guide](../scrapers/index.md)
