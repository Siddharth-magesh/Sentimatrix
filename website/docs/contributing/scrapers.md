---
title: Adding Scrapers
description: How to add new platform scrapers
---

# Adding Scrapers

Guide to adding new platform scrapers to Sentimatrix.

## Scraper Structure

```
sentimatrix/providers/scrapers/platforms/
├── base.py           # Base class
├── amazon.py         # Example scraper
└── your_platform.py  # New scraper
```

## Implement Base Class

```python
from sentimatrix.providers.scrapers.base import BaseScraper
from sentimatrix.models import Review

class YourPlatformScraper(BaseScraper):
    """Scraper for YourPlatform."""

    @property
    def platform_name(self) -> str:
        return "your_platform"

    @property
    def supported_url_patterns(self) -> list[str]:
        return [
            r"https?://yourplatform\.com/.*",
            r"https?://www\.yourplatform\.com/.*",
        ]

    async def validate_url(self, url: str) -> bool:
        return "yourplatform.com" in url

    async def scrape(
        self,
        url: str,
        max_reviews: int = 50,
        **kwargs
    ) -> list[Review]:
        reviews = []

        # Fetch page
        html = await self._fetch(url)

        # Parse reviews
        for element in self._parse_reviews(html):
            review = Review(
                text=element.text,
                rating=element.rating,
                author=element.author,
                posted_date=element.date,
                helpful_count=element.helpful,
                platform=self.platform_name,
                metadata={
                    "review_id": element.id,
                    # Platform-specific data
                }
            )
            reviews.append(review)

            if len(reviews) >= max_reviews:
                break

        return reviews
```

## Register Scraper

Add to `ScraperProvider` enum if needed:

```python
class ScraperProvider(str, Enum):
    YOUR_PLATFORM = "your_platform"
```

Register in scraper manager:

```python
# In scraper_manager.py
self._register_scraper("your_platform", YourPlatformScraper)
```

## Add Tests

```python
# tests/scrapers/test_your_platform.py
import pytest
from sentimatrix.providers.scrapers.platforms.your_platform import YourPlatformScraper

@pytest.mark.asyncio
async def test_scrape():
    scraper = YourPlatformScraper()
    reviews = await scraper.scrape(
        "https://yourplatform.com/product/123",
        max_reviews=10
    )
    assert len(reviews) <= 10
    assert all(r.platform == "your_platform" for r in reviews)

def test_validate_url():
    scraper = YourPlatformScraper()
    assert scraper.validate_url("https://yourplatform.com/product/123")
    assert not scraper.validate_url("https://other.com/product/123")
```

## Add Documentation

Create `docs/scrapers/your-platform.md` with:

- Quick start example
- Supported URL formats
- Review object fields
- Rate limits
- Platform-specific options

## Related

- [Base Scraper API](../api/base-scraper.md)
- [Adding Providers](providers.md)
- [Testing](testing.md)
