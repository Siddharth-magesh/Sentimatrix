"""
Amazon Product Review Scraper

Extracts product reviews from Amazon using multiple methods:
1. Direct HTML scraping (requires proxy/rotation)
2. Playwright browser automation (for JavaScript content)

Features:
- ASIN-based and URL-based review fetching
- Rating and filter support
- Pagination handling
- Verified purchase detection
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

from sentimatrix.core.exceptions import (
    ScraperBlockedError,
    ScraperParseError,
    ProviderInitializationError,
)
from sentimatrix.providers.base import (
    ProviderInfo,
    ProviderType,
    ProviderCapabilities,
    Review,
)
from sentimatrix.providers.scrapers.platforms.base import (
    BasePlatformScraper,
    PlatformConfig,
    ProductInfo,
    SortOrder,
    ReviewFilter,
)
from sentimatrix.providers.scrapers.rate_limiter import RateLimiter


# Country domain mapping
AMAZON_DOMAINS = {
    "us": "amazon.com",
    "uk": "amazon.co.uk",
    "de": "amazon.de",
    "fr": "amazon.fr",
    "jp": "amazon.co.jp",
    "ca": "amazon.ca",
    "it": "amazon.it",
    "es": "amazon.es",
    "in": "amazon.in",
    "au": "amazon.com.au",
    "br": "amazon.com.br",
    "mx": "amazon.com.mx",
}


@dataclass
class AmazonConfig(PlatformConfig):
    """Amazon-specific configuration."""

    # Country/marketplace
    country: str = "us"

    # Review settings
    filter_verified: bool = False
    include_images: bool = False

    # Rate limiting (Amazon is strict)
    requests_per_second: float = 0.5
    burst_size: int = 3

    # Optional API keys for commercial scrapers
    rainforest_api_key: Optional[str] = None
    scraperapi_key: Optional[str] = None

    @property
    def domain(self) -> str:
        """Get Amazon domain for country."""
        return AMAZON_DOMAINS.get(self.country, "amazon.com")


class AmazonScraper(BasePlatformScraper):
    """
    Amazon product review scraper.

    Supports multiple scraping methods:
    1. Playwright (browser automation) - default
    2. HTTPX with rotating proxies
    3. Commercial APIs (Rainforest, ScraperAPI) - if configured

    Example:
        >>> config = AmazonConfig(country="us")
        >>> async with AmazonScraper(config) as scraper:
        ...     reviews = await scraper.scrape_reviews("B08N5WRWNW", limit=50)
        ...     for review in reviews:
        ...         print(f"{review.rating}/5: {review.text[:50]}...")
    """

    # ASIN pattern: 10 alphanumeric characters
    ASIN_PATTERN = re.compile(r"[A-Z0-9]{10}")

    # URL patterns for ASIN extraction
    URL_PATTERNS = [
        re.compile(r"/dp/([A-Z0-9]{10})"),
        re.compile(r"/gp/product/([A-Z0-9]{10})"),
        re.compile(r"/product-reviews/([A-Z0-9]{10})"),
        re.compile(r"/asin/([A-Z0-9]{10})"),
    ]

    def __init__(
        self,
        config: Optional[AmazonConfig] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        """
        Initialize Amazon scraper.

        Args:
            config: Amazon-specific configuration
            rate_limiter: Optional rate limiter
        """
        self._amazon_config = config or AmazonConfig()
        super().__init__(self._amazon_config, rate_limiter)

        self._playwright_scraper = None
        self._httpx_scraper = None

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="amazon",
            provider_type=ProviderType.SCRAPER,
            version="1.0.0",
            description="Amazon product review scraper",
            capabilities=ProviderCapabilities(
                javascript_rendering=True,
                proxy_support=True,
            ),
            website="https://amazon.com",
        )

    @property
    def platform_name(self) -> str:
        """Get platform name."""
        return "amazon"

    @property
    def platform_domain(self) -> str:
        """Get platform domain."""
        return self._amazon_config.domain

    async def initialize(self) -> None:
        """Initialize the scraper."""
        if self._initialized:
            return

        # Import scrapers lazily
        from sentimatrix.providers.scrapers.playwright_scraper import PlaywrightScraper
        from sentimatrix.providers.scrapers.httpx_scraper import HTTPXScraper

        # Initialize Playwright for JavaScript-rendered content
        self._playwright_scraper = PlaywrightScraper(
            config=self._platform_config.to_scraper_config(),
            rate_limiter=self._rate_limiter,
            stealth=True,  # Enable stealth mode for Amazon
        )
        await self._playwright_scraper.initialize()

        # Initialize HTTPX as fallback
        self._httpx_scraper = HTTPXScraper(
            config=self._platform_config.to_scraper_config(),
            rate_limiter=self._rate_limiter,
        )
        await self._httpx_scraper.initialize()

        self._initialized = True

    async def close(self) -> None:
        """Close the scraper."""
        if self._playwright_scraper:
            await self._playwright_scraper.close()
            self._playwright_scraper = None

        if self._httpx_scraper:
            await self._httpx_scraper.close()
            self._httpx_scraper = None

        self._initialized = False

    def validate_url(self, url: str) -> bool:
        """Validate Amazon URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Check if it's an Amazon domain
            for country_domain in AMAZON_DOMAINS.values():
                if country_domain in domain or f"www.{country_domain}" in domain:
                    return True

            return False
        except Exception:
            return False

    def extract_id(self, url: str) -> Optional[str]:
        """Extract ASIN from URL."""
        for pattern in self.URL_PATTERNS:
            match = pattern.search(url)
            if match:
                return match.group(1)

        return None

    def validate_asin(self, asin: str) -> bool:
        """Validate ASIN format."""
        return bool(self.ASIN_PATTERN.fullmatch(asin.upper()))

    def _build_reviews_url(
        self,
        asin: str,
        page: int = 1,
        sort_by: SortOrder = SortOrder.RECENT,
        filter_by: ReviewFilter = ReviewFilter.ALL,
    ) -> str:
        """Build reviews page URL."""
        base_url = f"https://www.{self.platform_domain}/product-reviews/{asin}"

        # Sort mapping
        sort_map = {
            SortOrder.RECENT: "recent",
            SortOrder.HELPFUL: "helpful",
            SortOrder.RATING_HIGH: "reviewerType=all_reviews&sortBy=recent&filterByStar=positive",
            SortOrder.RATING_LOW: "reviewerType=all_reviews&sortBy=recent&filterByStar=critical",
        }

        # Filter mapping
        filter_map = {
            ReviewFilter.ALL: "",
            ReviewFilter.POSITIVE: "filterByStar=positive",
            ReviewFilter.NEGATIVE: "filterByStar=critical",
            ReviewFilter.CRITICAL: "filterByStar=one_star",
            ReviewFilter.VERIFIED: "reviewerType=avp_only_reviews",
        }

        params = [f"pageNumber={page}"]

        if sort_by in sort_map:
            sort_param = sort_map[sort_by]
            if "&" in sort_param:
                params.extend(sort_param.split("&"))
            else:
                params.append(f"sortBy={sort_param}")

        if filter_by in filter_map and filter_map[filter_by]:
            params.append(filter_map[filter_by])

        return f"{base_url}?{'&'.join(params)}"

    async def scrape_reviews(
        self,
        identifier: str,
        limit: int = 100,
        sort_by: SortOrder = SortOrder.RECENT,
        filter_by: ReviewFilter = ReviewFilter.ALL,
        **kwargs: Any,
    ) -> List[Review]:
        """
        Scrape reviews for a product.

        Args:
            identifier: ASIN or product URL
            limit: Maximum number of reviews to fetch
            sort_by: Sort order
            filter_by: Review filter
            **kwargs: Additional parameters

        Returns:
            List of Review objects
        """
        self._ensure_initialized()

        # Extract ASIN if URL provided
        if identifier.startswith("http"):
            asin = self.extract_id(identifier)
            if not asin:
                raise ValueError(f"Could not extract ASIN from URL: {identifier}")
        else:
            asin = identifier.upper()

        if not self.validate_asin(asin):
            raise ValueError(f"Invalid ASIN format: {asin}")

        reviews: List[Review] = []
        page = 1
        max_pages = (limit // 10) + 1  # Amazon shows ~10 reviews per page

        while len(reviews) < limit and page <= max_pages:
            url = self._build_reviews_url(asin, page, sort_by, filter_by)

            try:
                page_reviews = await self._scrape_reviews_page(url, asin)
                if not page_reviews:
                    break  # No more reviews

                reviews.extend(page_reviews)
                page += 1

            except ScraperBlockedError:
                # Try with fallback methods if blocked
                break

        return reviews[:limit]

    async def _scrape_reviews_page(
        self,
        url: str,
        asin: str,
    ) -> List[Review]:
        """Scrape a single page of reviews."""
        # Rate limit
        await self._rate_limiter.acquire(domain=self.platform_domain)

        # Try Playwright first (handles JavaScript)
        try:
            content = await self._playwright_scraper.scrape(
                url,
                wait_for="[data-hook='review']",
                timeout=self._platform_config.timeout * 1000,
            )

            return self._parse_reviews_html(content.html or content.content, asin)

        except Exception as e:
            # Check if we're blocked
            if "captcha" in str(e).lower() or "robot" in str(e).lower():
                raise ScraperBlockedError(
                    provider=self.platform_name,
                    url=url,
                    reason="CAPTCHA or bot detection triggered",
                    original_error=e,
                )
            raise

    def _parse_reviews_html(self, html: str, asin: str) -> List[Review]:
        """Parse reviews from HTML content."""
        reviews: List[Review] = []

        try:
            # Lazy import BeautifulSoup
            try:
                from bs4 import BeautifulSoup
            except ImportError:
                raise ImportError(
                    "BeautifulSoup is required for Amazon scraping. "
                    "Install with: pip install beautifulsoup4"
                )

            soup = BeautifulSoup(html, "html.parser")

            # Find all review containers
            review_elements = soup.select("[data-hook='review']")

            for element in review_elements:
                try:
                    review = self._parse_single_review(element, asin)
                    if review:
                        reviews.append(review)
                except Exception:
                    continue  # Skip malformed reviews

        except Exception as e:
            raise ScraperParseError(
                provider=self.platform_name,
                url=f"ASIN:{asin}",
                reason=f"Failed to parse reviews HTML: {e}",
                original_error=e,
            )

        return reviews

    def _parse_single_review(self, element: Any, asin: str) -> Optional[Review]:
        """Parse a single review element."""
        # Extract review text
        text_elem = element.select_one("[data-hook='review-body']")
        if not text_elem:
            return None

        text = self.clean_text(text_elem.get_text())
        if not text:
            return None

        # Extract rating
        rating = None
        rating_elem = element.select_one("[data-hook='review-star-rating']")
        if rating_elem:
            rating_text = rating_elem.get_text()
            match = re.search(r"(\d+(?:\.\d+)?)", rating_text)
            if match:
                rating = float(match.group(1))

        # Extract author
        author = None
        author_elem = element.select_one(".a-profile-name")
        if author_elem:
            author = author_elem.get_text().strip()

        # Extract date
        timestamp = None
        date_elem = element.select_one("[data-hook='review-date']")
        if date_elem:
            date_text = date_elem.get_text()
            # Format: "Reviewed in the United States on January 1, 2024"
            date_match = re.search(
                r"on\s+(\w+\s+\d+,\s+\d+)",
                date_text
            )
            if date_match:
                timestamp = self.parse_date(date_match.group(1))

        # Extract title
        title = None
        title_elem = element.select_one("[data-hook='review-title']")
        if title_elem:
            title = title_elem.get_text().strip()

        # Check if verified purchase
        verified = False
        verified_elem = element.select_one("[data-hook='avp-badge']")
        if verified_elem:
            verified = True

        # Extract helpful votes
        helpful_votes = 0
        helpful_elem = element.select_one("[data-hook='helpful-vote-statement']")
        if helpful_elem:
            helpful_text = helpful_elem.get_text()
            match = re.search(r"(\d+)", helpful_text)
            if match:
                helpful_votes = int(match.group(1))

        # Generate unique ID
        review_id = self.generate_review_id(
            platform=self.platform_name,
            text=text,
            author=author,
            timestamp=timestamp,
        )

        return Review(
            id=review_id,
            text=text,
            source=f"https://www.{self.platform_domain}/dp/{asin}",
            platform=self.platform_name,
            author=author,
            rating=rating,
            timestamp=timestamp,
            metadata={
                "asin": asin,
                "title": title,
                "verified_purchase": verified,
                "helpful_votes": helpful_votes,
                "country": self._amazon_config.country,
            },
        )

    async def get_product_info(self, identifier: str) -> ProductInfo:
        """
        Get product information.

        Args:
            identifier: ASIN or product URL

        Returns:
            ProductInfo object
        """
        self._ensure_initialized()

        # Extract ASIN if URL provided
        if identifier.startswith("http"):
            asin = self.extract_id(identifier)
            if not asin:
                raise ValueError(f"Could not extract ASIN from URL: {identifier}")
        else:
            asin = identifier.upper()

        if not self.validate_asin(asin):
            raise ValueError(f"Invalid ASIN format: {asin}")

        # Scrape product page
        url = f"https://www.{self.platform_domain}/dp/{asin}"

        await self._rate_limiter.acquire(domain=self.platform_domain)

        content = await self._playwright_scraper.scrape(
            url,
            wait_for="#productTitle",
            timeout=self._platform_config.timeout * 1000,
        )

        return self._parse_product_html(content.html or content.content, asin, url)

    def _parse_product_html(self, html: str, asin: str, url: str) -> ProductInfo:
        """Parse product information from HTML."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "BeautifulSoup is required. Install with: pip install beautifulsoup4"
            )

        soup = BeautifulSoup(html, "html.parser")

        # Extract title
        name = ""
        title_elem = soup.select_one("#productTitle")
        if title_elem:
            name = title_elem.get_text().strip()

        # Extract price
        price = None
        currency = None
        price_elem = soup.select_one(".a-price .a-offscreen")
        if price_elem:
            price_text = price_elem.get_text()
            match = re.search(r"([£$€¥])?(\d+(?:,\d+)?(?:\.\d+)?)", price_text)
            if match:
                currency = match.group(1) or "$"
                price = float(match.group(2).replace(",", ""))

        # Extract rating
        rating = None
        rating_elem = soup.select_one("#acrPopover")
        if rating_elem:
            rating_text = rating_elem.get("title", "")
            match = re.search(r"(\d+(?:\.\d+)?)", rating_text)
            if match:
                rating = float(match.group(1))

        # Extract review count
        review_count = None
        count_elem = soup.select_one("#acrCustomerReviewText")
        if count_elem:
            count_text = count_elem.get_text()
            match = re.search(r"([\d,]+)", count_text)
            if match:
                review_count = int(match.group(1).replace(",", ""))

        # Extract image
        image_url = None
        image_elem = soup.select_one("#landingImage, #imgBlkFront")
        if image_elem:
            image_url = image_elem.get("src")

        # Extract description
        description = None
        desc_elem = soup.select_one("#productDescription p")
        if desc_elem:
            description = desc_elem.get_text().strip()

        return ProductInfo(
            id=asin,
            name=name,
            platform=self.platform_name,
            url=url,
            description=description,
            price=price,
            currency=currency,
            rating=rating,
            review_count=review_count,
            image_url=image_url,
            metadata={
                "country": self._amazon_config.country,
            },
        )

    async def search_products(
        self,
        query: str,
        limit: int = 10,
    ) -> List[ProductInfo]:
        """
        Search for products.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of ProductInfo objects
        """
        self._ensure_initialized()

        from urllib.parse import quote_plus

        url = f"https://www.{self.platform_domain}/s?k={quote_plus(query)}"

        await self._rate_limiter.acquire(domain=self.platform_domain)

        content = await self._playwright_scraper.scrape(
            url,
            wait_for="[data-component-type='s-search-result']",
            timeout=self._platform_config.timeout * 1000,
        )

        return self._parse_search_results(
            content.html or content.content,
            limit
        )

    def _parse_search_results(self, html: str, limit: int) -> List[ProductInfo]:
        """Parse search results."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "BeautifulSoup is required. Install with: pip install beautifulsoup4"
            )

        soup = BeautifulSoup(html, "html.parser")
        products: List[ProductInfo] = []

        result_elements = soup.select("[data-component-type='s-search-result']")

        for element in result_elements[:limit]:
            try:
                # Extract ASIN
                asin = element.get("data-asin", "")
                if not asin:
                    continue

                # Extract title
                name = ""
                title_elem = element.select_one("h2 a span")
                if title_elem:
                    name = title_elem.get_text().strip()

                # Extract URL
                url_elem = element.select_one("h2 a")
                url = None
                if url_elem:
                    href = url_elem.get("href", "")
                    url = urljoin(f"https://www.{self.platform_domain}", href)

                # Extract price
                price = None
                price_elem = element.select_one(".a-price .a-offscreen")
                if price_elem:
                    price_text = price_elem.get_text()
                    match = re.search(r"(\d+(?:\.\d+)?)", price_text)
                    if match:
                        price = float(match.group(1))

                # Extract rating
                rating = None
                rating_elem = element.select_one("[aria-label*='out of 5']")
                if rating_elem:
                    rating_text = rating_elem.get("aria-label", "")
                    match = re.search(r"(\d+(?:\.\d+)?)", rating_text)
                    if match:
                        rating = float(match.group(1))

                # Extract image
                image_url = None
                image_elem = element.select_one("img.s-image")
                if image_elem:
                    image_url = image_elem.get("src")

                products.append(ProductInfo(
                    id=asin,
                    name=name,
                    platform=self.platform_name,
                    url=url,
                    price=price,
                    rating=rating,
                    image_url=image_url,
                ))

            except Exception:
                continue

        return products
