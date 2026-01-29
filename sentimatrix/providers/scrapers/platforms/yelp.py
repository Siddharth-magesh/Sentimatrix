"""
Yelp Business Review Scraper

Extracts business reviews from Yelp using:
1. Yelp website scraping via Playwright
2. Yelp Fusion API (optional, requires API key)

Features:
- Business reviews and ratings
- Search by location and category
- User ratings and elite status
- Photo and reaction counts
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from sentimatrix.core.exceptions import (
    ScraperError,
    ScraperParseError,
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


@dataclass
class YelpConfig(PlatformConfig):
    """Yelp-specific configuration."""

    # Yelp Fusion API key (optional)
    api_key: Optional[str] = None

    # Search location
    location: str = "New York, NY"

    # Rate limiting (Yelp is strict about scraping)
    requests_per_second: float = 0.5
    burst_size: int = 3


@dataclass
class BusinessInfo:
    """Yelp business information."""

    id: str
    name: str
    url: str
    rating: Optional[float] = None
    review_count: Optional[int] = None
    price: Optional[str] = None  # $, $$, $$$, $$$$
    categories: List[str] = field(default_factory=list)
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    country: str = "US"
    phone: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    image_url: Optional[str] = None
    is_closed: bool = False
    hours: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "url": self.url,
            "rating": self.rating,
            "review_count": self.review_count,
            "price": self.price,
            "categories": self.categories,
            "address": self.address,
            "city": self.city,
            "state": self.state,
            "zip_code": self.zip_code,
            "country": self.country,
            "phone": self.phone,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "image_url": self.image_url,
            "is_closed": self.is_closed,
            "hours": self.hours,
        }


class YelpScraper(BasePlatformScraper):
    """
    Yelp business review scraper.

    Uses Playwright for web scraping and optionally Yelp Fusion API
    for enhanced business information.

    Example:
        >>> config = YelpConfig(location="San Francisco, CA")
        >>> async with YelpScraper(config) as scraper:
        ...     reviews = await scraper.scrape_reviews("the-french-laundry-yountville")
        ...     for review in reviews:
        ...         print(f"{review.rating}/5: {review.text[:50]}...")
    """

    # URL patterns for business ID extraction
    URL_PATTERNS = [
        re.compile(r"yelp\.com/biz/([a-zA-Z0-9_-]+)"),
    ]

    # Base URLs
    BASE_URL = "https://www.yelp.com"
    API_URL = "https://api.yelp.com/v3"

    def __init__(
        self,
        config: Optional[YelpConfig] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        """
        Initialize Yelp scraper.

        Args:
            config: Yelp-specific configuration
            rate_limiter: Optional rate limiter
        """
        self._yelp_config = config or YelpConfig()
        super().__init__(self._yelp_config, rate_limiter)

        self._playwright_scraper = None
        self._httpx_scraper = None

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="yelp",
            provider_type=ProviderType.SCRAPER,
            version="1.0.0",
            description="Yelp business review scraper",
            capabilities=ProviderCapabilities(
                javascript_rendering=True,
                proxy_support=True,
            ),
            website="https://yelp.com",
        )

    @property
    def platform_name(self) -> str:
        """Get platform name."""
        return "yelp"

    @property
    def platform_domain(self) -> str:
        """Get platform domain."""
        return "yelp.com"

    async def initialize(self) -> None:
        """Initialize the scraper."""
        if self._initialized:
            return

        from sentimatrix.providers.scrapers.playwright_scraper import PlaywrightScraper
        from sentimatrix.providers.scrapers.httpx_scraper import HTTPXScraper

        self._playwright_scraper = PlaywrightScraper(
            config=self._platform_config.to_scraper_config(),
            rate_limiter=self._rate_limiter,
            stealth=True,
        )
        await self._playwright_scraper.initialize()

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
        """Validate Yelp URL."""
        for pattern in self.URL_PATTERNS:
            if pattern.search(url):
                return True
        return False

    def extract_id(self, url: str) -> Optional[str]:
        """Extract business ID from URL."""
        for pattern in self.URL_PATTERNS:
            match = pattern.search(url)
            if match:
                return match.group(1)
        return None

    def _build_reviews_url(
        self,
        business_id: str,
        page: int = 0,
        sort_by: SortOrder = SortOrder.RECENT,
    ) -> str:
        """Build reviews page URL."""
        base_url = f"{self.BASE_URL}/biz/{business_id}"

        # Sort mapping
        sort_map = {
            SortOrder.RECENT: "date_desc",
            SortOrder.HELPFUL: "relevance_desc",
            SortOrder.RATING_HIGH: "rating_desc",
            SortOrder.RATING_LOW: "rating_asc",
        }

        params = []

        if page > 0:
            params.append(f"start={page * 10}")

        if sort_by in sort_map:
            params.append(f"sort_by={sort_map[sort_by]}")

        if params:
            return f"{base_url}?{'&'.join(params)}"
        return base_url

    async def scrape_reviews(
        self,
        identifier: str,
        limit: int = 100,
        sort_by: SortOrder = SortOrder.RECENT,
        filter_by: ReviewFilter = ReviewFilter.ALL,
        **kwargs: Any,
    ) -> List[Review]:
        """
        Scrape reviews for a business.

        Args:
            identifier: Business ID or URL
            limit: Maximum number of reviews to fetch
            sort_by: Sort order
            filter_by: Review filter (star ratings)
            **kwargs: Additional parameters

        Returns:
            List of Review objects
        """
        self._ensure_initialized()

        # Extract business ID if URL provided
        if identifier.startswith("http"):
            business_id = self.extract_id(identifier)
            if not business_id:
                raise ValueError(f"Could not extract business ID from URL: {identifier}")
        else:
            business_id = identifier

        reviews: List[Review] = []
        page = 0
        max_pages = (limit // 10) + 1

        while len(reviews) < limit and page < max_pages:
            url = self._build_reviews_url(business_id, page, sort_by)

            try:
                await self._rate_limiter.acquire(domain=self.platform_domain)

                content = await self._playwright_scraper.scrape(
                    url,
                    wait_for="[data-review-id], .review__09f24__oHr9V",
                    timeout=self._platform_config.timeout * 1000,
                )

                page_reviews = self._parse_reviews_html(
                    content.html or content.content,
                    business_id
                )

                if not page_reviews:
                    break

                reviews.extend(page_reviews)
                page += 1

            except Exception as e:
                if "timeout" in str(e).lower():
                    break
                raise ScraperError(
                    f"Failed to scrape Yelp reviews: {e}",
                    provider=self.platform_name,
                    original_error=e,
                )

        return reviews[:limit]

    def _parse_reviews_html(self, html: str, business_id: str) -> List[Review]:
        """Parse reviews from HTML content."""
        reviews: List[Review] = []

        try:
            try:
                from bs4 import BeautifulSoup
            except ImportError:
                raise ImportError(
                    "BeautifulSoup is required for Yelp scraping. "
                    "Install with: pip install beautifulsoup4"
                )

            soup = BeautifulSoup(html, "html.parser")

            # Find review elements (Yelp frequently changes their classes)
            review_elements = soup.select("[data-review-id]")

            # Fallback selectors
            if not review_elements:
                review_elements = soup.select(".review__09f24__oHr9V")
            if not review_elements:
                review_elements = soup.select("[class*='review']")

            for element in review_elements:
                try:
                    review = self._parse_single_review(element, business_id)
                    if review:
                        reviews.append(review)
                except Exception:
                    continue

        except Exception as e:
            raise ScraperParseError(
                provider=self.platform_name,
                url=f"Business:{business_id}",
                reason=f"Failed to parse reviews HTML: {e}",
                original_error=e,
            )

        return reviews

    def _parse_single_review(self, element: Any, business_id: str) -> Optional[Review]:
        """Parse a single review element."""
        # Extract review text
        text = ""
        text_elem = element.select_one(
            "[class*='comment'] span, .raw__09f24__T4Ezm, p[class*='comment']"
        )
        if text_elem:
            text = self.clean_text(text_elem.get_text())

        if not text:
            return None

        # Extract rating (from aria-label or star count)
        rating = None
        rating_elem = element.select_one(
            "[aria-label*='star rating'], [class*='star']"
        )
        if rating_elem:
            aria_label = rating_elem.get("aria-label", "")
            match = re.search(r"(\d+(?:\.\d+)?)\s*star", aria_label)
            if match:
                rating = float(match.group(1))

        # If no aria-label, count filled stars
        if rating is None:
            filled_stars = element.select("[class*='star'][class*='fill']")
            if filled_stars:
                rating = float(len(filled_stars))

        # Extract author
        author = None
        author_elem = element.select_one(
            "[class*='user-passport'] a, [class*='author'] a, .user-name a"
        )
        if author_elem:
            author = author_elem.get_text().strip()

        # Extract date
        timestamp = None
        date_elem = element.select_one(
            "[class*='date'], span[class*='css-chan6m']"
        )
        if date_elem:
            date_text = date_elem.get_text()
            timestamp = self.parse_date(date_text)

        # Extract review ID
        review_id = element.get("data-review-id")
        if not review_id:
            review_id = self.generate_review_id(
                platform=self.platform_name,
                text=text,
                author=author,
                timestamp=timestamp,
            )

        # Check for elite badge
        is_elite = False
        elite_elem = element.select_one("[class*='elite'], .elite-badge")
        if elite_elem:
            is_elite = True

        # Extract reaction counts
        useful = 0
        funny = 0
        cool = 0
        reaction_elems = element.select("[class*='reaction']")
        for reaction in reaction_elems:
            reaction_text = reaction.get_text().lower()
            count_match = re.search(r"(\d+)", reaction_text)
            count = int(count_match.group(1)) if count_match else 0
            if "useful" in reaction_text:
                useful = count
            elif "funny" in reaction_text:
                funny = count
            elif "cool" in reaction_text:
                cool = count

        return Review(
            id=str(review_id),
            text=text,
            source=f"{self.BASE_URL}/biz/{business_id}",
            platform=self.platform_name,
            author=author,
            rating=rating,
            timestamp=timestamp,
            metadata={
                "business_id": business_id,
                "is_elite": is_elite,
                "useful_count": useful,
                "funny_count": funny,
                "cool_count": cool,
            },
        )

    async def get_product_info(self, identifier: str) -> ProductInfo:
        """
        Get business information.

        Args:
            identifier: Business ID or URL

        Returns:
            ProductInfo object
        """
        business_info = await self.get_business_info(identifier)

        return ProductInfo(
            id=business_info.id,
            name=business_info.name,
            platform=self.platform_name,
            url=business_info.url,
            rating=business_info.rating,
            review_count=business_info.review_count,
            image_url=business_info.image_url,
            category=", ".join(business_info.categories),
            metadata={
                "price": business_info.price,
                "address": business_info.address,
                "city": business_info.city,
                "state": business_info.state,
                "zip_code": business_info.zip_code,
                "phone": business_info.phone,
                "is_closed": business_info.is_closed,
                "latitude": business_info.latitude,
                "longitude": business_info.longitude,
            },
        )

    async def get_business_info(self, identifier: str) -> BusinessInfo:
        """
        Get detailed business information.

        Uses Yelp Fusion API if configured, otherwise scrapes.

        Args:
            identifier: Business ID or URL

        Returns:
            BusinessInfo object
        """
        self._ensure_initialized()

        # Extract business ID if URL provided
        if identifier.startswith("http"):
            business_id = self.extract_id(identifier)
            if not business_id:
                raise ValueError(f"Could not extract business ID from URL: {identifier}")
        else:
            business_id = identifier

        # Try Fusion API first if configured
        if self._yelp_config.api_key:
            try:
                return await self._get_business_info_api(business_id)
            except Exception:
                pass  # Fall back to scraping

        return await self._get_business_info_scrape(business_id)

    async def _get_business_info_api(self, business_id: str) -> BusinessInfo:
        """Get business info from Yelp Fusion API."""
        await self._rate_limiter.acquire(domain="api.yelp.com")

        url = f"{self.API_URL}/businesses/{business_id}"

        content = await self._httpx_scraper.scrape(
            url,
            headers={"Authorization": f"Bearer {self._yelp_config.api_key}"},
        )

        try:
            import json
            data = json.loads(content.content)

            # Extract categories
            categories = [c.get("title", "") for c in data.get("categories", [])]

            # Extract location
            location = data.get("location", {})

            # Extract coordinates
            coords = data.get("coordinates", {})

            return BusinessInfo(
                id=data.get("id", business_id),
                name=data.get("name", ""),
                url=data.get("url", f"{self.BASE_URL}/biz/{business_id}"),
                rating=data.get("rating"),
                review_count=data.get("review_count"),
                price=data.get("price"),
                categories=categories,
                address=location.get("address1"),
                city=location.get("city"),
                state=location.get("state"),
                zip_code=location.get("zip_code"),
                country=location.get("country", "US"),
                phone=data.get("display_phone"),
                latitude=coords.get("latitude"),
                longitude=coords.get("longitude"),
                image_url=data.get("image_url"),
                is_closed=data.get("is_closed", False),
            )

        except Exception as e:
            raise ScraperParseError(
                provider=self.platform_name,
                url=url,
                reason=f"Failed to parse Yelp API response: {e}",
                original_error=e,
            )

    async def _get_business_info_scrape(self, business_id: str) -> BusinessInfo:
        """Get business info by scraping Yelp."""
        await self._rate_limiter.acquire(domain=self.platform_domain)

        url = f"{self.BASE_URL}/biz/{business_id}"

        content = await self._playwright_scraper.scrape(
            url,
            wait_for="h1",
            timeout=self._platform_config.timeout * 1000,
        )

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "BeautifulSoup is required. Install with: pip install beautifulsoup4"
            )

        soup = BeautifulSoup(content.html or content.content, "html.parser")

        # Extract name
        name = ""
        name_elem = soup.select_one("h1")
        if name_elem:
            name = name_elem.get_text().strip()

        # Extract rating
        rating = None
        rating_elem = soup.select_one("[aria-label*='star rating']")
        if rating_elem:
            aria_label = rating_elem.get("aria-label", "")
            match = re.search(r"(\d+(?:\.\d+)?)", aria_label)
            if match:
                rating = float(match.group(1))

        # Extract review count
        review_count = None
        count_elem = soup.select_one("[class*='reviewCount'], .review-count")
        if count_elem:
            count_text = count_elem.get_text()
            match = re.search(r"(\d+)", count_text.replace(",", ""))
            if match:
                review_count = int(match.group(1))

        # Extract price
        price = None
        price_elem = soup.select_one("[class*='priceRange'], .price-range")
        if price_elem:
            price = price_elem.get_text().strip()

        # Extract categories
        categories = []
        cat_elems = soup.select("[class*='category'] a, .category-str-list a")
        for elem in cat_elems:
            cat_text = elem.get_text().strip()
            if cat_text:
                categories.append(cat_text)

        # Extract address
        address = None
        addr_elem = soup.select_one("address, [class*='address']")
        if addr_elem:
            address = addr_elem.get_text().strip()

        # Extract image
        image_url = None
        img_elem = soup.select_one("[class*='photo-header'] img, .biz-photo img")
        if img_elem:
            image_url = img_elem.get("src")

        return BusinessInfo(
            id=business_id,
            name=name,
            url=url,
            rating=rating,
            review_count=review_count,
            price=price,
            categories=categories,
            address=address,
            image_url=image_url,
        )

    async def search_businesses(
        self,
        term: str,
        location: Optional[str] = None,
        limit: int = 10,
        categories: Optional[str] = None,
        price: Optional[str] = None,
    ) -> List[BusinessInfo]:
        """
        Search for businesses.

        Args:
            term: Search term
            location: Location (defaults to config location)
            limit: Maximum number of results
            categories: Filter by category alias
            price: Filter by price (1, 2, 3, 4 or combination)

        Returns:
            List of BusinessInfo objects
        """
        self._ensure_initialized()

        location = location or self._yelp_config.location

        # Use API if configured
        if self._yelp_config.api_key:
            return await self._search_businesses_api(
                term, location, limit, categories, price
            )

        return await self._search_businesses_scrape(term, location, limit)

    async def _search_businesses_api(
        self,
        term: str,
        location: str,
        limit: int,
        categories: Optional[str],
        price: Optional[str],
    ) -> List[BusinessInfo]:
        """Search using Yelp Fusion API."""
        from urllib.parse import quote_plus

        await self._rate_limiter.acquire(domain="api.yelp.com")

        url = f"{self.API_URL}/businesses/search?term={quote_plus(term)}&location={quote_plus(location)}&limit={min(50, limit)}"

        if categories:
            url += f"&categories={categories}"
        if price:
            url += f"&price={price}"

        content = await self._httpx_scraper.scrape(
            url,
            headers={"Authorization": f"Bearer {self._yelp_config.api_key}"},
        )

        try:
            import json
            data = json.loads(content.content)

            results: List[BusinessInfo] = []

            for biz in data.get("businesses", [])[:limit]:
                location_data = biz.get("location", {})
                coords = biz.get("coordinates", {})
                categories_list = [c.get("title", "") for c in biz.get("categories", [])]

                results.append(BusinessInfo(
                    id=biz.get("id", ""),
                    name=biz.get("name", ""),
                    url=biz.get("url", ""),
                    rating=biz.get("rating"),
                    review_count=biz.get("review_count"),
                    price=biz.get("price"),
                    categories=categories_list,
                    address=location_data.get("address1"),
                    city=location_data.get("city"),
                    state=location_data.get("state"),
                    zip_code=location_data.get("zip_code"),
                    phone=biz.get("display_phone"),
                    latitude=coords.get("latitude"),
                    longitude=coords.get("longitude"),
                    image_url=biz.get("image_url"),
                    is_closed=biz.get("is_closed", False),
                ))

            return results

        except Exception:
            return []

    async def _search_businesses_scrape(
        self,
        term: str,
        location: str,
        limit: int,
    ) -> List[BusinessInfo]:
        """Search by scraping Yelp."""
        from urllib.parse import quote_plus

        await self._rate_limiter.acquire(domain=self.platform_domain)

        url = f"{self.BASE_URL}/search?find_desc={quote_plus(term)}&find_loc={quote_plus(location)}"

        content = await self._playwright_scraper.scrape(
            url,
            wait_for="[data-testid='serp-ia-card'], .result",
            timeout=self._platform_config.timeout * 1000,
        )

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "BeautifulSoup is required. Install with: pip install beautifulsoup4"
            )

        soup = BeautifulSoup(content.html or content.content, "html.parser")

        results: List[BusinessInfo] = []

        # Find search results
        result_elems = soup.select("[data-testid='serp-ia-card'], .result")[:limit]

        for elem in result_elems:
            # Extract business URL/ID
            link = elem.select_one("a[href*='/biz/']")
            if not link:
                continue

            href = link.get("href", "")
            match = re.search(r"/biz/([a-zA-Z0-9_-]+)", href)
            if not match:
                continue

            business_id = match.group(1)

            # Extract name
            name = link.get_text().strip()

            # Extract rating
            rating = None
            rating_elem = elem.select_one("[aria-label*='star rating']")
            if rating_elem:
                aria_label = rating_elem.get("aria-label", "")
                match = re.search(r"(\d+(?:\.\d+)?)", aria_label)
                if match:
                    rating = float(match.group(1))

            # Extract review count
            review_count = None
            count_elem = elem.select_one("[class*='reviewCount']")
            if count_elem:
                count_text = count_elem.get_text()
                match = re.search(r"(\d+)", count_text.replace(",", ""))
                if match:
                    review_count = int(match.group(1))

            # Extract price
            price = None
            price_elem = elem.select_one("[class*='priceRange']")
            if price_elem:
                price = price_elem.get_text().strip()

            # Extract image
            image_url = None
            img_elem = elem.select_one("img")
            if img_elem:
                image_url = img_elem.get("src")

            results.append(BusinessInfo(
                id=business_id,
                name=name,
                url=f"{self.BASE_URL}/biz/{business_id}",
                rating=rating,
                review_count=review_count,
                price=price,
                image_url=image_url,
            ))

        return results
