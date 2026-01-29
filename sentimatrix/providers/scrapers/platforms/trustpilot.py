"""
Trustpilot Company Review Scraper

Extracts company reviews from Trustpilot using:
1. Trustpilot website scraping via Playwright
2. Public data from Trustpilot pages

Features:
- Company reviews and ratings
- Search by company name or domain
- User verification status
- Reply tracking
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
class TrustpilotConfig(PlatformConfig):
    """Trustpilot-specific configuration."""

    # Country domain (us, uk, de, fr, etc.)
    country: str = "www"

    # Include company replies
    include_replies: bool = True

    # Rate limiting
    requests_per_second: float = 0.5
    burst_size: int = 3


@dataclass
class CompanyInfo:
    """Trustpilot company information."""

    id: str
    name: str
    url: str
    website: Optional[str] = None
    rating: Optional[float] = None  # TrustScore (1-5)
    review_count: Optional[int] = None
    category: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    logo_url: Optional[str] = None
    verified: bool = False
    claimed: bool = False
    rating_distribution: Dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "url": self.url,
            "website": self.website,
            "rating": self.rating,
            "review_count": self.review_count,
            "category": self.category,
            "location": self.location,
            "description": self.description,
            "logo_url": self.logo_url,
            "verified": self.verified,
            "claimed": self.claimed,
            "rating_distribution": self.rating_distribution,
        }


class TrustpilotScraper(BasePlatformScraper):
    """
    Trustpilot company review scraper.

    Scrapes company reviews and ratings from Trustpilot using Playwright.

    Example:
        >>> config = TrustpilotConfig()
        >>> async with TrustpilotScraper(config) as scraper:
        ...     reviews = await scraper.scrape_reviews("amazon.com", limit=50)
        ...     for review in reviews:
        ...         print(f"{review.rating}/5: {review.text[:50]}...")
    """

    # URL patterns for company ID extraction
    URL_PATTERNS = [
        re.compile(r"trustpilot\.com/review/([a-zA-Z0-9._-]+)"),
    ]

    # Base URL
    BASE_URL = "https://www.trustpilot.com"

    def __init__(
        self,
        config: Optional[TrustpilotConfig] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        """
        Initialize Trustpilot scraper.

        Args:
            config: Trustpilot-specific configuration
            rate_limiter: Optional rate limiter
        """
        self._trustpilot_config = config or TrustpilotConfig()
        super().__init__(self._trustpilot_config, rate_limiter)

        self._playwright_scraper = None

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="trustpilot",
            provider_type=ProviderType.SCRAPER,
            version="1.0.0",
            description="Trustpilot company review scraper",
            capabilities=ProviderCapabilities(
                javascript_rendering=True,
                proxy_support=True,
            ),
            website="https://trustpilot.com",
        )

    @property
    def platform_name(self) -> str:
        """Get platform name."""
        return "trustpilot"

    @property
    def platform_domain(self) -> str:
        """Get platform domain."""
        return "trustpilot.com"

    async def initialize(self) -> None:
        """Initialize the scraper."""
        if self._initialized:
            return

        from sentimatrix.providers.scrapers.playwright_scraper import PlaywrightScraper

        self._playwright_scraper = PlaywrightScraper(
            config=self._platform_config.to_scraper_config(),
            rate_limiter=self._rate_limiter,
            stealth=True,
        )
        await self._playwright_scraper.initialize()

        self._initialized = True

    async def close(self) -> None:
        """Close the scraper."""
        if self._playwright_scraper:
            await self._playwright_scraper.close()
            self._playwright_scraper = None

        self._initialized = False

    def validate_url(self, url: str) -> bool:
        """Validate Trustpilot URL."""
        for pattern in self.URL_PATTERNS:
            if pattern.search(url):
                return True
        return False

    def extract_id(self, url: str) -> Optional[str]:
        """Extract company ID/domain from URL."""
        for pattern in self.URL_PATTERNS:
            match = pattern.search(url)
            if match:
                return match.group(1)
        return None

    def _build_reviews_url(
        self,
        company_id: str,
        page: int = 1,
        sort_by: SortOrder = SortOrder.RECENT,
        filter_by: ReviewFilter = ReviewFilter.ALL,
    ) -> str:
        """Build reviews page URL."""
        base_url = f"{self.BASE_URL}/review/{company_id}"

        params = []

        if page > 1:
            params.append(f"page={page}")

        # Sort mapping
        sort_map = {
            SortOrder.RECENT: "recency",
            SortOrder.HELPFUL: "usefulness",
            SortOrder.RATING_HIGH: "rating",
            SortOrder.RATING_LOW: "rating",
        }

        if sort_by in sort_map:
            params.append(f"sort={sort_map[sort_by]}")

        # Filter by star rating
        filter_map = {
            ReviewFilter.POSITIVE: "stars=5,4",
            ReviewFilter.NEGATIVE: "stars=1,2",
            ReviewFilter.CRITICAL: "stars=1",
        }

        if filter_by in filter_map:
            params.append(filter_map[filter_by])

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
        Scrape reviews for a company.

        Args:
            identifier: Company domain (e.g., "amazon.com") or URL
            limit: Maximum number of reviews to fetch
            sort_by: Sort order
            filter_by: Review filter (star ratings)
            **kwargs: Additional parameters

        Returns:
            List of Review objects
        """
        self._ensure_initialized()

        # Extract company ID if URL provided
        if identifier.startswith("http"):
            company_id = self.extract_id(identifier)
            if not company_id:
                raise ValueError(f"Could not extract company ID from URL: {identifier}")
        else:
            # Assume it's a domain
            company_id = identifier.lower().replace("https://", "").replace("http://", "").replace("www.", "")

        reviews: List[Review] = []
        page = 1
        max_pages = (limit // 20) + 1  # Trustpilot shows ~20 reviews per page

        while len(reviews) < limit and page <= max_pages:
            url = self._build_reviews_url(company_id, page, sort_by, filter_by)

            try:
                await self._rate_limiter.acquire(domain=self.platform_domain)

                content = await self._playwright_scraper.scrape(
                    url,
                    wait_for="[data-review-id], .review-card",
                    timeout=self._platform_config.timeout * 1000,
                )

                page_reviews = self._parse_reviews_html(
                    content.html or content.content,
                    company_id
                )

                if not page_reviews:
                    break

                reviews.extend(page_reviews)
                page += 1

            except Exception as e:
                if "timeout" in str(e).lower():
                    break
                raise ScraperError(
                    f"Failed to scrape Trustpilot reviews: {e}",
                    provider=self.platform_name,
                    original_error=e,
                )

        return reviews[:limit]

    def _parse_reviews_html(self, html: str, company_id: str) -> List[Review]:
        """Parse reviews from HTML content."""
        reviews: List[Review] = []

        try:
            try:
                from bs4 import BeautifulSoup
            except ImportError:
                raise ImportError(
                    "BeautifulSoup is required for Trustpilot scraping. "
                    "Install with: pip install beautifulsoup4"
                )

            soup = BeautifulSoup(html, "html.parser")

            # Find review elements
            review_elements = soup.select("[data-review-id]")

            # Fallback selector
            if not review_elements:
                review_elements = soup.select("article[class*='review']")

            for element in review_elements:
                try:
                    review = self._parse_single_review(element, company_id)
                    if review:
                        reviews.append(review)
                except Exception:
                    continue

        except Exception as e:
            raise ScraperParseError(
                provider=self.platform_name,
                url=f"Company:{company_id}",
                reason=f"Failed to parse reviews HTML: {e}",
                original_error=e,
            )

        return reviews

    def _parse_single_review(self, element: Any, company_id: str) -> Optional[Review]:
        """Parse a single review element."""
        # Extract review text
        text = ""
        text_elem = element.select_one(
            "[data-service-review-text-typography], .review-content__text, p[data-service-review-text-typography='true']"
        )
        if text_elem:
            text = self.clean_text(text_elem.get_text())

        if not text:
            return None

        # Extract title
        title = None
        title_elem = element.select_one(
            "[data-service-review-title-typography], .review-content__title, h2"
        )
        if title_elem:
            title = title_elem.get_text().strip()

        # Extract rating (from star count or data attribute)
        rating = None
        rating_elem = element.select_one(
            "[data-rating], .star-rating img, [class*='star']"
        )
        if rating_elem:
            # Try data attribute first
            rating_val = rating_elem.get("data-rating")
            if rating_val:
                try:
                    rating = float(rating_val)
                except ValueError:
                    pass

            # Try alt text or aria-label
            if rating is None:
                alt_text = rating_elem.get("alt", "") or rating_elem.get("aria-label", "")
                match = re.search(r"(\d+)", alt_text)
                if match:
                    rating = float(match.group(1))

        # Extract author
        author = None
        author_elem = element.select_one(
            "[data-consumer-name-typography], .consumer-info__name, [class*='consumer'] a"
        )
        if author_elem:
            author = author_elem.get_text().strip()

        # Extract date
        timestamp = None
        date_elem = element.select_one(
            "[data-service-review-date-of-experience-typography], time, .review-date"
        )
        if date_elem:
            date_text = date_elem.get("datetime") or date_elem.get_text()
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

        # Check for verified badge
        verified = False
        verified_elem = element.select_one("[data-verified-review], .verified-buyer, [class*='verified']")
        if verified_elem:
            verified = True

        # Check for company reply
        has_reply = False
        reply_text = None
        reply_elem = element.select_one("[data-company-reply], .reply-content, [class*='reply']")
        if reply_elem:
            has_reply = True
            reply_text = reply_elem.get_text().strip()

        # Extract experience date
        experience_date = None
        exp_elem = element.select_one("[data-service-review-date-of-experience-typography]")
        if exp_elem:
            exp_text = exp_elem.get_text()
            match = re.search(r"(\w+\s+\d+,?\s+\d{4})", exp_text)
            if match:
                experience_date = self.parse_date(match.group(1))

        # Check for useful votes
        useful_count = 0
        useful_elem = element.select_one("[data-service-review-useful-count], .useful-count")
        if useful_elem:
            useful_text = useful_elem.get_text()
            match = re.search(r"(\d+)", useful_text)
            if match:
                useful_count = int(match.group(1))

        return Review(
            id=str(review_id),
            text=text,
            source=f"{self.BASE_URL}/review/{company_id}",
            platform=self.platform_name,
            author=author,
            rating=rating,
            timestamp=timestamp,
            metadata={
                "company_id": company_id,
                "title": title,
                "verified": verified,
                "has_reply": has_reply,
                "reply_text": reply_text if self._trustpilot_config.include_replies else None,
                "experience_date": experience_date.isoformat() if experience_date else None,
                "useful_count": useful_count,
            },
        )

    async def get_product_info(self, identifier: str) -> ProductInfo:
        """
        Get company information.

        Args:
            identifier: Company domain or URL

        Returns:
            ProductInfo object
        """
        company_info = await self.get_company_info(identifier)

        return ProductInfo(
            id=company_info.id,
            name=company_info.name,
            platform=self.platform_name,
            url=company_info.url,
            description=company_info.description,
            rating=company_info.rating,
            review_count=company_info.review_count,
            image_url=company_info.logo_url,
            category=company_info.category,
            metadata={
                "website": company_info.website,
                "location": company_info.location,
                "verified": company_info.verified,
                "claimed": company_info.claimed,
                "rating_distribution": company_info.rating_distribution,
            },
        )

    async def get_company_info(self, identifier: str) -> CompanyInfo:
        """
        Get detailed company information.

        Args:
            identifier: Company domain or URL

        Returns:
            CompanyInfo object
        """
        self._ensure_initialized()

        # Extract company ID if URL provided
        if identifier.startswith("http"):
            company_id = self.extract_id(identifier)
            if not company_id:
                raise ValueError(f"Could not extract company ID from URL: {identifier}")
        else:
            company_id = identifier.lower().replace("https://", "").replace("http://", "").replace("www.", "")

        await self._rate_limiter.acquire(domain=self.platform_domain)

        url = f"{self.BASE_URL}/review/{company_id}"

        content = await self._playwright_scraper.scrape(
            url,
            wait_for="[data-business-unit-name], h1",
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
        name_elem = soup.select_one("[data-business-unit-name], h1")
        if name_elem:
            name = name_elem.get_text().strip()

        # Extract TrustScore rating
        rating = None
        rating_elem = soup.select_one("[data-rating], .star-rating")
        if rating_elem:
            rating_val = rating_elem.get("data-rating")
            if rating_val:
                try:
                    rating = float(rating_val)
                except ValueError:
                    pass

        # Fallback: try to find in text
        if rating is None:
            score_elem = soup.select_one("[class*='trustScore'], [class*='TrustScore']")
            if score_elem:
                score_text = score_elem.get_text()
                match = re.search(r"(\d+\.?\d*)", score_text)
                if match:
                    rating = float(match.group(1))

        # Extract review count
        review_count = None
        count_elem = soup.select_one("[data-reviews-count], .review-count")
        if count_elem:
            count_text = count_elem.get_text().replace(",", "").replace(".", "")
            match = re.search(r"(\d+)", count_text)
            if match:
                review_count = int(match.group(1))

        # Extract category
        category = None
        cat_elem = soup.select_one("[data-category], .category a")
        if cat_elem:
            category = cat_elem.get_text().strip()

        # Extract website
        website = None
        website_elem = soup.select_one("a[href*='redirect'][data-business-unit-website]")
        if website_elem:
            website = website_elem.get("data-business-unit-website")
        if not website:
            # Company ID is often the website domain
            website = f"https://{company_id}"

        # Extract description/about
        description = None
        desc_elem = soup.select_one("[data-about-company], .about-company")
        if desc_elem:
            description = desc_elem.get_text().strip()

        # Extract logo
        logo_url = None
        logo_elem = soup.select_one("[data-business-unit-logo] img, .business-logo img")
        if logo_elem:
            logo_url = logo_elem.get("src")

        # Check for verified/claimed status
        verified = False
        claimed = False
        verified_elem = soup.select_one("[data-verified-business], .verified-badge")
        if verified_elem:
            verified = True
            claimed = True

        # Extract rating distribution
        rating_distribution = {}
        for star in range(1, 6):
            star_elem = soup.select_one(f"[data-star-rating='{star}']")
            if star_elem:
                count_text = star_elem.get_text()
                match = re.search(r"(\d+)", count_text.replace(",", ""))
                if match:
                    rating_distribution[star] = int(match.group(1))

        return CompanyInfo(
            id=company_id,
            name=name,
            url=url,
            website=website,
            rating=rating,
            review_count=review_count,
            category=category,
            description=description,
            logo_url=logo_url,
            verified=verified,
            claimed=claimed,
            rating_distribution=rating_distribution,
        )

    async def search_companies(
        self,
        query: str,
        limit: int = 10,
        category: Optional[str] = None,
    ) -> List[CompanyInfo]:
        """
        Search for companies.

        Args:
            query: Search query
            limit: Maximum number of results
            category: Filter by category

        Returns:
            List of CompanyInfo objects
        """
        self._ensure_initialized()

        from urllib.parse import quote_plus

        await self._rate_limiter.acquire(domain=self.platform_domain)

        url = f"{self.BASE_URL}/search?query={quote_plus(query)}"
        if category:
            url += f"&category={quote_plus(category)}"

        content = await self._playwright_scraper.scrape(
            url,
            wait_for=".search-results, [data-search-result]",
            timeout=self._platform_config.timeout * 1000,
        )

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "BeautifulSoup is required. Install with: pip install beautifulsoup4"
            )

        soup = BeautifulSoup(content.html or content.content, "html.parser")

        results: List[CompanyInfo] = []

        # Find search results
        result_elems = soup.select("[data-search-result], .search-result")[:limit]

        for elem in result_elems:
            # Extract company URL/ID
            link = elem.select_one("a[href*='/review/']")
            if not link:
                continue

            href = link.get("href", "")
            match = re.search(r"/review/([a-zA-Z0-9._-]+)", href)
            if not match:
                continue

            company_id = match.group(1)

            # Extract name
            name_elem = elem.select_one("[data-business-unit-name], h3, .business-name")
            name = name_elem.get_text().strip() if name_elem else company_id

            # Extract rating
            rating = None
            rating_elem = elem.select_one("[data-rating]")
            if rating_elem:
                try:
                    rating = float(rating_elem.get("data-rating"))
                except (ValueError, TypeError):
                    pass

            # Extract review count
            review_count = None
            count_elem = elem.select_one("[data-reviews-count], .review-count")
            if count_elem:
                count_text = count_elem.get_text().replace(",", "")
                match = re.search(r"(\d+)", count_text)
                if match:
                    review_count = int(match.group(1))

            # Extract category
            cat = None
            cat_elem = elem.select_one("[data-category], .category")
            if cat_elem:
                cat = cat_elem.get_text().strip()

            # Extract logo
            logo_url = None
            logo_elem = elem.select_one("img")
            if logo_elem:
                logo_url = logo_elem.get("src")

            results.append(CompanyInfo(
                id=company_id,
                name=name,
                url=f"{self.BASE_URL}/review/{company_id}",
                rating=rating,
                review_count=review_count,
                category=cat,
                logo_url=logo_url,
            ))

        return results
