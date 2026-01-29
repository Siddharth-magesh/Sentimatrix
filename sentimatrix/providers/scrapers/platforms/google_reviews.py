"""
Google Reviews Scraper

Extracts business reviews from Google using:
1. Google Places API (recommended, requires API key)
2. SerpAPI for search results (alternative, requires API key)

Note: Direct scraping of Google is not recommended due to aggressive
anti-bot measures. This scraper primarily uses official APIs.

Features:
- Business reviews and ratings
- Place search by query
- User ratings and review counts
- Photo references
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from sentimatrix.core.exceptions import (
    ScraperError,
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


@dataclass
class GoogleReviewsConfig(PlatformConfig):
    """Google Reviews-specific configuration."""

    # Google Places API key (required)
    api_key: Optional[str] = None

    # Alternative: SerpAPI key for search
    serpapi_key: Optional[str] = None

    # Default location for searches
    location: str = "New York, NY"

    # Language for reviews
    language: str = "en"

    # Rate limiting (API has quotas)
    requests_per_second: float = 1.0
    burst_size: int = 5


@dataclass
class PlaceInfo:
    """Google Place information."""

    place_id: str
    name: str
    formatted_address: Optional[str] = None
    rating: Optional[float] = None
    user_ratings_total: Optional[int] = None
    price_level: Optional[int] = None  # 0-4
    types: List[str] = field(default_factory=list)
    business_status: Optional[str] = None  # OPERATIONAL, CLOSED_TEMPORARILY, etc.
    phone: Optional[str] = None
    website: Optional[str] = None
    url: Optional[str] = None  # Google Maps URL
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    photo_reference: Optional[str] = None
    opening_hours: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "place_id": self.place_id,
            "name": self.name,
            "formatted_address": self.formatted_address,
            "rating": self.rating,
            "user_ratings_total": self.user_ratings_total,
            "price_level": self.price_level,
            "types": self.types,
            "business_status": self.business_status,
            "phone": self.phone,
            "website": self.website,
            "url": self.url,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "photo_reference": self.photo_reference,
            "opening_hours": self.opening_hours,
        }


class GoogleReviewsScraper(BasePlatformScraper):
    """
    Google Reviews scraper using Google Places API.

    Note: Requires a Google Places API key for full functionality.
    Free tier includes limited requests per month.

    API Documentation: https://developers.google.com/maps/documentation/places/web-service

    Example:
        >>> config = GoogleReviewsConfig(api_key="your_api_key")
        >>> async with GoogleReviewsScraper(config) as scraper:
        ...     # Search for a place
        ...     places = await scraper.search_places("Eiffel Tower")
        ...     # Get reviews for the place
        ...     reviews = await scraper.scrape_reviews(places[0].place_id)
    """

    # URL patterns for place ID extraction
    URL_PATTERNS = [
        re.compile(r"google\.com/maps/place/[^/]+/@[^/]+/data=.*!1s([^!]+)"),
        re.compile(r"place_id[=:]([a-zA-Z0-9_-]+)"),
    ]

    # API endpoints
    PLACES_API_BASE = "https://maps.googleapis.com/maps/api/place"
    SERPAPI_BASE = "https://serpapi.com/search"

    def __init__(
        self,
        config: Optional[GoogleReviewsConfig] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        """
        Initialize Google Reviews scraper.

        Args:
            config: Google Reviews-specific configuration
            rate_limiter: Optional rate limiter
        """
        self._google_config = config or GoogleReviewsConfig()
        super().__init__(self._google_config, rate_limiter)

        self._httpx_scraper = None

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="google_reviews",
            provider_type=ProviderType.SCRAPER,
            version="1.0.0",
            description="Google Reviews scraper using Places API",
            capabilities=ProviderCapabilities(
                javascript_rendering=False,  # Uses API
                proxy_support=True,
            ),
            website="https://google.com/maps",
            documentation="https://developers.google.com/maps/documentation/places/web-service",
        )

    @property
    def platform_name(self) -> str:
        """Get platform name."""
        return "google_reviews"

    @property
    def platform_domain(self) -> str:
        """Get platform domain."""
        return "maps.googleapis.com"

    def _check_api_key(self) -> None:
        """Check if API key is configured."""
        if not self._google_config.api_key and not self._google_config.serpapi_key:
            raise ProviderInitializationError(
                self.platform_name,
                "Google Places API key or SerpAPI key is required. "
                "Set via GoogleReviewsConfig(api_key='...') or "
                "environment variable GOOGLE_PLACES_API_KEY",
            )

    async def initialize(self) -> None:
        """Initialize the scraper."""
        if self._initialized:
            return

        from sentimatrix.providers.scrapers.httpx_scraper import HTTPXScraper

        self._httpx_scraper = HTTPXScraper(
            config=self._platform_config.to_scraper_config(),
            rate_limiter=self._rate_limiter,
        )
        await self._httpx_scraper.initialize()

        self._initialized = True

    async def close(self) -> None:
        """Close the scraper."""
        if self._httpx_scraper:
            await self._httpx_scraper.close()
            self._httpx_scraper = None

        self._initialized = False

    def validate_url(self, url: str) -> bool:
        """Validate Google Maps URL."""
        return "google.com/maps" in url or "maps.google.com" in url

    def extract_id(self, url: str) -> Optional[str]:
        """Extract place ID from URL."""
        for pattern in self.URL_PATTERNS:
            match = pattern.search(url)
            if match:
                return match.group(1)
        return None

    async def scrape_reviews(
        self,
        identifier: str,
        limit: int = 100,
        sort_by: SortOrder = SortOrder.RECENT,
        filter_by: ReviewFilter = ReviewFilter.ALL,
        **kwargs: Any,
    ) -> List[Review]:
        """
        Scrape reviews for a place.

        Note: Google Places API returns a maximum of 5 reviews per request.
        For more reviews, consider using SerpAPI or other alternatives.

        Args:
            identifier: Place ID or Google Maps URL
            limit: Maximum number of reviews (API limited to 5)
            sort_by: Sort order (limited support)
            filter_by: Review filter (not supported by API)
            **kwargs: Additional parameters

        Returns:
            List of Review objects
        """
        self._ensure_initialized()
        self._check_api_key()

        # Extract place ID if URL provided
        if identifier.startswith("http"):
            place_id = self.extract_id(identifier)
            if not place_id:
                raise ValueError(f"Could not extract place ID from URL: {identifier}")
        else:
            place_id = identifier

        # Use SerpAPI for more reviews if available
        if self._google_config.serpapi_key:
            return await self._scrape_reviews_serpapi(place_id, limit, sort_by)

        # Fall back to Places API (limited to 5 reviews)
        return await self._scrape_reviews_places_api(place_id, limit)

    async def _scrape_reviews_places_api(
        self,
        place_id: str,
        limit: int,
    ) -> List[Review]:
        """Scrape reviews using Google Places API."""
        await self._rate_limiter.acquire(domain=self.platform_domain)

        # Place Details request with reviews
        url = (
            f"{self.PLACES_API_BASE}/details/json"
            f"?place_id={place_id}"
            f"&fields=name,reviews,url,rating,user_ratings_total"
            f"&key={self._google_config.api_key}"
            f"&language={self._google_config.language}"
        )

        content = await self._httpx_scraper.scrape(url)

        try:
            import json
            data = json.loads(content.content)

            if data.get("status") != "OK":
                error_msg = data.get("error_message", data.get("status", "Unknown error"))
                raise ScraperError(
                    f"Google Places API error: {error_msg}",
                    provider=self.platform_name,
                )

            result = data.get("result", {})
            place_name = result.get("name", "")
            place_url = result.get("url", "")

            reviews: List[Review] = []

            for review_data in result.get("reviews", [])[:limit]:
                review = self._parse_api_review(review_data, place_id, place_name, place_url)
                if review:
                    reviews.append(review)

            return reviews

        except Exception as e:
            if isinstance(e, ScraperError):
                raise
            raise ScraperParseError(
                provider=self.platform_name,
                url=url,
                reason=f"Failed to parse Places API response: {e}",
                original_error=e,
            )

    async def _scrape_reviews_serpapi(
        self,
        place_id: str,
        limit: int,
        sort_by: SortOrder,
    ) -> List[Review]:
        """Scrape reviews using SerpAPI (supports more reviews)."""
        from urllib.parse import quote_plus

        await self._rate_limiter.acquire(domain="serpapi.com")

        # Map sort order
        sort_map = {
            SortOrder.RECENT: "newestFirst",
            SortOrder.HELPFUL: "mostRelevant",
            SortOrder.RATING_HIGH: "ratingHigh",
            SortOrder.RATING_LOW: "ratingLow",
        }

        sort_by_param = sort_map.get(sort_by, "mostRelevant")

        url = (
            f"{self.SERPAPI_BASE}"
            f"?engine=google_maps_reviews"
            f"&place_id={quote_plus(place_id)}"
            f"&sort_by={sort_by_param}"
            f"&hl={self._google_config.language}"
            f"&api_key={self._google_config.serpapi_key}"
        )

        content = await self._httpx_scraper.scrape(url)

        try:
            import json
            data = json.loads(content.content)

            if "error" in data:
                raise ScraperError(
                    f"SerpAPI error: {data['error']}",
                    provider=self.platform_name,
                )

            place_info = data.get("place_info", {})
            place_name = place_info.get("title", "")
            place_url = place_info.get("link", "")

            reviews: List[Review] = []

            for review_data in data.get("reviews", [])[:limit]:
                review = self._parse_serpapi_review(review_data, place_id, place_name, place_url)
                if review:
                    reviews.append(review)

            return reviews

        except Exception as e:
            if isinstance(e, ScraperError):
                raise
            raise ScraperParseError(
                provider=self.platform_name,
                url=url,
                reason=f"Failed to parse SerpAPI response: {e}",
                original_error=e,
            )

    def _parse_api_review(
        self,
        data: Dict[str, Any],
        place_id: str,
        place_name: str,
        place_url: str,
    ) -> Optional[Review]:
        """Parse a review from Places API response."""
        text = data.get("text", "")
        if not text:
            return None

        # Parse timestamp
        timestamp = None
        time_val = data.get("time")
        if time_val:
            try:
                timestamp = datetime.fromtimestamp(time_val)
            except (ValueError, TypeError):
                pass

        return Review(
            id=self.generate_review_id(
                platform=self.platform_name,
                text=text,
                author=data.get("author_name"),
                timestamp=timestamp,
            ),
            text=text,
            source=place_url or f"https://www.google.com/maps/place/?q=place_id:{place_id}",
            platform=self.platform_name,
            author=data.get("author_name"),
            rating=data.get("rating"),
            timestamp=timestamp,
            metadata={
                "place_id": place_id,
                "place_name": place_name,
                "author_url": data.get("author_url"),
                "profile_photo_url": data.get("profile_photo_url"),
                "relative_time_description": data.get("relative_time_description"),
                "language": data.get("language"),
            },
        )

    def _parse_serpapi_review(
        self,
        data: Dict[str, Any],
        place_id: str,
        place_name: str,
        place_url: str,
    ) -> Optional[Review]:
        """Parse a review from SerpAPI response."""
        text = data.get("snippet", "") or data.get("text", "")
        if not text:
            return None

        # Parse timestamp from ISO date
        timestamp = None
        date_str = data.get("iso_date") or data.get("date")
        if date_str:
            timestamp = self.parse_date(date_str)

        return Review(
            id=data.get("review_id") or self.generate_review_id(
                platform=self.platform_name,
                text=text,
                author=data.get("user", {}).get("name"),
                timestamp=timestamp,
            ),
            text=text,
            source=place_url or f"https://www.google.com/maps/place/?q=place_id:{place_id}",
            platform=self.platform_name,
            author=data.get("user", {}).get("name"),
            rating=data.get("rating"),
            timestamp=timestamp,
            metadata={
                "place_id": place_id,
                "place_name": place_name,
                "likes": data.get("likes"),
                "user_reviews": data.get("user", {}).get("reviews"),
                "user_photos": data.get("user", {}).get("photos"),
                "response": data.get("response"),
            },
        )

    async def get_product_info(self, identifier: str) -> ProductInfo:
        """
        Get place information.

        Args:
            identifier: Place ID or URL

        Returns:
            ProductInfo object
        """
        place_info = await self.get_place_info(identifier)

        # Convert price level to string
        price_str = None
        if place_info.price_level is not None:
            price_str = "$" * (place_info.price_level + 1)

        return ProductInfo(
            id=place_info.place_id,
            name=place_info.name,
            platform=self.platform_name,
            url=place_info.url or f"https://www.google.com/maps/place/?q=place_id:{place_info.place_id}",
            rating=place_info.rating,
            review_count=place_info.user_ratings_total,
            image_url=None,  # Would need separate photo request
            category=", ".join(place_info.types[:3]) if place_info.types else None,
            metadata={
                "formatted_address": place_info.formatted_address,
                "price_level": price_str,
                "business_status": place_info.business_status,
                "phone": place_info.phone,
                "website": place_info.website,
                "latitude": place_info.latitude,
                "longitude": place_info.longitude,
                "types": place_info.types,
                "opening_hours": place_info.opening_hours,
            },
        )

    async def get_place_info(self, identifier: str) -> PlaceInfo:
        """
        Get detailed place information.

        Args:
            identifier: Place ID or URL

        Returns:
            PlaceInfo object
        """
        self._ensure_initialized()
        self._check_api_key()

        # Extract place ID if URL provided
        if identifier.startswith("http"):
            place_id = self.extract_id(identifier)
            if not place_id:
                raise ValueError(f"Could not extract place ID from URL: {identifier}")
        else:
            place_id = identifier

        await self._rate_limiter.acquire(domain=self.platform_domain)

        # Place Details request
        fields = (
            "place_id,name,formatted_address,geometry,rating,"
            "user_ratings_total,price_level,types,business_status,"
            "formatted_phone_number,website,url,opening_hours"
        )

        url = (
            f"{self.PLACES_API_BASE}/details/json"
            f"?place_id={place_id}"
            f"&fields={fields}"
            f"&key={self._google_config.api_key}"
        )

        content = await self._httpx_scraper.scrape(url)

        try:
            import json
            data = json.loads(content.content)

            if data.get("status") != "OK":
                error_msg = data.get("error_message", data.get("status", "Unknown error"))
                raise ScraperError(
                    f"Google Places API error: {error_msg}",
                    provider=self.platform_name,
                )

            result = data.get("result", {})
            geometry = result.get("geometry", {}).get("location", {})

            return PlaceInfo(
                place_id=result.get("place_id", place_id),
                name=result.get("name", ""),
                formatted_address=result.get("formatted_address"),
                rating=result.get("rating"),
                user_ratings_total=result.get("user_ratings_total"),
                price_level=result.get("price_level"),
                types=result.get("types", []),
                business_status=result.get("business_status"),
                phone=result.get("formatted_phone_number"),
                website=result.get("website"),
                url=result.get("url"),
                latitude=geometry.get("lat"),
                longitude=geometry.get("lng"),
                opening_hours=result.get("opening_hours"),
            )

        except Exception as e:
            if isinstance(e, ScraperError):
                raise
            raise ScraperParseError(
                provider=self.platform_name,
                url=url,
                reason=f"Failed to parse Places API response: {e}",
                original_error=e,
            )

    async def search_places(
        self,
        query: str,
        location: Optional[str] = None,
        limit: int = 10,
        place_type: Optional[str] = None,
    ) -> List[PlaceInfo]:
        """
        Search for places.

        Args:
            query: Search query
            location: Location bias (optional)
            limit: Maximum number of results
            place_type: Filter by type (restaurant, store, etc.)

        Returns:
            List of PlaceInfo objects
        """
        self._ensure_initialized()
        self._check_api_key()

        from urllib.parse import quote_plus

        await self._rate_limiter.acquire(domain=self.platform_domain)

        # Text Search request
        url = (
            f"{self.PLACES_API_BASE}/textsearch/json"
            f"?query={quote_plus(query)}"
            f"&key={self._google_config.api_key}"
        )

        if location:
            url += f"&location={quote_plus(location)}"

        if place_type:
            url += f"&type={place_type}"

        content = await self._httpx_scraper.scrape(url)

        try:
            import json
            data = json.loads(content.content)

            if data.get("status") not in ("OK", "ZERO_RESULTS"):
                error_msg = data.get("error_message", data.get("status", "Unknown error"))
                raise ScraperError(
                    f"Google Places API error: {error_msg}",
                    provider=self.platform_name,
                )

            results: List[PlaceInfo] = []

            for place in data.get("results", [])[:limit]:
                geometry = place.get("geometry", {}).get("location", {})
                photos = place.get("photos", [])

                results.append(PlaceInfo(
                    place_id=place.get("place_id", ""),
                    name=place.get("name", ""),
                    formatted_address=place.get("formatted_address"),
                    rating=place.get("rating"),
                    user_ratings_total=place.get("user_ratings_total"),
                    price_level=place.get("price_level"),
                    types=place.get("types", []),
                    business_status=place.get("business_status"),
                    latitude=geometry.get("lat"),
                    longitude=geometry.get("lng"),
                    photo_reference=photos[0].get("photo_reference") if photos else None,
                ))

            return results

        except Exception as e:
            if isinstance(e, ScraperError):
                raise
            raise ScraperParseError(
                provider=self.platform_name,
                url=url,
                reason=f"Failed to parse Places API response: {e}",
                original_error=e,
            )

    async def find_place(self, input_text: str, input_type: str = "textquery") -> Optional[PlaceInfo]:
        """
        Find a single place by text or phone number.

        Args:
            input_text: Search text or phone number
            input_type: "textquery" or "phonenumber"

        Returns:
            PlaceInfo if found, None otherwise
        """
        self._ensure_initialized()
        self._check_api_key()

        from urllib.parse import quote_plus

        await self._rate_limiter.acquire(domain=self.platform_domain)

        url = (
            f"{self.PLACES_API_BASE}/findplacefromtext/json"
            f"?input={quote_plus(input_text)}"
            f"&inputtype={input_type}"
            f"&fields=place_id,name,formatted_address,rating,user_ratings_total"
            f"&key={self._google_config.api_key}"
        )

        content = await self._httpx_scraper.scrape(url)

        try:
            import json
            data = json.loads(content.content)

            if data.get("status") != "OK":
                return None

            candidates = data.get("candidates", [])
            if not candidates:
                return None

            place = candidates[0]
            return PlaceInfo(
                place_id=place.get("place_id", ""),
                name=place.get("name", ""),
                formatted_address=place.get("formatted_address"),
                rating=place.get("rating"),
                user_ratings_total=place.get("user_ratings_total"),
            )

        except Exception:
            return None
