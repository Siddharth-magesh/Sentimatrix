"""
Steam Game Review Scraper

Extracts game reviews from Steam using the official Steam Web API:
- Steam Reviews API (public, no key required for basic access)
- Store API for game information

Features:
- App ID and URL-based review fetching
- Language and filter support
- Playtime information
- Recommendation status (positive/negative)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
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
class SteamConfig(PlatformConfig):
    """Steam-specific configuration."""

    # Language for reviews
    language: str = "english"

    # Filter options
    purchase_type: str = "all"  # all, steam, non_steam_purchase

    # Review type
    review_type: str = "all"  # all, positive, negative

    # Day range (0 = all time)
    day_range: int = 0

    # Rate limiting (Steam is fairly permissive)
    requests_per_second: float = 2.0
    burst_size: int = 10

    # Steam Web API key (optional, for enhanced access)
    steam_api_key: Optional[str] = None


class SteamScraper(BasePlatformScraper):
    """
    Steam game review scraper.

    Uses Steam's public APIs:
    - Reviews API: https://store.steampowered.com/appreviews/{appid}
    - Store API: https://store.steampowered.com/api/appdetails

    Example:
        >>> config = SteamConfig(language="english")
        >>> async with SteamScraper(config) as scraper:
        ...     reviews = await scraper.scrape_reviews("730", limit=100)  # CS:GO
        ...     for review in reviews:
        ...         print(f"{'👍' if review.rating else '👎'}: {review.text[:50]}...")
    """

    # Steam store URL patterns
    URL_PATTERNS = [
        re.compile(r"store\.steampowered\.com/app/(\d+)"),
        re.compile(r"steamcommunity\.com/app/(\d+)"),
    ]

    # API endpoints
    REVIEWS_API = "https://store.steampowered.com/appreviews/{app_id}"
    STORE_API = "https://store.steampowered.com/api/appdetails"
    SEARCH_API = "https://store.steampowered.com/api/storesearch"

    def __init__(
        self,
        config: Optional[SteamConfig] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        """
        Initialize Steam scraper.

        Args:
            config: Steam-specific configuration
            rate_limiter: Optional rate limiter
        """
        self._steam_config = config or SteamConfig()
        super().__init__(self._steam_config, rate_limiter)

        self._httpx_scraper = None

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="steam",
            provider_type=ProviderType.SCRAPER,
            version="1.0.0",
            description="Steam game review scraper using official API",
            capabilities=ProviderCapabilities(
                javascript_rendering=False,  # Uses API, no JS needed
                proxy_support=True,
            ),
            website="https://store.steampowered.com",
        )

    @property
    def platform_name(self) -> str:
        """Get platform name."""
        return "steam"

    @property
    def platform_domain(self) -> str:
        """Get platform domain."""
        return "store.steampowered.com"

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
        """Validate Steam URL."""
        for pattern in self.URL_PATTERNS:
            if pattern.search(url):
                return True
        return False

    def extract_id(self, url: str) -> Optional[str]:
        """Extract app ID from URL."""
        for pattern in self.URL_PATTERNS:
            match = pattern.search(url)
            if match:
                return match.group(1)
        return None

    def validate_app_id(self, app_id: str) -> bool:
        """Validate app ID format."""
        try:
            int(app_id)
            return True
        except ValueError:
            return False

    async def scrape_reviews(
        self,
        identifier: str,
        limit: int = 100,
        sort_by: SortOrder = SortOrder.RECENT,
        filter_by: ReviewFilter = ReviewFilter.ALL,
        **kwargs: Any,
    ) -> List[Review]:
        """
        Scrape reviews for a game.

        Args:
            identifier: App ID or store URL
            limit: Maximum number of reviews to fetch
            sort_by: Sort order (RECENT, HELPFUL)
            filter_by: Review filter (ALL, POSITIVE, NEGATIVE)
            **kwargs: Additional parameters
                - language: Override language
                - playtime_min: Minimum playtime in hours
                - playtime_max: Maximum playtime in hours

        Returns:
            List of Review objects
        """
        self._ensure_initialized()

        # Extract app ID if URL provided
        if identifier.startswith("http"):
            app_id = self.extract_id(identifier)
            if not app_id:
                raise ValueError(f"Could not extract app ID from URL: {identifier}")
        else:
            app_id = identifier

        if not self.validate_app_id(app_id):
            raise ValueError(f"Invalid app ID format: {app_id}")

        reviews: List[Review] = []
        cursor = "*"  # Steam uses cursor-based pagination

        # Map sort order
        sort_map = {
            SortOrder.RECENT: "recent",
            SortOrder.HELPFUL: "all",  # 'all' sorts by helpfulness
            SortOrder.RELEVANCE: "all",
        }
        filter_map = {
            ReviewFilter.ALL: "all",
            ReviewFilter.POSITIVE: "positive",
            ReviewFilter.NEGATIVE: "negative",
        }

        language = kwargs.get("language", self._steam_config.language)
        review_type = filter_map.get(filter_by, "all")

        while len(reviews) < limit:
            # Build API URL
            params = {
                "json": "1",
                "language": language,
                "cursor": cursor,
                "num_per_page": min(100, limit - len(reviews)),
                "filter": sort_map.get(sort_by, "recent"),
                "review_type": review_type,
                "purchase_type": self._steam_config.purchase_type,
                "day_range": str(self._steam_config.day_range) if self._steam_config.day_range else "0",
            }

            url = self.REVIEWS_API.format(app_id=app_id)

            try:
                data = await self._fetch_json(url, params)

                if not data or data.get("success") != 1:
                    break

                page_reviews = data.get("reviews", [])
                if not page_reviews:
                    break

                for review_data in page_reviews:
                    review = self._parse_review(review_data, app_id)
                    if review:
                        reviews.append(review)

                # Get next cursor
                cursor = data.get("cursor", "")
                if not cursor:
                    break

            except Exception as e:
                raise ScraperError(
                    f"Failed to fetch Steam reviews: {e}",
                    provider=self.platform_name,
                    original_error=e,
                )

        return reviews[:limit]

    async def _fetch_json(
        self,
        url: str,
        params: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Fetch JSON from API."""
        await self._rate_limiter.acquire(domain=self.platform_domain)

        # Build URL with params
        if params:
            param_str = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{param_str}"

        content = await self._httpx_scraper.scrape(url)

        try:
            import json
            return json.loads(content.content)
        except Exception as e:
            raise ScraperParseError(
                provider=self.platform_name,
                url=url,
                reason=f"Invalid JSON response: {e}",
                original_error=e,
            )

    def _parse_review(self, data: Dict[str, Any], app_id: str) -> Optional[Review]:
        """Parse a single review from API response."""
        try:
            review_text = data.get("review", "")
            if not review_text:
                return None

            # Steam uses voted_up boolean for recommendation
            voted_up = data.get("voted_up", True)
            rating = 5.0 if voted_up else 1.0  # Convert to 5-star scale

            # Author info
            author_data = data.get("author", {})
            author_id = author_data.get("steamid", "")

            # Timestamp
            timestamp = None
            timestamp_created = data.get("timestamp_created")
            if timestamp_created:
                timestamp = datetime.fromtimestamp(timestamp_created)

            # Playtime
            playtime_forever = author_data.get("playtime_forever", 0)  # minutes
            playtime_at_review = author_data.get("playtime_at_review", 0)

            # Votes
            votes_up = data.get("votes_up", 0)
            votes_funny = data.get("votes_funny", 0)

            # Generate ID
            recommendation_id = data.get("recommendationid", "")
            review_id = recommendation_id or self.generate_review_id(
                platform=self.platform_name,
                text=review_text,
                author=author_id,
                timestamp=timestamp,
            )

            return Review(
                id=str(review_id),
                text=self.clean_text(review_text),
                source=f"https://store.steampowered.com/app/{app_id}",
                platform=self.platform_name,
                author=author_id,
                rating=rating,
                timestamp=timestamp,
                metadata={
                    "app_id": app_id,
                    "voted_up": voted_up,
                    "playtime_forever_hours": round(playtime_forever / 60, 1),
                    "playtime_at_review_hours": round(playtime_at_review / 60, 1),
                    "votes_up": votes_up,
                    "votes_funny": votes_funny,
                    "steam_purchase": data.get("steam_purchase", False),
                    "received_for_free": data.get("received_for_free", False),
                    "written_during_early_access": data.get("written_during_early_access", False),
                    "language": data.get("language", ""),
                    "weighted_vote_score": data.get("weighted_vote_score", 0),
                },
            )

        except Exception:
            return None

    async def get_product_info(self, identifier: str) -> ProductInfo:
        """
        Get game information.

        Args:
            identifier: App ID or store URL

        Returns:
            ProductInfo object
        """
        self._ensure_initialized()

        # Extract app ID if URL provided
        if identifier.startswith("http"):
            app_id = self.extract_id(identifier)
            if not app_id:
                raise ValueError(f"Could not extract app ID from URL: {identifier}")
        else:
            app_id = identifier

        if not self.validate_app_id(app_id):
            raise ValueError(f"Invalid app ID format: {app_id}")

        # Fetch from Store API
        url = f"{self.STORE_API}?appids={app_id}&cc=us&l=english"

        data = await self._fetch_json(url)

        app_data = data.get(str(app_id), {})
        if not app_data.get("success"):
            raise ScraperError(
                f"Game not found: {app_id}",
                provider=self.platform_name,
            )

        game_data = app_data.get("data", {})

        # Extract price
        price = None
        currency = None
        price_data = game_data.get("price_overview", {})
        if price_data:
            price = price_data.get("final", 0) / 100  # Price in cents
            currency = price_data.get("currency", "USD")

        # Extract review score
        rating = None
        review_count = None
        if "metacritic" in game_data:
            rating = self.normalize_rating(
                game_data["metacritic"].get("score", 0),
                0, 100, 5.0
            )

        # Get recommendations (reviews)
        recommendations = game_data.get("recommendations", {})
        review_count = recommendations.get("total", 0)

        return ProductInfo(
            id=app_id,
            name=game_data.get("name", ""),
            platform=self.platform_name,
            url=f"https://store.steampowered.com/app/{app_id}",
            description=game_data.get("short_description", ""),
            price=price,
            currency=currency,
            rating=rating,
            review_count=review_count,
            image_url=game_data.get("header_image"),
            category=", ".join(g.get("description", "") for g in game_data.get("genres", [])),
            metadata={
                "type": game_data.get("type", ""),
                "is_free": game_data.get("is_free", False),
                "required_age": game_data.get("required_age", 0),
                "developers": game_data.get("developers", []),
                "publishers": game_data.get("publishers", []),
                "platforms": game_data.get("platforms", {}),
                "release_date": game_data.get("release_date", {}),
                "categories": [c.get("description") for c in game_data.get("categories", [])],
            },
        )

    async def search_games(
        self,
        query: str,
        limit: int = 10,
    ) -> List[ProductInfo]:
        """
        Search for games.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of ProductInfo objects
        """
        self._ensure_initialized()

        from urllib.parse import quote_plus

        url = f"{self.SEARCH_API}?term={quote_plus(query)}&l=english&cc=US"

        data = await self._fetch_json(url)

        games: List[ProductInfo] = []

        for item in data.get("items", [])[:limit]:
            app_id = str(item.get("id", ""))

            # Extract price
            price = None
            price_data = item.get("price", {})
            if price_data:
                final = price_data.get("final", 0)
                if final:
                    price = final / 100

            games.append(ProductInfo(
                id=app_id,
                name=item.get("name", ""),
                platform=self.platform_name,
                url=f"https://store.steampowered.com/app/{app_id}",
                price=price,
                image_url=item.get("tiny_image"),
                metadata={
                    "metascore": item.get("metascore", ""),
                    "platforms": item.get("platforms", {}),
                },
            ))

        return games

    async def get_review_summary(self, app_id: str) -> Dict[str, Any]:
        """
        Get review summary for a game.

        Args:
            app_id: Steam app ID

        Returns:
            Dictionary with review summary
        """
        self._ensure_initialized()

        url = self.REVIEWS_API.format(app_id=app_id)
        params = {
            "json": "1",
            "language": "all",
            "num_per_page": "0",  # Just get summary
        }

        data = await self._fetch_json(url, params)

        if not data or data.get("success") != 1:
            raise ScraperError(
                f"Failed to get review summary for app {app_id}",
                provider=self.platform_name,
            )

        summary = data.get("query_summary", {})

        return {
            "total_positive": summary.get("total_positive", 0),
            "total_negative": summary.get("total_negative", 0),
            "total_reviews": summary.get("total_reviews", 0),
            "review_score": summary.get("review_score", 0),
            "review_score_desc": summary.get("review_score_desc", ""),
        }
