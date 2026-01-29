"""
IMDB Movie/TV Review Scraper

Extracts reviews from IMDB using:
1. IMDB website scraping via Playwright
2. OMDb API for movie information (optional, requires API key)

Features:
- Movie and TV show reviews
- Title ID and URL-based fetching
- Rating and spoiler filtering
- User ratings and helpfulness scores
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
class IMDBConfig(PlatformConfig):
    """IMDB-specific configuration."""

    # OMDb API key (optional, for enhanced movie info)
    omdb_api_key: Optional[str] = None

    # Review settings
    include_spoilers: bool = False
    filter_rating: Optional[int] = None  # 1-10, filter by user rating

    # Rate limiting (IMDB is fairly permissive)
    requests_per_second: float = 1.0
    burst_size: int = 5


@dataclass
class MovieInfo:
    """IMDB movie/TV show information."""

    id: str
    title: str
    year: Optional[str] = None
    type: str = "movie"  # movie, series, episode
    rating: Optional[float] = None
    votes: Optional[int] = None
    runtime: Optional[str] = None
    genres: List[str] = None
    director: Optional[str] = None
    plot: Optional[str] = None
    poster_url: Optional[str] = None
    awards: Optional[str] = None
    box_office: Optional[str] = None
    metascore: Optional[int] = None

    def __post_init__(self):
        if self.genres is None:
            self.genres = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "year": self.year,
            "type": self.type,
            "rating": self.rating,
            "votes": self.votes,
            "runtime": self.runtime,
            "genres": self.genres,
            "director": self.director,
            "plot": self.plot,
            "poster_url": self.poster_url,
            "awards": self.awards,
            "box_office": self.box_office,
            "metascore": self.metascore,
        }


class IMDBScraper(BasePlatformScraper):
    """
    IMDB movie and TV show review scraper.

    Uses Playwright for web scraping and optionally OMDb API for
    enhanced movie information.

    Example:
        >>> config = IMDBConfig()
        >>> async with IMDBScraper(config) as scraper:
        ...     reviews = await scraper.scrape_reviews("tt0111161", limit=50)  # Shawshank
        ...     for review in reviews:
        ...         print(f"{review.rating}/10: {review.text[:50]}...")
    """

    # IMDB title ID patterns
    TITLE_ID_PATTERN = re.compile(r"tt\d{7,}")

    # URL patterns for ID extraction
    URL_PATTERNS = [
        re.compile(r"imdb\.com/title/(tt\d{7,})"),
    ]

    # Base URLs
    BASE_URL = "https://www.imdb.com"
    OMDB_API_URL = "http://www.omdbapi.com"

    def __init__(
        self,
        config: Optional[IMDBConfig] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        """
        Initialize IMDB scraper.

        Args:
            config: IMDB-specific configuration
            rate_limiter: Optional rate limiter
        """
        self._imdb_config = config or IMDBConfig()
        super().__init__(self._imdb_config, rate_limiter)

        self._playwright_scraper = None
        self._httpx_scraper = None

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="imdb",
            provider_type=ProviderType.SCRAPER,
            version="1.0.0",
            description="IMDB movie and TV show review scraper",
            capabilities=ProviderCapabilities(
                javascript_rendering=True,
                proxy_support=True,
            ),
            website="https://imdb.com",
        )

    @property
    def platform_name(self) -> str:
        """Get platform name."""
        return "imdb"

    @property
    def platform_domain(self) -> str:
        """Get platform domain."""
        return "imdb.com"

    async def initialize(self) -> None:
        """Initialize the scraper."""
        if self._initialized:
            return

        from sentimatrix.providers.scrapers.playwright_scraper import PlaywrightScraper
        from sentimatrix.providers.scrapers.httpx_scraper import HTTPXScraper

        # Initialize Playwright for JavaScript-rendered content
        self._playwright_scraper = PlaywrightScraper(
            config=self._platform_config.to_scraper_config(),
            rate_limiter=self._rate_limiter,
            stealth=True,
        )
        await self._playwright_scraper.initialize()

        # Initialize HTTPX for API calls
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
        """Validate IMDB URL."""
        for pattern in self.URL_PATTERNS:
            if pattern.search(url):
                return True
        return False

    def extract_id(self, url: str) -> Optional[str]:
        """Extract title ID from URL."""
        for pattern in self.URL_PATTERNS:
            match = pattern.search(url)
            if match:
                return match.group(1)
        return None

    def validate_title_id(self, title_id: str) -> bool:
        """Validate title ID format."""
        return bool(self.TITLE_ID_PATTERN.fullmatch(title_id))

    def _build_reviews_url(
        self,
        title_id: str,
        sort_by: SortOrder = SortOrder.HELPFUL,
        filter_by: ReviewFilter = ReviewFilter.ALL,
    ) -> str:
        """Build reviews page URL."""
        base_url = f"{self.BASE_URL}/title/{title_id}/reviews"

        # Sort mapping
        sort_map = {
            SortOrder.RECENT: "submissionDate",
            SortOrder.HELPFUL: "helpfulnessScore",
            SortOrder.RATING_HIGH: "userRating",
            SortOrder.RATING_LOW: "userRating",
        }

        params = []

        if sort_by in sort_map:
            params.append(f"sort={sort_map[sort_by]}")
            if sort_by == SortOrder.RATING_LOW:
                params.append("dir=asc")
            else:
                params.append("dir=desc")

        # Filter by rating if specified
        if self._imdb_config.filter_rating:
            params.append(f"ratingFilter={self._imdb_config.filter_rating}")

        if params:
            return f"{base_url}?{'&'.join(params)}"
        return base_url

    async def scrape_reviews(
        self,
        identifier: str,
        limit: int = 100,
        sort_by: SortOrder = SortOrder.HELPFUL,
        filter_by: ReviewFilter = ReviewFilter.ALL,
        **kwargs: Any,
    ) -> List[Review]:
        """
        Scrape reviews for a movie or TV show.

        Args:
            identifier: Title ID (tt1234567) or URL
            limit: Maximum number of reviews to fetch
            sort_by: Sort order
            filter_by: Review filter
            **kwargs: Additional parameters

        Returns:
            List of Review objects
        """
        self._ensure_initialized()

        # Extract title ID if URL provided
        if identifier.startswith("http"):
            title_id = self.extract_id(identifier)
            if not title_id:
                raise ValueError(f"Could not extract title ID from URL: {identifier}")
        else:
            title_id = identifier.lower()

        if not self.validate_title_id(title_id):
            raise ValueError(f"Invalid IMDB title ID format: {title_id}")

        reviews: List[Review] = []
        url = self._build_reviews_url(title_id, sort_by, filter_by)

        try:
            # Rate limit
            await self._rate_limiter.acquire(domain=self.platform_domain)

            # Scrape reviews page with Playwright
            content = await self._playwright_scraper.scrape(
                url,
                wait_for="[data-testid='review-card'],.lister-item",
                timeout=self._platform_config.timeout * 1000,
            )

            page_reviews = self._parse_reviews_html(
                content.html or content.content,
                title_id
            )
            reviews.extend(page_reviews)

            # Load more reviews if needed (click "Load More" button)
            while len(reviews) < limit:
                # In a real implementation, we would use page interactions
                # to click "Load More" button and get additional reviews
                break

        except Exception as e:
            raise ScraperError(
                f"Failed to scrape IMDB reviews: {e}",
                provider=self.platform_name,
                original_error=e,
            )

        return reviews[:limit]

    def _parse_reviews_html(self, html: str, title_id: str) -> List[Review]:
        """Parse reviews from HTML content."""
        reviews: List[Review] = []

        try:
            try:
                from bs4 import BeautifulSoup
            except ImportError:
                raise ImportError(
                    "BeautifulSoup is required for IMDB scraping. "
                    "Install with: pip install beautifulsoup4"
                )

            soup = BeautifulSoup(html, "html.parser")

            # Try modern review cards first
            review_elements = soup.select("[data-testid='review-card']")

            # Fallback to legacy format
            if not review_elements:
                review_elements = soup.select(".lister-item.mode-detail")

            for element in review_elements:
                try:
                    review = self._parse_single_review(element, title_id)
                    if review:
                        reviews.append(review)
                except Exception:
                    continue

        except Exception as e:
            raise ScraperParseError(
                provider=self.platform_name,
                url=f"Title:{title_id}",
                reason=f"Failed to parse reviews HTML: {e}",
                original_error=e,
            )

        return reviews

    def _parse_single_review(self, element: Any, title_id: str) -> Optional[Review]:
        """Parse a single review element."""
        # Try to extract review text
        text = ""

        # Modern format
        text_elem = element.select_one("[data-testid='review-text'], .content .text")
        if text_elem:
            text = self.clean_text(text_elem.get_text())

        # Legacy format fallback
        if not text:
            text_elem = element.select_one(".text.show-more__control")
            if text_elem:
                text = self.clean_text(text_elem.get_text())

        if not text:
            return None

        # Check for spoilers
        spoiler_elem = element.select_one(".spoiler-warning, [data-testid='spoiler-warning']")
        has_spoiler = spoiler_elem is not None

        if has_spoiler and not self._imdb_config.include_spoilers:
            return None

        # Extract rating (out of 10)
        rating = None
        rating_elem = element.select_one(
            "[data-testid='review-rating'] span, .rating-other-user-rating span"
        )
        if rating_elem:
            rating_text = rating_elem.get_text()
            match = re.search(r"(\d+)", rating_text)
            if match:
                rating = float(match.group(1))

        # Extract author
        author = None
        author_elem = element.select_one(
            "[data-testid='review-author'], .display-name-link a"
        )
        if author_elem:
            author = author_elem.get_text().strip()

        # Extract title/headline
        title = None
        title_elem = element.select_one(
            "[data-testid='review-title'], .title"
        )
        if title_elem:
            title = title_elem.get_text().strip()

        # Extract date
        timestamp = None
        date_elem = element.select_one(
            "[data-testid='review-date'], .review-date"
        )
        if date_elem:
            date_text = date_elem.get_text()
            timestamp = self.parse_date(date_text)

        # Extract helpfulness
        helpful_votes = 0
        total_votes = 0
        helpful_elem = element.select_one(
            "[data-testid='review-helpful'], .actions"
        )
        if helpful_elem:
            helpful_text = helpful_elem.get_text()
            match = re.search(r"(\d+)\s+out\s+of\s+(\d+)", helpful_text)
            if match:
                helpful_votes = int(match.group(1))
                total_votes = int(match.group(2))

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
            source=f"{self.BASE_URL}/title/{title_id}",
            platform=self.platform_name,
            author=author,
            rating=rating,
            timestamp=timestamp,
            metadata={
                "title_id": title_id,
                "review_title": title,
                "has_spoiler": has_spoiler,
                "helpful_votes": helpful_votes,
                "total_votes": total_votes,
                "rating_scale": 10,
            },
        )

    async def get_product_info(self, identifier: str) -> ProductInfo:
        """
        Get movie/TV show information.

        Args:
            identifier: Title ID or URL

        Returns:
            ProductInfo object
        """
        movie_info = await self.get_movie_info(identifier)

        return ProductInfo(
            id=movie_info.id,
            name=movie_info.title,
            platform=self.platform_name,
            url=f"{self.BASE_URL}/title/{movie_info.id}",
            description=movie_info.plot,
            rating=movie_info.rating / 2 if movie_info.rating else None,  # Convert to 5-star
            review_count=movie_info.votes,
            image_url=movie_info.poster_url,
            category=", ".join(movie_info.genres) if movie_info.genres else None,
            metadata={
                "year": movie_info.year,
                "type": movie_info.type,
                "imdb_rating": movie_info.rating,
                "runtime": movie_info.runtime,
                "director": movie_info.director,
                "metascore": movie_info.metascore,
                "awards": movie_info.awards,
                "box_office": movie_info.box_office,
            },
        )

    async def get_movie_info(self, identifier: str) -> MovieInfo:
        """
        Get detailed movie/TV show information.

        Uses OMDb API if configured, otherwise scrapes IMDB directly.

        Args:
            identifier: Title ID or URL

        Returns:
            MovieInfo object
        """
        self._ensure_initialized()

        # Extract title ID if URL provided
        if identifier.startswith("http"):
            title_id = self.extract_id(identifier)
            if not title_id:
                raise ValueError(f"Could not extract title ID from URL: {identifier}")
        else:
            title_id = identifier.lower()

        if not self.validate_title_id(title_id):
            raise ValueError(f"Invalid IMDB title ID format: {title_id}")

        # Try OMDb API first if configured
        if self._imdb_config.omdb_api_key:
            try:
                return await self._get_movie_info_omdb(title_id)
            except Exception:
                pass  # Fall back to scraping

        # Scrape IMDB directly
        return await self._get_movie_info_scrape(title_id)

    async def _get_movie_info_omdb(self, title_id: str) -> MovieInfo:
        """Get movie info from OMDb API."""
        await self._rate_limiter.acquire(domain="omdbapi.com")

        url = f"{self.OMDB_API_URL}/?i={title_id}&apikey={self._imdb_config.omdb_api_key}&plot=full"

        content = await self._httpx_scraper.scrape(url)

        try:
            import json
            data = json.loads(content.content)

            if data.get("Response") == "False":
                raise ScraperError(
                    f"OMDb API error: {data.get('Error', 'Unknown error')}",
                    provider=self.platform_name,
                )

            # Parse rating
            rating = None
            if data.get("imdbRating") and data["imdbRating"] != "N/A":
                rating = float(data["imdbRating"])

            # Parse votes
            votes = None
            if data.get("imdbVotes") and data["imdbVotes"] != "N/A":
                votes = int(data["imdbVotes"].replace(",", ""))

            # Parse metascore
            metascore = None
            if data.get("Metascore") and data["Metascore"] != "N/A":
                metascore = int(data["Metascore"])

            # Parse genres
            genres = []
            if data.get("Genre") and data["Genre"] != "N/A":
                genres = [g.strip() for g in data["Genre"].split(",")]

            return MovieInfo(
                id=title_id,
                title=data.get("Title", ""),
                year=data.get("Year"),
                type=data.get("Type", "movie"),
                rating=rating,
                votes=votes,
                runtime=data.get("Runtime"),
                genres=genres,
                director=data.get("Director"),
                plot=data.get("Plot"),
                poster_url=data.get("Poster") if data.get("Poster") != "N/A" else None,
                awards=data.get("Awards"),
                box_office=data.get("BoxOffice"),
                metascore=metascore,
            )

        except Exception as e:
            raise ScraperParseError(
                provider=self.platform_name,
                url=url,
                reason=f"Failed to parse OMDb response: {e}",
                original_error=e,
            )

    async def _get_movie_info_scrape(self, title_id: str) -> MovieInfo:
        """Get movie info by scraping IMDB."""
        await self._rate_limiter.acquire(domain=self.platform_domain)

        url = f"{self.BASE_URL}/title/{title_id}"

        content = await self._playwright_scraper.scrape(
            url,
            wait_for="[data-testid='hero__pageTitle'], h1",
            timeout=self._platform_config.timeout * 1000,
        )

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "BeautifulSoup is required. Install with: pip install beautifulsoup4"
            )

        soup = BeautifulSoup(content.html or content.content, "html.parser")

        # Extract title
        title = ""
        title_elem = soup.select_one("[data-testid='hero__pageTitle'], h1")
        if title_elem:
            title = title_elem.get_text().strip()

        # Extract year
        year = None
        year_elem = soup.select_one("[data-testid='hero__releaseYear'] a, .title_wrapper a[href*='releaseinfo']")
        if year_elem:
            year_text = year_elem.get_text()
            match = re.search(r"(\d{4})", year_text)
            if match:
                year = match.group(1)

        # Extract rating
        rating = None
        rating_elem = soup.select_one("[data-testid='hero-rating-bar__aggregate-rating__score'] span")
        if rating_elem:
            try:
                rating = float(rating_elem.get_text())
            except ValueError:
                pass

        # Extract votes
        votes = None
        votes_elem = soup.select_one("[data-testid='hero-rating-bar__aggregate-rating__score'] + div")
        if votes_elem:
            votes_text = votes_elem.get_text()
            match = re.search(r"([\d,KMB]+)", votes_text)
            if match:
                votes_str = match.group(1).replace(",", "")
                if "K" in votes_str:
                    votes = int(float(votes_str.replace("K", "")) * 1000)
                elif "M" in votes_str:
                    votes = int(float(votes_str.replace("M", "")) * 1000000)
                else:
                    try:
                        votes = int(votes_str)
                    except ValueError:
                        pass

        # Extract genres
        genres = []
        genre_elems = soup.select("[data-testid='genres'] a, .genres a")
        for elem in genre_elems:
            genres.append(elem.get_text().strip())

        # Extract plot
        plot = None
        plot_elem = soup.select_one("[data-testid='plot'] span, .plot_summary")
        if plot_elem:
            plot = plot_elem.get_text().strip()

        # Extract poster
        poster_url = None
        poster_elem = soup.select_one("[data-testid='hero-media__poster'] img")
        if poster_elem:
            poster_url = poster_elem.get("src")

        # Determine type
        media_type = "movie"
        if "TV Series" in str(soup) or "TV Mini Series" in str(soup):
            media_type = "series"

        return MovieInfo(
            id=title_id,
            title=title,
            year=year,
            type=media_type,
            rating=rating,
            votes=votes,
            genres=genres,
            plot=plot,
            poster_url=poster_url,
        )

    async def search_titles(
        self,
        query: str,
        limit: int = 10,
        title_type: Optional[str] = None,
    ) -> List[MovieInfo]:
        """
        Search for movies/TV shows.

        Args:
            query: Search query
            limit: Maximum number of results
            title_type: Filter by type (movie, series, episode)

        Returns:
            List of MovieInfo objects
        """
        self._ensure_initialized()

        # If OMDb API is configured, use it
        if self._imdb_config.omdb_api_key:
            return await self._search_titles_omdb(query, limit, title_type)

        # Otherwise scrape IMDB search
        return await self._search_titles_scrape(query, limit, title_type)

    async def _search_titles_omdb(
        self,
        query: str,
        limit: int,
        title_type: Optional[str],
    ) -> List[MovieInfo]:
        """Search using OMDb API."""
        from urllib.parse import quote_plus

        await self._rate_limiter.acquire(domain="omdbapi.com")

        url = f"{self.OMDB_API_URL}/?s={quote_plus(query)}&apikey={self._imdb_config.omdb_api_key}"
        if title_type:
            url += f"&type={title_type}"

        content = await self._httpx_scraper.scrape(url)

        try:
            import json
            data = json.loads(content.content)

            if data.get("Response") == "False":
                return []

            results: List[MovieInfo] = []

            for item in data.get("Search", [])[:limit]:
                results.append(MovieInfo(
                    id=item.get("imdbID", ""),
                    title=item.get("Title", ""),
                    year=item.get("Year"),
                    type=item.get("Type", "movie"),
                    poster_url=item.get("Poster") if item.get("Poster") != "N/A" else None,
                ))

            return results

        except Exception:
            return []

    async def _search_titles_scrape(
        self,
        query: str,
        limit: int,
        title_type: Optional[str],
    ) -> List[MovieInfo]:
        """Search by scraping IMDB."""
        from urllib.parse import quote_plus

        await self._rate_limiter.acquire(domain=self.platform_domain)

        url = f"{self.BASE_URL}/find/?q={quote_plus(query)}&s=tt"
        if title_type:
            url += f"&ttype={title_type[:2]}"  # fe=feature, tv=tv series

        content = await self._playwright_scraper.scrape(
            url,
            wait_for=".ipc-metadata-list-summary-item",
            timeout=self._platform_config.timeout * 1000,
        )

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "BeautifulSoup is required. Install with: pip install beautifulsoup4"
            )

        soup = BeautifulSoup(content.html or content.content, "html.parser")

        results: List[MovieInfo] = []

        # Find search results
        result_elems = soup.select(".ipc-metadata-list-summary-item")[:limit]

        for elem in result_elems:
            # Extract title ID
            link = elem.select_one("a[href*='/title/']")
            if not link:
                continue

            href = link.get("href", "")
            match = re.search(r"/title/(tt\d+)", href)
            if not match:
                continue

            title_id = match.group(1)

            # Extract title
            title_elem = elem.select_one(".ipc-metadata-list-summary-item__t")
            title = title_elem.get_text().strip() if title_elem else ""

            # Extract year
            year = None
            year_elem = elem.select_one(".ipc-metadata-list-summary-item__li")
            if year_elem:
                year_text = year_elem.get_text()
                match = re.search(r"(\d{4})", year_text)
                if match:
                    year = match.group(1)

            # Extract poster
            poster_url = None
            img_elem = elem.select_one("img")
            if img_elem:
                poster_url = img_elem.get("src")

            results.append(MovieInfo(
                id=title_id,
                title=title,
                year=year,
                poster_url=poster_url,
            ))

        return results
