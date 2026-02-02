"""
Reddit Post and Comment Scraper

Extracts posts and comments from Reddit using:
1. Reddit JSON API (no authentication for public data)
2. PRAW (Python Reddit API Wrapper) for authenticated access

Features:
- Subreddit post scraping
- Comment extraction with threading
- Search functionality
- User activity scraping
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
class RedditConfig(PlatformConfig):
    """Reddit-specific configuration."""

    # OAuth credentials (optional, for authenticated access)
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    user_agent: str = "Sentimatrix/1.0"

    # Rate limiting (Reddit is strict: 60 req/min for OAuth, 30 for anonymous)
    requests_per_second: float = 0.5
    burst_size: int = 5

    # Comment settings
    comment_depth: int = 10  # Maximum depth to traverse
    comment_limit: int = 100  # Comments per request


@dataclass
class RedditPost:
    """Reddit post information."""

    id: str
    title: str
    selftext: str
    author: str
    subreddit: str
    score: int
    upvote_ratio: float
    num_comments: int
    created_utc: datetime
    url: str
    permalink: str
    is_self: bool
    is_video: bool
    over_18: bool
    spoiler: bool
    stickied: bool
    link_flair_text: Optional[str] = None
    awards: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "selftext": self.selftext,
            "author": self.author,
            "subreddit": self.subreddit,
            "score": self.score,
            "upvote_ratio": self.upvote_ratio,
            "num_comments": self.num_comments,
            "created_utc": self.created_utc.isoformat(),
            "url": self.url,
            "permalink": self.permalink,
            "is_self": self.is_self,
            "is_video": self.is_video,
            "over_18": self.over_18,
            "spoiler": self.spoiler,
            "stickied": self.stickied,
            "link_flair_text": self.link_flair_text,
            "awards": self.awards,
        }


@dataclass
class RedditComment:
    """Reddit comment."""

    id: str
    body: str
    author: str
    score: int
    created_utc: datetime
    permalink: str
    parent_id: str
    is_submitter: bool
    depth: int
    replies: List["RedditComment"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "body": self.body,
            "author": self.author,
            "score": self.score,
            "created_utc": self.created_utc.isoformat(),
            "permalink": self.permalink,
            "parent_id": self.parent_id,
            "is_submitter": self.is_submitter,
            "depth": self.depth,
            "replies": [r.to_dict() for r in self.replies],
        }


class RedditScraper(BasePlatformScraper):
    """
    Reddit post and comment scraper.

    Uses Reddit's JSON API for public data. For authenticated access
    and higher rate limits, provide OAuth credentials.

    Example:
        >>> config = RedditConfig()
        >>> async with RedditScraper(config) as scraper:
        ...     posts = await scraper.get_subreddit_posts("python", limit=25)
        ...     comments = await scraper.scrape_reviews(posts[0].id)
    """

    # URL patterns
    URL_PATTERNS = [
        re.compile(r"reddit\.com/r/(\w+)/comments/(\w+)"),
        re.compile(r"reddit\.com/r/(\w+)/?$"),
        re.compile(r"redd\.it/(\w+)"),
    ]

    # API endpoints (JSON API)
    BASE_URL = "https://www.reddit.com"
    OAUTH_URL = "https://oauth.reddit.com"

    def __init__(
        self,
        config: Optional[RedditConfig] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        """
        Initialize Reddit scraper.

        Args:
            config: Reddit-specific configuration
            rate_limiter: Optional rate limiter
        """
        self._reddit_config = config or RedditConfig()
        super().__init__(self._reddit_config, rate_limiter)

        self._httpx_scraper = None
        self._access_token: Optional[str] = None
        self._token_expires: Optional[datetime] = None

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="reddit",
            provider_type=ProviderType.SCRAPER,
            version="1.0.0",
            description="Reddit post and comment scraper",
            capabilities=ProviderCapabilities(
                javascript_rendering=False,
                proxy_support=True,
            ),
            website="https://reddit.com",
        )

    @property
    def platform_name(self) -> str:
        """Get platform name."""
        return "reddit"

    @property
    def platform_domain(self) -> str:
        """Get platform domain."""
        return "reddit.com"

    @property
    def _is_authenticated(self) -> bool:
        """Check if using OAuth authentication."""
        return bool(
            self._reddit_config.client_id and
            self._reddit_config.client_secret
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

        # Get OAuth token if credentials provided
        if self._is_authenticated:
            await self._refresh_token()

        self._initialized = True

    async def close(self) -> None:
        """Close the scraper."""
        if self._httpx_scraper:
            await self._httpx_scraper.close()
            self._httpx_scraper = None

        self._access_token = None
        self._initialized = False

    async def _refresh_token(self) -> None:
        """Refresh OAuth access token."""
        if not self._is_authenticated:
            return

        import base64

        auth = base64.b64encode(
            f"{self._reddit_config.client_id}:{self._reddit_config.client_secret}".encode()
        ).decode()

        # This would need actual HTTP POST, simplified here
        # In production, use proper OAuth flow
        self._access_token = None  # Placeholder

    def validate_url(self, url: str) -> bool:
        """Validate Reddit URL."""
        for pattern in self.URL_PATTERNS:
            if pattern.search(url):
                return True
        return False

    def extract_id(self, url: str) -> Optional[str]:
        """Extract post ID from URL."""
        # Try to extract from full URL
        match = re.search(r"/comments/(\w+)", url)
        if match:
            return match.group(1)

        # Try short URL
        match = re.search(r"redd\.it/(\w+)", url)
        if match:
            return match.group(1)

        return None

    def _extract_subreddit(self, url: str) -> Optional[str]:
        """Extract subreddit name from URL."""
        match = re.search(r"/r/(\w+)", url)
        if match:
            return match.group(1)
        return None

    async def _fetch_json(
        self,
        endpoint: str,
        params: Optional[Dict[str, str]] = None,
    ) -> Any:
        """Fetch JSON from Reddit API."""
        await self._rate_limiter.acquire(domain=self.platform_domain)

        # Build URL
        base = self.OAUTH_URL if self._access_token else self.BASE_URL
        url = f"{base}{endpoint}.json"

        if params:
            param_str = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{param_str}"

        # Add headers
        headers = {
            "User-Agent": self._reddit_config.user_agent,
        }
        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"

        try:
            content = await self._httpx_scraper.scrape(
                url,
                headers=headers,
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "rate" in error_msg:
                raise ScraperError(
                    f"Reddit rate limit exceeded. Please wait before retrying.",
                    provider=self.platform_name,
                )
            if "403" in error_msg or "forbidden" in error_msg:
                raise ScraperError(
                    f"Access forbidden. Post may be private or subreddit restricted.",
                    provider=self.platform_name,
                )
            if "404" in error_msg or "not found" in error_msg:
                raise ScraperError(
                    f"Post or subreddit not found: {endpoint}",
                    provider=self.platform_name,
                )
            raise ScraperError(
                f"Failed to fetch Reddit data: {e}",
                provider=self.platform_name,
            )

        # Check HTTP status code
        if hasattr(content, 'status_code'):
            if content.status_code == 429:
                raise ScraperError(
                    f"Reddit rate limit exceeded. Please wait before retrying.",
                    provider=self.platform_name,
                )
            if content.status_code == 403:
                raise ScraperError(
                    f"Access forbidden. Post may be private or subreddit restricted.",
                    provider=self.platform_name,
                )
            if content.status_code == 404:
                raise ScraperError(
                    f"Post or subreddit not found: {endpoint}",
                    provider=self.platform_name,
                )

        try:
            import json
            data = json.loads(content.content)

            # Check for Reddit error responses
            if isinstance(data, dict):
                if "error" in data:
                    error_code = data.get("error", "")
                    message = data.get("message", "Unknown error")
                    raise ScraperError(
                        f"Reddit API error ({error_code}): {message}",
                        provider=self.platform_name,
                    )
                # Handle empty/deleted post responses
                if data.get("kind") == "Listing" and not data.get("data", {}).get("children"):
                    raise ScraperError(
                        f"No data found. Post may be deleted or private.",
                        provider=self.platform_name,
                    )

            return data
        except json.JSONDecodeError as e:
            # Check if the response is HTML (often indicates an error page)
            if content.content.strip().startswith(("<!DOCTYPE", "<html", "<!doctype")):
                raise ScraperError(
                    f"Received HTML instead of JSON. Reddit may be blocking requests or post doesn't exist.",
                    provider=self.platform_name,
                )
            raise ScraperParseError(
                provider=self.platform_name,
                url=url,
                reason=f"Invalid JSON response: {e}",
                original_error=e,
            )

    async def scrape_reviews(
        self,
        identifier: str,
        limit: int = 100,
        sort_by: SortOrder = SortOrder.RELEVANCE,
        filter_by: ReviewFilter = ReviewFilter.ALL,
        **kwargs: Any,
    ) -> List[Review]:
        """
        Scrape comments from a post.

        Args:
            identifier: Post ID or URL
            limit: Maximum number of comments
            sort_by: Sort order (RELEVANCE, RECENT, etc.)
            filter_by: Not used for Reddit
            **kwargs: Additional parameters
                - depth: Maximum comment depth

        Returns:
            List of Review objects (comments)
        """
        self._ensure_initialized()

        # Extract post ID if URL provided
        if identifier.startswith("http"):
            post_id = self.extract_id(identifier)
            subreddit = self._extract_subreddit(identifier)
            if not post_id:
                raise ValueError(f"Could not extract post ID from URL: {identifier}")
        else:
            post_id = identifier
            # Handle full Reddit ID format (e.g., "t3_abc123" -> "abc123")
            if post_id.startswith("t3_"):
                post_id = post_id[3:]
            subreddit = None

        # Map sort order
        sort_map = {
            SortOrder.RELEVANCE: "confidence",
            SortOrder.RECENT: "new",
            SortOrder.HELPFUL: "top",
            SortOrder.RATING_HIGH: "top",
        }
        sort = sort_map.get(sort_by, "confidence")

        depth = kwargs.get("depth", self._reddit_config.comment_depth)

        # Fetch post and comments
        if subreddit:
            endpoint = f"/r/{subreddit}/comments/{post_id}"
        else:
            endpoint = f"/comments/{post_id}"

        params = {
            "sort": sort,
            "limit": str(self._reddit_config.comment_limit),
            "depth": str(depth),
        }

        data = await self._fetch_json(endpoint, params)

        if not isinstance(data, list) or len(data) < 2:
            # Check if it's an error response
            if isinstance(data, dict) and "error" in data:
                raise ScraperError(
                    f"Reddit API error: {data.get('message', 'Unknown error')}",
                    provider=self.platform_name,
                )
            raise ScraperError(
                f"Post not found or has no comments: {post_id}. "
                f"Make sure the post ID is correct and the post is publicly accessible.",
                provider=self.platform_name,
            )

        # Parse comments (second element in response)
        comments_data = data[1].get("data", {}).get("children", [])

        reviews: List[Review] = []
        self._parse_comments_recursive(comments_data, post_id, reviews, limit)

        return reviews[:limit]

    def _parse_comments_recursive(
        self,
        comments: List[Dict[str, Any]],
        post_id: str,
        reviews: List[Review],
        limit: int,
        depth: int = 0,
    ) -> None:
        """Recursively parse comments."""
        for item in comments:
            if len(reviews) >= limit:
                break

            kind = item.get("kind")
            if kind != "t1":  # t1 = comment
                continue

            data = item.get("data", {})
            review = self._parse_comment_data(data, post_id, depth)

            if review:
                reviews.append(review)

            # Parse replies
            replies = data.get("replies")
            if replies and isinstance(replies, dict):
                reply_children = replies.get("data", {}).get("children", [])
                self._parse_comments_recursive(
                    reply_children, post_id, reviews, limit, depth + 1
                )

    def _parse_comment_data(
        self,
        data: Dict[str, Any],
        post_id: str,
        depth: int,
    ) -> Optional[Review]:
        """Parse a single comment."""
        body = data.get("body", "")
        if not body or body == "[deleted]" or body == "[removed]":
            return None

        # Parse timestamp
        timestamp = None
        created_utc = data.get("created_utc")
        if created_utc:
            timestamp = datetime.fromtimestamp(created_utc)

        author = data.get("author", "[deleted]")
        score = data.get("score", 0)
        comment_id = data.get("id", "")

        # Generate permalink
        permalink = data.get("permalink", "")
        if permalink:
            permalink = f"https://www.reddit.com{permalink}"

        return Review(
            id=comment_id,
            text=self.clean_text(body),
            source=f"https://www.reddit.com/comments/{post_id}",
            platform=self.platform_name,
            author=author,
            rating=None,  # Reddit uses upvotes/downvotes, not ratings
            timestamp=timestamp,
            metadata={
                "post_id": post_id,
                "score": score,
                "depth": depth,
                "is_submitter": data.get("is_submitter", False),
                "permalink": permalink,
                "parent_id": data.get("parent_id", ""),
                "controversiality": data.get("controversiality", 0),
                "edited": data.get("edited", False),
            },
        )

    async def get_product_info(self, identifier: str) -> ProductInfo:
        """
        Get post information.

        Args:
            identifier: Post ID or URL

        Returns:
            ProductInfo object
        """
        post = await self.get_post(identifier)

        return ProductInfo(
            id=post.id,
            name=post.title,
            platform=self.platform_name,
            url=f"https://www.reddit.com{post.permalink}",
            description=post.selftext[:500] if post.selftext else None,
            review_count=post.num_comments,
            category=post.subreddit,
            metadata={
                "author": post.author,
                "score": post.score,
                "upvote_ratio": post.upvote_ratio,
                "is_self": post.is_self,
                "created_utc": post.created_utc.isoformat(),
                "link_flair_text": post.link_flair_text,
            },
        )

    async def get_post(self, identifier: str) -> RedditPost:
        """
        Get detailed post information.

        Args:
            identifier: Post ID or URL

        Returns:
            RedditPost object
        """
        self._ensure_initialized()

        # Extract post ID if URL provided
        if identifier.startswith("http"):
            post_id = self.extract_id(identifier)
            subreddit = self._extract_subreddit(identifier)
            if not post_id:
                raise ValueError(f"Could not extract post ID from URL: {identifier}")
        else:
            post_id = identifier
            # Handle full Reddit ID format (e.g., "t3_abc123" -> "abc123")
            if post_id.startswith("t3_"):
                post_id = post_id[3:]
            subreddit = None

        # Fetch post
        if subreddit:
            endpoint = f"/r/{subreddit}/comments/{post_id}"
        else:
            endpoint = f"/comments/{post_id}"

        data = await self._fetch_json(endpoint, {"limit": "0"})

        if not isinstance(data, list) or not data:
            raise ScraperError(
                f"Post not found: {post_id}",
                provider=self.platform_name,
            )

        post_data = data[0].get("data", {}).get("children", [])
        if not post_data:
            raise ScraperError(
                f"Post not found: {post_id}",
                provider=self.platform_name,
            )

        return self._parse_post_data(post_data[0].get("data", {}))

    def _parse_post_data(self, data: Dict[str, Any]) -> RedditPost:
        """Parse post data."""
        # Parse timestamp
        created_utc = datetime.fromtimestamp(data.get("created_utc", 0))

        # Extract awards
        awards = []
        for award in data.get("all_awardings", []):
            awards.append(award.get("name", ""))

        return RedditPost(
            id=data.get("id", ""),
            title=data.get("title", ""),
            selftext=data.get("selftext", ""),
            author=data.get("author", "[deleted]"),
            subreddit=data.get("subreddit", ""),
            score=data.get("score", 0),
            upvote_ratio=data.get("upvote_ratio", 0.0),
            num_comments=data.get("num_comments", 0),
            created_utc=created_utc,
            url=data.get("url", ""),
            permalink=data.get("permalink", ""),
            is_self=data.get("is_self", True),
            is_video=data.get("is_video", False),
            over_18=data.get("over_18", False),
            spoiler=data.get("spoiler", False),
            stickied=data.get("stickied", False),
            link_flair_text=data.get("link_flair_text"),
            awards=awards,
        )

    async def get_subreddit_posts(
        self,
        subreddit: str,
        limit: int = 25,
        sort: str = "hot",
        time_filter: str = "all",
    ) -> List[RedditPost]:
        """
        Get posts from a subreddit.

        Args:
            subreddit: Subreddit name (without r/)
            limit: Maximum number of posts
            sort: Sort order (hot, new, top, rising)
            time_filter: Time filter for top (hour, day, week, month, year, all)

        Returns:
            List of RedditPost objects
        """
        self._ensure_initialized()

        endpoint = f"/r/{subreddit}/{sort}"
        params = {
            "limit": str(min(100, limit)),
        }

        if sort == "top":
            params["t"] = time_filter

        data = await self._fetch_json(endpoint, params)

        posts: List[RedditPost] = []

        for item in data.get("data", {}).get("children", []):
            if item.get("kind") == "t3":  # t3 = post
                post = self._parse_post_data(item.get("data", {}))
                posts.append(post)

        return posts[:limit]

    async def search_posts(
        self,
        query: str,
        subreddit: Optional[str] = None,
        limit: int = 25,
        sort: str = "relevance",
        time_filter: str = "all",
    ) -> List[RedditPost]:
        """
        Search for posts.

        Args:
            query: Search query
            subreddit: Limit to specific subreddit (optional)
            limit: Maximum number of results
            sort: Sort order (relevance, hot, top, new, comments)
            time_filter: Time filter (hour, day, week, month, year, all)

        Returns:
            List of RedditPost objects
        """
        self._ensure_initialized()

        from urllib.parse import quote_plus

        if subreddit:
            endpoint = f"/r/{subreddit}/search"
            params = {
                "q": quote_plus(query),
                "restrict_sr": "on",
                "limit": str(min(100, limit)),
                "sort": sort,
                "t": time_filter,
            }
        else:
            endpoint = "/search"
            params = {
                "q": quote_plus(query),
                "limit": str(min(100, limit)),
                "sort": sort,
                "t": time_filter,
            }

        data = await self._fetch_json(endpoint, params)

        posts: List[RedditPost] = []

        for item in data.get("data", {}).get("children", []):
            if item.get("kind") == "t3":
                post = self._parse_post_data(item.get("data", {}))
                posts.append(post)

        return posts[:limit]

    async def get_subreddit_info(self, subreddit: str) -> Dict[str, Any]:
        """
        Get subreddit information.

        Args:
            subreddit: Subreddit name

        Returns:
            Dictionary with subreddit info
        """
        self._ensure_initialized()

        endpoint = f"/r/{subreddit}/about"
        data = await self._fetch_json(endpoint)

        sub_data = data.get("data", {})

        return {
            "name": sub_data.get("display_name", ""),
            "title": sub_data.get("title", ""),
            "description": sub_data.get("public_description", ""),
            "subscribers": sub_data.get("subscribers", 0),
            "active_users": sub_data.get("accounts_active", 0),
            "created_utc": datetime.fromtimestamp(
                sub_data.get("created_utc", 0)
            ).isoformat(),
            "over18": sub_data.get("over18", False),
            "url": f"https://www.reddit.com/r/{subreddit}",
        }
