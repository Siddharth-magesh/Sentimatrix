"""
YouTube Comment Scraper

Extracts video comments from YouTube using:
1. YouTube Data API v3 (requires API key)
2. youtube-transcript-api for transcripts (no API key needed)

Features:
- Comment and reply extraction
- Video transcript extraction
- Like count and author info
- Pagination support
"""

from __future__ import annotations

import re
from dataclasses import dataclass
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
class YouTubeConfig(PlatformConfig):
    """YouTube-specific configuration."""

    # YouTube Data API key (required for comments)
    api_key: Optional[str] = None

    # Include replies to comments
    include_replies: bool = True

    # Maximum reply depth
    max_reply_depth: int = 1

    # Rate limiting (YouTube API has strict quotas)
    requests_per_second: float = 1.0
    burst_size: int = 5

    # Transcript language preference
    transcript_languages: List[str] = None

    def __post_init__(self):
        if self.transcript_languages is None:
            self.transcript_languages = ["en", "en-US"]


@dataclass
class VideoInfo:
    """YouTube video information."""

    id: str
    title: str
    channel_id: str
    channel_title: str
    description: str
    published_at: Optional[datetime] = None
    view_count: int = 0
    like_count: int = 0
    comment_count: int = 0
    duration: Optional[str] = None
    thumbnail_url: Optional[str] = None
    tags: List[str] = None
    category_id: Optional[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "channel_id": self.channel_id,
            "channel_title": self.channel_title,
            "description": self.description,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "view_count": self.view_count,
            "like_count": self.like_count,
            "comment_count": self.comment_count,
            "duration": self.duration,
            "thumbnail_url": self.thumbnail_url,
            "tags": self.tags,
            "category_id": self.category_id,
        }


@dataclass
class Transcript:
    """Video transcript."""

    video_id: str
    language: str
    is_generated: bool
    segments: List[Dict[str, Any]]
    full_text: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "video_id": self.video_id,
            "language": self.language,
            "is_generated": self.is_generated,
            "segments": self.segments,
            "full_text": self.full_text,
        }


class YouTubeScraper(BasePlatformScraper):
    """
    YouTube comment and transcript scraper.

    Uses:
    - YouTube Data API v3 for comments (requires API key)
    - youtube-transcript-api for transcripts (no key needed)

    Example:
        >>> config = YouTubeConfig(api_key="your_api_key")
        >>> async with YouTubeScraper(config) as scraper:
        ...     comments = await scraper.scrape_reviews("dQw4w9WgXcQ", limit=100)
        ...     transcript = await scraper.get_transcript("dQw4w9WgXcQ")
    """

    # Video ID patterns
    URL_PATTERNS = [
        re.compile(r"youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})"),
        re.compile(r"youtu\.be/([a-zA-Z0-9_-]{11})"),
        re.compile(r"youtube\.com/embed/([a-zA-Z0-9_-]{11})"),
        re.compile(r"youtube\.com/v/([a-zA-Z0-9_-]{11})"),
        re.compile(r"youtube\.com/shorts/([a-zA-Z0-9_-]{11})"),
    ]

    # API endpoints
    API_BASE = "https://www.googleapis.com/youtube/v3"
    COMMENTS_ENDPOINT = f"{API_BASE}/commentThreads"
    VIDEOS_ENDPOINT = f"{API_BASE}/videos"
    SEARCH_ENDPOINT = f"{API_BASE}/search"
    REPLIES_ENDPOINT = f"{API_BASE}/comments"

    def __init__(
        self,
        config: Optional[YouTubeConfig] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        """
        Initialize YouTube scraper.

        Args:
            config: YouTube-specific configuration
            rate_limiter: Optional rate limiter
        """
        self._youtube_config = config or YouTubeConfig()
        super().__init__(self._youtube_config, rate_limiter)

        self._httpx_scraper = None

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="youtube",
            provider_type=ProviderType.SCRAPER,
            version="1.0.0",
            description="YouTube comment and transcript scraper",
            capabilities=ProviderCapabilities(
                javascript_rendering=False,
                proxy_support=True,
            ),
            website="https://youtube.com",
        )

    @property
    def platform_name(self) -> str:
        """Get platform name."""
        return "youtube"

    @property
    def platform_domain(self) -> str:
        """Get platform domain."""
        return "youtube.com"

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
        """Validate YouTube URL."""
        for pattern in self.URL_PATTERNS:
            if pattern.search(url):
                return True
        return False

    def extract_id(self, url: str) -> Optional[str]:
        """Extract video ID from URL."""
        for pattern in self.URL_PATTERNS:
            match = pattern.search(url)
            if match:
                return match.group(1)
        return None

    def validate_video_id(self, video_id: str) -> bool:
        """Validate video ID format."""
        return bool(re.match(r"^[a-zA-Z0-9_-]{11}$", video_id))

    def _check_api_key(self) -> None:
        """Check if API key is configured."""
        if not self._youtube_config.api_key:
            raise ProviderInitializationError(
                self.platform_name,
                "YouTube Data API key is required. "
                "Set it via YouTubeConfig(api_key='your_key') or "
                "environment variable YOUTUBE_API_KEY",
            )

    async def scrape_reviews(
        self,
        identifier: str,
        limit: int = 100,
        sort_by: SortOrder = SortOrder.RECENT,
        filter_by: ReviewFilter = ReviewFilter.ALL,
        **kwargs: Any,
    ) -> List[Review]:
        """
        Scrape comments from a video.

        Args:
            identifier: Video ID or URL
            limit: Maximum number of comments
            sort_by: Sort order (RECENT, RELEVANCE)
            filter_by: Not used for YouTube
            **kwargs: Additional parameters
                - include_replies: Override config setting

        Returns:
            List of Review objects (comments)
        """
        self._ensure_initialized()
        self._check_api_key()

        # Extract video ID if URL provided
        if identifier.startswith("http"):
            video_id = self.extract_id(identifier)
            if not video_id:
                raise ValueError(f"Could not extract video ID from URL: {identifier}")
        else:
            video_id = identifier

        if not self.validate_video_id(video_id):
            raise ValueError(f"Invalid video ID format: {video_id}")

        comments: List[Review] = []
        page_token = None

        # Map sort order
        order = "time" if sort_by == SortOrder.RECENT else "relevance"

        include_replies = kwargs.get(
            "include_replies",
            self._youtube_config.include_replies
        )

        while len(comments) < limit:
            params = {
                "key": self._youtube_config.api_key,
                "videoId": video_id,
                "part": "snippet,replies",
                "maxResults": str(min(100, limit - len(comments))),
                "order": order,
                "textFormat": "plainText",
            }

            if page_token:
                params["pageToken"] = page_token

            try:
                data = await self._fetch_api(self.COMMENTS_ENDPOINT, params)

                items = data.get("items", [])
                if not items:
                    break

                for item in items:
                    # Parse top-level comment
                    comment = self._parse_comment_thread(item, video_id)
                    if comment:
                        comments.append(comment)

                    # Parse replies if configured
                    if include_replies:
                        replies = item.get("replies", {}).get("comments", [])
                        for reply in replies:
                            reply_comment = self._parse_comment(
                                reply, video_id, is_reply=True
                            )
                            if reply_comment:
                                comments.append(reply_comment)

                # Get next page
                page_token = data.get("nextPageToken")
                if not page_token:
                    break

            except Exception as e:
                if "quotaExceeded" in str(e):
                    raise ScraperError(
                        "YouTube API quota exceeded",
                        provider=self.platform_name,
                        original_error=e,
                    )
                raise

        return comments[:limit]

    async def _fetch_api(
        self,
        endpoint: str,
        params: Dict[str, str],
    ) -> Dict[str, Any]:
        """Fetch from YouTube API."""
        await self._rate_limiter.acquire(domain="googleapis.com")

        # Build URL with params
        param_str = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{endpoint}?{param_str}"

        content = await self._httpx_scraper.scrape(url)

        try:
            import json
            data = json.loads(content.content)

            # Check for API errors
            if "error" in data:
                error = data["error"]
                raise ScraperError(
                    f"YouTube API error: {error.get('message', 'Unknown error')}",
                    provider=self.platform_name,
                )

            return data

        except Exception as e:
            if isinstance(e, ScraperError):
                raise
            raise ScraperParseError(
                provider=self.platform_name,
                url=endpoint,
                reason=f"Invalid JSON response: {e}",
                original_error=e,
            )

    def _parse_comment_thread(
        self,
        item: Dict[str, Any],
        video_id: str,
    ) -> Optional[Review]:
        """Parse a comment thread."""
        snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
        return self._parse_comment_snippet(snippet, video_id, item.get("id", ""))

    def _parse_comment(
        self,
        item: Dict[str, Any],
        video_id: str,
        is_reply: bool = False,
    ) -> Optional[Review]:
        """Parse a single comment."""
        snippet = item.get("snippet", {})
        return self._parse_comment_snippet(
            snippet, video_id, item.get("id", ""), is_reply
        )

    def _parse_comment_snippet(
        self,
        snippet: Dict[str, Any],
        video_id: str,
        comment_id: str,
        is_reply: bool = False,
    ) -> Optional[Review]:
        """Parse comment snippet data."""
        text = snippet.get("textDisplay", "")
        if not text:
            return None

        # Parse timestamp
        timestamp = None
        published_at = snippet.get("publishedAt")
        if published_at:
            timestamp = self.parse_date(published_at)

        # Author info
        author = snippet.get("authorDisplayName", "")
        author_channel_id = snippet.get("authorChannelId", {}).get("value", "")

        # Engagement
        like_count = snippet.get("likeCount", 0)

        return Review(
            id=comment_id,
            text=self.clean_text(text),
            source=f"https://www.youtube.com/watch?v={video_id}",
            platform=self.platform_name,
            author=author,
            rating=None,  # YouTube comments don't have ratings
            timestamp=timestamp,
            metadata={
                "video_id": video_id,
                "is_reply": is_reply,
                "like_count": like_count,
                "author_channel_id": author_channel_id,
                "updated_at": snippet.get("updatedAt"),
                "parent_id": snippet.get("parentId"),
            },
        )

    async def get_product_info(self, identifier: str) -> ProductInfo:
        """
        Get video information.

        Args:
            identifier: Video ID or URL

        Returns:
            ProductInfo object
        """
        video_info = await self.get_video_info(identifier)

        return ProductInfo(
            id=video_info.id,
            name=video_info.title,
            platform=self.platform_name,
            url=f"https://www.youtube.com/watch?v={video_info.id}",
            description=video_info.description,
            review_count=video_info.comment_count,
            image_url=video_info.thumbnail_url,
            category=video_info.category_id,
            metadata={
                "channel_id": video_info.channel_id,
                "channel_title": video_info.channel_title,
                "view_count": video_info.view_count,
                "like_count": video_info.like_count,
                "duration": video_info.duration,
                "tags": video_info.tags,
                "published_at": video_info.published_at.isoformat() if video_info.published_at else None,
            },
        )

    async def get_video_info(self, identifier: str) -> VideoInfo:
        """
        Get detailed video information.

        Args:
            identifier: Video ID or URL

        Returns:
            VideoInfo object
        """
        self._ensure_initialized()
        self._check_api_key()

        # Extract video ID if URL provided
        if identifier.startswith("http"):
            video_id = self.extract_id(identifier)
            if not video_id:
                raise ValueError(f"Could not extract video ID from URL: {identifier}")
        else:
            video_id = identifier

        if not self.validate_video_id(video_id):
            raise ValueError(f"Invalid video ID format: {video_id}")

        params = {
            "key": self._youtube_config.api_key,
            "id": video_id,
            "part": "snippet,statistics,contentDetails",
        }

        data = await self._fetch_api(self.VIDEOS_ENDPOINT, params)

        items = data.get("items", [])
        if not items:
            raise ScraperError(
                f"Video not found: {video_id}",
                provider=self.platform_name,
            )

        item = items[0]
        snippet = item.get("snippet", {})
        statistics = item.get("statistics", {})
        content_details = item.get("contentDetails", {})

        # Parse published date
        published_at = None
        if snippet.get("publishedAt"):
            published_at = self.parse_date(snippet["publishedAt"])

        # Get best thumbnail
        thumbnails = snippet.get("thumbnails", {})
        thumbnail_url = (
            thumbnails.get("maxres", {}).get("url") or
            thumbnails.get("high", {}).get("url") or
            thumbnails.get("default", {}).get("url")
        )

        return VideoInfo(
            id=video_id,
            title=snippet.get("title", ""),
            channel_id=snippet.get("channelId", ""),
            channel_title=snippet.get("channelTitle", ""),
            description=snippet.get("description", ""),
            published_at=published_at,
            view_count=int(statistics.get("viewCount", 0)),
            like_count=int(statistics.get("likeCount", 0)),
            comment_count=int(statistics.get("commentCount", 0)),
            duration=content_details.get("duration"),
            thumbnail_url=thumbnail_url,
            tags=snippet.get("tags", []),
            category_id=snippet.get("categoryId"),
        )

    async def get_transcript(
        self,
        identifier: str,
        languages: Optional[List[str]] = None,
    ) -> Transcript:
        """
        Get video transcript.

        Uses youtube-transcript-api (no API key required).

        Args:
            identifier: Video ID or URL
            languages: Preferred languages (defaults to config)

        Returns:
            Transcript object
        """
        # Extract video ID if URL provided
        if identifier.startswith("http"):
            video_id = self.extract_id(identifier)
            if not video_id:
                raise ValueError(f"Could not extract video ID from URL: {identifier}")
        else:
            video_id = identifier

        if not self.validate_video_id(video_id):
            raise ValueError(f"Invalid video ID format: {video_id}")

        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            raise ImportError(
                "youtube-transcript-api is required for transcripts. "
                "Install with: pip install youtube-transcript-api"
            )

        languages = languages or self._youtube_config.transcript_languages

        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            # Try to find transcript in preferred languages
            transcript = None
            is_generated = False

            for lang in languages:
                try:
                    transcript = transcript_list.find_transcript([lang])
                    is_generated = transcript.is_generated
                    break
                except Exception:
                    continue

            # Fall back to any available transcript
            if transcript is None:
                try:
                    transcript = transcript_list.find_generated_transcript(languages)
                    is_generated = True
                except Exception:
                    # Get first available
                    for t in transcript_list:
                        transcript = t
                        is_generated = t.is_generated
                        break

            if transcript is None:
                raise ScraperError(
                    f"No transcript available for video: {video_id}",
                    provider=self.platform_name,
                )

            segments = transcript.fetch()

            # Build full text
            full_text = " ".join(seg.get("text", "") for seg in segments)

            return Transcript(
                video_id=video_id,
                language=transcript.language_code,
                is_generated=is_generated,
                segments=segments,
                full_text=full_text,
            )

        except Exception as e:
            if "TranscriptsDisabled" in str(type(e).__name__):
                raise ScraperError(
                    f"Transcripts are disabled for video: {video_id}",
                    provider=self.platform_name,
                    original_error=e,
                )
            if isinstance(e, ScraperError):
                raise
            raise ScraperError(
                f"Failed to get transcript: {e}",
                provider=self.platform_name,
                original_error=e,
            )

    async def search_videos(
        self,
        query: str,
        limit: int = 10,
        order: str = "relevance",
    ) -> List[VideoInfo]:
        """
        Search for videos.

        Args:
            query: Search query
            limit: Maximum number of results
            order: Sort order (relevance, date, rating, viewCount)

        Returns:
            List of VideoInfo objects
        """
        self._ensure_initialized()
        self._check_api_key()

        from urllib.parse import quote_plus

        params = {
            "key": self._youtube_config.api_key,
            "q": quote_plus(query),
            "part": "snippet",
            "type": "video",
            "maxResults": str(min(50, limit)),
            "order": order,
        }

        data = await self._fetch_api(self.SEARCH_ENDPOINT, params)

        videos: List[VideoInfo] = []

        for item in data.get("items", []):
            video_id = item.get("id", {}).get("videoId", "")
            if not video_id:
                continue

            snippet = item.get("snippet", {})

            # Parse published date
            published_at = None
            if snippet.get("publishedAt"):
                published_at = self.parse_date(snippet["publishedAt"])

            thumbnails = snippet.get("thumbnails", {})
            thumbnail_url = thumbnails.get("high", {}).get("url")

            videos.append(VideoInfo(
                id=video_id,
                title=snippet.get("title", ""),
                channel_id=snippet.get("channelId", ""),
                channel_title=snippet.get("channelTitle", ""),
                description=snippet.get("description", ""),
                published_at=published_at,
                thumbnail_url=thumbnail_url,
            ))

        return videos[:limit]
