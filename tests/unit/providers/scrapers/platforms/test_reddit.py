"""
Unit tests for Reddit Scraper.

Tests cover:
- Post ID validation and extraction
- URL validation
- Comment parsing
- RedditPost and RedditComment dataclasses
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from sentimatrix.providers.base import ProviderType


class TestRedditConfig:
    """Test RedditConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from sentimatrix.providers.scrapers.platforms.reddit import RedditConfig

        config = RedditConfig()

        assert config.client_id is None
        assert config.client_secret is None
        assert config.user_agent == "Sentimatrix/1.0"
        assert config.requests_per_second == 0.5
        assert config.comment_depth == 10
        assert config.comment_limit == 100

    def test_authenticated_config(self):
        """Test configuration with OAuth credentials."""
        from sentimatrix.providers.scrapers.platforms.reddit import RedditConfig

        config = RedditConfig(
            client_id="my_client_id",
            client_secret="my_secret",
            user_agent="MyApp/1.0",
        )

        assert config.client_id == "my_client_id"
        assert config.client_secret == "my_secret"


class TestRedditPost:
    """Test RedditPost dataclass."""

    def test_post_creation(self):
        """Test RedditPost creation."""
        from sentimatrix.providers.scrapers.platforms.reddit import RedditPost

        post = RedditPost(
            id="abc123",
            title="Test Post",
            selftext="This is the content",
            author="testuser",
            subreddit="python",
            score=100,
            upvote_ratio=0.95,
            num_comments=50,
            created_utc=datetime(2024, 1, 15),
            url="https://reddit.com/r/python/comments/abc123",
            permalink="/r/python/comments/abc123",
            is_self=True,
            is_video=False,
            over_18=False,
            spoiler=False,
            stickied=False,
        )

        assert post.id == "abc123"
        assert post.score == 100
        assert post.upvote_ratio == 0.95
        assert post.is_self is True

    def test_post_to_dict(self):
        """Test RedditPost to_dict."""
        from sentimatrix.providers.scrapers.platforms.reddit import RedditPost

        post = RedditPost(
            id="xyz",
            title="Title",
            selftext="Body",
            author="user",
            subreddit="test",
            score=10,
            upvote_ratio=0.9,
            num_comments=5,
            created_utc=datetime(2024, 1, 1),
            url="https://reddit.com",
            permalink="/r/test",
            is_self=True,
            is_video=False,
            over_18=False,
            spoiler=False,
            stickied=False,
        )

        data = post.to_dict()

        assert data["id"] == "xyz"
        assert data["title"] == "Title"
        assert "2024-01-01" in data["created_utc"]


class TestRedditComment:
    """Test RedditComment dataclass."""

    def test_comment_creation(self):
        """Test RedditComment creation."""
        from sentimatrix.providers.scrapers.platforms.reddit import RedditComment

        comment = RedditComment(
            id="comment1",
            body="Great post!",
            author="commenter",
            score=50,
            created_utc=datetime(2024, 1, 15),
            permalink="/r/python/comments/abc123/comment1",
            parent_id="t3_abc123",
            is_submitter=False,
            depth=0,
        )

        assert comment.id == "comment1"
        assert comment.score == 50
        assert comment.depth == 0
        assert comment.replies == []

    def test_comment_with_replies(self):
        """Test RedditComment with nested replies."""
        from sentimatrix.providers.scrapers.platforms.reddit import RedditComment

        reply = RedditComment(
            id="reply1",
            body="Thanks!",
            author="op",
            score=10,
            created_utc=datetime(2024, 1, 15),
            permalink="/r/python/comments/abc123/reply1",
            parent_id="t1_comment1",
            is_submitter=True,
            depth=1,
        )

        comment = RedditComment(
            id="comment1",
            body="Great post!",
            author="commenter",
            score=50,
            created_utc=datetime(2024, 1, 15),
            permalink="/r/python/comments/abc123/comment1",
            parent_id="t3_abc123",
            is_submitter=False,
            depth=0,
            replies=[reply],
        )

        assert len(comment.replies) == 1
        assert comment.replies[0].is_submitter is True

    def test_comment_to_dict(self):
        """Test RedditComment to_dict."""
        from sentimatrix.providers.scrapers.platforms.reddit import RedditComment

        comment = RedditComment(
            id="c1",
            body="Test",
            author="user",
            score=5,
            created_utc=datetime(2024, 1, 1),
            permalink="/r/test",
            parent_id="t3_post",
            is_submitter=False,
            depth=0,
        )

        data = comment.to_dict()

        assert data["id"] == "c1"
        assert data["body"] == "Test"
        assert data["replies"] == []


class TestRedditScraperInit:
    """Test Reddit scraper initialization."""

    def test_init_default(self):
        """Test default initialization."""
        from sentimatrix.providers.scrapers.platforms.reddit import RedditScraper

        scraper = RedditScraper()

        assert scraper.platform_name == "reddit"
        assert scraper.platform_domain == "reddit.com"
        assert not scraper._initialized
        assert not scraper._is_authenticated

    def test_init_authenticated(self):
        """Test initialization with credentials."""
        from sentimatrix.providers.scrapers.platforms.reddit import (
            RedditScraper,
            RedditConfig,
        )

        config = RedditConfig(
            client_id="id",
            client_secret="secret",
        )
        scraper = RedditScraper(config)

        assert scraper._is_authenticated is True

    def test_provider_info(self):
        """Test provider information."""
        from sentimatrix.providers.scrapers.platforms.reddit import RedditScraper

        scraper = RedditScraper()
        info = scraper.info

        assert info.name == "reddit"
        assert info.provider_type == ProviderType.SCRAPER
        assert info.capabilities.javascript_rendering is False


class TestRedditScraperValidation:
    """Test URL and ID validation."""

    def test_validate_url_post(self):
        """Test post URL validation."""
        from sentimatrix.providers.scrapers.platforms.reddit import RedditScraper

        scraper = RedditScraper()

        assert scraper.validate_url("https://reddit.com/r/python/comments/abc123") is True
        assert scraper.validate_url("https://www.reddit.com/r/gaming/comments/xyz789/title") is True

    def test_validate_url_subreddit(self):
        """Test subreddit URL validation."""
        from sentimatrix.providers.scrapers.platforms.reddit import RedditScraper

        scraper = RedditScraper()

        assert scraper.validate_url("https://reddit.com/r/python") is True
        assert scraper.validate_url("https://www.reddit.com/r/programming/") is True

    def test_validate_url_short(self):
        """Test short URL validation."""
        from sentimatrix.providers.scrapers.platforms.reddit import RedditScraper

        scraper = RedditScraper()

        assert scraper.validate_url("https://redd.it/abc123") is True

    def test_validate_url_invalid(self):
        """Test invalid URL validation."""
        from sentimatrix.providers.scrapers.platforms.reddit import RedditScraper

        scraper = RedditScraper()

        assert scraper.validate_url("https://example.com") is False
        assert scraper.validate_url("https://twitter.com/user") is False

    def test_extract_id_from_post_url(self):
        """Test post ID extraction from URL."""
        from sentimatrix.providers.scrapers.platforms.reddit import RedditScraper

        scraper = RedditScraper()

        post_id = scraper.extract_id("https://reddit.com/r/python/comments/abc123/title")
        assert post_id == "abc123"

    def test_extract_id_from_short_url(self):
        """Test post ID extraction from short URL."""
        from sentimatrix.providers.scrapers.platforms.reddit import RedditScraper

        scraper = RedditScraper()

        post_id = scraper.extract_id("https://redd.it/xyz789")
        assert post_id == "xyz789"

    def test_extract_subreddit(self):
        """Test subreddit extraction."""
        from sentimatrix.providers.scrapers.platforms.reddit import RedditScraper

        scraper = RedditScraper()

        subreddit = scraper._extract_subreddit("https://reddit.com/r/python/comments/abc123")
        assert subreddit == "python"


class TestRedditScraperParsing:
    """Test comment parsing."""

    def test_parse_comment_data(self):
        """Test parsing comment data."""
        from sentimatrix.providers.scrapers.platforms.reddit import RedditScraper

        scraper = RedditScraper()

        data = {
            "id": "comment1",
            "body": "This is a great discussion!",
            "author": "testuser",
            "score": 42,
            "created_utc": 1704067200,
            "permalink": "/r/python/comments/abc123/post/comment1",
            "parent_id": "t3_abc123",
            "is_submitter": False,
            "controversiality": 0,
            "edited": False,
        }

        review = scraper._parse_comment_data(data, "abc123", 0)

        assert review is not None
        assert review.id == "comment1"
        assert "great discussion" in review.text
        assert review.author == "testuser"
        assert review.metadata["score"] == 42
        assert review.metadata["depth"] == 0

    def test_parse_comment_deleted(self):
        """Test parsing deleted comment returns None."""
        from sentimatrix.providers.scrapers.platforms.reddit import RedditScraper

        scraper = RedditScraper()

        data = {
            "id": "deleted",
            "body": "[deleted]",
            "author": "[deleted]",
        }

        review = scraper._parse_comment_data(data, "abc123", 0)

        assert review is None

    def test_parse_comment_removed(self):
        """Test parsing removed comment returns None."""
        from sentimatrix.providers.scrapers.platforms.reddit import RedditScraper

        scraper = RedditScraper()

        data = {
            "id": "removed",
            "body": "[removed]",
            "author": "user",
        }

        review = scraper._parse_comment_data(data, "abc123", 0)

        assert review is None


class TestRedditScraperPostParsing:
    """Test post parsing."""

    def test_parse_post_data(self):
        """Test parsing post data."""
        from sentimatrix.providers.scrapers.platforms.reddit import RedditScraper

        scraper = RedditScraper()

        data = {
            "id": "abc123",
            "title": "Test Post Title",
            "selftext": "Post content here",
            "author": "poster",
            "subreddit": "python",
            "score": 500,
            "upvote_ratio": 0.92,
            "num_comments": 100,
            "created_utc": 1704067200,
            "url": "https://reddit.com/r/python/comments/abc123",
            "permalink": "/r/python/comments/abc123",
            "is_self": True,
            "is_video": False,
            "over_18": False,
            "spoiler": False,
            "stickied": False,
            "link_flair_text": "Discussion",
            "all_awardings": [{"name": "Helpful"}],
        }

        post = scraper._parse_post_data(data)

        assert post.id == "abc123"
        assert post.title == "Test Post Title"
        assert post.score == 500
        assert post.upvote_ratio == 0.92
        assert post.link_flair_text == "Discussion"
        assert "Helpful" in post.awards


class TestRedditScraperScrape:
    """Test scraping functionality."""

    @pytest.mark.asyncio
    async def test_scrape_not_initialized(self):
        """Test scraping without initialization raises error."""
        from sentimatrix.providers.scrapers.platforms.reddit import RedditScraper
        from sentimatrix.core.exceptions import ProviderInitializationError

        scraper = RedditScraper()

        with pytest.raises(ProviderInitializationError):
            await scraper.scrape_reviews("abc123")


class TestRedditScraperMethods:
    """Test available methods."""

    def test_has_required_methods(self):
        """Test scraper has all required methods."""
        from sentimatrix.providers.scrapers.platforms.reddit import RedditScraper

        scraper = RedditScraper()

        assert hasattr(scraper, "scrape_reviews")
        assert hasattr(scraper, "get_product_info")
        assert hasattr(scraper, "get_post")
        assert hasattr(scraper, "get_subreddit_posts")
        assert hasattr(scraper, "search_posts")
        assert hasattr(scraper, "get_subreddit_info")


class TestRedditScraperClose:
    """Test scraper close method."""

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing scraper."""
        from sentimatrix.providers.scrapers.platforms.reddit import RedditScraper

        scraper = RedditScraper()
        scraper._initialized = True
        scraper._httpx_scraper = AsyncMock()
        scraper._access_token = "token"

        await scraper.close()

        assert not scraper._initialized
        assert scraper._httpx_scraper is None
        assert scraper._access_token is None
