"""
Unit tests for Rate Limiter.

Tests cover:
- Token bucket limiter
- Fixed window limiter
- Sliding window limiter
- Per-domain limiting
- Statistics tracking
- Cooldown handling
"""

import asyncio
import time

import pytest

from sentimatrix.providers.scrapers.rate_limiter import (
    RateLimiter,
    RateLimitStrategy,
    TokenBucketLimiter,
    FixedWindowLimiter,
    SlidingWindowLimiter,
    create_rate_limiter,
)


class TestTokenBucketLimiter:
    """Test token bucket rate limiter."""

    def test_init_default(self):
        """Test default initialization."""
        limiter = TokenBucketLimiter()
        assert limiter._capacity > 0
        assert limiter._rate > 0

    def test_init_custom(self):
        """Test custom initialization."""
        limiter = TokenBucketLimiter(
            requests_per_second=2.0,
            burst_size=10,
        )
        assert limiter._rate == 2.0
        assert limiter._capacity == 10

    @pytest.mark.asyncio
    async def test_acquire_within_limit(self):
        """Test acquiring when within limits."""
        limiter = TokenBucketLimiter(
            requests_per_second=10.0,
            burst_size=5,
        )

        # Should not wait for first few requests
        wait_time = await limiter.acquire()
        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_acquire_with_wait(self):
        """Test acquiring when bucket is empty."""
        limiter = TokenBucketLimiter(
            requests_per_second=10.0,
            burst_size=1,
        )

        # First request - no wait
        await limiter.acquire()

        # Second request - should wait
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start

        # Should have waited approximately 0.1 seconds
        assert elapsed >= 0.05

    def test_try_acquire_success(self):
        """Test try_acquire when tokens available."""
        limiter = TokenBucketLimiter(
            requests_per_second=10.0,
            burst_size=5,
        )

        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is True

    def test_try_acquire_fail(self):
        """Test try_acquire when no tokens."""
        limiter = TokenBucketLimiter(
            requests_per_second=10.0,
            burst_size=1,
        )

        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is False

    @pytest.mark.asyncio
    async def test_per_domain_limiting(self):
        """Test per-domain rate limiting."""
        limiter = TokenBucketLimiter(
            requests_per_second=10.0,
            burst_size=2,
        )

        # Domain A
        await limiter.acquire(domain="domain-a.com")
        await limiter.acquire(domain="domain-a.com")

        # Domain B should have its own bucket
        wait_time = await limiter.acquire(domain="domain-b.com")
        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        """Test statistics tracking."""
        limiter = TokenBucketLimiter(
            requests_per_second=10.0,
            burst_size=5,
        )

        await limiter.acquire()
        await limiter.acquire(domain="example.com")

        stats = limiter.stats
        assert stats.total_requests == 2
        assert stats.last_request_at is not None
        assert "example.com" in stats.requests_per_domain

    def test_available_tokens(self):
        """Test available tokens property."""
        limiter = TokenBucketLimiter(
            requests_per_second=10.0,
            burst_size=5,
        )

        initial = limiter.available_tokens
        assert initial == 5.0

        limiter.try_acquire()
        assert limiter.available_tokens < initial


class TestFixedWindowLimiter:
    """Test fixed window rate limiter."""

    def test_init_default(self):
        """Test default initialization."""
        limiter = FixedWindowLimiter()
        assert limiter._max_requests > 0
        assert limiter._window_seconds > 0

    @pytest.mark.asyncio
    async def test_acquire_within_window(self):
        """Test acquiring within window limit."""
        limiter = FixedWindowLimiter(
            max_requests=10,
            window_seconds=60.0,
        )

        for _ in range(5):
            wait_time = await limiter.acquire()
            assert wait_time == 0.0

    def test_try_acquire_at_limit(self):
        """Test try_acquire at window limit."""
        limiter = FixedWindowLimiter(
            max_requests=2,
            window_seconds=60.0,
        )

        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is False

    def test_remaining_requests(self):
        """Test remaining requests property."""
        limiter = FixedWindowLimiter(
            max_requests=5,
            window_seconds=60.0,
        )

        assert limiter.remaining_requests == 5
        limiter.try_acquire()
        assert limiter.remaining_requests == 4

    def test_window_reset(self):
        """Test window resets after expiry."""
        limiter = FixedWindowLimiter(
            max_requests=1,
            window_seconds=0.1,
        )

        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is False

        # Wait for window to reset
        time.sleep(0.15)
        assert limiter.try_acquire() is True


class TestSlidingWindowLimiter:
    """Test sliding window rate limiter."""

    def test_init_default(self):
        """Test default initialization."""
        limiter = SlidingWindowLimiter()
        assert limiter._max_requests > 0
        assert limiter._window_seconds > 0

    @pytest.mark.asyncio
    async def test_acquire_within_limit(self):
        """Test acquiring within limit."""
        limiter = SlidingWindowLimiter(
            max_requests=10,
            window_seconds=60.0,
        )

        for _ in range(5):
            wait_time = await limiter.acquire()
            assert wait_time == 0.0

    def test_try_acquire_sliding(self):
        """Test sliding window behavior."""
        limiter = SlidingWindowLimiter(
            max_requests=2,
            window_seconds=0.5,
        )

        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is False

        # Wait for first request to expire
        time.sleep(0.6)
        assert limiter.try_acquire() is True

    def test_current_count(self):
        """Test current count property."""
        limiter = SlidingWindowLimiter(
            max_requests=5,
            window_seconds=60.0,
        )

        assert limiter.current_count == 0
        limiter.try_acquire()
        assert limiter.current_count == 1


class TestRateLimiter:
    """Test high-level rate limiter."""

    def test_init_token_bucket(self):
        """Test initialization with token bucket strategy."""
        limiter = RateLimiter(
            strategy=RateLimitStrategy.TOKEN_BUCKET,
            requests_per_second=2.0,
            burst_size=5,
        )
        assert limiter._strategy == RateLimitStrategy.TOKEN_BUCKET

    def test_init_fixed_window(self):
        """Test initialization with fixed window strategy."""
        limiter = RateLimiter(
            strategy=RateLimitStrategy.FIXED_WINDOW,
            requests_per_minute=60,
        )
        assert limiter._strategy == RateLimitStrategy.FIXED_WINDOW

    def test_init_sliding_window(self):
        """Test initialization with sliding window strategy."""
        limiter = RateLimiter(
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            requests_per_minute=60,
        )
        assert limiter._strategy == RateLimitStrategy.SLIDING_WINDOW

    @pytest.mark.asyncio
    async def test_acquire(self):
        """Test acquire method."""
        limiter = RateLimiter(
            strategy=RateLimitStrategy.TOKEN_BUCKET,
            requests_per_second=10.0,
            burst_size=5,
        )

        wait_time = await limiter.acquire()
        assert wait_time >= 0.0

    def test_try_acquire(self):
        """Test try_acquire method."""
        limiter = RateLimiter(
            strategy=RateLimitStrategy.TOKEN_BUCKET,
            requests_per_second=10.0,
            burst_size=5,
        )

        assert limiter.try_acquire() is True

    def test_report_429(self):
        """Test reporting 429 response."""
        limiter = RateLimiter(
            cooldown_on_429=1.0,
        )

        limiter.report_429("example.com")
        assert limiter._domain_cooldown["example.com"] > time.time()

    def test_clear_cooldown(self):
        """Test clearing cooldown."""
        limiter = RateLimiter(cooldown_on_429=60.0)

        limiter.report_429("example.com")
        limiter.clear_cooldown("example.com")
        assert "example.com" not in limiter._domain_cooldown

    @pytest.mark.asyncio
    async def test_cooldown_wait(self):
        """Test that acquire waits for cooldown."""
        limiter = RateLimiter(
            cooldown_on_429=0.1,
            requests_per_second=100.0,
            burst_size=100,
        )

        limiter.report_429("example.com")

        start = time.monotonic()
        await limiter.acquire("example.com")
        elapsed = time.monotonic() - start

        assert elapsed >= 0.05

    @pytest.mark.asyncio
    async def test_request_context_manager(self):
        """Test request context manager."""
        limiter = RateLimiter(
            requests_per_second=10.0,
            burst_size=5,
        )

        async with limiter.request("example.com") as wait_time:
            assert wait_time >= 0.0

    def test_stats(self):
        """Test stats property."""
        limiter = RateLimiter()
        limiter.try_acquire()

        stats = limiter.stats
        assert stats.total_requests >= 1

    def test_reset_stats(self):
        """Test reset_stats method."""
        limiter = RateLimiter()
        limiter.try_acquire()
        limiter.reset_stats()

        assert limiter.stats.total_requests == 0


class TestCreateRateLimiter:
    """Test factory function."""

    def test_create_token_bucket(self):
        """Test creating token bucket limiter."""
        limiter = create_rate_limiter(
            strategy="token_bucket",
            requests_per_second=2.0,
            burst_size=5,
        )
        assert limiter._strategy == RateLimitStrategy.TOKEN_BUCKET

    def test_create_fixed_window(self):
        """Test creating fixed window limiter."""
        limiter = create_rate_limiter(strategy="fixed_window")
        assert limiter._strategy == RateLimitStrategy.FIXED_WINDOW

    def test_create_sliding_window(self):
        """Test creating sliding window limiter."""
        limiter = create_rate_limiter(strategy="sliding_window")
        assert limiter._strategy == RateLimitStrategy.SLIDING_WINDOW


class TestConcurrency:
    """Test concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_acquire(self):
        """Test multiple concurrent acquires."""
        limiter = RateLimiter(
            requests_per_second=10.0,
            burst_size=5,
        )

        async def acquire_task():
            await limiter.acquire()
            return True

        tasks = [acquire_task() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        assert all(results)
        assert limiter.stats.total_requests == 10
