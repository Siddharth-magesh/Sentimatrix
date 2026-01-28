"""
Sentimatrix Rate Limiter

Provides rate limiting functionality for web scraping with multiple strategies:
- Token Bucket: Burst-friendly limiting with sustained rate
- Fixed Window: Simple N requests per time window
- Sliding Window: Rolling window for smoother distribution

Also includes per-domain rate limiting for respectful scraping.

Example:
    >>> limiter = RateLimiter(requests_per_second=1.0, burst_size=5)
    >>> await limiter.acquire()  # Blocks if rate limit exceeded
    >>> await limiter.acquire(domain="example.com")  # Per-domain limiting
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from sentimatrix.core.config import RateLimitConfig
from sentimatrix.core.exceptions import RateLimitError


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies."""

    TOKEN_BUCKET = "token_bucket"
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"


@dataclass
class RateLimitStats:
    """Statistics for rate limiting."""

    total_requests: int = 0
    total_blocked: int = 0
    total_wait_time_ms: float = 0.0
    last_request_at: Optional[float] = None
    requests_per_domain: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "total_blocked": self.total_blocked,
            "total_wait_time_ms": self.total_wait_time_ms,
            "last_request_at": self.last_request_at,
            "requests_per_domain": dict(self.requests_per_domain),
        }


class BaseRateLimiter(ABC):
    """Abstract base class for rate limiters."""

    def __init__(self, config: Optional[RateLimitConfig] = None) -> None:
        """Initialize rate limiter with configuration."""
        self._config = config or RateLimitConfig()
        self._stats = RateLimitStats()
        self._lock = asyncio.Lock()

    @abstractmethod
    async def acquire(self, domain: Optional[str] = None, weight: float = 1.0) -> float:
        """
        Acquire permission to make a request.

        Args:
            domain: Optional domain for per-domain limiting
            weight: Request weight (default 1.0)

        Returns:
            Time waited in seconds

        Raises:
            RateLimitError: If rate limit cannot be satisfied
        """
        pass

    @abstractmethod
    def try_acquire(self, domain: Optional[str] = None, weight: float = 1.0) -> bool:
        """
        Try to acquire permission without blocking.

        Args:
            domain: Optional domain for per-domain limiting
            weight: Request weight

        Returns:
            True if permission granted, False otherwise
        """
        pass

    @property
    def stats(self) -> RateLimitStats:
        """Get rate limiter statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = RateLimitStats()


class TokenBucketLimiter(BaseRateLimiter):
    """
    Token bucket rate limiter.

    Allows burst traffic up to bucket capacity, then enforces
    sustained rate. Tokens refill at a constant rate.

    Good for: APIs that allow bursts but have sustained limits.
    """

    def __init__(
        self,
        config: Optional[RateLimitConfig] = None,
        requests_per_second: Optional[float] = None,
        burst_size: Optional[int] = None,
    ) -> None:
        """
        Initialize token bucket limiter.

        Args:
            config: Rate limit configuration
            requests_per_second: Override config requests_per_second
            burst_size: Maximum burst size (bucket capacity)
        """
        super().__init__(config)

        self._rate = requests_per_second or self._config.requests_per_second
        self._capacity = burst_size or self._config.concurrent_requests
        self._tokens = float(self._capacity)
        self._last_refill = time.monotonic()

        # Per-domain buckets
        self._domain_tokens: Dict[str, float] = defaultdict(lambda: float(self._capacity))
        self._domain_last_refill: Dict[str, float] = defaultdict(time.monotonic)

    def _refill(self, domain: Optional[str] = None) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()

        if domain:
            elapsed = now - self._domain_last_refill[domain]
            self._domain_tokens[domain] = min(
                self._capacity,
                self._domain_tokens[domain] + elapsed * self._rate,
            )
            self._domain_last_refill[domain] = now
        else:
            elapsed = now - self._last_refill
            self._tokens = min(
                self._capacity,
                self._tokens + elapsed * self._rate,
            )
            self._last_refill = now

    async def acquire(self, domain: Optional[str] = None, weight: float = 1.0) -> float:
        """
        Acquire tokens, waiting if necessary.

        Args:
            domain: Optional domain for per-domain limiting
            weight: Number of tokens to acquire

        Returns:
            Time waited in seconds
        """
        wait_time = 0.0

        async with self._lock:
            self._refill(domain)

            tokens = self._domain_tokens[domain] if domain else self._tokens

            if tokens < weight:
                # Calculate wait time
                deficit = weight - tokens
                wait_time = deficit / self._rate

                # Release lock while waiting
                self._lock.release()
                try:
                    await asyncio.sleep(wait_time)
                finally:
                    await self._lock.acquire()

                # Refill after waiting
                self._refill(domain)

            # Consume tokens
            if domain:
                self._domain_tokens[domain] -= weight
            else:
                self._tokens -= weight

            # Update stats
            self._stats.total_requests += 1
            self._stats.total_wait_time_ms += wait_time * 1000
            self._stats.last_request_at = time.time()
            if domain:
                self._stats.requests_per_domain[domain] = (
                    self._stats.requests_per_domain.get(domain, 0) + 1
                )

            if wait_time > 0:
                self._stats.total_blocked += 1

        return wait_time

    def try_acquire(self, domain: Optional[str] = None, weight: float = 1.0) -> bool:
        """Try to acquire tokens without blocking."""
        self._refill(domain)

        tokens = self._domain_tokens[domain] if domain else self._tokens

        if tokens >= weight:
            if domain:
                self._domain_tokens[domain] -= weight
            else:
                self._tokens -= weight

            self._stats.total_requests += 1
            self._stats.last_request_at = time.time()
            if domain:
                self._stats.requests_per_domain[domain] = (
                    self._stats.requests_per_domain.get(domain, 0) + 1
                )
            return True

        return False

    @property
    def available_tokens(self) -> float:
        """Get current available tokens."""
        self._refill()
        return self._tokens

    def get_domain_tokens(self, domain: str) -> float:
        """Get available tokens for a domain."""
        self._refill(domain)
        return self._domain_tokens[domain]


class FixedWindowLimiter(BaseRateLimiter):
    """
    Fixed window rate limiter.

    Allows N requests per fixed time window. Simple but can
    allow bursts at window boundaries.

    Good for: APIs with strict per-minute/per-hour limits.
    """

    def __init__(
        self,
        config: Optional[RateLimitConfig] = None,
        max_requests: Optional[int] = None,
        window_seconds: float = 60.0,
    ) -> None:
        """
        Initialize fixed window limiter.

        Args:
            config: Rate limit configuration
            max_requests: Maximum requests per window
            window_seconds: Window duration in seconds
        """
        super().__init__(config)

        self._max_requests = max_requests or self._config.requests_per_minute
        self._window_seconds = window_seconds
        self._window_start = time.monotonic()
        self._request_count = 0

        # Per-domain windows
        self._domain_window_start: Dict[str, float] = defaultdict(time.monotonic)
        self._domain_request_count: Dict[str, int] = defaultdict(int)

    def _reset_if_needed(self, domain: Optional[str] = None) -> None:
        """Reset window if expired."""
        now = time.monotonic()

        if domain:
            if now - self._domain_window_start[domain] >= self._window_seconds:
                self._domain_window_start[domain] = now
                self._domain_request_count[domain] = 0
        else:
            if now - self._window_start >= self._window_seconds:
                self._window_start = now
                self._request_count = 0

    async def acquire(self, domain: Optional[str] = None, weight: float = 1.0) -> float:
        """Acquire permission, waiting until next window if needed."""
        wait_time = 0.0

        async with self._lock:
            self._reset_if_needed(domain)

            count = self._domain_request_count[domain] if domain else self._request_count
            window_start = (
                self._domain_window_start[domain] if domain else self._window_start
            )

            if count >= self._max_requests:
                # Wait for next window
                elapsed = time.monotonic() - window_start
                wait_time = self._window_seconds - elapsed

                if wait_time > 0:
                    self._stats.total_blocked += 1
                    self._lock.release()
                    try:
                        await asyncio.sleep(wait_time)
                    finally:
                        await self._lock.acquire()

                    self._reset_if_needed(domain)

            # Increment count
            if domain:
                self._domain_request_count[domain] += int(weight)
            else:
                self._request_count += int(weight)

            # Update stats
            self._stats.total_requests += 1
            self._stats.total_wait_time_ms += wait_time * 1000
            self._stats.last_request_at = time.time()
            if domain:
                self._stats.requests_per_domain[domain] = (
                    self._stats.requests_per_domain.get(domain, 0) + 1
                )

        return wait_time

    def try_acquire(self, domain: Optional[str] = None, weight: float = 1.0) -> bool:
        """Try to acquire without blocking."""
        self._reset_if_needed(domain)

        count = self._domain_request_count[domain] if domain else self._request_count

        if count < self._max_requests:
            if domain:
                self._domain_request_count[domain] += int(weight)
            else:
                self._request_count += int(weight)

            self._stats.total_requests += 1
            self._stats.last_request_at = time.time()
            if domain:
                self._stats.requests_per_domain[domain] = (
                    self._stats.requests_per_domain.get(domain, 0) + 1
                )
            return True

        return False

    @property
    def remaining_requests(self) -> int:
        """Get remaining requests in current window."""
        self._reset_if_needed()
        return max(0, self._max_requests - self._request_count)

    @property
    def window_reset_in(self) -> float:
        """Get seconds until window resets."""
        elapsed = time.monotonic() - self._window_start
        return max(0.0, self._window_seconds - elapsed)


class SlidingWindowLimiter(BaseRateLimiter):
    """
    Sliding window rate limiter.

    Uses a sliding window for smoother rate limiting without
    boundary burst issues.

    Good for: Smooth, consistent rate limiting.
    """

    def __init__(
        self,
        config: Optional[RateLimitConfig] = None,
        max_requests: Optional[int] = None,
        window_seconds: float = 60.0,
    ) -> None:
        """
        Initialize sliding window limiter.

        Args:
            config: Rate limit configuration
            max_requests: Maximum requests per window
            window_seconds: Window duration in seconds
        """
        super().__init__(config)

        self._max_requests = max_requests or self._config.requests_per_minute
        self._window_seconds = window_seconds
        self._request_times: List[float] = []

        # Per-domain request times
        self._domain_request_times: Dict[str, List[float]] = defaultdict(list)

    def _clean_old_requests(self, domain: Optional[str] = None) -> None:
        """Remove requests outside the window."""
        now = time.monotonic()
        cutoff = now - self._window_seconds

        if domain:
            self._domain_request_times[domain] = [
                t for t in self._domain_request_times[domain] if t > cutoff
            ]
        else:
            self._request_times = [t for t in self._request_times if t > cutoff]

    async def acquire(self, domain: Optional[str] = None, weight: float = 1.0) -> float:
        """Acquire permission, waiting if window is full."""
        wait_time = 0.0

        async with self._lock:
            self._clean_old_requests(domain)

            times = self._domain_request_times[domain] if domain else self._request_times

            if len(times) >= self._max_requests:
                # Wait until oldest request expires
                oldest = times[0]
                wait_time = (oldest + self._window_seconds) - time.monotonic()

                if wait_time > 0:
                    self._stats.total_blocked += 1
                    self._lock.release()
                    try:
                        await asyncio.sleep(wait_time)
                    finally:
                        await self._lock.acquire()

                    self._clean_old_requests(domain)

            # Record request
            now = time.monotonic()
            for _ in range(int(weight)):
                if domain:
                    self._domain_request_times[domain].append(now)
                else:
                    self._request_times.append(now)

            # Update stats
            self._stats.total_requests += 1
            self._stats.total_wait_time_ms += wait_time * 1000
            self._stats.last_request_at = time.time()
            if domain:
                self._stats.requests_per_domain[domain] = (
                    self._stats.requests_per_domain.get(domain, 0) + 1
                )

        return wait_time

    def try_acquire(self, domain: Optional[str] = None, weight: float = 1.0) -> bool:
        """Try to acquire without blocking."""
        self._clean_old_requests(domain)

        times = self._domain_request_times[domain] if domain else self._request_times

        if len(times) < self._max_requests:
            now = time.monotonic()
            for _ in range(int(weight)):
                if domain:
                    self._domain_request_times[domain].append(now)
                else:
                    self._request_times.append(now)

            self._stats.total_requests += 1
            self._stats.last_request_at = time.time()
            if domain:
                self._stats.requests_per_domain[domain] = (
                    self._stats.requests_per_domain.get(domain, 0) + 1
                )
            return True

        return False

    @property
    def current_count(self) -> int:
        """Get current request count in window."""
        self._clean_old_requests()
        return len(self._request_times)


class RateLimiter:
    """
    High-level rate limiter with strategy selection.

    Provides a unified interface for all rate limiting strategies
    with additional features like:
    - Per-domain limiting
    - Cooldown on 429 responses
    - Statistics tracking
    - Backoff handling

    Example:
        >>> limiter = RateLimiter(
        ...     strategy=RateLimitStrategy.TOKEN_BUCKET,
        ...     requests_per_second=2.0,
        ...     burst_size=10
        ... )
        >>> async with limiter.request("example.com"):
        ...     # Make request
        ...     pass
    """

    def __init__(
        self,
        config: Optional[RateLimitConfig] = None,
        strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET,
        requests_per_second: float = 1.0,
        burst_size: int = 5,
        requests_per_minute: int = 60,
        per_domain: bool = True,
        cooldown_on_429: float = 60.0,
    ) -> None:
        """
        Initialize rate limiter.

        Args:
            config: Rate limit configuration
            strategy: Rate limiting strategy
            requests_per_second: Requests per second for token bucket
            burst_size: Burst capacity for token bucket
            requests_per_minute: Requests per minute for window limiters
            per_domain: Enable per-domain rate limiting
            cooldown_on_429: Cooldown time after 429 response
        """
        self._config = config or RateLimitConfig()
        self._strategy = strategy
        self._per_domain = per_domain
        self._cooldown_on_429 = cooldown_on_429

        # Domain cooldowns after 429
        self._domain_cooldown: Dict[str, float] = {}
        self._global_cooldown: float = 0.0

        # Create limiter based on strategy
        if strategy == RateLimitStrategy.TOKEN_BUCKET:
            self._limiter = TokenBucketLimiter(
                config,
                requests_per_second=requests_per_second,
                burst_size=burst_size,
            )
        elif strategy == RateLimitStrategy.FIXED_WINDOW:
            self._limiter = FixedWindowLimiter(
                config,
                max_requests=requests_per_minute,
            )
        elif strategy == RateLimitStrategy.SLIDING_WINDOW:
            self._limiter = SlidingWindowLimiter(
                config,
                max_requests=requests_per_minute,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    async def acquire(self, domain: Optional[str] = None, weight: float = 1.0) -> float:
        """
        Acquire permission to make a request.

        Args:
            domain: Domain for per-domain limiting
            weight: Request weight

        Returns:
            Total wait time in seconds
        """
        total_wait = 0.0

        # Check cooldowns
        now = time.time()

        if self._global_cooldown > now:
            wait = self._global_cooldown - now
            await asyncio.sleep(wait)
            total_wait += wait

        if domain and domain in self._domain_cooldown:
            if self._domain_cooldown[domain] > now:
                wait = self._domain_cooldown[domain] - now
                await asyncio.sleep(wait)
                total_wait += wait
            else:
                del self._domain_cooldown[domain]

        # Acquire from underlying limiter
        effective_domain = domain if self._per_domain else None
        wait = await self._limiter.acquire(effective_domain, weight)
        total_wait += wait

        return total_wait

    def try_acquire(self, domain: Optional[str] = None, weight: float = 1.0) -> bool:
        """Try to acquire without blocking."""
        now = time.time()

        # Check cooldowns
        if self._global_cooldown > now:
            return False

        if domain and domain in self._domain_cooldown:
            if self._domain_cooldown[domain] > now:
                return False

        effective_domain = domain if self._per_domain else None
        return self._limiter.try_acquire(effective_domain, weight)

    def report_429(self, domain: Optional[str] = None) -> None:
        """
        Report a 429 response to trigger cooldown.

        Args:
            domain: Domain that returned 429
        """
        cooldown_until = time.time() + self._cooldown_on_429

        if domain:
            self._domain_cooldown[domain] = cooldown_until
        else:
            self._global_cooldown = cooldown_until

    def clear_cooldown(self, domain: Optional[str] = None) -> None:
        """Clear cooldown for a domain or globally."""
        if domain:
            self._domain_cooldown.pop(domain, None)
        else:
            self._global_cooldown = 0.0
            self._domain_cooldown.clear()

    @property
    def stats(self) -> RateLimitStats:
        """Get statistics."""
        return self._limiter.stats

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._limiter.reset_stats()

    class _RequestContext:
        """Context manager for rate-limited requests."""

        def __init__(
            self, limiter: "RateLimiter", domain: Optional[str], weight: float
        ) -> None:
            self._limiter = limiter
            self._domain = domain
            self._weight = weight

        async def __aenter__(self) -> float:
            """Acquire rate limit on entry."""
            return await self._limiter.acquire(self._domain, self._weight)

        async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
            """Handle exceptions, report 429 if needed."""
            # Check if we got a 429 response
            if exc_type is not None:
                # Could check for specific HTTP exceptions here
                pass

    def request(
        self, domain: Optional[str] = None, weight: float = 1.0
    ) -> _RequestContext:
        """
        Context manager for rate-limited requests.

        Args:
            domain: Optional domain
            weight: Request weight

        Returns:
            Async context manager

        Example:
            >>> async with limiter.request("api.example.com"):
            ...     response = await client.get(url)
        """
        return self._RequestContext(self, domain, weight)


def create_rate_limiter(
    strategy: str = "token_bucket",
    requests_per_second: float = 1.0,
    burst_size: int = 5,
    **kwargs,
) -> RateLimiter:
    """
    Factory function to create a rate limiter.

    Args:
        strategy: "token_bucket", "fixed_window", or "sliding_window"
        requests_per_second: Requests per second
        burst_size: Burst capacity
        **kwargs: Additional arguments for RateLimiter

    Returns:
        Configured RateLimiter instance
    """
    strategy_enum = RateLimitStrategy(strategy.lower())
    return RateLimiter(
        strategy=strategy_enum,
        requests_per_second=requests_per_second,
        burst_size=burst_size,
        **kwargs,
    )
