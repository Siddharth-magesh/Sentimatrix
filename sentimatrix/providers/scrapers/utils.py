"""
Sentimatrix Scraper Utilities

Provides utility classes for web scraping:
- ProxyManager: Proxy rotation and management
- UserAgentRotator: User agent rotation for anti-detection
- RetryHandler: Retry logic with exponential backoff

Example:
    >>> proxy_manager = ProxyManager(proxies=["http://proxy1:8080"])
    >>> ua_rotator = UserAgentRotator(device_type="desktop")
    >>> proxy = proxy_manager.get_proxy()
    >>> user_agent = ua_rotator.get_user_agent()
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
from urllib.parse import urlparse

from sentimatrix.core.config import ProxyConfig, RetryConfig
from sentimatrix.core.exceptions import (
    ProxyError,
    ScraperError,
    ConnectionTimeoutError,
)


class ProxyProtocol(str, Enum):
    """Proxy protocols."""

    HTTP = "http"
    HTTPS = "https"
    SOCKS4 = "socks4"
    SOCKS5 = "socks5"


class RotationStrategy(str, Enum):
    """Proxy rotation strategies."""

    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_USED = "least_used"
    WEIGHTED = "weighted"
    STICKY = "sticky"


@dataclass
class ProxyInfo:
    """Information about a proxy."""

    url: str
    protocol: ProxyProtocol = ProxyProtocol.HTTP
    username: Optional[str] = None
    password: Optional[str] = None
    country: Optional[str] = None

    # Statistics
    success_count: int = 0
    failure_count: int = 0
    total_response_time_ms: float = 0.0
    last_used_at: Optional[float] = None
    last_checked_at: Optional[float] = None
    is_healthy: bool = True

    @property
    def use_count(self) -> int:
        """Total usage count."""
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float:
        """Success rate (0.0 to 1.0)."""
        if self.use_count == 0:
            return 1.0
        return self.success_count / self.use_count

    @property
    def avg_response_time_ms(self) -> float:
        """Average response time in milliseconds."""
        if self.success_count == 0:
            return 0.0
        return self.total_response_time_ms / self.success_count

    def record_success(self, response_time_ms: float) -> None:
        """Record a successful request."""
        self.success_count += 1
        self.total_response_time_ms += response_time_ms
        self.last_used_at = time.time()
        self.is_healthy = True

    def record_failure(self) -> None:
        """Record a failed request."""
        self.failure_count += 1
        self.last_used_at = time.time()

        # Mark unhealthy after too many failures
        if self.failure_count > 5 and self.success_rate < 0.3:
            self.is_healthy = False

    def get_auth_url(self) -> str:
        """Get URL with authentication if credentials exist."""
        if not self.username:
            return self.url

        parsed = urlparse(self.url)
        auth = f"{self.username}:{self.password}@" if self.password else f"{self.username}@"
        return f"{parsed.scheme}://{auth}{parsed.netloc}{parsed.path}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "protocol": self.protocol.value,
            "country": self.country,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "avg_response_time_ms": self.avg_response_time_ms,
            "is_healthy": self.is_healthy,
        }


class ProxyManager:
    """
    Manages proxy rotation and health tracking.

    Supports multiple rotation strategies and tracks proxy
    performance for intelligent selection.

    Example:
        >>> manager = ProxyManager(
        ...     proxies=["http://proxy1:8080", "http://proxy2:8080"],
        ...     strategy=RotationStrategy.WEIGHTED
        ... )
        >>> proxy = manager.get_proxy()
        >>> # Use proxy...
        >>> manager.report_success(proxy.url, response_time_ms=150)
    """

    def __init__(
        self,
        config: Optional[ProxyConfig] = None,
        proxies: Optional[List[str]] = None,
        strategy: RotationStrategy = RotationStrategy.ROUND_ROBIN,
        health_check_interval: float = 60.0,
    ) -> None:
        """
        Initialize proxy manager.

        Args:
            config: Proxy configuration
            proxies: List of proxy URLs
            strategy: Rotation strategy
            health_check_interval: Seconds between health checks
        """
        self._config = config or ProxyConfig()
        self._strategy = strategy
        self._health_check_interval = health_check_interval
        self._current_index = 0
        self._sticky_proxy: Optional[str] = None
        self._lock = asyncio.Lock()

        # Parse and store proxies
        self._proxies: Dict[str, ProxyInfo] = {}

        if proxies:
            for proxy_url in proxies:
                self.add_proxy(proxy_url)
        elif self._config.url:
            self.add_proxy(self._config.url)

    def add_proxy(
        self,
        url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        country: Optional[str] = None,
    ) -> None:
        """
        Add a proxy to the pool.

        Args:
            url: Proxy URL
            username: Optional username
            password: Optional password
            country: Optional country code
        """
        # Parse URL to detect protocol
        parsed = urlparse(url)
        protocol = ProxyProtocol.HTTP
        if parsed.scheme.lower() == "https":
            protocol = ProxyProtocol.HTTPS
        elif parsed.scheme.lower() == "socks4":
            protocol = ProxyProtocol.SOCKS4
        elif parsed.scheme.lower() == "socks5":
            protocol = ProxyProtocol.SOCKS5

        # Extract credentials from URL if present
        if parsed.username:
            username = username or parsed.username
            password = password or parsed.password

        self._proxies[url] = ProxyInfo(
            url=url,
            protocol=protocol,
            username=username or self._config.username,
            password=password or self._config.password,
            country=country or self._config.country,
        )

    def remove_proxy(self, url: str) -> None:
        """Remove a proxy from the pool."""
        self._proxies.pop(url, None)
        if self._sticky_proxy == url:
            self._sticky_proxy = None

    def get_proxy(self, country: Optional[str] = None) -> Optional[ProxyInfo]:
        """
        Get a proxy based on rotation strategy.

        Args:
            country: Filter by country code

        Returns:
            ProxyInfo or None if no proxies available
        """
        if not self._proxies:
            return None

        # Filter by country and health
        available = [
            p for p in self._proxies.values()
            if p.is_healthy and (country is None or p.country == country)
        ]

        if not available:
            # Fall back to unhealthy proxies
            available = [
                p for p in self._proxies.values()
                if country is None or p.country == country
            ]

        if not available:
            return None

        # Select based on strategy
        if self._strategy == RotationStrategy.STICKY:
            if self._sticky_proxy and self._sticky_proxy in self._proxies:
                return self._proxies[self._sticky_proxy]
            # Set new sticky proxy
            proxy = available[0]
            self._sticky_proxy = proxy.url
            return proxy

        elif self._strategy == RotationStrategy.ROUND_ROBIN:
            self._current_index = self._current_index % len(available)
            proxy = available[self._current_index]
            self._current_index += 1
            return proxy

        elif self._strategy == RotationStrategy.RANDOM:
            return random.choice(available)

        elif self._strategy == RotationStrategy.LEAST_USED:
            return min(available, key=lambda p: p.use_count)

        elif self._strategy == RotationStrategy.WEIGHTED:
            # Weight by success rate
            weights = [p.success_rate + 0.1 for p in available]
            return random.choices(available, weights=weights, k=1)[0]

        return available[0]

    def report_success(self, proxy_url: str, response_time_ms: float) -> None:
        """Report successful request through proxy."""
        if proxy_url in self._proxies:
            self._proxies[proxy_url].record_success(response_time_ms)

    def report_failure(self, proxy_url: str) -> None:
        """Report failed request through proxy."""
        if proxy_url in self._proxies:
            self._proxies[proxy_url].record_failure()

    def get_proxy_for_httpx(self, country: Optional[str] = None) -> Optional[Dict[str, str]]:
        """
        Get proxy configuration for HTTPX.

        Returns:
            Dict suitable for httpx proxies parameter
        """
        proxy = self.get_proxy(country)
        if not proxy:
            return None

        url = proxy.get_auth_url()
        return {
            "http://": url,
            "https://": url,
        }

    def get_proxy_for_playwright(self, country: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get proxy configuration for Playwright.

        Returns:
            Dict suitable for playwright proxy parameter
        """
        proxy = self.get_proxy(country)
        if not proxy:
            return None

        result = {"server": proxy.url}
        if proxy.username:
            result["username"] = proxy.username
        if proxy.password:
            result["password"] = proxy.password
        return result

    @property
    def proxy_count(self) -> int:
        """Get number of proxies in pool."""
        return len(self._proxies)

    @property
    def healthy_count(self) -> int:
        """Get number of healthy proxies."""
        return sum(1 for p in self._proxies.values() if p.is_healthy)

    def get_stats(self) -> Dict[str, Any]:
        """Get proxy pool statistics."""
        return {
            "total_proxies": self.proxy_count,
            "healthy_proxies": self.healthy_count,
            "proxies": [p.to_dict() for p in self._proxies.values()],
        }

    def reset_stats(self) -> None:
        """Reset all proxy statistics."""
        for proxy in self._proxies.values():
            proxy.success_count = 0
            proxy.failure_count = 0
            proxy.total_response_time_ms = 0.0
            proxy.is_healthy = True


# User Agent Data
_DESKTOP_USER_AGENTS = [
    # Chrome on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    # Chrome on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    # Firefox on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
    # Firefox on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    # Safari on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    # Edge on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    # Chrome on Linux
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

_MOBILE_USER_AGENTS = [
    # iOS Safari
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
    # Android Chrome
    "Mozilla/5.0 (Linux; Android 14; SM-S918B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.6099.144 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.6099.144 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 13; SM-A536B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.6099.144 Mobile Safari/537.36",
]


class DeviceType(str, Enum):
    """Device types for user agent rotation."""

    DESKTOP = "desktop"
    MOBILE = "mobile"
    MIXED = "mixed"


class UserAgentRotator:
    """
    Rotates user agents for anti-detection.

    Provides realistic user agent strings with proper distribution
    of browser versions and operating systems.

    Example:
        >>> rotator = UserAgentRotator(device_type=DeviceType.DESKTOP)
        >>> ua = rotator.get_user_agent()
        >>> # Different UA each call (random)
        >>> ua2 = rotator.get_user_agent()
    """

    def __init__(
        self,
        device_type: DeviceType = DeviceType.DESKTOP,
        custom_user_agents: Optional[List[str]] = None,
        rotation: str = "random",  # "random", "round_robin"
    ) -> None:
        """
        Initialize user agent rotator.

        Args:
            device_type: Type of device to simulate
            custom_user_agents: Custom user agent list
            rotation: Rotation strategy
        """
        self._device_type = device_type
        self._rotation = rotation
        self._index = 0

        if custom_user_agents:
            self._user_agents = custom_user_agents
        else:
            self._user_agents = self._get_user_agents_for_type(device_type)

    def _get_user_agents_for_type(self, device_type: DeviceType) -> List[str]:
        """Get user agents for device type."""
        if device_type == DeviceType.DESKTOP:
            return _DESKTOP_USER_AGENTS.copy()
        elif device_type == DeviceType.MOBILE:
            return _MOBILE_USER_AGENTS.copy()
        else:  # MIXED
            return _DESKTOP_USER_AGENTS + _MOBILE_USER_AGENTS

    def get_user_agent(self) -> str:
        """
        Get a user agent string.

        Returns:
            User agent string
        """
        if not self._user_agents:
            return _DESKTOP_USER_AGENTS[0]

        if self._rotation == "round_robin":
            ua = self._user_agents[self._index % len(self._user_agents)]
            self._index += 1
            return ua
        else:  # random
            return random.choice(self._user_agents)

    def add_user_agent(self, user_agent: str) -> None:
        """Add a custom user agent."""
        self._user_agents.append(user_agent)

    @property
    def count(self) -> int:
        """Get number of user agents in pool."""
        return len(self._user_agents)


T = TypeVar("T")


class RetryHandler:
    """
    Handles retry logic with exponential backoff.

    Provides configurable retry behavior for transient errors
    with support for:
    - Exponential backoff with jitter
    - Retry on specific exceptions
    - Retry on specific HTTP status codes
    - Custom retry conditions

    Example:
        >>> retry = RetryHandler(max_retries=3, retry_on_status=[429, 500])
        >>> async for attempt in retry.attempts():
        ...     try:
        ...         response = await client.get(url)
        ...         if attempt.should_retry(status_code=response.status_code):
        ...             continue
        ...         return response
        ...     except Exception as e:
        ...         if not attempt.should_retry(exception=e):
        ...             raise
    """

    def __init__(
        self,
        config: Optional[RetryConfig] = None,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_on_status: Optional[List[int]] = None,
        retry_on_exceptions: Optional[List[type]] = None,
    ) -> None:
        """
        Initialize retry handler.

        Args:
            config: Retry configuration
            max_retries: Maximum retry attempts
            initial_delay: Initial delay between retries
            max_delay: Maximum delay between retries
            exponential_base: Base for exponential backoff
            jitter: Add random jitter to delays
            retry_on_status: HTTP status codes to retry on
            retry_on_exceptions: Exception types to retry on
        """
        self._config = config or RetryConfig()
        self._max_retries = max_retries or self._config.max_retries
        self._initial_delay = initial_delay or self._config.initial_delay
        self._max_delay = max_delay or self._config.max_delay
        self._exponential_base = exponential_base or self._config.exponential_base
        self._jitter = jitter if jitter is not None else self._config.jitter

        self._retry_on_status = retry_on_status or [429, 500, 502, 503, 504]
        self._retry_on_exceptions = retry_on_exceptions or [
            ConnectionError,
            TimeoutError,
            ConnectionTimeoutError,
        ]

    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for an attempt.

        Args:
            attempt: Attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        delay = self._initial_delay * (self._exponential_base ** attempt)
        delay = min(delay, self._max_delay)

        if self._jitter:
            # Add 0-50% jitter
            jitter = random.uniform(0, delay * 0.5)
            delay += jitter

        return delay

    def should_retry_status(self, status_code: int) -> bool:
        """Check if status code should be retried."""
        return status_code in self._retry_on_status

    def should_retry_exception(self, exception: Exception) -> bool:
        """Check if exception should be retried."""
        return any(
            isinstance(exception, exc_type)
            for exc_type in self._retry_on_exceptions
        )

    async def execute(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute a function with retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Last exception if all retries exhausted
        """
        last_exception: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                if attempt >= self._max_retries:
                    raise

                if not self.should_retry_exception(e):
                    raise

                delay = self.get_delay(attempt)
                await asyncio.sleep(delay)

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        raise ScraperError("Retry exhausted without result")

    @dataclass
    class Attempt:
        """Represents a retry attempt."""

        number: int
        max_retries: int
        handler: "RetryHandler"
        _should_stop: bool = False

        @property
        def is_last(self) -> bool:
            """Check if this is the last attempt."""
            return self.number >= self.max_retries

        def should_retry(
            self,
            status_code: Optional[int] = None,
            exception: Optional[Exception] = None,
        ) -> bool:
            """
            Check if should retry based on status or exception.

            Returns:
                True if should retry, False otherwise
            """
            if self.is_last:
                return False

            if status_code is not None:
                return self.handler.should_retry_status(status_code)

            if exception is not None:
                return self.handler.should_retry_exception(exception)

            return False

        async def delay(self) -> None:
            """Wait for the appropriate delay."""
            delay = self.handler.get_delay(self.number)
            await asyncio.sleep(delay)

        def stop(self) -> None:
            """Stop retrying."""
            self._should_stop = True

    async def attempts(self):
        """
        Async generator for retry attempts.

        Yields:
            Attempt objects

        Example:
            >>> async for attempt in retry.attempts():
            ...     try:
            ...         result = await do_something()
            ...         attempt.stop()
            ...     except Exception as e:
            ...         if attempt.is_last or not attempt.should_retry(exception=e):
            ...             raise
            ...         await attempt.delay()
        """
        for i in range(self._max_retries + 1):
            attempt = self.Attempt(
                number=i,
                max_retries=self._max_retries,
                handler=self,
            )
            yield attempt

            if attempt._should_stop:
                break


def extract_domain(url: str) -> str:
    """
    Extract domain from URL.

    Args:
        url: Full URL

    Returns:
        Domain name
    """
    parsed = urlparse(url)
    return parsed.netloc or parsed.path.split("/")[0]


def normalize_url(url: str) -> str:
    """
    Normalize a URL.

    Args:
        url: URL to normalize

    Returns:
        Normalized URL
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)
    # Remove trailing slash
    path = parsed.path.rstrip("/") or "/"
    # Reconstruct
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def parse_cookies(cookie_string: str) -> Dict[str, str]:
    """
    Parse cookie string into dictionary.

    Args:
        cookie_string: Cookie header value

    Returns:
        Dictionary of cookie names to values
    """
    cookies = {}
    for item in cookie_string.split(";"):
        item = item.strip()
        if "=" in item:
            key, value = item.split("=", 1)
            cookies[key.strip()] = value.strip()
    return cookies
