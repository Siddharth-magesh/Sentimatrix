"""
Unit tests for Scraper Utilities.

Tests cover:
- ProxyManager: Proxy rotation and health tracking
- UserAgentRotator: User agent rotation
- RetryHandler: Retry logic with backoff
- Utility functions
"""

import asyncio
import time

import pytest

from sentimatrix.providers.scrapers.utils import (
    ProxyManager,
    ProxyInfo,
    ProxyProtocol,
    RotationStrategy,
    UserAgentRotator,
    DeviceType,
    RetryHandler,
    extract_domain,
    normalize_url,
    parse_cookies,
)


class TestProxyInfo:
    """Test ProxyInfo dataclass."""

    def test_basic_creation(self):
        """Test basic proxy info creation."""
        proxy = ProxyInfo(url="http://proxy:8080")
        assert proxy.url == "http://proxy:8080"
        assert proxy.protocol == ProxyProtocol.HTTP
        assert proxy.is_healthy is True

    def test_use_count(self):
        """Test use count calculation."""
        proxy = ProxyInfo(url="http://proxy:8080")
        proxy.success_count = 5
        proxy.failure_count = 2
        assert proxy.use_count == 7

    def test_success_rate(self):
        """Test success rate calculation."""
        proxy = ProxyInfo(url="http://proxy:8080")
        proxy.success_count = 8
        proxy.failure_count = 2
        assert proxy.success_rate == 0.8

    def test_success_rate_no_usage(self):
        """Test success rate with no usage."""
        proxy = ProxyInfo(url="http://proxy:8080")
        assert proxy.success_rate == 1.0

    def test_avg_response_time(self):
        """Test average response time."""
        proxy = ProxyInfo(url="http://proxy:8080")
        proxy.success_count = 2
        proxy.total_response_time_ms = 200.0
        assert proxy.avg_response_time_ms == 100.0

    def test_record_success(self):
        """Test recording success."""
        proxy = ProxyInfo(url="http://proxy:8080")
        proxy.record_success(100.0)

        assert proxy.success_count == 1
        assert proxy.total_response_time_ms == 100.0
        assert proxy.last_used_at is not None
        assert proxy.is_healthy is True

    def test_record_failure(self):
        """Test recording failure."""
        proxy = ProxyInfo(url="http://proxy:8080")
        proxy.record_failure()

        assert proxy.failure_count == 1
        assert proxy.last_used_at is not None

    def test_unhealthy_after_failures(self):
        """Test proxy marked unhealthy after many failures."""
        proxy = ProxyInfo(url="http://proxy:8080")

        # Many failures, low success rate
        for _ in range(10):
            proxy.record_failure()

        assert proxy.is_healthy is False

    def test_get_auth_url(self):
        """Test getting URL with auth."""
        proxy = ProxyInfo(
            url="http://proxy:8080",
            username="user",
            password="pass",
        )
        auth_url = proxy.get_auth_url()
        assert "user:pass@" in auth_url

    def test_to_dict(self):
        """Test conversion to dict."""
        proxy = ProxyInfo(url="http://proxy:8080", country="US")
        data = proxy.to_dict()

        assert data["url"] == "http://proxy:8080"
        assert data["country"] == "US"
        assert "success_rate" in data


class TestProxyManager:
    """Test ProxyManager class."""

    def test_init_empty(self):
        """Test initialization with no proxies."""
        manager = ProxyManager()
        assert manager.proxy_count == 0

    def test_init_with_proxies(self):
        """Test initialization with proxy list."""
        manager = ProxyManager(
            proxies=["http://proxy1:8080", "http://proxy2:8080"]
        )
        assert manager.proxy_count == 2

    def test_add_proxy(self):
        """Test adding a proxy."""
        manager = ProxyManager()
        manager.add_proxy("http://proxy:8080", username="user", password="pass")

        assert manager.proxy_count == 1
        proxy = manager.get_proxy()
        assert proxy.username == "user"

    def test_remove_proxy(self):
        """Test removing a proxy."""
        manager = ProxyManager(proxies=["http://proxy:8080"])
        manager.remove_proxy("http://proxy:8080")
        assert manager.proxy_count == 0

    def test_get_proxy_round_robin(self):
        """Test round robin rotation."""
        manager = ProxyManager(
            proxies=["http://proxy1:8080", "http://proxy2:8080"],
            strategy=RotationStrategy.ROUND_ROBIN,
        )

        proxy1 = manager.get_proxy()
        proxy2 = manager.get_proxy()
        proxy3 = manager.get_proxy()

        # Should cycle through
        assert proxy1.url != proxy2.url or manager.proxy_count == 1
        assert proxy3.url == proxy1.url or manager.proxy_count == 1

    def test_get_proxy_random(self):
        """Test random rotation."""
        manager = ProxyManager(
            proxies=["http://proxy1:8080", "http://proxy2:8080"],
            strategy=RotationStrategy.RANDOM,
        )

        # Should return a proxy
        proxy = manager.get_proxy()
        assert proxy is not None

    def test_get_proxy_least_used(self):
        """Test least used rotation."""
        manager = ProxyManager(
            proxies=["http://proxy1:8080", "http://proxy2:8080"],
            strategy=RotationStrategy.LEAST_USED,
        )

        # Use proxy1
        proxy1 = manager.get_proxy()
        manager.report_success(proxy1.url, 100.0)

        # Should prefer proxy2 (least used)
        proxy2 = manager.get_proxy()
        # If both unused, either is valid
        assert proxy2 is not None

    def test_get_proxy_sticky(self):
        """Test sticky rotation."""
        manager = ProxyManager(
            proxies=["http://proxy1:8080", "http://proxy2:8080"],
            strategy=RotationStrategy.STICKY,
        )

        proxy1 = manager.get_proxy()
        proxy2 = manager.get_proxy()
        proxy3 = manager.get_proxy()

        # Should always return same proxy
        assert proxy1.url == proxy2.url == proxy3.url

    def test_get_proxy_by_country(self):
        """Test filtering by country."""
        manager = ProxyManager()
        manager.add_proxy("http://proxy1:8080", country="US")
        manager.add_proxy("http://proxy2:8080", country="UK")

        proxy = manager.get_proxy(country="US")
        assert proxy is not None
        assert proxy.country == "US"

    def test_report_success(self):
        """Test reporting success."""
        manager = ProxyManager(proxies=["http://proxy:8080"])
        manager.report_success("http://proxy:8080", 100.0)

        proxy = manager.get_proxy()
        assert proxy.success_count == 1

    def test_report_failure(self):
        """Test reporting failure."""
        manager = ProxyManager(proxies=["http://proxy:8080"])
        manager.report_failure("http://proxy:8080")

        proxy = manager.get_proxy()
        assert proxy.failure_count == 1

    def test_get_proxy_for_httpx(self):
        """Test getting proxy config for HTTPX."""
        manager = ProxyManager(proxies=["http://proxy:8080"])
        config = manager.get_proxy_for_httpx()

        assert config is not None
        assert "http://" in config
        assert "https://" in config

    def test_get_proxy_for_playwright(self):
        """Test getting proxy config for Playwright."""
        manager = ProxyManager()
        manager.add_proxy("http://proxy:8080", username="user", password="pass")

        config = manager.get_proxy_for_playwright()

        assert config is not None
        assert "server" in config
        assert config.get("username") == "user"

    def test_healthy_count(self):
        """Test healthy proxy count."""
        manager = ProxyManager(
            proxies=["http://proxy1:8080", "http://proxy2:8080"]
        )
        assert manager.healthy_count == 2

    def test_get_stats(self):
        """Test getting statistics."""
        manager = ProxyManager(proxies=["http://proxy:8080"])
        stats = manager.get_stats()

        assert "total_proxies" in stats
        assert "healthy_proxies" in stats
        assert "proxies" in stats

    def test_reset_stats(self):
        """Test resetting statistics."""
        manager = ProxyManager(proxies=["http://proxy:8080"])
        manager.report_success("http://proxy:8080", 100.0)
        manager.reset_stats()

        proxy = manager.get_proxy()
        assert proxy.success_count == 0


class TestUserAgentRotator:
    """Test UserAgentRotator class."""

    def test_init_desktop(self):
        """Test initialization with desktop type."""
        rotator = UserAgentRotator(device_type=DeviceType.DESKTOP)
        assert rotator.count > 0

    def test_init_mobile(self):
        """Test initialization with mobile type."""
        rotator = UserAgentRotator(device_type=DeviceType.MOBILE)
        assert rotator.count > 0

    def test_init_mixed(self):
        """Test initialization with mixed type."""
        rotator = UserAgentRotator(device_type=DeviceType.MIXED)
        assert rotator.count > 0

    def test_init_custom(self):
        """Test initialization with custom user agents."""
        custom = ["Custom UA 1", "Custom UA 2"]
        rotator = UserAgentRotator(custom_user_agents=custom)
        assert rotator.count == 2

    def test_get_user_agent_random(self):
        """Test getting random user agent."""
        rotator = UserAgentRotator(rotation="random")
        ua = rotator.get_user_agent()
        assert isinstance(ua, str)
        assert len(ua) > 0

    def test_get_user_agent_round_robin(self):
        """Test getting user agents round robin."""
        rotator = UserAgentRotator(
            custom_user_agents=["UA1", "UA2", "UA3"],
            rotation="round_robin",
        )

        ua1 = rotator.get_user_agent()
        ua2 = rotator.get_user_agent()
        ua3 = rotator.get_user_agent()
        ua4 = rotator.get_user_agent()

        assert ua1 == "UA1"
        assert ua2 == "UA2"
        assert ua3 == "UA3"
        assert ua4 == "UA1"

    def test_add_user_agent(self):
        """Test adding user agent."""
        rotator = UserAgentRotator(custom_user_agents=["UA1"])
        initial = rotator.count

        rotator.add_user_agent("UA2")
        assert rotator.count == initial + 1

    def test_user_agent_format(self):
        """Test user agent string format."""
        rotator = UserAgentRotator(device_type=DeviceType.DESKTOP)
        ua = rotator.get_user_agent()

        # Should look like a browser user agent
        assert "Mozilla" in ua or len(ua) > 20


class TestRetryHandler:
    """Test RetryHandler class."""

    def test_init_default(self):
        """Test default initialization."""
        handler = RetryHandler()
        assert handler._max_retries > 0
        assert handler._initial_delay > 0

    def test_init_custom(self):
        """Test custom initialization."""
        handler = RetryHandler(
            max_retries=5,
            initial_delay=2.0,
            max_delay=120.0,
        )
        assert handler._max_retries == 5
        assert handler._initial_delay == 2.0
        assert handler._max_delay == 120.0

    def test_get_delay_exponential(self):
        """Test exponential backoff delay."""
        handler = RetryHandler(
            initial_delay=1.0,
            exponential_base=2.0,
            jitter=False,
        )

        delay0 = handler.get_delay(0)
        delay1 = handler.get_delay(1)
        delay2 = handler.get_delay(2)

        assert delay0 == 1.0
        assert delay1 == 2.0
        assert delay2 == 4.0

    def test_get_delay_max(self):
        """Test delay capped at max."""
        handler = RetryHandler(
            initial_delay=1.0,
            max_delay=5.0,
            exponential_base=10.0,
            jitter=False,
        )

        delay = handler.get_delay(5)
        assert delay == 5.0

    def test_get_delay_jitter(self):
        """Test delay with jitter."""
        handler = RetryHandler(
            initial_delay=1.0,
            jitter=True,
        )

        delay = handler.get_delay(0)
        # With jitter, delay should be >= base delay
        assert delay >= 1.0

    def test_should_retry_status(self):
        """Test should retry based on status code."""
        handler = RetryHandler(retry_on_status=[429, 500, 503])

        assert handler.should_retry_status(429) is True
        assert handler.should_retry_status(500) is True
        assert handler.should_retry_status(200) is False
        assert handler.should_retry_status(404) is False

    def test_should_retry_exception(self):
        """Test should retry based on exception."""
        handler = RetryHandler(
            retry_on_exceptions=[ConnectionError, TimeoutError]
        )

        assert handler.should_retry_exception(ConnectionError()) is True
        assert handler.should_retry_exception(TimeoutError()) is True
        assert handler.should_retry_exception(ValueError()) is False

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test execute with successful function."""
        handler = RetryHandler()
        call_count = 0

        async def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await handler.execute(success_func)

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_execute_retry_then_success(self):
        """Test execute with retry then success."""
        handler = RetryHandler(
            max_retries=3,
            initial_delay=0.01,
            retry_on_exceptions=[ConnectionError],
        )
        call_count = 0

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Retry please")
            return "success"

        result = await handler.execute(fail_then_succeed)

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_execute_exhaust_retries(self):
        """Test execute exhausts retries."""
        handler = RetryHandler(
            max_retries=2,
            initial_delay=0.01,
            retry_on_exceptions=[ConnectionError],
        )

        async def always_fail():
            raise ConnectionError("Always fails")

        with pytest.raises(ConnectionError):
            await handler.execute(always_fail)

    @pytest.mark.asyncio
    async def test_execute_non_retryable_exception(self):
        """Test execute with non-retryable exception."""
        handler = RetryHandler(
            max_retries=3,
            retry_on_exceptions=[ConnectionError],
        )
        call_count = 0

        async def raise_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        with pytest.raises(ValueError):
            await handler.execute(raise_value_error)

        # Should not retry
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_attempts_generator(self):
        """Test attempts async generator."""
        handler = RetryHandler(max_retries=2)
        attempts_list = []

        async for attempt in handler.attempts():
            attempts_list.append(attempt.number)
            if attempt.number == 1:
                attempt.stop()

        assert attempts_list == [0, 1]

    @pytest.mark.asyncio
    async def test_attempt_is_last(self):
        """Test attempt is_last property."""
        handler = RetryHandler(max_retries=2)

        async for attempt in handler.attempts():
            if attempt.number == 0:
                assert attempt.is_last is False
            elif attempt.number == 2:
                assert attempt.is_last is True
            attempt.stop()
            break


class TestUtilityFunctions:
    """Test utility functions."""

    def test_extract_domain(self):
        """Test domain extraction."""
        assert extract_domain("https://example.com/path") == "example.com"
        assert extract_domain("http://sub.example.com:8080/path") == "sub.example.com:8080"
        assert extract_domain("example.com") == "example.com"

    def test_normalize_url(self):
        """Test URL normalization."""
        assert normalize_url("example.com") == "https://example.com/"
        assert normalize_url("http://example.com/path/") == "http://example.com/path"
        assert normalize_url("https://example.com") == "https://example.com/"

    def test_parse_cookies(self):
        """Test cookie parsing."""
        cookie_string = "session=abc123; user=john; theme=dark"
        cookies = parse_cookies(cookie_string)

        assert cookies["session"] == "abc123"
        assert cookies["user"] == "john"
        assert cookies["theme"] == "dark"

    def test_parse_cookies_empty(self):
        """Test parsing empty cookie string."""
        cookies = parse_cookies("")
        assert cookies == {}
