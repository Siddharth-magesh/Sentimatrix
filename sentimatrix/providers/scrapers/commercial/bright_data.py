"""
Bright Data Integration

Bright Data (formerly Luminati) is an enterprise-grade web scraping platform with:
- 72M+ residential IPs across 195 countries
- Datacenter, ISP, and mobile proxies
- Web Scraper IDE for custom scrapers
- Automatic CAPTCHA solving and unblocking

Features:
- Web Scraper API: Pre-built scrapers for popular sites
- Scraping Browser: Full browser automation
- SERP API: Search engine results
- Dataset marketplace

Pricing: From $500/mo for enterprise features

API Documentation: https://docs.brightdata.com/

Example:
    >>> from sentimatrix.providers.scrapers.commercial import BrightDataClient
    >>>
    >>> async with BrightDataClient(api_token="your_token") as client:
    ...     # Scrape any URL
    ...     result = await client.scrape("https://example.com")
    ...
    ...     # Use platform-specific scraper
    ...     products = await client.scrape_amazon(url="https://amazon.com/dp/...")
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from sentimatrix.providers.base import (
    ProviderCapabilities,
    ProviderInfo,
    ProviderType,
)
from sentimatrix.providers.scrapers.commercial.base import (
    BaseCommercialClient,
    CommercialAPIConfig,
    ScrapeResult,
    _get_httpx,
)


class BrightDataZone(str, Enum):
    """Bright Data proxy zones."""
    DATACENTER = "datacenter"
    RESIDENTIAL = "residential"
    ISP = "isp"
    MOBILE = "mobile"
    UNLOCKER = "unlocker"  # Web Unlocker


@dataclass
class BrightDataConfig(CommercialAPIConfig):
    """Bright Data-specific configuration."""

    api_token: Optional[str] = None
    customer_id: Optional[str] = None

    # Zone settings
    zone: BrightDataZone = BrightDataZone.UNLOCKER
    zone_name: Optional[str] = None  # Custom zone name

    # Proxy settings
    country: Optional[str] = None
    city: Optional[str] = None
    asn: Optional[int] = None
    session_id: Optional[str] = None

    # Unlocker settings
    render_js: bool = False
    format: str = "raw"  # raw, json, html

    # Auto-create zone if not exists
    auto_create_zone: bool = True


# Supported scrapers by platform
PLATFORM_SCRAPERS = {
    "amazon": {
        "products": "/datasets/v3/trigger/amazon",
        "reviews": "/datasets/v3/trigger/amazon_reviews",
    },
    "linkedin": {
        "profiles": "/datasets/v3/trigger/linkedin_profiles",
        "companies": "/datasets/v3/trigger/linkedin_companies",
    },
    "google": {
        "search": "/serp/v1/google",
        "maps": "/datasets/v3/trigger/google_maps",
        "reviews": "/datasets/v3/trigger/google_maps_reviews",
    },
    "facebook": {
        "posts": "/datasets/v3/trigger/facebook_posts",
        "profiles": "/datasets/v3/trigger/facebook_profiles",
    },
    "instagram": {
        "profiles": "/datasets/v3/trigger/instagram_profiles",
        "posts": "/datasets/v3/trigger/instagram_posts",
    },
    "twitter": {
        "profiles": "/datasets/v3/trigger/twitter_profiles",
        "tweets": "/datasets/v3/trigger/twitter_tweets",
    },
}


class BrightDataClient(BaseCommercialClient):
    """
    Bright Data API client.

    Bright Data provides:
    - World's largest proxy network (72M+ IPs)
    - Web Unlocker for automatic CAPTCHA solving
    - Pre-built scrapers for popular platforms
    - SERP API for search engines

    Enterprise-grade with 99.9% uptime SLA.
    """

    SERVICE_NAME = "bright_data"
    BASE_URL = "https://api.brightdata.com"
    SCRAPER_URL = "https://api.brightdata.com/datasets/v3"
    SERP_URL = "https://api.brightdata.com/serp/v1"

    def __init__(
        self,
        config: Optional[BrightDataConfig] = None,
        api_token: Optional[str] = None,
        customer_id: Optional[str] = None,
    ) -> None:
        """
        Initialize Bright Data client.

        Args:
            config: Bright Data configuration
            api_token: API token
            customer_id: Customer ID (optional)
        """
        if config is None:
            config = BrightDataConfig()

        if api_token:
            config.api_token = api_token
        if customer_id:
            config.customer_id = customer_id

        super().__init__(config)
        self._bd_config: BrightDataConfig = config

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="bright_data",
            provider_type=ProviderType.SCRAPER,
            version="1.0.0",
            description="Bright Data - Enterprise web scraping with 72M+ proxies",
            capabilities=ProviderCapabilities(
                javascript_rendering=True,
                screenshots=True,
                pdf_generation=False,
                proxy_support=True,
                batch_processing=True,
            ),
        )

    def _get_headers(self) -> Dict[str, str]:
        """Get authorization headers."""
        return {
            "Authorization": f"Bearer {self._bd_config.api_token}",
            "Content-Type": "application/json",
        }

    async def _make_request(
        self,
        url: str,
        render_js: bool = False,
        country: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        wait_for: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> ScrapeResult:
        """
        Make request using Bright Data Web Unlocker.

        Args:
            url: Target URL
            render_js: Enable JavaScript rendering
            country: Target country code
            headers: Custom headers
            cookies: Custom cookies
            wait_for: CSS selector to wait for
            timeout: Request timeout

        Returns:
            ScrapeResult with response
        """
        httpx = _get_httpx()

        if not self._bd_config.api_token:
            raise ValueError("Bright Data API token required.")

        # Build request body
        request_body: Dict[str, Any] = {
            "url": url,
            "format": self._bd_config.format,
        }

        # JavaScript rendering
        if render_js or self._bd_config.render_js:
            request_body["render_js"] = True

        # Geo-targeting
        target_country = country or self._bd_config.country
        if target_country:
            request_body["country"] = target_country

        if self._bd_config.city:
            request_body["city"] = self._bd_config.city

        # Custom headers
        if headers:
            request_body["headers"] = headers

        # Wait for selector
        if wait_for:
            request_body["wait_for"] = wait_for

        try:
            response = await self._client.post(
                f"{self.BASE_URL}/request",
                json=request_body,
                headers=self._get_headers(),
                timeout=timeout or self._bd_config.timeout,
            )

            response.raise_for_status()

            content = response.text
            json_data = None

            if self._bd_config.format == "json":
                try:
                    json_data = response.json()
                    content = str(json_data)
                except Exception:
                    pass

            return ScrapeResult(
                url=url,
                content=content,
                status_code=response.status_code,
                html=content if self._bd_config.format in ("raw", "html") else None,
                json_data=json_data,
                headers=dict(response.headers),
                provider=self.SERVICE_NAME,
            )

        except Exception as e:
            return ScrapeResult(
                url=url,
                content="",
                status_code=0,
                error=str(e),
                provider=self.SERVICE_NAME,
            )

    async def scrape_generic(
        self,
        url: str,
        **kwargs: Any,
    ) -> ScrapeResult:
        """
        Scrape any URL using Web Unlocker.

        Args:
            url: Target URL
            **kwargs: Additional options

        Returns:
            ScrapeResult
        """
        return await self._make_request(url, **kwargs)

    async def scrape_amazon(
        self,
        url: Optional[str] = None,
        asin: Optional[str] = None,
        product_type: str = "products",
        country: str = "us",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Scrape Amazon products or reviews.

        Args:
            url: Product URL
            asin: Product ASIN
            product_type: "products" or "reviews"
            country: Target country
            **kwargs: Additional options

        Returns:
            Scraped data
        """
        self._ensure_initialized()

        endpoint = PLATFORM_SCRAPERS["amazon"][product_type]

        input_data = []
        if url:
            input_data.append({"url": url})
        elif asin:
            input_data.append({"asin": asin, "country": country})

        return await self._trigger_scraper(endpoint, input_data, **kwargs)

    async def scrape_google(
        self,
        query: str,
        search_type: str = "search",
        country: str = "us",
        language: str = "en",
        num_results: int = 10,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Scrape Google search results or maps.

        Args:
            query: Search query
            search_type: "search", "maps", or "reviews"
            country: Target country
            language: Result language
            num_results: Number of results
            **kwargs: Additional options

        Returns:
            Search results
        """
        self._ensure_initialized()

        if search_type == "search":
            # SERP API
            response = await self._client.post(
                f"{self.SERP_URL}/google",
                json={
                    "query": query,
                    "country": country,
                    "language": language,
                    "results": num_results,
                },
                headers=self._get_headers(),
            )
            response.raise_for_status()
            return response.json()
        else:
            endpoint = PLATFORM_SCRAPERS["google"][search_type]
            input_data = [{"query": query, "country": country}]
            return await self._trigger_scraper(endpoint, input_data, **kwargs)

    async def scrape_linkedin(
        self,
        url: Optional[str] = None,
        profile_type: str = "profiles",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Scrape LinkedIn profiles or companies.

        Args:
            url: Profile/company URL
            profile_type: "profiles" or "companies"
            **kwargs: Additional options

        Returns:
            Profile data
        """
        self._ensure_initialized()

        endpoint = PLATFORM_SCRAPERS["linkedin"][profile_type]
        input_data = [{"url": url}] if url else []

        return await self._trigger_scraper(endpoint, input_data, **kwargs)

    async def scrape_social(
        self,
        platform: str,
        url: Optional[str] = None,
        username: Optional[str] = None,
        content_type: str = "profiles",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Scrape social media platforms.

        Args:
            platform: "facebook", "instagram", or "twitter"
            url: Profile/post URL
            username: Username (alternative to URL)
            content_type: "profiles", "posts", or "tweets"
            **kwargs: Additional options

        Returns:
            Social media data
        """
        self._ensure_initialized()

        if platform not in PLATFORM_SCRAPERS:
            raise ValueError(f"Unsupported platform: {platform}")

        endpoint = PLATFORM_SCRAPERS[platform].get(content_type)
        if not endpoint:
            raise ValueError(f"Unsupported content type for {platform}: {content_type}")

        input_data = []
        if url:
            input_data.append({"url": url})
        elif username:
            input_data.append({"username": username})

        return await self._trigger_scraper(endpoint, input_data, **kwargs)

    async def _trigger_scraper(
        self,
        endpoint: str,
        input_data: List[Dict[str, Any]],
        wait: bool = True,
        webhook_url: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Trigger a Bright Data scraper.

        Args:
            endpoint: Scraper endpoint
            input_data: Input data array
            wait: Wait for completion
            webhook_url: Webhook for async results
            **kwargs: Additional options

        Returns:
            Scraper results or job info
        """
        request_body = {
            "input": input_data,
        }

        if webhook_url:
            request_body["webhook"] = webhook_url

        params = {}
        if wait:
            params["wait"] = "true"

        response = await self._client.post(
            f"{self.BASE_URL}{endpoint}",
            params=params,
            json=request_body,
            headers=self._get_headers(),
            timeout=self._bd_config.timeout,
        )

        response.raise_for_status()
        return response.json()

    async def get_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        """
        Get snapshot results.

        Args:
            snapshot_id: Snapshot ID from async request

        Returns:
            Snapshot data
        """
        self._ensure_initialized()

        response = await self._client.get(
            f"{self.SCRAPER_URL}/snapshots/{snapshot_id}",
            headers=self._get_headers(),
        )

        response.raise_for_status()
        return response.json()

    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information and balance.

        Returns:
            Account info
        """
        self._ensure_initialized()

        response = await self._client.get(
            f"{self.BASE_URL}/customer",
            headers=self._get_headers(),
        )

        response.raise_for_status()
        return response.json()

    def get_proxy_url(
        self,
        zone: Optional[BrightDataZone] = None,
        country: Optional[str] = None,
        session: Optional[str] = None,
    ) -> str:
        """
        Get proxy URL for use with other libraries.

        Args:
            zone: Proxy zone type
            country: Target country
            session: Session ID for sticky sessions

        Returns:
            Proxy URL string
        """
        zone_name = self._bd_config.zone_name or (zone or self._bd_config.zone).value
        username = f"brd-customer-{self._bd_config.customer_id}-zone-{zone_name}"

        if country or self._bd_config.country:
            username += f"-country-{country or self._bd_config.country}"

        if session:
            username += f"-session-{session}"

        return f"http://{username}:{self._bd_config.api_token}@brd.superproxy.io:22225"
