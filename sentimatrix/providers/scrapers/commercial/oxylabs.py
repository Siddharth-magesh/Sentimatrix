"""
Oxylabs Integration

Oxylabs is an enterprise web scraping platform with:
- 100M+ residential and datacenter proxies
- Web Scraper API for e-commerce and SERP
- Real-time and batch processing
- Automatic parsing and data extraction

Features:
- E-Commerce Scraper: Amazon, eBay, Walmart, etc.
- SERP Scraper: Google, Bing, Yahoo, etc.
- Real Estate Scraper: Zillow, Realtor, etc.
- Universal Scraper: Any website

Pricing: From $49/month for Web Scraper API

API Documentation: https://developers.oxylabs.io/

Example:
    >>> from sentimatrix.providers.scrapers.commercial import OxylabsClient
    >>>
    >>> async with OxylabsClient(username="user", password="pass") as client:
    ...     # Scrape any URL
    ...     result = await client.scrape("https://example.com")
    ...
    ...     # E-commerce scraping
    ...     products = await client.scrape_amazon(query="laptop", pages=3)
    ...
    ...     # SERP scraping
    ...     results = await client.scrape_google("best laptops 2024")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

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


class OxylabsSource(str, Enum):
    """Oxylabs scraping sources."""
    UNIVERSAL = "universal"
    UNIVERSAL_ECOMMERCE = "universal_ecommerce"

    # E-commerce
    AMAZON = "amazon"
    AMAZON_SEARCH = "amazon_search"
    AMAZON_PRODUCT = "amazon_product"
    AMAZON_PRICING = "amazon_pricing"
    AMAZON_REVIEWS = "amazon_reviews"
    AMAZON_QUESTIONS = "amazon_questions"
    AMAZON_BESTSELLERS = "amazon_bestsellers"
    AMAZON_SELLERS = "amazon_sellers"

    EBAY = "ebay"
    EBAY_SEARCH = "ebay_search"
    EBAY_PRODUCT = "ebay_product"

    WALMART = "walmart"
    WALMART_SEARCH = "walmart_search"
    WALMART_PRODUCT = "walmart_product"

    # SERP
    GOOGLE = "google"
    GOOGLE_SEARCH = "google_search"
    GOOGLE_ADS = "google_ads"
    GOOGLE_SHOPPING = "google_shopping"
    GOOGLE_IMAGES = "google_images"
    GOOGLE_TRENDS = "google_trends"

    BING = "bing"
    BING_SEARCH = "bing_search"

    # Real Estate
    ZILLOW = "zillow"
    REALTOR = "realtor"


@dataclass
class OxylabsConfig(CommercialAPIConfig):
    """Oxylabs-specific configuration."""

    username: Optional[str] = None
    password: Optional[str] = None

    # Request settings
    source: OxylabsSource = OxylabsSource.UNIVERSAL
    render_js: bool = False
    parse: bool = True  # Return parsed JSON data

    # Geo-targeting
    geo_location: Optional[str] = None
    locale: Optional[str] = None
    domain: Optional[str] = None

    # User agent
    user_agent_type: str = "desktop"  # desktop, mobile, desktop_chrome, etc.

    # Context options (for specific sources)
    context: Dict[str, Any] = field(default_factory=dict)


class OxylabsClient(BaseCommercialClient):
    """
    Oxylabs Web Scraper API client.

    Oxylabs provides:
    - 100M+ proxy pool
    - Specialized scrapers for e-commerce and SERP
    - Real-time and batch processing
    - Automatic data parsing

    Two integration methods:
    - Realtime (sync): Immediate results
    - Push-Pull (async): For large-scale scraping
    """

    SERVICE_NAME = "oxylabs"
    REALTIME_URL = "https://realtime.oxylabs.io/v1/queries"
    ASYNC_URL = "https://data.oxylabs.io/v1/queries"

    def __init__(
        self,
        config: Optional[OxylabsConfig] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        """
        Initialize Oxylabs client.

        Args:
            config: Oxylabs configuration
            username: API username
            password: API password
        """
        if config is None:
            config = OxylabsConfig()

        if username:
            config.username = username
        if password:
            config.password = password

        super().__init__(config)
        self._oxy_config: OxylabsConfig = config

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="oxylabs",
            provider_type=ProviderType.SCRAPER,
            version="1.0.0",
            description="Oxylabs - Web Scraper API with 100M+ proxies",
            capabilities=ProviderCapabilities(
                javascript_rendering=True,
                screenshots=False,
                pdf_generation=False,
                proxy_support=True,
                batch_processing=True,
            ),
        )

    def _get_auth(self) -> tuple:
        """Get basic auth credentials."""
        return (self._oxy_config.username, self._oxy_config.password)

    async def _make_request(
        self,
        url: str,
        render_js: bool = False,
        source: Optional[OxylabsSource] = None,
        parse: Optional[bool] = None,
        geo_location: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        wait_for: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> ScrapeResult:
        """
        Make request to Oxylabs Realtime API.

        Args:
            url: Target URL
            render_js: Enable JavaScript rendering
            source: Scraping source type
            parse: Return parsed data
            geo_location: Target location
            headers: Custom headers
            cookies: Custom cookies
            wait_for: Not supported (ignored)
            timeout: Request timeout

        Returns:
            ScrapeResult with response
        """
        httpx = _get_httpx()

        if not self._oxy_config.username or not self._oxy_config.password:
            raise ValueError("Oxylabs username and password required.")

        # Build request body
        request_body: Dict[str, Any] = {
            "source": (source or self._oxy_config.source).value,
            "url": url,
        }

        # Rendering
        if render_js or self._oxy_config.render_js:
            request_body["render"] = "html"

        # Parsing
        should_parse = parse if parse is not None else self._oxy_config.parse
        if should_parse:
            request_body["parse"] = True

        # Geo-targeting
        target_geo = geo_location or self._oxy_config.geo_location
        if target_geo:
            request_body["geo_location"] = target_geo

        if self._oxy_config.locale:
            request_body["locale"] = self._oxy_config.locale

        if self._oxy_config.domain:
            request_body["domain"] = self._oxy_config.domain

        # User agent
        request_body["user_agent_type"] = self._oxy_config.user_agent_type

        # Context options
        if self._oxy_config.context:
            request_body["context"] = self._oxy_config.context

        # Custom headers
        if headers:
            request_body["custom_headers"] = headers

        try:
            response = await self._client.post(
                self.REALTIME_URL,
                json=request_body,
                auth=self._get_auth(),
                timeout=timeout or self._oxy_config.timeout,
            )

            response.raise_for_status()
            data = response.json()

            # Extract results
            results = data.get("results", [{}])
            result = results[0] if results else {}

            content = result.get("content", "")
            parsed_data = result.get("parsed", None)

            return ScrapeResult(
                url=url,
                content=str(parsed_data) if parsed_data else content,
                status_code=result.get("status_code", response.status_code),
                html=content if not parsed_data else None,
                json_data=parsed_data,
                headers=dict(response.headers),
                credits_used=data.get("cost", 1),
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

    async def scrape_amazon(
        self,
        url: Optional[str] = None,
        query: Optional[str] = None,
        asin: Optional[str] = None,
        source_type: str = "search",
        domain: str = "com",
        pages: int = 1,
        start_page: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Scrape Amazon products, search results, or reviews.

        Args:
            url: Direct product URL
            query: Search query
            asin: Product ASIN
            source_type: "search", "product", "reviews", "pricing", "questions"
            domain: Amazon domain (com, co.uk, de, etc.)
            pages: Number of pages to scrape
            start_page: Starting page number
            **kwargs: Additional options

        Returns:
            Scraped Amazon data
        """
        self._ensure_initialized()

        # Determine source
        source_map = {
            "search": OxylabsSource.AMAZON_SEARCH,
            "product": OxylabsSource.AMAZON_PRODUCT,
            "reviews": OxylabsSource.AMAZON_REVIEWS,
            "pricing": OxylabsSource.AMAZON_PRICING,
            "questions": OxylabsSource.AMAZON_QUESTIONS,
            "bestsellers": OxylabsSource.AMAZON_BESTSELLERS,
            "sellers": OxylabsSource.AMAZON_SELLERS,
        }
        source = source_map.get(source_type, OxylabsSource.AMAZON)

        # Build request
        request_body: Dict[str, Any] = {
            "source": source.value,
            "domain": domain,
            "parse": True,
        }

        if url:
            request_body["url"] = url
        elif query:
            request_body["query"] = query
            request_body["pages"] = pages
            request_body["start_page"] = start_page
        elif asin:
            request_body["query"] = asin

        return await self._realtime_request(request_body, **kwargs)

    async def scrape_google(
        self,
        query: str,
        source_type: str = "search",
        domain: str = "com",
        locale: str = "en-us",
        geo_location: Optional[str] = None,
        pages: int = 1,
        start_page: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Scrape Google search results.

        Args:
            query: Search query
            source_type: "search", "ads", "shopping", "images", "trends"
            domain: Google domain
            locale: Result locale
            geo_location: Target location
            pages: Number of pages
            start_page: Starting page
            **kwargs: Additional options

        Returns:
            Search results
        """
        self._ensure_initialized()

        source_map = {
            "search": OxylabsSource.GOOGLE_SEARCH,
            "ads": OxylabsSource.GOOGLE_ADS,
            "shopping": OxylabsSource.GOOGLE_SHOPPING,
            "images": OxylabsSource.GOOGLE_IMAGES,
            "trends": OxylabsSource.GOOGLE_TRENDS,
        }
        source = source_map.get(source_type, OxylabsSource.GOOGLE_SEARCH)

        request_body: Dict[str, Any] = {
            "source": source.value,
            "domain": domain,
            "query": query,
            "locale": locale,
            "pages": pages,
            "start_page": start_page,
            "parse": True,
        }

        if geo_location:
            request_body["geo_location"] = geo_location

        return await self._realtime_request(request_body, **kwargs)

    async def scrape_ebay(
        self,
        url: Optional[str] = None,
        query: Optional[str] = None,
        source_type: str = "search",
        domain: str = "com",
        pages: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Scrape eBay products or search results.

        Args:
            url: Direct URL
            query: Search query
            source_type: "search" or "product"
            domain: eBay domain
            pages: Number of pages
            **kwargs: Additional options

        Returns:
            eBay data
        """
        self._ensure_initialized()

        source = OxylabsSource.EBAY_SEARCH if source_type == "search" else OxylabsSource.EBAY_PRODUCT

        request_body: Dict[str, Any] = {
            "source": source.value,
            "domain": domain,
            "parse": True,
        }

        if url:
            request_body["url"] = url
        elif query:
            request_body["query"] = query
            request_body["pages"] = pages

        return await self._realtime_request(request_body, **kwargs)

    async def scrape_walmart(
        self,
        url: Optional[str] = None,
        query: Optional[str] = None,
        source_type: str = "search",
        pages: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Scrape Walmart products or search results.

        Args:
            url: Direct URL
            query: Search query
            source_type: "search" or "product"
            pages: Number of pages
            **kwargs: Additional options

        Returns:
            Walmart data
        """
        self._ensure_initialized()

        source = OxylabsSource.WALMART_SEARCH if source_type == "search" else OxylabsSource.WALMART_PRODUCT

        request_body: Dict[str, Any] = {
            "source": source.value,
            "parse": True,
        }

        if url:
            request_body["url"] = url
        elif query:
            request_body["query"] = query
            request_body["pages"] = pages

        return await self._realtime_request(request_body, **kwargs)

    async def _realtime_request(
        self,
        request_body: Dict[str, Any],
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Make realtime API request.

        Args:
            request_body: Request body
            timeout: Request timeout
            **kwargs: Additional options

        Returns:
            API response
        """
        response = await self._client.post(
            self.REALTIME_URL,
            json=request_body,
            auth=self._get_auth(),
            timeout=timeout or self._oxy_config.timeout,
        )

        response.raise_for_status()
        return response.json()

    async def submit_async_job(
        self,
        request_body: Dict[str, Any],
        callback_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit async scraping job.

        Args:
            request_body: Job request body
            callback_url: Webhook URL for results

        Returns:
            Job submission response
        """
        self._ensure_initialized()

        if callback_url:
            request_body["callback_url"] = callback_url

        response = await self._client.post(
            self.ASYNC_URL,
            json=request_body,
            auth=self._get_auth(),
        )

        response.raise_for_status()
        return response.json()

    async def get_job_results(self, job_id: str) -> Dict[str, Any]:
        """
        Get async job results.

        Args:
            job_id: Job ID

        Returns:
            Job results
        """
        self._ensure_initialized()

        response = await self._client.get(
            f"{self.ASYNC_URL}/{job_id}/results",
            auth=self._get_auth(),
        )

        response.raise_for_status()
        return response.json()
