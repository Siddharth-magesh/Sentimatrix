"""
ScrapingBee Integration

ScrapingBee is a web scraping API that handles:
- Headless browser rendering
- Proxy rotation
- CAPTCHA solving
- JavaScript execution

Features:
- JavaScript rendering with Chrome
- Screenshots and PDFs
- Custom JavaScript execution
- Geo-targeting (190+ countries)
- AI web scraping with natural language

Pricing: From $49/mo for 150k API credits

API Documentation: https://www.scrapingbee.com/documentation/

Example:
    >>> from sentimatrix.providers.scrapers.commercial import ScrapingBeeClient
    >>>
    >>> async with ScrapingBeeClient(api_key="your_key") as client:
    ...     # Basic scraping
    ...     result = await client.scrape("https://example.com")
    ...
    ...     # With JS rendering
    ...     result = await client.scrape("https://spa.com", render_js=True)
    ...
    ...     # Take screenshot
    ...     screenshot = await client.take_screenshot("https://example.com")
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
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


@dataclass
class ScrapingBeeConfig(CommercialAPIConfig):
    """ScrapingBee-specific configuration."""

    api_key: Optional[str] = None

    # Rendering options
    render_js: bool = False
    premium_proxy: bool = False
    stealth_proxy: bool = False

    # JavaScript options
    js_snippet: Optional[str] = None
    js_scenario: Optional[Dict[str, Any]] = None
    wait: int = 0  # Wait milliseconds after load
    wait_for: Optional[str] = None  # CSS selector to wait for
    wait_browser: str = "load"  # load, domcontentloaded, networkidle0, networkidle2

    # Geo-targeting
    country_code: Optional[str] = None

    # Output options
    return_page_source: bool = False
    json_response: bool = False
    extract_rules: Optional[Dict[str, Any]] = None

    # Screenshot/PDF
    screenshot: bool = False
    screenshot_full_page: bool = False

    # Window size
    window_width: int = 1920
    window_height: int = 1080

    # AI Extraction
    ai_query: Optional[str] = None


class ScrapingBeeClient(BaseCommercialClient):
    """
    ScrapingBee API client.

    ScrapingBee provides:
    - Simple REST API
    - JavaScript rendering with Chrome
    - CAPTCHA solving
    - AI-powered content extraction

    Credits system:
    - 1 credit: Basic request
    - 5 credits: JS rendering
    - 10-25 credits: Premium/Stealth proxy
    """

    SERVICE_NAME = "scrapingbee"
    BASE_URL = "https://app.scrapingbee.com/api/v1"

    def __init__(
        self,
        config: Optional[ScrapingBeeConfig] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize ScrapingBee client.

        Args:
            config: ScrapingBee configuration
            api_key: API key
        """
        if config is None:
            config = ScrapingBeeConfig()

        if api_key:
            config.api_key = api_key

        super().__init__(config)
        self._sb_config: ScrapingBeeConfig = config

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="scrapingbee",
            provider_type=ProviderType.SCRAPER,
            version="1.0.0",
            description="ScrapingBee - Web scraping API with JS rendering and AI extraction",
            capabilities=ProviderCapabilities(
                javascript_rendering=True,
                screenshots=True,
                pdf_generation=True,
                proxy_support=True,
                batch_processing=True,
            ),
        )

    async def _make_request(
        self,
        url: str,
        render_js: bool = False,
        premium_proxy: bool = False,
        stealth_proxy: bool = False,
        country_code: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        wait_for: Optional[str] = None,
        timeout: Optional[int] = None,
        js_snippet: Optional[str] = None,
        js_scenario: Optional[Dict[str, Any]] = None,
        extract_rules: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ScrapeResult:
        """
        Make request to ScrapingBee API.

        Args:
            url: Target URL
            render_js: Enable JavaScript rendering
            premium_proxy: Use premium residential proxies
            stealth_proxy: Use stealth proxies (best for anti-bot)
            country_code: Target country
            headers: Custom headers
            cookies: Custom cookies
            wait_for: CSS selector to wait for
            timeout: Request timeout
            js_snippet: JavaScript to execute
            js_scenario: JS scenario with instructions
            extract_rules: CSS extraction rules

        Returns:
            ScrapeResult with response
        """
        httpx = _get_httpx()

        if not self._sb_config.api_key:
            raise ValueError("ScrapingBee API key required.")

        # Build params
        params: Dict[str, Any] = {
            "api_key": self._sb_config.api_key,
            "url": url,
        }

        # JavaScript rendering
        if render_js or self._sb_config.render_js:
            params["render_js"] = "true"

        # Proxy options
        if premium_proxy or self._sb_config.premium_proxy:
            params["premium_proxy"] = "true"

        if stealth_proxy or self._sb_config.stealth_proxy:
            params["stealth_proxy"] = "true"

        # Geo-targeting
        target_country = country_code or self._sb_config.country_code
        if target_country:
            params["country_code"] = target_country

        # Wait options
        if wait_for or self._sb_config.wait_for:
            params["wait_for"] = wait_for or self._sb_config.wait_for

        if self._sb_config.wait > 0:
            params["wait"] = self._sb_config.wait

        params["wait_browser"] = self._sb_config.wait_browser

        # JavaScript execution
        target_js = js_snippet or self._sb_config.js_snippet
        if target_js:
            params["js_snippet"] = base64.b64encode(target_js.encode()).decode()

        target_scenario = js_scenario or self._sb_config.js_scenario
        if target_scenario:
            params["js_scenario"] = json.dumps(target_scenario)

        # Output options
        if self._sb_config.return_page_source:
            params["return_page_source"] = "true"

        if self._sb_config.json_response:
            params["json_response"] = "true"

        # Extract rules
        rules = extract_rules or self._sb_config.extract_rules
        if rules:
            params["extract_rules"] = json.dumps(rules)

        # Window size
        params["window_width"] = self._sb_config.window_width
        params["window_height"] = self._sb_config.window_height

        # Screenshot
        if self._sb_config.screenshot:
            params["screenshot"] = "true"
            if self._sb_config.screenshot_full_page:
                params["screenshot_full_page"] = "true"

        # AI query
        if self._sb_config.ai_query:
            params["ai_query"] = self._sb_config.ai_query

        # Custom headers
        if headers:
            for key, value in headers.items():
                params[f"forward_headers_{key}"] = value

        # Custom cookies
        if cookies:
            cookie_str = "; ".join(f"{k}={v}" for k, v in cookies.items())
            params["cookies"] = cookie_str

        try:
            response = await self._client.get(
                self.BASE_URL,
                params=params,
                timeout=timeout or self._sb_config.timeout,
            )

            # Calculate credits used
            credits = 1
            if render_js or self._sb_config.render_js:
                credits = 5
            if premium_proxy or self._sb_config.premium_proxy:
                credits = 10
            if stealth_proxy or self._sb_config.stealth_proxy:
                credits = 25

            content = response.text
            json_data = None
            screenshot_data = None

            # Check if screenshot response
            content_type = response.headers.get("content-type", "")
            if "image" in content_type:
                screenshot_data = response.content
                content = ""
            elif self._sb_config.json_response or rules:
                try:
                    json_data = response.json()
                    content = str(json_data)
                except Exception:
                    pass

            return ScrapeResult(
                url=url,
                content=content,
                status_code=response.status_code,
                html=content if not screenshot_data else None,
                json_data=json_data,
                headers=dict(response.headers),
                screenshot=screenshot_data,
                credits_used=credits,
                cost_usd=credits * 0.000327,  # $49/150k credits
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

    async def take_screenshot(
        self,
        url: str,
        full_page: bool = True,
        width: int = 1920,
        height: int = 1080,
        **kwargs: Any,
    ) -> bytes:
        """
        Take screenshot of a webpage.

        Args:
            url: URL to screenshot
            full_page: Capture full page
            width: Viewport width
            height: Viewport height
            **kwargs: Additional options

        Returns:
            Screenshot as PNG bytes
        """
        self._ensure_initialized()
        httpx = _get_httpx()

        params = {
            "api_key": self._sb_config.api_key,
            "url": url,
            "screenshot": "true",
            "render_js": "true",
            "window_width": width,
            "window_height": height,
        }

        if full_page:
            params["screenshot_full_page"] = "true"

        response = await self._client.get(
            self.BASE_URL,
            params=params,
        )

        response.raise_for_status()
        return response.content

    async def extract_data(
        self,
        url: str,
        rules: Dict[str, Any],
        render_js: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Extract structured data using CSS selectors.

        Args:
            url: Target URL
            rules: Extraction rules dict
            render_js: Enable JS rendering
            **kwargs: Additional options

        Returns:
            Extracted data
        """
        self._ensure_initialized()
        httpx = _get_httpx()

        params = {
            "api_key": self._sb_config.api_key,
            "url": url,
            "extract_rules": json.dumps(rules),
        }

        if render_js:
            params["render_js"] = "true"

        response = await self._client.get(
            self.BASE_URL,
            params=params,
        )

        response.raise_for_status()
        return response.json()

    async def ai_extract(
        self,
        url: str,
        query: str,
        render_js: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Extract data using AI with natural language query.

        Args:
            url: Target URL
            query: Natural language query (e.g., "Extract all product prices")
            render_js: Enable JS rendering
            **kwargs: Additional options

        Returns:
            AI-extracted data
        """
        self._ensure_initialized()
        httpx = _get_httpx()

        params = {
            "api_key": self._sb_config.api_key,
            "url": url,
            "ai_query": query,
        }

        if render_js:
            params["render_js"] = "true"

        response = await self._client.get(
            self.BASE_URL,
            params=params,
        )

        response.raise_for_status()
        return response.json()

    async def execute_scenario(
        self,
        url: str,
        scenario: Dict[str, Any],
        **kwargs: Any,
    ) -> ScrapeResult:
        """
        Execute JavaScript scenario on a page.

        Scenario instructions:
        - click: {"click": "#button"}
        - wait: {"wait": 1000}
        - wait_for_and_click: {"wait_for_and_click": "#element"}
        - fill: {"fill": ["#input", "text"]}
        - evaluate: {"evaluate": "javascript code"}
        - scroll_x: {"scroll_x": 100}
        - scroll_y: {"scroll_y": 500}

        Args:
            url: Target URL
            scenario: Scenario with instructions list
            **kwargs: Additional options

        Returns:
            ScrapeResult after scenario execution
        """
        return await self._make_request(
            url,
            render_js=True,
            js_scenario=scenario,
            **kwargs,
        )

    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account credit balance.

        Returns:
            Account info with credits remaining
        """
        self._ensure_initialized()

        response = await self._client.get(
            "https://app.scrapingbee.com/api/v1/usage",
            params={"api_key": self._sb_config.api_key},
        )

        response.raise_for_status()
        return response.json()
