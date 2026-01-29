"""
Apify Integration

Apify is a full-stack web scraping and automation platform with:
- 2000+ pre-built actors (scrapers)
- Custom actor development
- Cloud execution with autoscaling
- Built-in storage (key-value, datasets, request queues)

Features:
- Actors: Serverless scraping programs
- Schedules: Automated recurring tasks
- Webhooks: Real-time notifications
- API access to datasets

Pricing: Pay-per-use model starting at $49/mo

API Documentation: https://docs.apify.com/api/v2

Example:
    >>> from sentimatrix.providers.scrapers.commercial import ApifyClient
    >>>
    >>> async with ApifyClient(api_token="your_token") as client:
    ...     # Run a pre-built actor
    ...     result = await client.run_actor(
    ...         "apify/web-scraper",
    ...         input={"startUrls": [{"url": "https://example.com"}]}
    ...     )
    ...
    ...     # Get dataset items
    ...     items = await client.get_dataset_items(result["defaultDatasetId"])
"""

from __future__ import annotations

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
class ApifyConfig(CommercialAPIConfig):
    """Apify-specific configuration."""

    api_token: Optional[str] = None

    # Default actor settings
    default_actor_id: str = "apify/web-scraper"
    memory_mbytes: int = 1024
    timeout_secs: int = 300
    max_items: int = 1000

    # Storage options
    build: str = "latest"
    wait_for_finish: int = 120  # Wait up to 2 minutes for actor to finish


# Popular Apify Actors
POPULAR_ACTORS = {
    "web-scraper": "apify/web-scraper",
    "cheerio-scraper": "apify/cheerio-scraper",
    "playwright-scraper": "apify/playwright-scraper",
    "puppeteer-scraper": "apify/puppeteer-scraper",
    "google-search": "apify/google-search-scraper",
    "instagram": "apify/instagram-scraper",
    "twitter": "apify/twitter-scraper",
    "youtube": "apify/youtube-scraper",
    "amazon": "apify/amazon-scraper",
    "tripadvisor": "apify/tripadvisor-scraper",
    "yelp": "apify/yelp-scraper",
}


class ApifyClient(BaseCommercialClient):
    """
    Apify API client.

    Apify provides:
    - Pre-built actors for popular sites
    - Custom actor development in JS/Python
    - Cloud execution with autoscaling
    - Dataset storage and export

    Supports serverless functions (Actors) that can be:
    - Run synchronously or asynchronously
    - Scheduled for recurring execution
    - Triggered via webhooks
    """

    SERVICE_NAME = "apify"
    BASE_URL = "https://api.apify.com/v2"

    def __init__(self, config: Optional[ApifyConfig] = None, api_token: Optional[str] = None) -> None:
        """
        Initialize Apify client.

        Args:
            config: Apify configuration
            api_token: API token (alternative to config.api_token)
        """
        if config is None:
            config = ApifyConfig()

        if api_token:
            config.api_token = api_token

        super().__init__(config)
        self._apify_config: ApifyConfig = config

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="apify",
            provider_type=ProviderType.SCRAPER,
            version="1.0.0",
            description="Apify - Full-stack web scraping platform with 2000+ actors",
            capabilities=ProviderCapabilities(
                javascript_rendering=True,
                screenshots=True,
                pdf_generation=True,
                proxy_support=True,
                batch_processing=True,
            ),
        )

    def _get_headers(self) -> Dict[str, str]:
        """Get authorization headers."""
        return {
            "Authorization": f"Bearer {self._apify_config.api_token}",
            "Content-Type": "application/json",
        }

    async def _make_request(
        self,
        url: str,
        render_js: bool = False,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        wait_for: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> ScrapeResult:
        """
        Make scraping request using Apify web-scraper actor.

        Args:
            url: Target URL
            render_js: Use Playwright (True) or Cheerio (False)
            headers: Custom headers
            cookies: Custom cookies
            wait_for: CSS selector to wait for
            timeout: Request timeout

        Returns:
            ScrapeResult with scraped content
        """
        httpx = _get_httpx()

        if not self._apify_config.api_token:
            raise ValueError("Apify API token required. Set via config.api_token or api_token parameter.")

        # Choose actor based on render_js
        actor_id = "apify/playwright-scraper" if render_js else "apify/cheerio-scraper"

        # Build input
        actor_input = {
            "startUrls": [{"url": url}],
            "maxRequestsPerCrawl": 1,
            "maxConcurrency": 1,
        }

        if render_js and wait_for:
            actor_input["waitUntil"] = "domcontentloaded"
            actor_input["preNavigationHooks"] = f"""
            async (crawlingContext) => {{
                await crawlingContext.page.waitForSelector('{wait_for}');
            }}
            """

        if headers:
            actor_input["customHeaders"] = headers

        # Run actor synchronously
        try:
            run_result = await self.run_actor(
                actor_id,
                input=actor_input,
                timeout_secs=timeout or self._apify_config.timeout_secs,
                memory_mbytes=self._apify_config.memory_mbytes,
                wait_for_finish=self._apify_config.wait_for_finish,
            )

            # Get results from dataset
            if run_result.get("defaultDatasetId"):
                items = await self.get_dataset_items(
                    run_result["defaultDatasetId"],
                    limit=1,
                )

                if items:
                    item = items[0]
                    return ScrapeResult(
                        url=item.get("url", url),
                        content=item.get("text", ""),
                        status_code=200,
                        html=item.get("html", ""),
                        json_data=item,
                        provider=self.SERVICE_NAME,
                    )

            return ScrapeResult(
                url=url,
                content="",
                status_code=200,
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

    async def run_actor(
        self,
        actor_id: str,
        input: Optional[Dict[str, Any]] = None,
        timeout_secs: Optional[int] = None,
        memory_mbytes: Optional[int] = None,
        build: Optional[str] = None,
        wait_for_finish: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run an Apify actor.

        Args:
            actor_id: Actor ID or name (e.g., "apify/web-scraper")
            input: Actor input data
            timeout_secs: Actor timeout
            memory_mbytes: Memory allocation
            build: Actor build version
            wait_for_finish: Seconds to wait for completion

        Returns:
            Actor run result
        """
        self._ensure_initialized()

        # Resolve actor shorthand
        if actor_id in POPULAR_ACTORS:
            actor_id = POPULAR_ACTORS[actor_id]

        endpoint = f"{self.BASE_URL}/acts/{actor_id}/runs"

        params = {}
        if timeout_secs or self._apify_config.timeout_secs:
            params["timeout"] = timeout_secs or self._apify_config.timeout_secs
        if memory_mbytes or self._apify_config.memory_mbytes:
            params["memory"] = memory_mbytes or self._apify_config.memory_mbytes
        if build or self._apify_config.build:
            params["build"] = build or self._apify_config.build

        wait_secs = wait_for_finish or self._apify_config.wait_for_finish
        if wait_secs:
            params["waitForFinish"] = wait_secs

        response = await self._client.post(
            endpoint,
            params=params,
            json=input or {},
            headers=self._get_headers(),
            timeout=wait_secs + 30 if wait_secs else None,
        )

        response.raise_for_status()
        return response.json().get("data", {})

    async def get_run(self, run_id: str) -> Dict[str, Any]:
        """
        Get actor run details.

        Args:
            run_id: Run ID

        Returns:
            Run details
        """
        self._ensure_initialized()

        response = await self._client.get(
            f"{self.BASE_URL}/actor-runs/{run_id}",
            headers=self._get_headers(),
        )

        response.raise_for_status()
        return response.json().get("data", {})

    async def get_dataset_items(
        self,
        dataset_id: str,
        offset: int = 0,
        limit: int = 100,
        format: str = "json",
    ) -> List[Dict[str, Any]]:
        """
        Get items from a dataset.

        Args:
            dataset_id: Dataset ID
            offset: Starting offset
            limit: Maximum items to return
            format: Output format (json, csv, xlsx, etc.)

        Returns:
            List of dataset items
        """
        self._ensure_initialized()

        response = await self._client.get(
            f"{self.BASE_URL}/datasets/{dataset_id}/items",
            params={
                "offset": offset,
                "limit": limit,
                "format": format,
            },
            headers=self._get_headers(),
        )

        response.raise_for_status()

        if format == "json":
            return response.json()
        return [{"content": response.text}]

    async def get_key_value_store_record(
        self,
        store_id: str,
        key: str,
    ) -> Any:
        """
        Get a record from key-value store.

        Args:
            store_id: Store ID
            key: Record key

        Returns:
            Record value
        """
        self._ensure_initialized()

        response = await self._client.get(
            f"{self.BASE_URL}/key-value-stores/{store_id}/records/{key}",
            headers=self._get_headers(),
        )

        response.raise_for_status()
        return response.json()

    async def list_actors(self, my: bool = True) -> List[Dict[str, Any]]:
        """
        List available actors.

        Args:
            my: If True, list only user's actors

        Returns:
            List of actors
        """
        self._ensure_initialized()

        endpoint = f"{self.BASE_URL}/acts" if my else f"{self.BASE_URL}/store"

        response = await self._client.get(
            endpoint,
            headers=self._get_headers(),
        )

        response.raise_for_status()
        return response.json().get("data", {}).get("items", [])

    async def get_user_info(self) -> Dict[str, Any]:
        """
        Get current user information.

        Returns:
            User info including usage limits
        """
        self._ensure_initialized()

        response = await self._client.get(
            f"{self.BASE_URL}/users/me",
            headers=self._get_headers(),
        )

        response.raise_for_status()
        return response.json().get("data", {})

    async def scrape_with_actor(
        self,
        actor_name: str,
        urls: List[str],
        **actor_options: Any,
    ) -> List[Dict[str, Any]]:
        """
        Convenience method to scrape URLs with a specific actor.

        Args:
            actor_name: Actor shorthand or full ID
            urls: URLs to scrape
            **actor_options: Additional actor input options

        Returns:
            List of scraped items
        """
        actor_id = POPULAR_ACTORS.get(actor_name, actor_name)

        input_data = {
            "startUrls": [{"url": url} for url in urls],
            "maxRequestsPerCrawl": len(urls),
            **actor_options,
        }

        run = await self.run_actor(actor_id, input=input_data)

        if run.get("defaultDatasetId"):
            return await self.get_dataset_items(run["defaultDatasetId"])

        return []
