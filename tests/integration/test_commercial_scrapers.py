"""
Commercial Scraper API Tests for Sentimatrix

Tests the commercial web scraping services:
1. ScraperAPI - Simple API with JS rendering and CAPTCHA handling
2. Apify - Actor-based scraping with 2000+ pre-built scrapers
3. Bright Data - Enterprise-grade with 72M+ residential proxies
4. Oxylabs - Web scraper API with e-commerce specialization
5. Zyte - All-in-one API with automatic extraction
6. ScrapingBee - Simple API with headless browser support
7. ScrapingAnt - Budget-friendly scraping with JS rendering

Set API keys via environment variables:
- SCRAPERAPI_KEY
- APIFY_TOKEN
- BRIGHTDATA_USER / BRIGHTDATA_PASSWORD
- OXYLABS_USER / OXYLABS_PASSWORD
- ZYTE_API_KEY
- SCRAPINGBEE_API_KEY
- SCRAPINGANT_API_KEY

Run with: python tests/integration/test_commercial_scrapers.py
"""

import asyncio
import os
from datetime import datetime
from typing import Optional, Dict, Any

# API Keys from environment
SCRAPERAPI_KEY = os.getenv("SCRAPERAPI_KEY", "")
APIFY_TOKEN = os.getenv("APIFY_TOKEN", "")
BRIGHTDATA_USER = os.getenv("BRIGHTDATA_USER", "")
BRIGHTDATA_PASSWORD = os.getenv("BRIGHTDATA_PASSWORD", "")
OXYLABS_USER = os.getenv("OXYLABS_USER", "")
OXYLABS_PASSWORD = os.getenv("OXYLABS_PASSWORD", "")
ZYTE_API_KEY = os.getenv("ZYTE_API_KEY", "")
SCRAPINGBEE_API_KEY = os.getenv("SCRAPINGBEE_API_KEY", "")
SCRAPINGANT_API_KEY = os.getenv("SCRAPINGANT_API_KEY", "")

# Test URL
TEST_URL = "https://httpbin.org/get"


def print_separator(title: str) -> None:
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")


def print_result(result: Any, service_name: str) -> None:
    """Print scrape result."""
    print(f"\n  {service_name} Result:")
    print(f"    Status Code: {result.status_code}")
    print(f"    Content Length: {len(result.content) if result.content else 0} chars")
    print(f"    Response Time: {result.response_time_ms:.0f}ms" if hasattr(result, 'response_time_ms') else "")
    if hasattr(result, 'provider'):
        print(f"    Provider: {result.provider}")
    # Check for credits in headers (ScraperAPI puts it there)
    if hasattr(result, 'headers') and result.headers:
        if 'sa-credit-cost' in result.headers:
            print(f"    Credits Used: {result.headers['sa-credit-cost']}")


async def test_scraperapi():
    """Test ScraperAPI integration."""
    print_separator("ScraperAPI Test")

    if not SCRAPERAPI_KEY:
        print("  SKIPPED: No SCRAPERAPI_KEY environment variable")
        print("  Get free API key at: https://www.scraperapi.com/")
        return None

    from sentimatrix.providers.scrapers.commercial import ScraperAPIClient

    try:
        async with ScraperAPIClient(api_key=SCRAPERAPI_KEY) as client:
            # Test basic scraping
            print("\n  Testing basic scrape...")
            result = await client.scrape(TEST_URL)
            print_result(result, "ScraperAPI")

            if result.status_code == 200:
                print("    SUCCESS: Basic scraping works!")

                # Show a snippet of content
                content_preview = result.content[:200] if result.content else ""
                print(f"    Content preview: {content_preview}...")
                return True
            else:
                print(f"    FAILED: Status {result.status_code}")
                return False

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_apify():
    """Test Apify integration."""
    print_separator("Apify Test")

    if not APIFY_TOKEN:
        print("  SKIPPED: No APIFY_TOKEN environment variable")
        print("  Get free token at: https://apify.com/")
        return None

    from sentimatrix.providers.scrapers.commercial import ApifyClient, POPULAR_ACTORS

    try:
        async with ApifyClient(api_token=APIFY_TOKEN) as client:
            print("\n  Available actors:")
            for name, actor_id in list(POPULAR_ACTORS.items())[:5]:
                print(f"    - {name}: {actor_id}")

            # Test basic scrape (uses cheerio-scraper internally)
            print("\n  Testing basic scrape...")
            result = await client.scrape(TEST_URL)
            print_result(result, "Apify")

            if result.status_code == 200:
                print("    SUCCESS: Basic scraping works!")

                # Get user info
                print("\n  Getting user info...")
                try:
                    user_info = await client.get_user_info()
                    print(f"    Username: {user_info.get('username', 'N/A')}")
                except Exception as e:
                    print(f"    User info error: {e}")

                return True
            else:
                print(f"    FAILED: Status {result.status_code}")
                return False

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_brightdata():
    """Test Bright Data integration."""
    print_separator("Bright Data Test")

    if not BRIGHTDATA_USER or not BRIGHTDATA_PASSWORD:
        print("  SKIPPED: No BRIGHTDATA_USER/BRIGHTDATA_PASSWORD environment variables")
        print("  Get account at: https://brightdata.com/")
        return None

    from sentimatrix.providers.scrapers.commercial import BrightDataClient, BrightDataConfig

    try:
        config = BrightDataConfig(
            username=BRIGHTDATA_USER,
            password=BRIGHTDATA_PASSWORD,
        )

        async with BrightDataClient(config=config) as client:
            print("\n  Testing Bright Data scrape...")
            result = await client.scrape(TEST_URL)
            print_result(result, "Bright Data")

            if result.status_code == 200:
                print("    SUCCESS!")
                return True
            return False

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


async def test_oxylabs():
    """Test Oxylabs integration."""
    print_separator("Oxylabs Test")

    if not OXYLABS_USER or not OXYLABS_PASSWORD:
        print("  SKIPPED: No OXYLABS_USER/OXYLABS_PASSWORD environment variables")
        print("  Get account at: https://oxylabs.io/")
        return None

    from sentimatrix.providers.scrapers.commercial import OxylabsClient, OxylabsConfig

    try:
        config = OxylabsConfig(
            username=OXYLABS_USER,
            password=OXYLABS_PASSWORD,
        )

        async with OxylabsClient(config=config) as client:
            print("\n  Testing Oxylabs scrape...")
            result = await client.scrape(TEST_URL)
            print_result(result, "Oxylabs")

            if result.status_code == 200:
                print("    SUCCESS!")
                return True
            return False

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


async def test_zyte():
    """Test Zyte integration."""
    print_separator("Zyte Test")

    if not ZYTE_API_KEY:
        print("  SKIPPED: No ZYTE_API_KEY environment variable")
        print("  Get API key at: https://www.zyte.com/")
        return None

    from sentimatrix.providers.scrapers.commercial import ZyteClient

    try:
        async with ZyteClient(api_key=ZYTE_API_KEY) as client:
            print("\n  Testing Zyte scrape...")
            result = await client.scrape(TEST_URL)
            print_result(result, "Zyte")

            if result.status_code == 200:
                print("    SUCCESS!")
                return True
            return False

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


async def test_scrapingbee():
    """Test ScrapingBee integration."""
    print_separator("ScrapingBee Test")

    if not SCRAPINGBEE_API_KEY:
        print("  SKIPPED: No SCRAPINGBEE_API_KEY environment variable")
        print("  Get API key at: https://www.scrapingbee.com/")
        return None

    from sentimatrix.providers.scrapers.commercial import ScrapingBeeClient

    try:
        async with ScrapingBeeClient(api_key=SCRAPINGBEE_API_KEY) as client:
            print("\n  Testing ScrapingBee scrape...")
            result = await client.scrape(TEST_URL)
            print_result(result, "ScrapingBee")

            if result.status_code == 200:
                print("    SUCCESS: Basic scraping works!")

                # Show a snippet of content
                content_preview = result.content[:200] if result.content else ""
                print(f"    Content preview: {content_preview}...")
                return True
            else:
                print(f"    FAILED: Status {result.status_code}")
                return False

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_scrapingant():
    """Test ScrapingAnt integration."""
    print_separator("ScrapingAnt Test")

    if not SCRAPINGANT_API_KEY:
        print("  SKIPPED: No SCRAPINGANT_API_KEY environment variable")
        print("  Get API key at: https://scrapingant.com/")
        return None

    from sentimatrix.providers.scrapers.commercial import ScrapingAntClient

    try:
        async with ScrapingAntClient(api_key=SCRAPINGANT_API_KEY) as client:
            print("\n  Testing ScrapingAnt scrape...")
            result = await client.scrape(TEST_URL)
            print_result(result, "ScrapingAnt")

            if result.status_code == 200:
                print("    SUCCESS!")
                return True
            return False

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


async def main():
    """Run all commercial scraper tests."""
    print("\n" + "=" * 70)
    print("  SENTIMATRIX - COMMERCIAL SCRAPER API TESTS")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print("\n  Environment Variables Detected:")
    print(f"    SCRAPERAPI_KEY: {'Set' if SCRAPERAPI_KEY else 'Not set'}")
    print(f"    APIFY_TOKEN: {'Set' if APIFY_TOKEN else 'Not set'}")
    print(f"    BRIGHTDATA_USER: {'Set' if BRIGHTDATA_USER else 'Not set'}")
    print(f"    OXYLABS_USER: {'Set' if OXYLABS_USER else 'Not set'}")
    print(f"    ZYTE_API_KEY: {'Set' if ZYTE_API_KEY else 'Not set'}")
    print(f"    SCRAPINGBEE_API_KEY: {'Set' if SCRAPINGBEE_API_KEY else 'Not set'}")
    print(f"    SCRAPINGANT_API_KEY: {'Set' if SCRAPINGANT_API_KEY else 'Not set'}")

    results = {}

    # Test each service
    results["ScraperAPI"] = await test_scraperapi()
    results["Apify"] = await test_apify()
    results["Bright Data"] = await test_brightdata()
    results["Oxylabs"] = await test_oxylabs()
    results["Zyte"] = await test_zyte()
    results["ScrapingBee"] = await test_scrapingbee()
    results["ScrapingAnt"] = await test_scrapingant()

    # Summary
    print_separator("FINAL RESULTS SUMMARY")

    total_passed = 0
    total_failed = 0
    total_skipped = 0

    for service_name, passed in results.items():
        if passed is None:
            status = "SKIPPED"
            total_skipped += 1
        elif passed:
            status = "PASSED"
            total_passed += 1
        else:
            status = "FAILED"
            total_failed += 1
        print(f"  {status:8} - {service_name}")

    print(f"\n  {'='*50}")
    print(f"  TOTAL: {total_passed} passed, {total_failed} failed, {total_skipped} skipped")
    print(f"  {'='*50}")

    return total_failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
