"""
Comprehensive Scraper Tests for Sentimatrix PyPI Package

Tests ALL available scrapers:
1. Steam - No API key needed
2. Reddit - No API key needed
3. Amazon - Needs Playwright (browser automation)
4. YouTube - Needs YouTube Data API key

Run with: python tests/integration/test_all_scrapers.py
"""

import asyncio
import os
from datetime import datetime

# API Keys
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

def print_separator(title):
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")

def print_review(review, index):
    """Pretty print a review object."""
    print(f"\n  Review #{index}:")
    print(f"    ID: {review.id}")
    print(f"    Text: {review.text[:100]}..." if review.text and len(review.text) > 100 else f"    Text: {review.text}")
    print(f"    Rating: {review.rating}")
    print(f"    Author: {review.author}")
    print(f"    Platform: {review.platform}")
    print(f"    Source: {review.source}")
    if review.timestamp:
        print(f"    Timestamp: {review.timestamp}")
    if review.metadata:
        print(f"    Metadata: {dict(list(review.metadata.items())[:3])}...")


async def test_steam_scraper():
    """Test Steam scraper with multiple games."""
    print_separator("STEAM SCRAPER TEST")

    from sentimatrix import Sentimatrix

    test_cases = [
        ("570", "Dota 2"),
        ("730", "Counter-Strike 2"),
        ("1172470", "Apex Legends"),
    ]

    results = {}

    async with Sentimatrix() as sm:
        for app_id, game_name in test_cases:
            print(f"\n  Testing: {game_name} (App ID: {app_id})")
            try:
                reviews = await sm.scrape_steam(app_id=app_id, limit=3)
                print(f"    SUCCESS: Got {len(reviews)} reviews")
                if reviews:
                    print_review(reviews[0], 1)
                results[game_name] = True
            except Exception as e:
                print(f"    FAILED: {e}")
                results[game_name] = False

    return results


async def test_reddit_scraper():
    """Test Reddit scraper with various posts."""
    print_separator("REDDIT SCRAPER TEST")

    from sentimatrix import Sentimatrix

    # Various Reddit post IDs to test
    # Format: (post_id, description)
    test_cases = [
        ("16h1ymq", "r/gaming post"),
        ("1i7xz8y", "Random post"),
        ("t3_16h1ymq", "Full post ID format"),
    ]

    results = {}

    async with Sentimatrix() as sm:
        for post_id, desc in test_cases:
            print(f"\n  Testing: {desc} (Post ID: {post_id})")
            try:
                reviews = await sm.scrape_reddit(post_id=post_id, limit=3)
                print(f"    SUCCESS: Got {len(reviews)} comments")
                if reviews:
                    print_review(reviews[0], 1)
                results[desc] = True
            except Exception as e:
                print(f"    FAILED: {e}")
                results[desc] = False

    return results


async def test_amazon_scraper():
    """Test Amazon scraper with various products."""
    print_separator("AMAZON SCRAPER TEST")

    from sentimatrix import Sentimatrix

    # Various Amazon ASINs to test
    test_cases = [
        ("B09V3KXJPB", "Echo Dot", "us"),
        ("B08N5WRWNW", "Fire TV Stick", "us"),
        ("B0BDHX8Z4X", "Kindle Paperwhite", "us"),
    ]

    results = {}

    async with Sentimatrix() as sm:
        for asin, product_name, country in test_cases:
            print(f"\n  Testing: {product_name} (ASIN: {asin}, Country: {country})")
            try:
                reviews = await sm.scrape_amazon(asin=asin, limit=3, country=country)
                print(f"    SUCCESS: Got {len(reviews)} reviews")
                if reviews:
                    print_review(reviews[0], 1)
                results[product_name] = True
            except Exception as e:
                error_msg = str(e)
                if "Playwright" in error_msg or "browser" in error_msg.lower():
                    print(f"    SKIPPED: Playwright not installed")
                    print(f"    To fix: sudo playwright install-deps && playwright install chromium")
                    results[product_name] = None
                else:
                    print(f"    FAILED: {e}")
                    results[product_name] = False

    return results


async def test_youtube_scraper():
    """Test YouTube scraper with various videos."""
    print_separator("YOUTUBE SCRAPER TEST")

    if not YOUTUBE_API_KEY:
        print("\n  SKIPPED: No YOUTUBE_API_KEY environment variable set")
        print("  To test: export YOUTUBE_API_KEY='your-api-key'")
        print("  Get key at: https://console.cloud.google.com/apis/credentials")
        return {"YouTube": None}

    from sentimatrix import Sentimatrix

    # Various YouTube video IDs to test
    test_cases = [
        ("dQw4w9WgXcQ", "Rick Astley - Never Gonna Give You Up"),
        ("jNQXAC9IVRw", "Me at the zoo (first YouTube video)"),
        ("kJQP7kiw5Fk", "Luis Fonsi - Despacito"),
    ]

    results = {}

    async with Sentimatrix() as sm:
        for video_id, video_name in test_cases:
            print(f"\n  Testing: {video_name} (Video ID: {video_id})")
            try:
                reviews = await sm.scrape_youtube(
                    video_id=video_id,
                    limit=3,
                    api_key=YOUTUBE_API_KEY
                )
                print(f"    SUCCESS: Got {len(reviews)} comments")
                if reviews:
                    print_review(reviews[0], 1)
                results[video_name] = True
            except Exception as e:
                print(f"    FAILED: {e}")
                results[video_name] = False

    return results


async def main():
    """Run all scraper tests."""
    print("\n" + "=" * 70)
    print("  SENTIMATRIX - COMPREHENSIVE SCRAPER TESTS")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_results = {}

    # Test each scraper
    print("\n" + "-" * 70)
    print(" Running Steam Scraper Tests...")
    print("-" * 70)
    all_results["Steam"] = await test_steam_scraper()

    print("\n" + "-" * 70)
    print(" Running Reddit Scraper Tests...")
    print("-" * 70)
    all_results["Reddit"] = await test_reddit_scraper()

    print("\n" + "-" * 70)
    print(" Running Amazon Scraper Tests...")
    print("-" * 70)
    all_results["Amazon"] = await test_amazon_scraper()

    print("\n" + "-" * 70)
    print(" Running YouTube Scraper Tests...")
    print("-" * 70)
    all_results["YouTube"] = await test_youtube_scraper()

    # Summary
    print_separator("FINAL RESULTS SUMMARY")

    total_passed = 0
    total_failed = 0
    total_skipped = 0

    for scraper_name, results in all_results.items():
        print(f"\n  {scraper_name} Scraper:")
        for test_name, passed in results.items():
            if passed is None:
                status = "SKIPPED"
                total_skipped += 1
            elif passed:
                status = "PASSED"
                total_passed += 1
            else:
                status = "FAILED"
                total_failed += 1
            print(f"    {status:8} - {test_name}")

    print(f"\n  " + "=" * 50)
    print(f"  TOTAL: {total_passed} passed, {total_failed} failed, {total_skipped} skipped")
    print(f"  " + "=" * 50)

    # Return overall success
    return total_failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
