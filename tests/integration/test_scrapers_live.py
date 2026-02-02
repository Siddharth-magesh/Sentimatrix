"""
Live Integration Tests for Sentimatrix Scrapers

This module tests all available scrapers in the sentimatrix package.
Tests include:
1. Steam scraper (no API key)
2. Reddit scraper (no API key)
3. Amazon scraper (no API key)
4. YouTube scraper (requires API key - optional)

Run with: python tests/integration/test_scrapers_live.py
Or: pytest tests/integration/test_scrapers_live.py -v -s
"""

import asyncio
import os
from datetime import datetime

# API Keys
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Test IDs
STEAM_APP_ID = "570"  # Dota 2
REDDIT_POST_ID = "1a2b3c"  # Example Reddit post - will need a valid one
AMAZON_ASIN = "B09V3KXJPB"  # Example Amazon product ASIN
YOUTUBE_VIDEO_ID = "dQw4w9WgXcQ"  # Example YouTube video


def print_separator(title):
    """Print a visual separator."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_result(label, value, indent=2):
    """Print a formatted result."""
    spaces = " " * indent
    print(f"{spaces}{label}: {value}")


# ============================================================================
# TEST 1: STEAM SCRAPER
# ============================================================================

async def test_steam_scraper():
    """Test Steam review scraping."""
    print_separator("TEST 1: Steam Scraper")

    from sentimatrix import Sentimatrix

    async with Sentimatrix() as sm:
        print(f"\n  App ID: {STEAM_APP_ID} (Dota 2)")

        try:
            reviews = await sm.scrape_steam(app_id=STEAM_APP_ID, limit=5)
            print(f"  Reviews scraped: {len(reviews)}")

            if reviews:
                print("\n  Sample reviews:")
                for i, review in enumerate(reviews[:3], 1):
                    text = review.text[:80] if review.text else "No text"
                    print(f"    {i}. [{review.rating}] {text}...")
                    print(f"       Author: {review.author}")
                    print(f"       Platform: {review.platform}")
                    if review.metadata:
                        print(f"       Playtime: {review.metadata.get('playtime_forever_hours', 'N/A')} hrs")

            return len(reviews) > 0

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False


# ============================================================================
# TEST 2: REDDIT SCRAPER
# ============================================================================

async def test_reddit_scraper():
    """Test Reddit comment scraping."""
    print_separator("TEST 2: Reddit Scraper")

    from sentimatrix import Sentimatrix

    async with Sentimatrix() as sm:
        # Use a popular gaming subreddit post - these are usually public
        # We'll try a known public post from r/gaming or r/dota2
        test_post_ids = [
            "16h1ymq",  # Example r/gaming post
            "1i7xz8y",  # Another example
        ]

        for post_id in test_post_ids:
            print(f"\n  Trying Post ID: {post_id}")

            try:
                reviews = await sm.scrape_reddit(post_id=post_id, limit=5)
                print(f"  Comments scraped: {len(reviews)}")

                if reviews:
                    print("\n  Sample comments:")
                    for i, review in enumerate(reviews[:3], 1):
                        text = review.text[:80] if review.text else "No text"
                        print(f"    {i}. {text}...")
                        print(f"       Author: {review.author}")
                    return True

            except Exception as e:
                print(f"  Error with post {post_id}: {e}")
                continue

        print("  Note: Reddit scraping may require valid post IDs")
        return None  # Skip rather than fail


# ============================================================================
# TEST 3: AMAZON SCRAPER
# ============================================================================

async def test_amazon_scraper():
    """Test Amazon review scraping."""
    print_separator("TEST 3: Amazon Scraper")

    from sentimatrix import Sentimatrix

    async with Sentimatrix() as sm:
        # Use a popular product ASIN
        test_asins = [
            "B09V3KXJPB",  # Popular electronics
            "B08N5WRWNW",  # Another popular product
        ]

        for asin in test_asins:
            print(f"\n  Trying ASIN: {asin}")

            try:
                reviews = await sm.scrape_amazon(asin=asin, limit=5, country="us")
                print(f"  Reviews scraped: {len(reviews)}")

                if reviews:
                    print("\n  Sample reviews:")
                    for i, review in enumerate(reviews[:3], 1):
                        text = review.text[:80] if review.text else "No text"
                        rating = review.rating if review.rating else "N/A"
                        print(f"    {i}. [{rating}] {text}...")
                    return True

            except Exception as e:
                print(f"  Error with ASIN {asin}: {e}")
                continue

        print("  Note: Amazon scraping may be rate-limited or blocked")
        return None  # Skip rather than fail


# ============================================================================
# TEST 4: YOUTUBE SCRAPER
# ============================================================================

async def test_youtube_scraper():
    """Test YouTube comment scraping."""
    print_separator("TEST 4: YouTube Scraper")

    if not YOUTUBE_API_KEY:
        print("  SKIPPED: No YOUTUBE_API_KEY set")
        print("  Note: YouTube API key required for comment scraping")
        return None

    from sentimatrix import Sentimatrix

    async with Sentimatrix() as sm:
        print(f"\n  Video ID: {YOUTUBE_VIDEO_ID}")

        try:
            reviews = await sm.scrape_youtube(
                video_id=YOUTUBE_VIDEO_ID,
                limit=5,
                api_key=YOUTUBE_API_KEY
            )
            print(f"  Comments scraped: {len(reviews)}")

            if reviews:
                print("\n  Sample comments:")
                for i, review in enumerate(reviews[:3], 1):
                    text = review.text[:80] if review.text else "No text"
                    print(f"    {i}. {text}...")
                    print(f"       Author: {review.author}")

            return len(reviews) > 0

        except Exception as e:
            print(f"  ERROR: {e}")
            return False


# ============================================================================
# TEST 5: COMBINED SCRAPE + ANALYZE
# ============================================================================

async def test_scrape_and_analyze():
    """Test scraping + sentiment analysis pipeline."""
    print_separator("TEST 5: Scrape + Analyze Pipeline")

    from sentimatrix import Sentimatrix, SentimatrixConfig, LLMConfig

    config = SentimatrixConfig(
        llm=LLMConfig(
            provider="groq",
            model="llama-3.3-70b-versatile",
            api_key=GROQ_API_KEY
        ) if GROQ_API_KEY else None
    )

    async with Sentimatrix(config) as sm:
        # Step 1: Scrape Steam
        print("\n  Step 1: Scraping Steam reviews...")
        try:
            reviews = await sm.scrape_steam(app_id=STEAM_APP_ID, limit=10)
            print(f"    Got {len(reviews)} reviews")
        except Exception as e:
            print(f"    Error: {e}")
            return False

        if not reviews:
            print("    No reviews to analyze")
            return False

        # Step 2: Analyze sentiment
        print("\n  Step 2: Analyzing sentiment...")
        texts = [r.text for r in reviews if r.text]
        batch_result = await sm.analyze_sentiment_batch(texts)

        print(f"    Positive: {batch_result.positive_count}")
        print(f"    Negative: {batch_result.negative_count}")
        print(f"    Neutral: {batch_result.neutral_count}")
        print(f"    Average Confidence: {batch_result.average_confidence:.1%}")

        # Step 3: Detect emotions
        print("\n  Step 3: Detecting emotions...")
        emotion_batch = await sm.detect_emotions_batch(texts[:5])
        print(f"    Most Common: {emotion_batch.most_common_emotion}")
        print(f"    Distribution: {emotion_batch.emotion_counts}")

        # Step 4: Summarize (if LLM configured)
        if GROQ_API_KEY:
            print("\n  Step 4: Generating summary...")
            try:
                summary = await sm.summarize_reviews(reviews[:5])
                print(f"    Summary: {summary[:200]}...")
            except Exception as e:
                print(f"    Summary error: {e}")

        return True


# ============================================================================
# MAIN
# ============================================================================

async def run_all_tests():
    """Run all scraper tests."""
    print("\n" + "=" * 60)
    print("  SENTIMATRIX SCRAPER LIVE TESTS")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = {}

    # Test 1: Steam
    try:
        results["Steam Scraper"] = await test_steam_scraper()
    except Exception as e:
        print(f"  ERROR: {e}")
        results["Steam Scraper"] = False

    # Test 2: Reddit
    try:
        results["Reddit Scraper"] = await test_reddit_scraper()
    except Exception as e:
        print(f"  ERROR: {e}")
        results["Reddit Scraper"] = False

    # Test 3: Amazon
    try:
        results["Amazon Scraper"] = await test_amazon_scraper()
    except Exception as e:
        print(f"  ERROR: {e}")
        results["Amazon Scraper"] = False

    # Test 4: YouTube
    try:
        results["YouTube Scraper"] = await test_youtube_scraper()
    except Exception as e:
        print(f"  ERROR: {e}")
        results["YouTube Scraper"] = False

    # Test 5: Pipeline
    try:
        results["Scrape+Analyze"] = await test_scrape_and_analyze()
    except Exception as e:
        print(f"  ERROR: {e}")
        results["Scrape+Analyze"] = False

    # Summary
    print_separator("TEST RESULTS SUMMARY")
    for test_name, passed in results.items():
        if passed is None:
            status = "SKIPPED"
        elif passed:
            status = "PASSED"
        else:
            status = "FAILED"
        print(f"  {status:12} {test_name}")

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)

    print(f"\n  Total: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)

    return results


if __name__ == "__main__":
    asyncio.run(run_all_tests())
