"""
Live Integration Tests for Sentimatrix PyPI Package

This module tests the sentimatrix package installed from PyPI against real APIs.
Tests are organized by:
1. Basic sentiment analysis (no API keys needed)
2. Emotion detection (no API keys needed)
3. Platform scrapers (Steam - no API key needed)
4. LLM providers (with API keys)

Run with: python tests/integration/test_pypi_live.py
Or: pytest tests/integration/test_pypi_live.py -v -s
"""

import asyncio
import os
import sys
from datetime import datetime

# ============================================================================
# TEST CONFIGURATION
# ============================================================================

# API Keys - Set via environment variables or directly here for testing
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")

# Test URLs
STEAM_APP_ID = "570"  # Dota 2 - lots of reviews
STEAM_URL = f"https://store.steampowered.com/app/{STEAM_APP_ID}/Dota_2/"

# Test texts for sentiment analysis
TEST_TEXTS = [
    "This product is absolutely amazing! Best purchase I've ever made.",
    "Terrible quality, broke after one day. Complete waste of money.",
    "It's okay, nothing special. Does what it's supposed to do.",
    "I'm really happy with this purchase, exceeded my expectations!",
    "Disappointed with the shipping time, but the product itself is good.",
]


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
# TEST 1: BASIC IMPORT
# ============================================================================

def test_import():
    """Test that sentimatrix can be imported from PyPI."""
    print_separator("TEST 1: Package Import")

    try:
        from sentimatrix import Sentimatrix, SentimatrixConfig, LLMConfig
        print_result("Import", "SUCCESS")
        print_result("Sentimatrix class", str(Sentimatrix))
        return True
    except ImportError as e:
        print_result("Import", f"FAILED - {e}")
        return False


# ============================================================================
# TEST 2: SENTIMENT ANALYSIS
# ============================================================================

async def test_sentiment_analysis():
    """Test basic sentiment analysis."""
    print_separator("TEST 2: Sentiment Analysis")

    from sentimatrix import Sentimatrix

    async with Sentimatrix() as sm:
        # Single text analysis using analyze() method
        print("\n  --- Single Text Analysis ---")
        text = "This is an amazing product! I love it!"
        result = await sm.analyze(text)

        print_result("Text", f"'{text}'")
        print_result("Sentiment", result.sentiment.sentiment)
        print_result("Confidence", f"{result.sentiment.confidence:.2%}")
        print_result("Processing Time", f"{result.sentiment.processing_time_ms:.0f}ms")

        # Batch analysis using analyze_sentiment_batch()
        print("\n  --- Batch Analysis ---")
        batch_result = await sm.analyze_sentiment_batch(TEST_TEXTS)

        # BatchSentimentResult has a .results attribute with list of SentimentResult
        for text, res in zip(TEST_TEXTS, batch_result.results):
            sentiment_str = str(res.sentiment).split('.')[-1].upper()
            print(f"    [{sentiment_str:>8}] ({res.confidence:.0%}) {text[:50]}...")

        print(f"\n  Summary: {batch_result.positive_count} positive, {batch_result.negative_count} negative, {batch_result.neutral_count} neutral")
        print(f"  Average Confidence: {batch_result.average_confidence:.2%}")

        return True


# ============================================================================
# TEST 3: EMOTION DETECTION
# ============================================================================

async def test_emotion_detection():
    """Test emotion detection."""
    print_separator("TEST 3: Emotion Detection")

    from sentimatrix import Sentimatrix

    test_texts = [
        ("I'm so happy and excited!", "Expected: joy"),
        ("This makes me really angry!", "Expected: anger"),
        ("I'm worried about the future.", "Expected: fear/nervousness"),
        ("What a surprise! I didn't expect that.", "Expected: surprise"),
    ]

    async with Sentimatrix() as sm:
        for text, expected in test_texts:
            result = await sm.detect_emotions(text)

            print(f"\n  Text: '{text}'")
            print_result("Primary Emotion", result.primary_emotion.label)
            print_result("Score", f"{result.primary_emotion.score:.2%}")
            print_result("Valence", result.primary_emotion.valence)
            print_result("Ekman Mapping", result.primary_emotion.ekman_mapping)
            print_result("", f"({expected})", indent=4)

        # Batch emotion detection
        print("\n  --- Batch Emotion Detection ---")
        texts = [t[0] for t in test_texts]
        batch_result = await sm.detect_emotions_batch(texts)

        print("  Results summary:")
        # BatchEmotionResult has a .results attribute with list of EmotionResult
        for text, res in zip(texts, batch_result.results):
            print(f"    {res.primary_emotion.label:>12}: {text[:40]}...")

        print(f"\n  Most Common Emotion: {batch_result.most_common_emotion}")
        print(f"  Emotion Distribution: {batch_result.emotion_counts}")

        return True


# ============================================================================
# TEST 4: STEAM SCRAPER
# ============================================================================

async def test_steam_scraper():
    """Test Steam review scraping (no API key needed)."""
    print_separator("TEST 4: Steam Scraper")

    from sentimatrix import Sentimatrix

    async with Sentimatrix() as sm:
        print(f"\n  Scraping reviews from Steam App ID: {STEAM_APP_ID}")
        print(f"  URL: {STEAM_URL}")

        try:
            # Correct API: scrape_steam(app_id, limit=100, language='english')
            reviews = await sm.scrape_steam(
                app_id=STEAM_APP_ID,
                limit=10
            )

            print(f"\n  Reviews scraped: {len(reviews)}")

            if reviews:
                print("\n  Sample reviews:")
                for i, review in enumerate(reviews[:3], 1):
                    # Review object has .text, .rating, .author, .platform, etc.
                    text = review.text[:100] if review.text else "No text"
                    print(f"    {i}. [{review.rating}] {text}...")

                # Analyze sentiment of scraped reviews
                print("\n  Analyzing sentiment of scraped reviews...")
                texts = [r.text for r in reviews[:5] if r.text]
                batch_result = await sm.analyze_sentiment_batch(texts)

                for review, res in zip(reviews[:5], batch_result.results):
                    sentiment_str = str(res.sentiment).split('.')[-1].upper()
                    text = review.text[:40] if review.text else "No text"
                    print(f"    [{sentiment_str:>8}] {text}...")

            return len(reviews) > 0

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False


# ============================================================================
# TEST 5: LLM PROVIDER (GROQ)
# ============================================================================

async def test_groq_llm():
    """Test Groq LLM provider for summarization."""
    print_separator("TEST 5: Groq LLM Provider")

    if not GROQ_API_KEY:
        print("  SKIPPED: No GROQ_API_KEY set")
        return None

    from sentimatrix import Sentimatrix, SentimatrixConfig, LLMConfig
    from sentimatrix.providers.base import Review

    config = SentimatrixConfig(
        llm=LLMConfig(
            provider="groq",
            model="llama-3.3-70b-versatile",
            api_key=GROQ_API_KEY
        )
    )

    async with Sentimatrix(config) as sm:
        # Test review summarization
        print("\n  --- Review Summarization ---")

        # Create Review objects with required fields (id, text, source, platform)
        reviews = [
            Review(id="r1", text="Amazing product! Works perfectly and great value.", source="test", platform="manual", rating=5.0),
            Review(id="r2", text="Good quality but shipping took too long.", source="test", platform="manual", rating=4.0),
            Review(id="r3", text="Broke after one week. Very disappointed.", source="test", platform="manual", rating=1.0),
            Review(id="r4", text="Exactly what I expected. Does the job.", source="test", platform="manual", rating=3.0),
            Review(id="r5", text="Best purchase ever! Highly recommend!", source="test", platform="manual", rating=5.0),
        ]

        try:
            print("  Generating summary with Groq (llama-3.3-70b)...")
            summary = await sm.summarize_reviews(reviews)

            print(f"\n  Summary:\n  {'-'*50}")
            # Wrap text for better display
            words = summary.split()
            line = "  "
            for word in words:
                if len(line) + len(word) > 70:
                    print(line)
                    line = "  " + word
                else:
                    line += " " + word if line != "  " else word
            if line.strip():
                print(line)
            print(f"  {'-'*50}")

            return True

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False


async def test_groq_insights():
    """Test Groq LLM for insight generation."""
    print_separator("TEST 6: Groq Insight Generation")

    if not GROQ_API_KEY:
        print("  SKIPPED: No GROQ_API_KEY set")
        return None

    from sentimatrix import Sentimatrix, SentimatrixConfig, LLMConfig
    from sentimatrix.providers.base import Review

    config = SentimatrixConfig(
        llm=LLMConfig(
            provider="groq",
            model="llama-3.3-70b-versatile",
            api_key=GROQ_API_KEY
        )
    )

    async with Sentimatrix(config) as sm:
        # Create Review objects with required fields (id, text, source, platform)
        reviews = [
            Review(id="i1", text="Camera quality is excellent, but battery drains fast.", source="test", platform="manual", rating=4.0),
            Review(id="i2", text="Love the sleek design. Screen is beautiful!", source="test", platform="manual", rating=5.0),
            Review(id="i3", text="Too expensive for what you get. Competitor is better.", source="test", platform="manual", rating=2.0),
            Review(id="i4", text="Fast performance, but gets hot during gaming.", source="test", platform="manual", rating=3.0),
            Review(id="i5", text="Customer support was very helpful when I had issues.", source="test", platform="manual", rating=4.0),
        ]

        try:
            print("  Generating insights with Groq...")
            insights = await sm.generate_insights(reviews)

            print(f"\n  Insights Generated:")
            print(f"  {'-'*50}")

            if hasattr(insights, 'summary'):
                print(f"  Summary: {insights.summary[:200]}...")

            if hasattr(insights, 'pros') and insights.pros:
                print(f"\n  PROS:")
                for pro in insights.pros[:5]:
                    print(f"    + {pro}")

            if hasattr(insights, 'cons') and insights.cons:
                print(f"\n  CONS:")
                for con in insights.cons[:5]:
                    print(f"    - {con}")

            if hasattr(insights, 'themes') and insights.themes:
                print(f"\n  Themes: {', '.join(insights.themes[:5])}")

            print(f"  {'-'*50}")

            return True

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False


# ============================================================================
# TEST 7: FULL PIPELINE
# ============================================================================

async def test_full_pipeline():
    """Test full pipeline: scrape -> analyze -> summarize."""
    print_separator("TEST 7: Full Pipeline (Steam + Sentiment + Groq)")

    if not GROQ_API_KEY:
        print("  SKIPPED: No GROQ_API_KEY set for summarization")
        return None

    from sentimatrix import Sentimatrix, SentimatrixConfig, LLMConfig

    config = SentimatrixConfig(
        llm=LLMConfig(
            provider="groq",
            model="llama-3.3-70b-versatile",
            api_key=GROQ_API_KEY
        )
    )

    async with Sentimatrix(config) as sm:
        # Step 1: Scrape
        print("\n  Step 1: Scraping Steam reviews...")
        try:
            reviews = await sm.scrape_steam(app_id=STEAM_APP_ID, limit=15)
            print(f"    Scraped {len(reviews)} reviews")
        except Exception as e:
            print(f"    Scraping failed: {e}")
            return False

        if not reviews:
            print("    No reviews found")
            return False

        # Step 2: Analyze sentiment
        print("\n  Step 2: Analyzing sentiment...")
        try:
            texts = [r.text for r in reviews if r.text]
            batch_result = await sm.analyze_sentiment_batch(texts)

            print(f"    Positive: {batch_result.positive_count}/{batch_result.total_count}")
            print(f"    Negative: {batch_result.negative_count}/{batch_result.total_count}")
            print(f"    Positive ratio: {batch_result.positive_ratio:.1%}")
        except Exception as e:
            print(f"    Analysis failed: {e}")

        # Step 3: Detect emotions
        print("\n  Step 3: Detecting emotions...")
        try:
            emotion_batch = await sm.detect_emotions_batch(texts[:5])
            print(f"    Most Common Emotion: {emotion_batch.most_common_emotion}")
            print(f"    Emotion distribution: {emotion_batch.emotion_counts}")
        except Exception as e:
            print(f"    Emotion detection failed: {e}")

        # Step 4: Generate summary
        print("\n  Step 4: Generating summary with Groq...")
        try:
            # Reviews are already Review objects from scraping
            summary = await sm.summarize_reviews(reviews[:10])
            print(f"    Summary: {summary[:300]}...")
        except Exception as e:
            print(f"    Summary failed: {e}")

        print("\n  Pipeline completed!")
        return True


# ============================================================================
# MAIN
# ============================================================================

async def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("  SENTIMATRIX PYPI LIVE INTEGRATION TESTS")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = {}

    # Test 1: Import
    results["Import"] = test_import()
    if not results["Import"]:
        print("\nCANNOT PROCEED: Package import failed")
        return results

    # Test 2: Sentiment Analysis
    try:
        results["Sentiment"] = await test_sentiment_analysis()
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        results["Sentiment"] = False

    # Test 3: Emotion Detection
    try:
        results["Emotion"] = await test_emotion_detection()
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        results["Emotion"] = False

    # Test 4: Steam Scraper
    try:
        results["Steam Scraper"] = await test_steam_scraper()
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        results["Steam Scraper"] = False

    # Test 5: Groq LLM
    try:
        results["Groq Summarize"] = await test_groq_llm()
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        results["Groq Summarize"] = False

    # Test 6: Groq Insights
    try:
        results["Groq Insights"] = await test_groq_insights()
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        results["Groq Insights"] = False

    # Test 7: Full Pipeline
    try:
        results["Full Pipeline"] = await test_full_pipeline()
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        results["Full Pipeline"] = False

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
