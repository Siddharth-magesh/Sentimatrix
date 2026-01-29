#!/usr/bin/env python3
"""
Full Pipeline Test - Review Scraping and Sentiment Analysis

This script demonstrates the complete Sentimatrix pipeline:
1. Scraping reviews from Steam (uses JSON API, no browser needed)
2. Analyzing sentiment using local transformer models
3. Generating insights using Groq LLM
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_steam_reviews():
    """Test scraping and analyzing Steam game reviews."""
    print("=" * 60)
    print("STEAM REVIEWS - FULL PIPELINE TEST")
    print("=" * 60)

    from sentimatrix.providers.scrapers.platforms.steam import SteamScraper, SteamConfig

    # Popular game: Counter-Strike 2 (app_id: 730)
    app_id = "730"

    print(f"\n1. Scraping reviews for Steam App ID: {app_id}")
    print("-" * 40)

    config = SteamConfig(
        language="english",
        review_type="all",  # positive, negative, or all
    )

    async with SteamScraper(config) as scraper:
        reviews = await scraper.scrape_reviews(app_id, limit=10)

        print(f"   Scraped {len(reviews)} reviews")

        if reviews:
            print("\n   Sample reviews:")
            for i, review in enumerate(reviews[:3], 1):
                # Truncate long reviews
                text = review.text[:150] + "..." if len(review.text) > 150 else review.text
                rating = "Positive" if review.rating and review.rating > 0 else "Negative"
                author = review.author or "Anonymous"
                print(f"\n   [{i}] {rating} - by {author}")
                print(f"       {text}")

        return reviews


async def test_sentiment_analysis(reviews):
    """Test sentiment analysis on scraped reviews."""
    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS")
    print("=" * 60)

    from sentimatrix.analysis.sentiment import SentimentAnalyzer

    analyzer = SentimentAnalyzer()
    await analyzer.initialize()

    print("\n2. Analyzing sentiment of scraped reviews...")
    print("-" * 40)

    try:
        # Analyze each review
        results = []
        for i, review in enumerate(reviews[:5], 1):  # Analyze first 5
            result = await analyzer.analyze(review.text)
            results.append(result)

            print(f"\n   Review {i}:")
            print(f"   Text: {review.text[:80]}...")
            print(f"   Sentiment: {result.sentiment} (confidence: {result.confidence:.2%})")
            print(f"   Polarity: {result.polarity:.2f}")

        # Batch analysis summary
        batch_result = await analyzer.analyze_batch([r.text for r in reviews])

        print("\n   === BATCH ANALYSIS SUMMARY ===")
        print(f"   Total reviews analyzed: {len(reviews)}")
        print(f"   Positive: {batch_result.positive_count} ({batch_result.positive_ratio:.1%})")
        print(f"   Negative: {batch_result.negative_count} ({batch_result.negative_ratio:.1%})")
        print(f"   Neutral: {batch_result.neutral_count} ({batch_result.neutral_ratio:.1%})")
        print(f"   Average polarity: {batch_result.average_polarity:.2f}")

        return batch_result

    finally:
        await analyzer.close()


async def test_emotion_detection(reviews):
    """Test emotion detection on reviews."""
    print("\n" + "=" * 60)
    print("EMOTION DETECTION")
    print("=" * 60)

    from sentimatrix.analysis.emotion import EmotionDetector

    detector = EmotionDetector()
    await detector.initialize()

    print("\n3. Detecting emotions in reviews...")
    print("-" * 40)

    try:
        for i, review in enumerate(reviews[:3], 1):  # Analyze first 3
            result = await detector.detect(review.text)

            print(f"\n   Review {i}:")
            print(f"   Text: {review.text[:80]}...")
            print(f"   Primary emotion: {result.primary_emotion.label} ({result.primary_emotion.score:.2%})")

            if len(result.emotions) > 1:
                secondary = result.emotions[1]
                print(f"   Secondary emotion: {secondary.label} ({secondary.score:.2%})")

        return True

    finally:
        await detector.close()


async def test_llm_insights(reviews, groq_api_key: str):
    """Test LLM-powered insights generation."""
    print("\n" + "=" * 60)
    print("LLM-POWERED INSIGHTS (Groq)")
    print("=" * 60)

    from sentimatrix.providers.llm.groq_provider import GroqProvider
    from sentimatrix.core.config import LLMConfig

    config = LLMConfig(
        provider="groq",
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0.7,
        max_tokens=500,
    )

    print("\n4. Generating insights using Groq LLM...")
    print("-" * 40)

    async with GroqProvider(config) as provider:
        # Prepare review summaries for the prompt
        review_summaries = []
        for i, review in enumerate(reviews[:10], 1):
            rating = "Positive" if review.rating and review.rating > 0 else "Negative"
            text = review.text[:200] + "..." if len(review.text) > 200 else review.text
            review_summaries.append(f"{i}. [{rating}] {text}")

        reviews_text = "\n".join(review_summaries)

        prompt = f"""Analyze these game reviews and provide:
1. A 2-3 sentence summary of overall sentiment
2. Top 3 things players love (PROS)
3. Top 3 things players dislike (CONS)
4. Key themes mentioned

Reviews:
{reviews_text}

Provide a structured analysis:"""

        system_prompt = "You are a game review analyst. Provide concise, actionable insights."

        response = await provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
        )

        print("\n   === LLM ANALYSIS ===")
        print(f"\n{response.content}")
        print(f"\n   Tokens used: {response.usage.total_tokens}")
        print(f"   Response time: {response.response_time_ms:.0f}ms")

        return response


async def test_full_sentimatrix_pipeline(groq_api_key: str):
    """Test the full Sentimatrix high-level API."""
    print("\n" + "=" * 60)
    print("FULL SENTIMATRIX PIPELINE")
    print("=" * 60)

    from sentimatrix import Sentimatrix, LLMConfig

    # Configure with Groq
    llm_config = LLMConfig(
        provider="groq",
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
    )

    print("\n5. Running full Sentimatrix pipeline...")
    print("-" * 40)

    async with Sentimatrix(llm_config=llm_config) as sm:
        # Scrape Steam reviews
        print("\n   a) Scraping Steam reviews...")
        reviews = await sm.scrape_steam("730", limit=10)
        print(f"      Got {len(reviews)} reviews")

        # Analyze reviews
        print("\n   b) Analyzing sentiment and emotions...")
        analysis = await sm.analyze_reviews(reviews)

        print(f"      Positive ratio: {analysis.positive_ratio:.1%}")
        print(f"      Negative ratio: {analysis.negative_ratio:.1%}")
        print(f"      Average polarity: {analysis.average_polarity:.2f}")

        # Generate insights
        print("\n   c) Generating LLM insights...")
        insights = await sm.generate_insights(reviews, analysis=analysis)

        print("\n   === INSIGHTS ===")
        print(f"\n   Summary: {insights.summary}")

        if insights.pros:
            print("\n   PROS:")
            for pro in insights.pros[:3]:
                print(f"   + {pro}")

        if insights.cons:
            print("\n   CONS:")
            for con in insights.cons[:3]:
                print(f"   - {con}")

        if insights.themes:
            print("\n   THEMES:")
            for theme in insights.themes[:3]:
                print(f"   * {theme}")

        return analysis, insights


async def main():
    """Run all tests."""
    # Groq API key from user
    groq_api_key = "your_groq_api_key_here"  # Replace with your actual Groq API key

    print("\n" + "#" * 60)
    print("#  SENTIMATRIX - FULL PIPELINE DEMONSTRATION")
    print("#" * 60)
    print("""
This test demonstrates the complete review processing pipeline:

1. SCRAPING: Fetch reviews from Steam's API (JSON-based, no browser needed)
2. SENTIMENT: Analyze positive/negative sentiment using transformer models
3. EMOTION: Detect emotions (joy, anger, surprise, etc.)
4. LLM INSIGHTS: Generate actionable insights using Groq's LLaMA model

Note: Amazon reviews require JavaScript rendering (Playwright + browser deps).
      Steam reviews use a JSON API and work without additional dependencies.
""")

    try:
        # Test 1: Scrape Steam reviews
        reviews = await test_steam_reviews()

        if not reviews:
            print("\nNo reviews scraped. Exiting.")
            return

        # Test 2: Sentiment analysis
        sentiment_result = await test_sentiment_analysis(reviews)

        # Test 3: Emotion detection
        await test_emotion_detection(reviews)

        # Test 4: LLM insights
        await test_llm_insights(reviews, groq_api_key)

        # Test 5: Full pipeline
        analysis, insights = await test_full_sentimatrix_pipeline(groq_api_key)

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        print("""
SUMMARY:
--------
The Sentimatrix library successfully:
1. Scraped reviews from Steam (JSON API)
2. Analyzed sentiment using local transformer models
3. Detected emotions in review text
4. Generated insights using Groq LLM (LLaMA 3.3 70B)

For Amazon reviews (requires JavaScript):
- Install Playwright browser deps: sudo playwright install-deps
- Then run: playwright install chromium
- Or use a commercial scraping API (ScraperAPI, Bright Data, etc.)
""")

    except ImportError as e:
        print(f"\nImport error: {e}")
        print("Make sure all dependencies are installed: pip install -e .")
    except Exception as e:
        print(f"\nError: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
