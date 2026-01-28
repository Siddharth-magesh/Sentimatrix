# Sentimatrix V2 Examples

Comprehensive examples demonstrating Sentimatrix capabilities.

## Table of Contents

1. [Basic Analysis](#basic-analysis)
2. [Batch Processing](#batch-processing)
3. [Pipeline Workflows](#pipeline-workflows)
4. [Web Scraping](#web-scraping)
5. [Multi-Modal Analysis](#multi-modal-analysis)
6. [Advanced Configurations](#advanced-configurations)
7. [Real-World Use Cases](#real-world-use-cases)

---

## Basic Analysis

### Simple Sentiment Analysis

```python
import asyncio
from sentimatrix.analysis.sentiment import SentimentAnalyzer, SentimentLabel

async def analyze_review():
    analyzer = SentimentAnalyzer()

    text = "This is the best product I've ever purchased!"
    result = await analyzer.analyze(text)

    print(f"Text: {text}")
    print(f"Sentiment: {result.sentiment.name}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Scores: {result.scores}")

asyncio.run(analyze_review())
```

### Combined Sentiment and Emotion Analysis

```python
import asyncio
from sentimatrix.analysis.sentiment import SentimentAnalyzer
from sentimatrix.analysis.emotion import EmotionDetector

async def full_analysis(text: str):
    sentiment_analyzer = SentimentAnalyzer()
    emotion_detector = EmotionDetector()

    # Run both analyses concurrently
    sentiment_result, emotion_result = await asyncio.gather(
        sentiment_analyzer.analyze(text),
        emotion_detector.detect(text)
    )

    return {
        "text": text,
        "sentiment": {
            "label": sentiment_result.sentiment.name,
            "confidence": sentiment_result.confidence,
        },
        "emotion": {
            "primary": emotion_result.primary_emotion,
            "all_emotions": [
                {"label": e.label, "score": e.score}
                for e in emotion_result.emotions
            ],
        },
    }

async def main():
    texts = [
        "I'm absolutely thrilled with my purchase!",
        "This product is a complete disappointment.",
        "It works as expected, nothing special.",
    ]

    for text in texts:
        result = await full_analysis(text)
        print(f"\n{result['text']}")
        print(f"  Sentiment: {result['sentiment']['label']}")
        print(f"  Primary Emotion: {result['emotion']['primary']}")

asyncio.run(main())
```

---

## Batch Processing

### Processing Multiple Reviews

```python
import asyncio
from sentimatrix.analysis.sentiment import SentimentAnalyzer

async def batch_analyze():
    analyzer = SentimentAnalyzer()

    reviews = [
        "Excellent quality and fast shipping!",
        "Product broke after two days. Terrible!",
        "It's okay for the price.",
        "Best purchase I've made this year!",
        "Don't waste your money on this.",
    ]

    results = await analyzer.analyze_batch(reviews)

    # Aggregate results
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    for result in results:
        sentiment_counts[result.sentiment.name.lower()] += 1

    print("Batch Analysis Results:")
    print(f"  Total: {len(results)}")
    print(f"  Positive: {sentiment_counts['positive']}")
    print(f"  Negative: {sentiment_counts['negative']}")
    print(f"  Neutral: {sentiment_counts['neutral']}")

asyncio.run(batch_analyze())
```

### Processing from CSV File

```python
import asyncio
import csv
from sentimatrix.analysis.sentiment import SentimentAnalyzer
from sentimatrix.output import CSVFormatter

async def process_csv(input_file: str, output_file: str):
    analyzer = SentimentAnalyzer()

    # Read input
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        reviews = list(reader)

    # Extract texts
    texts = [r['text'] for r in reviews]

    # Analyze
    results = await analyzer.analyze_batch(texts)

    # Combine with original data
    for review, result in zip(reviews, results):
        review['sentiment'] = result.sentiment.name
        review['confidence'] = f"{result.confidence:.2%}"

    # Write output
    formatter = CSVFormatter()
    formatter.write(output_file, reviews)

    print(f"Processed {len(reviews)} reviews -> {output_file}")

asyncio.run(process_csv("reviews.csv", "analyzed_reviews.csv"))
```

---

## Pipeline Workflows

### Review Analysis Pipeline

```python
import asyncio
from sentimatrix.core.pipeline import (
    Pipeline, FunctionStep, ParallelSteps, PipelineContext
)

async def main():
    pipeline = Pipeline(name="review_analysis_pipeline")

    # Step 1: Load reviews
    async def load_reviews(ctx: PipelineContext, _):
        reviews = [
            {"id": 1, "text": "Great product!", "rating": 5},
            {"id": 2, "text": "Terrible quality.", "rating": 1},
            {"id": 3, "text": "Average, nothing special.", "rating": 3},
            {"id": 4, "text": "Exceeded expectations!", "rating": 5},
            {"id": 5, "text": "Waste of money.", "rating": 1},
        ]
        ctx.set("review_count", len(reviews))
        return reviews

    # Step 2: Filter valid reviews
    async def filter_reviews(ctx: PipelineContext, reviews):
        valid = [r for r in reviews if r.get("text") and len(r["text"]) > 5]
        ctx.set("filtered_count", len(valid))
        return valid

    # Step 3: Analyze sentiment
    async def analyze_sentiment(ctx: PipelineContext, reviews):
        results = []
        for review in reviews:
            rating = review.get("rating", 3)
            if rating >= 4:
                sentiment = "positive"
                confidence = 0.9
            elif rating <= 2:
                sentiment = "negative"
                confidence = 0.85
            else:
                sentiment = "neutral"
                confidence = 0.7

            results.append({
                **review,
                "sentiment": sentiment,
                "confidence": confidence,
            })
        return results

    # Step 4: Generate summary
    async def generate_summary(ctx: PipelineContext, results):
        total = ctx.get("review_count", 0)
        sentiment_counts = {}
        for r in results:
            s = r["sentiment"]
            sentiment_counts[s] = sentiment_counts.get(s, 0) + 1

        return {
            "total_reviews": total,
            "processed_reviews": len(results),
            "sentiment_distribution": sentiment_counts,
            "average_confidence": sum(r["confidence"] for r in results) / len(results),
        }

    pipeline.add_step(FunctionStep("load", load_reviews))
    pipeline.add_step(FunctionStep("filter", filter_reviews))
    pipeline.add_step(FunctionStep("analyze", analyze_sentiment))
    pipeline.add_step(FunctionStep("summarize", generate_summary))

    result = await pipeline.run()

    if result.success:
        print("Pipeline completed successfully!")
        print(f"Summary: {result.output}")
    else:
        print(f"Pipeline failed: {result.error}")

asyncio.run(main())
```

### Parallel Analysis Pipeline

```python
import asyncio
from sentimatrix.core.pipeline import (
    Pipeline, FunctionStep, ParallelSteps, PipelineContext
)

async def main():
    pipeline = Pipeline(name="parallel_analysis")

    # Load data
    async def load_data(ctx: PipelineContext, _):
        return ["Great product!", "Terrible experience.", "It's okay."]

    pipeline.add_step(FunctionStep("load", load_data))

    # Parallel analysis steps
    async def model_a_analysis(ctx: PipelineContext, texts):
        await asyncio.sleep(0.1)  # Simulate processing
        return {"model": "A", "results": ["pos", "neg", "neu"]}

    async def model_b_analysis(ctx: PipelineContext, texts):
        await asyncio.sleep(0.1)  # Simulate processing
        return {"model": "B", "results": ["pos", "neg", "neu"]}

    parallel = ParallelSteps(
        name="parallel_models",
        steps=[
            FunctionStep("model_a", model_a_analysis),
            FunctionStep("model_b", model_b_analysis),
        ]
    )
    pipeline.add_step(parallel)

    result = await pipeline.run()
    print(f"Parallel results: {result.output}")

asyncio.run(main())
```

---

## Web Scraping

### Scraping Amazon Reviews

```python
import asyncio
from sentimatrix.providers.scrapers import AmazonScraper
from sentimatrix.analysis.sentiment import SentimentAnalyzer
from sentimatrix.output import JSONFormatter

async def scrape_and_analyze(product_id: str, max_reviews: int = 50):
    scraper = AmazonScraper()
    analyzer = SentimentAnalyzer()

    # Scrape reviews
    print(f"Scraping reviews for {product_id}...")
    reviews = await scraper.scrape(product_id, max_reviews=max_reviews)
    print(f"Found {len(reviews)} reviews")

    # Analyze each review
    results = []
    for review in reviews:
        sentiment = await analyzer.analyze(review.text)
        results.append({
            "id": review.id,
            "text": review.text[:100] + "...",
            "rating": review.rating,
            "sentiment": sentiment.sentiment.name,
            "confidence": sentiment.confidence,
        })

    # Export results
    formatter = JSONFormatter()
    formatter.write(f"{product_id}_analysis.json", results)

    # Print summary
    positive = sum(1 for r in results if r["sentiment"] == "POSITIVE")
    negative = sum(1 for r in results if r["sentiment"] == "NEGATIVE")

    print(f"\nAnalysis Summary:")
    print(f"  Total: {len(results)}")
    print(f"  Positive: {positive} ({positive/len(results)*100:.1f}%)")
    print(f"  Negative: {negative} ({negative/len(results)*100:.1f}%)")

asyncio.run(scrape_and_analyze("B08N5WRWNW"))
```

### Multi-Platform Scraping

```python
import asyncio
from sentimatrix.providers.scrapers import (
    AmazonScraper, YelpScraper, GooglePlayScraper
)

async def scrape_multiple_platforms():
    amazon = AmazonScraper()
    yelp = YelpScraper()
    google = GooglePlayScraper()

    # Scrape from all platforms concurrently
    amazon_reviews, yelp_reviews, google_reviews = await asyncio.gather(
        amazon.scrape("B08N5WRWNW", max_reviews=20),
        yelp.scrape("best-restaurant", max_reviews=20),
        google.scrape("com.example.app", max_reviews=20),
    )

    print(f"Amazon: {len(amazon_reviews)} reviews")
    print(f"Yelp: {len(yelp_reviews)} reviews")
    print(f"Google Play: {len(google_reviews)} reviews")

    # Combine all reviews
    all_reviews = [
        *[{"source": "amazon", **r.__dict__} for r in amazon_reviews],
        *[{"source": "yelp", **r.__dict__} for r in yelp_reviews],
        *[{"source": "google", **r.__dict__} for r in google_reviews],
    ]

    return all_reviews

asyncio.run(scrape_multiple_platforms())
```

---

## Multi-Modal Analysis

### Audio Transcription and Analysis

```python
import asyncio
from sentimatrix.providers.models.audio import AudioTranscriber
from sentimatrix.analysis.sentiment import SentimentAnalyzer

async def analyze_audio(audio_file: str):
    transcriber = AudioTranscriber()
    analyzer = SentimentAnalyzer()

    # Transcribe audio
    transcription = await transcriber.transcribe(audio_file)
    print(f"Transcription: {transcription.text}")

    # Analyze sentiment
    sentiment = await analyzer.analyze(transcription.text)

    return {
        "audio_file": audio_file,
        "transcription": transcription.text,
        "language": transcription.language,
        "duration": transcription.duration_seconds,
        "sentiment": sentiment.sentiment.name,
        "confidence": sentiment.confidence,
    }

result = asyncio.run(analyze_audio("customer_feedback.mp3"))
print(result)
```

### Image Analysis

```python
import asyncio
from sentimatrix.providers.models.vision import ImageAnalyzer
from sentimatrix.analysis.sentiment import SentimentAnalyzer

async def analyze_image(image_path: str):
    image_analyzer = ImageAnalyzer()
    sentiment_analyzer = SentimentAnalyzer()

    # Generate caption
    caption = await image_analyzer.caption(image_path)
    print(f"Caption: {caption.text}")

    # Analyze sentiment of caption
    sentiment = await sentiment_analyzer.analyze(caption.text)

    return {
        "image": image_path,
        "caption": caption.text,
        "caption_confidence": caption.confidence,
        "sentiment": sentiment.sentiment.name,
    }

result = asyncio.run(analyze_image("product_photo.jpg"))
print(result)
```

---

## Advanced Configurations

### Custom Provider Configuration

```python
import asyncio
from sentimatrix.providers.llm import OpenAIProvider
from sentimatrix.analysis.sentiment import SentimentAnalyzer

async def main():
    # Custom OpenAI configuration
    provider = OpenAIProvider(
        api_key="sk-...",
        model="gpt-4-turbo-preview",
        temperature=0.1,  # Low temperature for consistency
        max_tokens=100,
        timeout=30.0,
    )

    analyzer = SentimentAnalyzer(provider=provider)

    result = await analyzer.analyze("This product is fantastic!")
    print(f"Sentiment: {result.sentiment.name}")

asyncio.run(main())
```

### Caching with Redis

```python
import asyncio
from sentimatrix.utils.cache import RedisCache
from sentimatrix.analysis.sentiment import SentimentAnalyzer

async def main():
    # Configure Redis cache
    cache = RedisCache(
        host="localhost",
        port=6379,
        db=0,
        ttl=3600,  # 1 hour
        prefix="sentimatrix:"
    )

    analyzer = SentimentAnalyzer(cache=cache)

    # First call - cache miss, makes API request
    result1 = await analyzer.analyze("Great product!")
    print(f"First call: {result1.sentiment.name}")

    # Second call - cache hit, instant response
    result2 = await analyzer.analyze("Great product!")
    print(f"Second call (cached): {result2.sentiment.name}")

asyncio.run(main())
```

### Rate Limiting Configuration

```python
import asyncio
from sentimatrix.providers.scrapers import AmazonScraper
from sentimatrix.providers.scrapers.rate_limiter import (
    RateLimiter, RateLimitStrategy
)

async def main():
    # Configure rate limiter
    limiter = RateLimiter(
        strategy=RateLimitStrategy.SLIDING_WINDOW,
        requests_per_second=2.0,
        burst_size=5,
        max_retries=3,
        retry_delay=1.0,
    )

    scraper = AmazonScraper(rate_limiter=limiter)

    # Scrape with rate limiting
    reviews = await scraper.scrape("B08N5WRWNW", max_reviews=100)
    print(f"Scraped {len(reviews)} reviews with rate limiting")

asyncio.run(main())
```

---

## Real-World Use Cases

### E-commerce Review Dashboard

```python
import asyncio
from datetime import datetime
from sentimatrix.providers.scrapers import AmazonScraper
from sentimatrix.analysis.sentiment import SentimentAnalyzer
from sentimatrix.analysis.aggregator import ResultAggregator
from sentimatrix.output import HTMLFormatter

async def generate_dashboard(product_ids: list):
    scraper = AmazonScraper()
    analyzer = SentimentAnalyzer()
    aggregator = ResultAggregator()

    all_results = []

    for product_id in product_ids:
        # Scrape reviews
        reviews = await scraper.scrape(product_id, max_reviews=100)

        # Analyze
        for review in reviews:
            sentiment = await analyzer.analyze(review.text)
            all_results.append({
                "product_id": product_id,
                "review_text": review.text,
                "rating": review.rating,
                "sentiment": sentiment.sentiment.name,
                "confidence": sentiment.confidence,
            })

    # Aggregate by product
    aggregated = aggregator.aggregate(all_results)

    # Generate HTML dashboard
    formatter = HTMLFormatter(template="dashboard")
    html = formatter.format({
        "title": "E-commerce Review Dashboard",
        "generated_at": datetime.now().isoformat(),
        "products": aggregated,
    })

    with open("dashboard.html", "w") as f:
        f.write(html)

    print("Dashboard generated: dashboard.html")

asyncio.run(generate_dashboard(["B08N5WRWNW", "B09V3KXJPB"]))
```

### Social Media Monitoring

```python
import asyncio
from sentimatrix.analysis.sentiment import SentimentAnalyzer
from sentimatrix.analysis.emotion import EmotionDetector

async def monitor_social_media(posts: list):
    sentiment_analyzer = SentimentAnalyzer()
    emotion_detector = EmotionDetector()

    results = []

    for post in posts:
        # Run both analyses
        sentiment, emotion = await asyncio.gather(
            sentiment_analyzer.analyze(post["text"]),
            emotion_detector.detect(post["text"])
        )

        results.append({
            "post_id": post["id"],
            "text": post["text"],
            "sentiment": sentiment.sentiment.name,
            "primary_emotion": emotion.primary_emotion,
            "requires_attention": (
                sentiment.sentiment.name == "NEGATIVE" and
                emotion.primary_emotion in ["anger", "sadness"]
            ),
        })

    # Find posts requiring attention
    urgent = [r for r in results if r["requires_attention"]]

    print(f"Total posts analyzed: {len(results)}")
    print(f"Posts requiring attention: {len(urgent)}")

    for post in urgent:
        print(f"  - {post['post_id']}: {post['text'][:50]}...")

    return results

# Example posts
posts = [
    {"id": "1", "text": "Love your new product! Amazing quality!"},
    {"id": "2", "text": "This is terrible! I want a refund immediately!"},
    {"id": "3", "text": "The product is okay, works as expected."},
    {"id": "4", "text": "Worst customer service ever! So angry!"},
]

asyncio.run(monitor_social_media(posts))
```

### Customer Feedback Analysis

```python
import asyncio
from sentimatrix.core.pipeline import Pipeline, FunctionStep, PipelineContext
from sentimatrix.output import ReportGenerator

async def analyze_customer_feedback():
    pipeline = Pipeline(name="feedback_analysis")

    # Step 1: Load feedback
    async def load_feedback(ctx: PipelineContext, _):
        return [
            {"id": 1, "category": "product", "text": "Great quality!"},
            {"id": 2, "category": "support", "text": "Terrible response time."},
            {"id": 3, "category": "shipping", "text": "Fast delivery!"},
            {"id": 4, "category": "product", "text": "Broke after a week."},
            {"id": 5, "category": "support", "text": "Very helpful staff!"},
        ]

    # Step 2: Categorize and analyze
    async def analyze(ctx: PipelineContext, feedback):
        categories = {}
        for item in feedback:
            cat = item["category"]
            if cat not in categories:
                categories[cat] = {"positive": 0, "negative": 0, "items": []}

            # Simple sentiment based on keywords
            text = item["text"].lower()
            if any(w in text for w in ["great", "fast", "helpful", "love"]):
                sentiment = "positive"
                categories[cat]["positive"] += 1
            elif any(w in text for w in ["terrible", "broke", "worst", "bad"]):
                sentiment = "negative"
                categories[cat]["negative"] += 1
            else:
                sentiment = "neutral"

            categories[cat]["items"].append({**item, "sentiment": sentiment})

        return categories

    # Step 3: Generate insights
    async def generate_insights(ctx: PipelineContext, categories):
        insights = []
        for cat, data in categories.items():
            total = data["positive"] + data["negative"]
            if total > 0:
                positive_ratio = data["positive"] / total
                if positive_ratio < 0.5:
                    insights.append(f"ALERT: {cat} has low satisfaction ({positive_ratio:.0%})")
                else:
                    insights.append(f"OK: {cat} satisfaction is good ({positive_ratio:.0%})")

        return {"categories": categories, "insights": insights}

    pipeline.add_step(FunctionStep("load", load_feedback))
    pipeline.add_step(FunctionStep("analyze", analyze))
    pipeline.add_step(FunctionStep("insights", generate_insights))

    result = await pipeline.run()

    if result.success:
        print("Insights:")
        for insight in result.output["insights"]:
            print(f"  - {insight}")

        # Generate report
        generator = ReportGenerator()
        report = generator.generate(result.output, format="html")
        generator.write("feedback_report.html", report)

asyncio.run(analyze_customer_feedback())
```
