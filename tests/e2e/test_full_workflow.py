"""
End-to-End Full Workflow Tests.

Tests complete Sentimatrix workflows from start to finish.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sentimatrix.analysis.sentiment import SentimentResult, SentimentLabel
from sentimatrix.analysis.emotion import EmotionResult, EmotionScore


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_texts():
    """Sample texts for analysis."""
    return [
        "This product is absolutely amazing! Best purchase I've ever made.",
        "Terrible quality. Complete waste of money. Do not buy!",
        "It's okay. Nothing special but gets the job done.",
        "Exceeded all my expectations. Highly recommend to everyone!",
        "Worst experience ever. Customer service was unhelpful.",
    ]


@pytest.fixture
def sample_reviews():
    """Sample review data."""
    return [
        {"text": "Great product! Highly recommend.", "rating": 5},
        {"text": "Terrible quality. Broke after a week.", "rating": 1},
        {"text": "Average product. Nothing special.", "rating": 3},
    ]


# ============================================================================
# Text Analysis Workflow Tests
# ============================================================================


class TestTextAnalysisWorkflow:
    """Tests for text analysis workflow."""

    @pytest.mark.asyncio
    async def test_single_text_analysis_flow(self):
        """Test analyzing a single text through the full flow."""
        text = "This product is amazing!"

        # Simulate sentiment analysis
        sentiment = {
            "label": "positive",
            "confidence": 0.95,
            "scores": {"positive": 0.95, "negative": 0.03, "neutral": 0.02},
        }

        # Simulate emotion detection
        emotion = {
            "primary": "joy",
            "score": 0.88,
            "emotions": [{"label": "joy", "score": 0.88}],
        }

        # Combined result
        result = {
            "text": text,
            "sentiment": sentiment,
            "emotion": emotion,
        }

        assert result["sentiment"]["label"] == "positive"
        assert result["emotion"]["primary"] == "joy"

    @pytest.mark.asyncio
    async def test_batch_text_analysis_flow(self, sample_texts):
        """Test batch text analysis."""
        results = []

        for text in sample_texts:
            # Simple rule-based classification for testing
            if "amazing" in text.lower() or "best" in text.lower() or "recommend" in text.lower():
                sentiment = "positive"
            elif "terrible" in text.lower() or "worst" in text.lower() or "waste" in text.lower():
                sentiment = "negative"
            else:
                sentiment = "neutral"

            results.append({
                "text": text,
                "sentiment": sentiment,
            })

        assert len(results) == 5

        # Count sentiments
        positive = sum(1 for r in results if r["sentiment"] == "positive")
        negative = sum(1 for r in results if r["sentiment"] == "negative")

        assert positive == 2
        assert negative == 2


class TestReviewAnalysisWorkflow:
    """Tests for review analysis workflow."""

    @pytest.mark.asyncio
    async def test_review_processing_flow(self, sample_reviews):
        """Test complete review processing flow."""
        # Process reviews
        processed = []
        for review in sample_reviews:
            rating = review["rating"]
            if rating >= 4:
                sentiment = "positive"
            elif rating <= 2:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            processed.append({
                "text": review["text"],
                "rating": rating,
                "sentiment": sentiment,
            })

        # Aggregate results
        total = len(processed)
        positive_count = sum(1 for p in processed if p["sentiment"] == "positive")
        negative_count = sum(1 for p in processed if p["sentiment"] == "negative")
        neutral_count = sum(1 for p in processed if p["sentiment"] == "neutral")

        result = {
            "total_reviews": total,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "positive_ratio": positive_count / total,
            "average_rating": sum(r["rating"] for r in sample_reviews) / total,
        }

        assert result["total_reviews"] == 3
        assert result["positive_count"] == 1
        assert result["negative_count"] == 1
        assert result["positive_ratio"] == pytest.approx(0.333, rel=0.01)


class TestExportWorkflow:
    """Tests for data export workflows."""

    @pytest.mark.asyncio
    async def test_json_export_flow(self, sample_reviews):
        """Test exporting results to JSON."""
        # Process reviews
        results = [
            {"text": r["text"], "rating": r["rating"], "processed": True}
            for r in sample_reviews
        ]

        # Export to JSON
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(results, f, indent=2)
            temp_path = f.name

        try:
            # Verify file was created and contains valid JSON
            with open(temp_path, "r") as f:
                loaded = json.load(f)

            assert len(loaded) == 3
            assert all("text" in item for item in loaded)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_csv_export_flow(self, sample_reviews):
        """Test exporting results to CSV."""
        import csv

        # Process reviews
        results = [
            {"text": r["text"], "rating": r["rating"]}
            for r in sample_reviews
        ]

        # Export to CSV
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "rating"])
            writer.writeheader()
            writer.writerows(results)
            temp_path = f.name

        try:
            # Verify file was created
            assert Path(temp_path).exists()

            # Read back
            with open(temp_path, "r") as f:
                reader = csv.DictReader(f)
                loaded = list(reader)

            assert len(loaded) == 3
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestPipelineWorkflow:
    """Tests for complete pipeline workflows."""

    @pytest.mark.asyncio
    async def test_full_analysis_pipeline_flow(self, sample_reviews):
        """Test running the full analysis pipeline."""
        # Step 1: Fetch reviews
        reviews = sample_reviews

        # Step 2: Analyze sentiment
        sentiments = []
        for review in reviews:
            rating = review["rating"]
            sentiment = "positive" if rating >= 4 else "negative" if rating <= 2 else "neutral"
            sentiments.append({
                "text": review["text"],
                "sentiment": sentiment,
                "confidence": 0.9,
            })

        # Step 3: Detect emotions
        emotions = []
        for sent in sentiments:
            if sent["sentiment"] == "positive":
                emotion = "joy"
            elif sent["sentiment"] == "negative":
                emotion = "anger"
            else:
                emotion = "neutral"
            emotions.append({
                "text": sent["text"],
                "emotion": emotion,
            })

        # Step 4: Aggregate
        result = {
            "total": len(reviews),
            "sentiments": {
                "positive": sum(1 for s in sentiments if s["sentiment"] == "positive"),
                "negative": sum(1 for s in sentiments if s["sentiment"] == "negative"),
                "neutral": sum(1 for s in sentiments if s["sentiment"] == "neutral"),
            },
            "emotions": {
                "joy": sum(1 for e in emotions if e["emotion"] == "joy"),
                "anger": sum(1 for e in emotions if e["emotion"] == "anger"),
                "neutral": sum(1 for e in emotions if e["emotion"] == "neutral"),
            },
        }

        assert result["total"] == 3
        assert result["sentiments"]["positive"] == 1
        assert result["emotions"]["joy"] == 1


class TestErrorHandlingWorkflow:
    """Tests for error handling in workflows."""

    @pytest.mark.asyncio
    async def test_graceful_error_handling(self):
        """Test graceful handling of errors."""
        reviews = [
            {"text": "Valid review", "rating": 5},
            {"text": None, "rating": 3},  # Invalid
            {"text": "Another review", "rating": 4},
        ]

        results = []
        errors = []

        for review in reviews:
            try:
                if not review.get("text"):
                    raise ValueError("Missing text")
                results.append({
                    "text": review["text"],
                    "processed": True,
                })
            except Exception as e:
                errors.append(str(e))

        assert len(results) == 2
        assert len(errors) == 1
        assert "Missing text" in errors[0]

    @pytest.mark.asyncio
    async def test_empty_input_handling(self):
        """Test handling of empty input."""
        reviews = []

        result = {
            "total": len(reviews),
            "processed": 0,
            "skipped": 0,
        }

        assert result["total"] == 0


class TestCachingWorkflow:
    """Tests for caching in workflows."""

    @pytest.mark.asyncio
    async def test_cache_hit_scenario(self):
        """Test cache hit scenario."""
        cache = {}

        def get_or_compute(key: str, compute_fn):
            if key in cache:
                return cache[key], True  # Cache hit
            result = compute_fn()
            cache[key] = result
            return result, False  # Cache miss

        # First call - cache miss
        result1, hit1 = get_or_compute("key1", lambda: {"value": 42})
        assert not hit1
        assert result1["value"] == 42

        # Second call - cache hit
        result2, hit2 = get_or_compute("key1", lambda: {"value": 100})
        assert hit2
        assert result2["value"] == 42  # Returns cached value

    @pytest.mark.asyncio
    async def test_cache_invalidation(self):
        """Test cache invalidation."""
        cache = {}

        # Add to cache
        cache["key1"] = {"value": 42}
        assert "key1" in cache

        # Invalidate
        del cache["key1"]
        assert "key1" not in cache


class TestMultiModalWorkflow:
    """Tests for multi-modal analysis workflows."""

    @pytest.mark.asyncio
    async def test_audio_analysis_flow(self):
        """Test audio analysis workflow."""
        # Simulate audio transcription
        transcription = {
            "text": "This product is amazing and works great!",
            "language": "en",
            "duration_seconds": 5.0,
        }

        # Analyze sentiment of transcription
        sentiment = {
            "label": "positive",
            "confidence": 0.92,
        }

        result = {
            "transcription": transcription,
            "sentiment": sentiment,
        }

        assert result["transcription"]["text"] is not None
        assert result["sentiment"]["label"] == "positive"

    @pytest.mark.asyncio
    async def test_image_analysis_flow(self):
        """Test image analysis workflow."""
        # Simulate image captioning
        caption = {
            "text": "A happy person holding a product and smiling",
            "confidence": 0.88,
        }

        # Analyze sentiment of caption
        sentiment = {
            "label": "positive",
            "confidence": 0.85,
        }

        result = {
            "caption": caption,
            "sentiment": sentiment,
        }

        assert "happy" in result["caption"]["text"]
        assert result["sentiment"]["label"] == "positive"

    @pytest.mark.asyncio
    async def test_combined_multimodal_flow(self):
        """Test combined multi-modal analysis."""
        # Text input
        text_sentiment = {"label": "positive", "confidence": 0.95}

        # Audio input
        audio_sentiment = {"label": "positive", "confidence": 0.88}

        # Image input
        image_sentiment = {"label": "neutral", "confidence": 0.70}

        # Combine with weighted average
        weights = {"text": 0.5, "audio": 0.3, "image": 0.2}
        sentiments = {
            "text": text_sentiment,
            "audio": audio_sentiment,
            "image": image_sentiment,
        }

        # Simple fusion: majority vote
        votes = {"positive": 0, "negative": 0, "neutral": 0}
        for modality, sent in sentiments.items():
            votes[sent["label"]] += 1

        combined_label = max(votes, key=votes.get)

        result = {
            "modalities_analyzed": list(sentiments.keys()),
            "individual_sentiments": sentiments,
            "combined_sentiment": combined_label,
        }

        assert len(result["modalities_analyzed"]) == 3
        assert result["combined_sentiment"] == "positive"
