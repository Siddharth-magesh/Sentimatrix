"""
Integration Tests for Pipeline System.

Tests the complete pipeline flow from input to output,
including step chaining, error handling, and context propagation.
"""

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sentimatrix.core.pipeline import (
    Pipeline,
    PipelineStep,
    FunctionStep,
    ParallelSteps,
    PipelineContext,
    PipelineResult,
)
from sentimatrix.core.exceptions import PipelineError, PipelineStepError
from sentimatrix.analysis.sentiment import SentimentResult, SentimentLabel


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_reviews():
    """Sample review data for testing."""
    return [
        {"text": "Great product! Highly recommend.", "rating": 5},
        {"text": "Terrible experience, never buying again.", "rating": 1},
        {"text": "It's okay, nothing special.", "rating": 3},
        {"text": "Amazing quality and fast shipping!", "rating": 5},
        {"text": "Waste of money, broke after a week.", "rating": 1},
    ]


# ============================================================================
# Pipeline Integration Tests
# ============================================================================


class TestPipelineIntegration:
    """Integration tests for pipeline execution."""

    @pytest.mark.asyncio
    async def test_simple_function_step_flow(self, sample_reviews):
        """Test a simple pipeline flow with function steps."""
        pipeline = Pipeline(name="simple_flow")

        # Step 1: Fetch reviews - NOTE: FunctionStep signature is (input_data, context)
        async def fetch_reviews(input_data, ctx: PipelineContext) -> List[Dict]:
            return sample_reviews

        pipeline.add_step(FunctionStep("fetch_reviews", fetch_reviews))

        # Step 2: Analyze sentiment (mock)
        async def analyze_sentiment(reviews, ctx: PipelineContext) -> List[Dict]:
            results = []
            for review in reviews:
                rating = review.get("rating", 3)
                if rating >= 4:
                    sentiment = "positive"
                elif rating <= 2:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
                results.append({"text": review["text"], "sentiment": sentiment})
            return results

        pipeline.add_step(FunctionStep("analyze_sentiment", analyze_sentiment))

        # Step 3: Aggregate results
        async def aggregate(sentiments, ctx: PipelineContext) -> Dict:
            total = len(sentiments)
            counts = {"positive": 0, "negative": 0, "neutral": 0}
            for s in sentiments:
                counts[s["sentiment"]] += 1
            return {"total": total, **counts}

        pipeline.add_step(FunctionStep("aggregate", aggregate))

        result = await pipeline.run()

        assert result.success is True
        assert result.output is not None
        assert result.output["total"] == 5
        assert result.output["positive"] == 2
        assert result.output["negative"] == 2
        assert result.output["neutral"] == 1

    @pytest.mark.asyncio
    async def test_pipeline_with_filter(self, sample_reviews):
        """Test pipeline with filtering step."""
        pipeline = Pipeline(name="filtered_flow")

        # Fetch - NOTE: FunctionStep signature is (input_data, context)
        async def fetch(input_data, ctx: PipelineContext) -> List[Dict]:
            return sample_reviews

        # Filter only high ratings
        async def filter_positive(reviews, ctx: PipelineContext) -> List[Dict]:
            return [r for r in reviews if r.get("rating", 0) >= 4]

        # Count
        async def count(reviews, ctx: PipelineContext) -> Dict:
            return {"count": len(reviews)}

        pipeline.add_step(FunctionStep("fetch", fetch))
        pipeline.add_step(FunctionStep("filter", filter_positive))
        pipeline.add_step(FunctionStep("count", count))

        result = await pipeline.run()

        assert result.success is True
        assert result.output["count"] == 2  # Only 2 reviews with rating >= 4

    @pytest.mark.asyncio
    async def test_pipeline_context_propagation(self, sample_reviews):
        """Test that context is properly propagated between steps."""
        pipeline = Pipeline(name="context_test")

        # Step that writes to context - NOTE: FunctionStep signature is (input_data, context)
        async def write_metadata(input_data, ctx: PipelineContext) -> str:
            ctx.set("source", "test_source")
            ctx.set("version", "1.0")
            return "metadata_written"

        # Step that reads from context
        async def read_metadata(input_data, ctx: PipelineContext) -> Dict:
            return {
                "source": ctx.get("source"),
                "version": ctx.get("version"),
            }

        pipeline.add_step(FunctionStep("write", write_metadata))
        pipeline.add_step(FunctionStep("read", read_metadata))

        result = await pipeline.run()

        assert result.success is True
        assert result.output["source"] == "test_source"
        assert result.output["version"] == "1.0"

    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self):
        """Test pipeline error handling with failing step."""
        pipeline = Pipeline(name="error_test")

        async def success_step(input_data, ctx: PipelineContext) -> str:
            return "success"

        async def failing_step(input_data, ctx: PipelineContext) -> str:
            raise RuntimeError("Test error")

        pipeline.add_step(FunctionStep("success", success_step))
        pipeline.add_step(FunctionStep("fail", failing_step))

        result = await pipeline.run()

        assert result.success is False
        assert result.error is not None
        assert "Test error" in str(result.error)

    @pytest.mark.asyncio
    async def test_pipeline_empty_input(self):
        """Test pipeline with empty input."""
        pipeline = Pipeline(name="empty_test")

        async def return_empty(input_data, ctx: PipelineContext) -> List:
            return []

        async def count_items(items, ctx: PipelineContext) -> Dict:
            return {"count": len(items)}

        pipeline.add_step(FunctionStep("empty", return_empty))
        pipeline.add_step(FunctionStep("count", count_items))

        result = await pipeline.run()

        assert result.success is True
        assert result.output["count"] == 0


class TestParallelStepsIntegration:
    """Integration tests for parallel step execution."""

    @pytest.mark.asyncio
    async def test_parallel_analysis_steps(self, sample_reviews):
        """Test parallel execution of multiple analysis steps."""
        pipeline = Pipeline(name="parallel_test")

        # Fetch step - NOTE: FunctionStep signature is (input_data, context)
        async def fetch(input_data, ctx: PipelineContext) -> List[Dict]:
            return sample_reviews

        pipeline.add_step(FunctionStep("fetch", fetch))

        # Parallel analysis steps
        async def analyze_a(reviews, ctx: PipelineContext) -> Dict:
            await asyncio.sleep(0.01)
            return {"model": "A", "count": len(reviews)}

        async def analyze_b(reviews, ctx: PipelineContext) -> Dict:
            await asyncio.sleep(0.01)
            return {"model": "B", "count": len(reviews)}

        parallel = ParallelSteps(
            name="parallel_analysis",
            steps=[
                FunctionStep("model_a", analyze_a),
                FunctionStep("model_b", analyze_b),
            ]
        )
        pipeline.add_step(parallel)

        result = await pipeline.run()

        assert result.success is True
        # Check that parallel results are available
        assert result.output is not None


class TestPipelineWithRealComponents:
    """Integration tests using real (mocked) Sentimatrix components."""

    @pytest.mark.asyncio
    async def test_full_analysis_pipeline(self, sample_reviews):
        """Test full analysis pipeline with mocked analyzers."""
        pipeline = Pipeline(name="full_analysis")

        # Fetch step - NOTE: FunctionStep signature is (input_data, context)
        async def fetch_step(input_data, ctx: PipelineContext) -> List[Dict]:
            return sample_reviews

        pipeline.add_step(FunctionStep("fetch", fetch_step))

        # Sentiment analysis step
        async def sentiment_step(reviews, ctx: PipelineContext) -> List[Dict]:
            results = []
            for review in reviews:
                rating = review.get("rating", 3)
                sentiment = "positive" if rating >= 4 else "negative" if rating <= 2 else "neutral"
                results.append({
                    "text": review["text"],
                    "sentiment": sentiment,
                    "confidence": 0.9,
                })
            # Store in context for later
            ctx.set("sentiments", results)
            return results

        pipeline.add_step(FunctionStep("sentiment", sentiment_step))

        # Emotion detection step
        async def emotion_step(sentiments, ctx: PipelineContext) -> List[Dict]:
            results = []
            for sent in sentiments:
                if sent["sentiment"] == "positive":
                    emotion = "joy"
                elif sent["sentiment"] == "negative":
                    emotion = "anger"
                else:
                    emotion = "neutral"
                results.append({
                    "text": sent["text"],
                    "emotion": emotion,
                    "score": 0.8,
                })
            return results

        pipeline.add_step(FunctionStep("emotions", emotion_step))

        # Combine results
        async def combine_step(emotions, ctx: PipelineContext) -> Dict:
            # Get sentiments from context
            sentiments = ctx.get("sentiments") or []
            return {
                "total_reviews": len(emotions),
                "sentiment_counts": {
                    "positive": sum(1 for s in sentiments if s["sentiment"] == "positive"),
                    "negative": sum(1 for s in sentiments if s["sentiment"] == "negative"),
                    "neutral": sum(1 for s in sentiments if s["sentiment"] == "neutral"),
                },
                "emotion_counts": {
                    "joy": sum(1 for e in emotions if e["emotion"] == "joy"),
                    "anger": sum(1 for e in emotions if e["emotion"] == "anger"),
                    "neutral": sum(1 for e in emotions if e["emotion"] == "neutral"),
                },
            }

        pipeline.add_step(FunctionStep("combine", combine_step))

        result = await pipeline.run()

        assert result.success is True
        assert result.output["total_reviews"] == 5
        assert result.output["emotion_counts"]["joy"] == 2
        assert result.output["emotion_counts"]["anger"] == 2
