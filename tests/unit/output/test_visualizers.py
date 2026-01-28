"""
Unit tests for Sentimatrix Output Visualizers.

Tests:
- ChartVisualizer: Chart creation and saving
- Visualization options and themes
"""

import asyncio
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from sentimatrix.output.visualizers import (
    ChartType,
    Theme,
    VisualizationOptions,
    VisualizationResult,
    ChartVisualizer,
    get_visualizer,
    create_sentiment_chart,
    create_emotion_chart,
)


# ============================================================================
# Test Data
# ============================================================================


@dataclass
class MockAnalysisResult:
    """Mock analysis result for testing."""

    total_count: int
    positive_ratio: float
    negative_ratio: float
    average_polarity: float
    emotion_summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_count": self.total_count,
            "positive_ratio": self.positive_ratio,
            "negative_ratio": self.negative_ratio,
            "average_polarity": self.average_polarity,
            "emotion_summary": self.emotion_summary,
        }


@pytest.fixture
def sample_analysis():
    """Create sample analysis result for testing."""
    return MockAnalysisResult(
        total_count=100,
        positive_ratio=0.65,
        negative_ratio=0.15,
        average_polarity=0.45,
        emotion_summary={
            "distribution": {
                "joy": 0.4,
                "anger": 0.1,
                "sadness": 0.15,
                "surprise": 0.2,
                "fear": 0.05,
                "disgust": 0.1,
            }
        },
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def has_matplotlib():
    """Check if matplotlib is available."""
    try:
        import matplotlib
        return True
    except ImportError:
        return False


# ============================================================================
# VisualizationOptions Tests
# ============================================================================


class TestVisualizationOptions:
    """Tests for VisualizationOptions dataclass."""

    def test_default_values(self):
        """Test default option values."""
        options = VisualizationOptions()
        assert options.width == 10
        assert options.height == 6
        assert options.dpi == 150
        assert options.theme == Theme.DEFAULT
        assert options.show_values is True

    def test_custom_values(self):
        """Test custom option values."""
        options = VisualizationOptions(
            width=12,
            height=8,
            theme=Theme.DARK,
            title="Custom Chart",
            show_legend=False,
        )
        assert options.width == 12
        assert options.height == 8
        assert options.theme == Theme.DARK
        assert options.title == "Custom Chart"
        assert options.show_legend is False


# ============================================================================
# ChartVisualizer Tests
# ============================================================================


class TestChartVisualizer:
    """Tests for ChartVisualizer."""

    @pytest.mark.asyncio
    async def test_create_sentiment_bar_chart(self, sample_analysis, has_matplotlib):
        """Test creating sentiment bar chart."""
        if not has_matplotlib:
            pytest.skip("matplotlib not installed")

        visualizer = ChartVisualizer()
        fig = await visualizer.create_sentiment_bar_chart(sample_analysis)

        assert fig is not None
        # Close figure to free memory
        import matplotlib.pyplot as plt
        plt.close(fig)

    @pytest.mark.asyncio
    async def test_create_sentiment_bar_chart_with_title(self, sample_analysis, has_matplotlib):
        """Test creating sentiment bar chart with custom title."""
        if not has_matplotlib:
            pytest.skip("matplotlib not installed")

        visualizer = ChartVisualizer()
        fig = await visualizer.create_sentiment_bar_chart(
            sample_analysis,
            title="Custom Sentiment Chart",
        )

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    @pytest.mark.asyncio
    async def test_create_emotion_bar_chart(self, sample_analysis, has_matplotlib):
        """Test creating emotion bar chart."""
        if not has_matplotlib:
            pytest.skip("matplotlib not installed")

        visualizer = ChartVisualizer()
        fig = await visualizer.create_emotion_bar_chart(sample_analysis)

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    @pytest.mark.asyncio
    async def test_create_sentiment_pie_chart(self, sample_analysis, has_matplotlib):
        """Test creating sentiment pie chart."""
        if not has_matplotlib:
            pytest.skip("matplotlib not installed")

        visualizer = ChartVisualizer()
        fig = await visualizer.create_sentiment_pie_chart(sample_analysis)

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    @pytest.mark.asyncio
    async def test_create_donut_chart(self, sample_analysis, has_matplotlib):
        """Test creating donut chart."""
        if not has_matplotlib:
            pytest.skip("matplotlib not installed")

        visualizer = ChartVisualizer()
        fig = await visualizer.create_sentiment_pie_chart(sample_analysis, donut=True)

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    @pytest.mark.asyncio
    async def test_create_score_histogram(self, has_matplotlib):
        """Test creating score histogram."""
        if not has_matplotlib:
            pytest.skip("matplotlib not installed")

        visualizer = ChartVisualizer()
        scores = [0.1, 0.3, 0.5, 0.7, 0.9, 0.4, 0.6, 0.8, 0.2, 0.5]
        fig = await visualizer.create_score_histogram(scores)

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    @pytest.mark.asyncio
    async def test_create_bar_chart_generic(self, has_matplotlib):
        """Test creating generic bar chart."""
        if not has_matplotlib:
            pytest.skip("matplotlib not installed")

        visualizer = ChartVisualizer()
        data = {"Category A": 30, "Category B": 45, "Category C": 25}
        fig = await visualizer.create_bar_chart(data)

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    @pytest.mark.asyncio
    async def test_create_horizontal_bar_chart(self, has_matplotlib):
        """Test creating horizontal bar chart."""
        if not has_matplotlib:
            pytest.skip("matplotlib not installed")

        visualizer = ChartVisualizer()
        data = {"Item 1": 100, "Item 2": 80, "Item 3": 60}
        fig = await visualizer.create_horizontal_bar_chart(data)

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    @pytest.mark.asyncio
    async def test_create_line_chart(self, has_matplotlib):
        """Test creating line chart."""
        if not has_matplotlib:
            pytest.skip("matplotlib not installed")

        visualizer = ChartVisualizer()
        data = {"Series A": [1, 2, 3, 4, 5], "Series B": [2, 3, 4, 5, 6]}
        fig = await visualizer.create_line_chart(data)

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    @pytest.mark.asyncio
    async def test_save_chart(self, sample_analysis, temp_dir, has_matplotlib):
        """Test saving chart to file."""
        if not has_matplotlib:
            pytest.skip("matplotlib not installed")

        visualizer = ChartVisualizer()
        fig = await visualizer.create_sentiment_bar_chart(sample_analysis)
        path = os.path.join(temp_dir, "chart.png")

        result = await visualizer.save(fig, path)

        assert result.success is True
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    @pytest.mark.asyncio
    async def test_save_chart_svg(self, sample_analysis, temp_dir, has_matplotlib):
        """Test saving chart as SVG."""
        if not has_matplotlib:
            pytest.skip("matplotlib not installed")

        visualizer = ChartVisualizer()
        fig = await visualizer.create_sentiment_bar_chart(sample_analysis)
        path = os.path.join(temp_dir, "chart.svg")

        result = await visualizer.save(fig, path, format="svg")

        assert result.success is True
        assert os.path.exists(path)

    @pytest.mark.asyncio
    async def test_to_bytes(self, sample_analysis, has_matplotlib):
        """Test converting chart to bytes."""
        if not has_matplotlib:
            pytest.skip("matplotlib not installed")

        visualizer = ChartVisualizer()
        fig = await visualizer.create_sentiment_bar_chart(sample_analysis)

        image_bytes = await visualizer.to_bytes(fig)

        assert len(image_bytes) > 0
        # PNG magic bytes
        assert image_bytes[:4] == b'\x89PNG'

    @pytest.mark.asyncio
    async def test_dark_theme(self, sample_analysis, has_matplotlib):
        """Test chart with dark theme."""
        if not has_matplotlib:
            pytest.skip("matplotlib not installed")

        options = VisualizationOptions(theme=Theme.DARK)
        visualizer = ChartVisualizer(options)
        fig = await visualizer.create_sentiment_bar_chart(sample_analysis)

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    @pytest.mark.asyncio
    async def test_colorful_theme(self, sample_analysis, has_matplotlib):
        """Test chart with colorful theme."""
        if not has_matplotlib:
            pytest.skip("matplotlib not installed")

        options = VisualizationOptions(theme=Theme.COLORFUL)
        visualizer = ChartVisualizer(options)
        fig = await visualizer.create_sentiment_bar_chart(sample_analysis)

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    @pytest.mark.asyncio
    async def test_custom_color_palette(self, sample_analysis, has_matplotlib):
        """Test chart with custom color palette."""
        if not has_matplotlib:
            pytest.skip("matplotlib not installed")

        options = VisualizationOptions(
            color_palette=["#ff6b6b", "#4ecdc4", "#45b7d1"]
        )
        visualizer = ChartVisualizer(options)
        fig = await visualizer.create_sentiment_bar_chart(sample_analysis)

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


# ============================================================================
# Comparison Chart Tests
# ============================================================================


class TestComparisonChart:
    """Tests for comparison chart."""

    @dataclass
    class MockComparisonResult:
        """Mock comparison result."""
        item_a: str = "Product A"
        item_b: str = "Product B"
        item_a_analysis: Any = None
        item_b_analysis: Any = None

    @pytest.fixture
    def comparison_data(self, sample_analysis):
        """Create comparison data."""
        analysis_b = MockAnalysisResult(
            total_count=80,
            positive_ratio=0.55,
            negative_ratio=0.25,
            average_polarity=0.30,
        )
        result = self.MockComparisonResult()
        result.item_a_analysis = sample_analysis
        result.item_b_analysis = analysis_b
        return result

    @pytest.mark.asyncio
    async def test_create_comparison_chart(self, comparison_data, has_matplotlib):
        """Test creating comparison chart."""
        if not has_matplotlib:
            pytest.skip("matplotlib not installed")

        visualizer = ChartVisualizer()
        fig = await visualizer.create_comparison_chart(comparison_data)

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Tests for convenience visualization functions."""

    @pytest.mark.asyncio
    async def test_get_visualizer(self, has_matplotlib):
        """Test get_visualizer function."""
        visualizer = get_visualizer(theme="dark")
        assert isinstance(visualizer, ChartVisualizer)
        assert visualizer.options.theme == Theme.DARK

    @pytest.mark.asyncio
    async def test_create_sentiment_chart_bar(self, sample_analysis, temp_dir, has_matplotlib):
        """Test create_sentiment_chart for bar chart."""
        if not has_matplotlib:
            pytest.skip("matplotlib not installed")

        path = os.path.join(temp_dir, "sentiment.png")
        result = await create_sentiment_chart(sample_analysis, path, chart_type="bar")

        assert result.success is True
        assert os.path.exists(path)

    @pytest.mark.asyncio
    async def test_create_sentiment_chart_pie(self, sample_analysis, temp_dir, has_matplotlib):
        """Test create_sentiment_chart for pie chart."""
        if not has_matplotlib:
            pytest.skip("matplotlib not installed")

        path = os.path.join(temp_dir, "pie.png")
        result = await create_sentiment_chart(sample_analysis, path, chart_type="pie")

        assert result.success is True
        assert os.path.exists(path)

    @pytest.mark.asyncio
    async def test_create_sentiment_chart_donut(self, sample_analysis, temp_dir, has_matplotlib):
        """Test create_sentiment_chart for donut chart."""
        if not has_matplotlib:
            pytest.skip("matplotlib not installed")

        path = os.path.join(temp_dir, "donut.png")
        result = await create_sentiment_chart(sample_analysis, path, chart_type="donut")

        assert result.success is True
        assert os.path.exists(path)

    @pytest.mark.asyncio
    async def test_create_emotion_chart_function(self, sample_analysis, temp_dir, has_matplotlib):
        """Test create_emotion_chart convenience function."""
        if not has_matplotlib:
            pytest.skip("matplotlib not installed")

        path = os.path.join(temp_dir, "emotions.png")
        result = await create_emotion_chart(sample_analysis, path)

        assert result.success is True
        assert os.path.exists(path)

    @pytest.mark.asyncio
    async def test_create_sentiment_chart_no_path(self, sample_analysis, has_matplotlib):
        """Test create_sentiment_chart without path returns result."""
        if not has_matplotlib:
            pytest.skip("matplotlib not installed")

        result = await create_sentiment_chart(sample_analysis, chart_type="bar")

        assert result.success is True


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_scores_histogram(self, has_matplotlib):
        """Test histogram with empty scores."""
        if not has_matplotlib:
            pytest.skip("matplotlib not installed")

        visualizer = ChartVisualizer()
        scores = []

        # Should handle empty gracefully
        try:
            fig = await visualizer.create_score_histogram(scores)
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass  # Empty data may raise exception, which is fine

    @pytest.mark.asyncio
    async def test_single_value_data(self, has_matplotlib):
        """Test chart with single value."""
        if not has_matplotlib:
            pytest.skip("matplotlib not installed")

        visualizer = ChartVisualizer()
        data = {"Single": 100}
        fig = await visualizer.create_bar_chart(data)

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    @pytest.mark.asyncio
    async def test_large_dataset(self, has_matplotlib):
        """Test chart with large dataset."""
        if not has_matplotlib:
            pytest.skip("matplotlib not installed")

        visualizer = ChartVisualizer()
        data = {f"Item {i}": i * 10 for i in range(20)}
        fig = await visualizer.create_horizontal_bar_chart(data)

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    @pytest.mark.asyncio
    async def test_negative_values(self, has_matplotlib):
        """Test chart with negative values."""
        if not has_matplotlib:
            pytest.skip("matplotlib not installed")

        visualizer = ChartVisualizer()
        data = {"Positive": 50, "Neutral": 0, "Negative": -30}
        fig = await visualizer.create_bar_chart(data)

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    @pytest.mark.asyncio
    async def test_zero_ratios(self, has_matplotlib):
        """Test sentiment chart with zero ratios."""
        if not has_matplotlib:
            pytest.skip("matplotlib not installed")

        visualizer = ChartVisualizer()
        data = MockAnalysisResult(
            total_count=0,
            positive_ratio=0,
            negative_ratio=0,
            average_polarity=0,
        )
        fig = await visualizer.create_sentiment_bar_chart(data)

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    @pytest.mark.asyncio
    async def test_save_creates_directory(self, sample_analysis, temp_dir, has_matplotlib):
        """Test that save creates parent directories."""
        if not has_matplotlib:
            pytest.skip("matplotlib not installed")

        visualizer = ChartVisualizer()
        fig = await visualizer.create_sentiment_bar_chart(sample_analysis)
        path = os.path.join(temp_dir, "subdir", "nested", "chart.png")

        result = await visualizer.save(fig, path)

        assert result.success is True
        assert os.path.exists(path)

    @pytest.mark.asyncio
    async def test_invalid_chart_type(self, sample_analysis, has_matplotlib):
        """Test invalid chart type."""
        if not has_matplotlib:
            pytest.skip("matplotlib not installed")

        with pytest.raises(ValueError):
            await create_sentiment_chart(sample_analysis, chart_type="invalid")
