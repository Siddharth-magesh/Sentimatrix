"""
Unit tests for Sentimatrix Output Formatters.

Tests:
- HTMLFormatter: HTML report generation
- TextFormatter: Plain text formatting
- MarkdownFormatter: Markdown formatting
"""

import asyncio
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest

from sentimatrix.output.formatters import (
    FormatOptions,
    FormatResult,
    HTMLFormatter,
    TextFormatter,
    MarkdownFormatter,
    get_formatter,
    format_as_html,
    format_as_text,
    format_as_markdown,
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
    sentiment_summary: Optional[Dict[str, Any]] = None
    emotion_summary: Optional[Dict[str, Any]] = None
    reviews: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_count": self.total_count,
            "positive_ratio": self.positive_ratio,
            "negative_ratio": self.negative_ratio,
            "average_polarity": self.average_polarity,
            "sentiment_summary": self.sentiment_summary,
            "emotion_summary": self.emotion_summary,
            "reviews": self.reviews,
        }


@pytest.fixture
def sample_analysis():
    """Create sample analysis result for testing."""
    return MockAnalysisResult(
        total_count=100,
        positive_ratio=0.65,
        negative_ratio=0.15,
        average_polarity=0.45,
        sentiment_summary={
            "positive_count": 65,
            "negative_count": 15,
            "neutral_count": 20,
        },
        emotion_summary={
            "distribution": {
                "joy": 0.4,
                "anger": 0.1,
                "sadness": 0.15,
                "surprise": 0.2,
            }
        },
        reviews=[
            {
                "text": "Great product!",
                "sentiment": {"sentiment": "positive", "confidence": 0.95},
            },
            {
                "text": "Not worth it.",
                "sentiment": {"sentiment": "negative", "confidence": 0.88},
            },
        ],
    )


@pytest.fixture
def sample_insights():
    """Create sample insights for testing."""
    return {
        "summary": "Overall positive reception with some concerns about price.",
        "pros": ["Great quality", "Fast shipping", "Good customer service"],
        "cons": ["High price", "Limited colors"],
        "themes": ["quality", "value", "shipping"],
        "recommendations": ["Consider offering discounts", "Add more color options"],
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ============================================================================
# FormatOptions Tests
# ============================================================================


class TestFormatOptions:
    """Tests for FormatOptions dataclass."""

    def test_default_values(self):
        """Test default option values."""
        options = FormatOptions()
        assert options.title == "Sentimatrix Analysis Report"
        assert options.include_charts is True
        assert options.theme == "default"

    def test_custom_values(self):
        """Test custom option values."""
        options = FormatOptions(
            title="Custom Report",
            theme="dark",
            include_raw_data=True,
            max_text_length=200,
        )
        assert options.title == "Custom Report"
        assert options.theme == "dark"
        assert options.include_raw_data is True


# ============================================================================
# HTMLFormatter Tests
# ============================================================================


class TestHTMLFormatter:
    """Tests for HTMLFormatter."""

    @pytest.mark.asyncio
    async def test_format_dict(self):
        """Test formatting a dictionary."""
        formatter = HTMLFormatter()
        data = {"key": "value", "number": 42}

        html = await formatter.format(data)

        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "key" in html
        assert "value" in html

    @pytest.mark.asyncio
    async def test_format_analysis_result(self, sample_analysis):
        """Test formatting analysis result."""
        formatter = HTMLFormatter()
        html = await formatter.format(sample_analysis)

        assert "<!DOCTYPE html>" in html
        assert "65.0%" in html  # positive_ratio
        assert "Total" in html or "100" in html

    @pytest.mark.asyncio
    async def test_format_with_custom_title(self, sample_analysis):
        """Test formatting with custom title."""
        options = FormatOptions(title="My Custom Report")
        formatter = HTMLFormatter(options)

        html = await formatter.format(sample_analysis)

        assert "My Custom Report" in html

    @pytest.mark.asyncio
    async def test_format_with_dark_theme(self, sample_analysis):
        """Test formatting with dark theme."""
        options = FormatOptions(theme="dark")
        formatter = HTMLFormatter(options)

        html = await formatter.format(sample_analysis)

        assert "<!DOCTYPE html>" in html
        # Dark theme CSS should be included
        assert "#1a1a2e" in html or "--bg-primary" in html

    @pytest.mark.asyncio
    async def test_format_insights(self, sample_insights):
        """Test formatting insights with pros/cons."""
        formatter = HTMLFormatter()
        html = await formatter.format(sample_insights)

        assert "Great quality" in html
        assert "High price" in html
        assert "Pros" in html or "pros" in html

    @pytest.mark.asyncio
    async def test_format_reviews_table(self, sample_analysis):
        """Test that reviews are formatted as table."""
        formatter = HTMLFormatter()
        html = await formatter.format(sample_analysis)

        assert "Great product!" in html
        assert "<table" in html

    @pytest.mark.asyncio
    async def test_save_to_file(self, sample_analysis, temp_dir):
        """Test saving formatted HTML to file."""
        formatter = HTMLFormatter()
        html = await formatter.format(sample_analysis)
        path = os.path.join(temp_dir, "report.html")

        result = await formatter.save(html, path)

        assert result is True
        assert os.path.exists(path)

        with open(path) as f:
            content = f.read()
        assert "<!DOCTYPE html>" in content

    @pytest.mark.asyncio
    async def test_format_empty_data(self):
        """Test formatting empty data."""
        formatter = HTMLFormatter()
        html = await formatter.format({})

        assert "<!DOCTYPE html>" in html

    @pytest.mark.asyncio
    async def test_format_list(self):
        """Test formatting a list."""
        formatter = HTMLFormatter()
        data = [{"id": 1, "text": "Item 1"}, {"id": 2, "text": "Item 2"}]

        html = await formatter.format(data)

        assert "<!DOCTYPE html>" in html
        assert "Item 1" in html


# ============================================================================
# TextFormatter Tests
# ============================================================================


class TestTextFormatter:
    """Tests for TextFormatter."""

    @pytest.mark.asyncio
    async def test_format_dict(self):
        """Test formatting a dictionary."""
        formatter = TextFormatter()
        data = {"key": "value", "number": 42}

        text = await formatter.format(data)

        assert "key" in text
        assert "value" in text
        assert "42" in text

    @pytest.mark.asyncio
    async def test_format_analysis_result(self, sample_analysis):
        """Test formatting analysis result."""
        formatter = TextFormatter()
        text = await formatter.format(sample_analysis)

        assert "100" in text  # total_count
        assert "=" in text  # separator

    @pytest.mark.asyncio
    async def test_format_with_title(self, sample_analysis):
        """Test formatting with custom title."""
        options = FormatOptions(title="Custom Title")
        formatter = TextFormatter(options)

        text = await formatter.format(sample_analysis)

        assert "Custom Title" in text

    @pytest.mark.asyncio
    async def test_format_list(self):
        """Test formatting a list."""
        formatter = TextFormatter()
        data = [{"id": 1}, {"id": 2}, {"id": 3}]

        text = await formatter.format(data)

        assert "1" in text
        assert "2" in text


# ============================================================================
# MarkdownFormatter Tests
# ============================================================================


class TestMarkdownFormatter:
    """Tests for MarkdownFormatter."""

    @pytest.mark.asyncio
    async def test_format_dict(self):
        """Test formatting a dictionary."""
        formatter = MarkdownFormatter()
        data = {"key": "value", "number": 42}

        md = await formatter.format(data)

        assert "#" in md  # Has headings
        assert "key" in md.lower() or "Key" in md

    @pytest.mark.asyncio
    async def test_format_analysis_result(self, sample_analysis):
        """Test formatting analysis result."""
        formatter = MarkdownFormatter()
        md = await formatter.format(sample_analysis)

        assert "# " in md  # H1 heading
        assert "100" in md
        # Should have table for summary
        assert "|" in md

    @pytest.mark.asyncio
    async def test_format_with_title(self, sample_analysis):
        """Test formatting with custom title."""
        options = FormatOptions(title="Analysis Report")
        formatter = MarkdownFormatter(options)

        md = await formatter.format(sample_analysis)

        assert "# Analysis Report" in md

    @pytest.mark.asyncio
    async def test_format_summary_table(self, sample_analysis):
        """Test that summary is formatted as table."""
        formatter = MarkdownFormatter()
        md = await formatter.format(sample_analysis)

        # Markdown table format
        assert "|" in md
        assert "---" in md or "Metric" in md

    @pytest.mark.asyncio
    async def test_format_list(self):
        """Test formatting a list."""
        formatter = MarkdownFormatter()
        data = [{"text": "Item 1"}, {"text": "Item 2"}]

        md = await formatter.format(data)

        assert "- " in md  # List items


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Tests for convenience formatting functions."""

    @pytest.mark.asyncio
    async def test_get_formatter_html(self):
        """Test get_formatter for HTML."""
        formatter = get_formatter("html")
        assert isinstance(formatter, HTMLFormatter)

    @pytest.mark.asyncio
    async def test_get_formatter_text(self):
        """Test get_formatter for text."""
        formatter = get_formatter("text")
        assert isinstance(formatter, TextFormatter)

    @pytest.mark.asyncio
    async def test_get_formatter_markdown(self):
        """Test get_formatter for markdown."""
        formatter = get_formatter("markdown")
        assert isinstance(formatter, MarkdownFormatter)

    @pytest.mark.asyncio
    async def test_get_formatter_md_alias(self):
        """Test get_formatter with 'md' alias."""
        formatter = get_formatter("md")
        assert isinstance(formatter, MarkdownFormatter)

    @pytest.mark.asyncio
    async def test_format_as_html_function(self, sample_analysis):
        """Test format_as_html convenience function."""
        html = await format_as_html(sample_analysis, title="Quick Report")

        assert "<!DOCTYPE html>" in html
        assert "Quick Report" in html

    @pytest.mark.asyncio
    async def test_format_as_text_function(self, sample_analysis):
        """Test format_as_text convenience function."""
        text = await format_as_text(sample_analysis, title="Quick Report")

        assert "Quick Report" in text

    @pytest.mark.asyncio
    async def test_format_as_markdown_function(self, sample_analysis):
        """Test format_as_markdown convenience function."""
        md = await format_as_markdown(sample_analysis, title="Quick Report")

        assert "# Quick Report" in md

    @pytest.mark.asyncio
    async def test_get_formatter_invalid(self):
        """Test get_formatter with invalid format."""
        with pytest.raises(ValueError):
            get_formatter("invalid_format")


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_format_unicode(self):
        """Test formatting unicode characters."""
        formatter = HTMLFormatter()
        data = {"text": "Hello 世界 🌍"}

        html = await formatter.format(data)

        # Unicode should be escaped or preserved
        assert "Hello" in html

    @pytest.mark.asyncio
    async def test_format_html_special_chars(self):
        """Test formatting with HTML special characters."""
        formatter = HTMLFormatter()
        data = {"text": "<script>alert('xss')</script>"}

        html = await formatter.format(data)

        # Should be escaped
        assert "&lt;script" in html or "<script>" not in html

    @pytest.mark.asyncio
    async def test_format_long_text(self):
        """Test formatting with long text (should truncate)."""
        formatter = HTMLFormatter(FormatOptions(max_text_length=50))
        data = {"text": "A" * 1000}

        html = await formatter.format(data)

        # Text should be truncated or handled
        assert len(html) < 10000

    @pytest.mark.asyncio
    async def test_format_none_values(self):
        """Test formatting data with None values."""
        formatter = HTMLFormatter()
        data = {"key": None, "value": "test"}

        html = await formatter.format(data)

        assert "<!DOCTYPE html>" in html

    @pytest.mark.asyncio
    async def test_save_creates_directory(self, temp_dir):
        """Test that save creates parent directories."""
        formatter = HTMLFormatter()
        html = "<html><body>Test</body></html>"
        path = os.path.join(temp_dir, "subdir", "nested", "report.html")

        result = await formatter.save(html, path)

        assert result is True
        assert os.path.exists(path)

    @pytest.mark.asyncio
    async def test_format_datetime(self):
        """Test formatting datetime values."""
        formatter = HTMLFormatter()
        data = {"timestamp": datetime(2024, 1, 15, 10, 30)}

        html = await formatter.format(data)

        assert "2024" in html

    @pytest.mark.asyncio
    async def test_format_nested_structure(self):
        """Test formatting deeply nested structure."""
        formatter = HTMLFormatter()
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": "deep"
                    }
                }
            }
        }

        html = await formatter.format(data)

        assert "<!DOCTYPE html>" in html
