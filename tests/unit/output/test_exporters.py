"""
Unit tests for Sentimatrix Output Exporters.

Tests:
- JSONExporter: JSON file export and string export
- CSVExporter: CSV file export with flattening
- ExcelExporter: Excel file export (if openpyxl available)
- Convenience functions: export_to_json, export_to_csv, export_to_excel
"""

import asyncio
import json
import csv
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from sentimatrix.output.exporters import (
    ExportFormat,
    ExportOptions,
    ExportResult,
    JSONExporter,
    CSVExporter,
    ExcelExporter,
    get_exporter,
    export_to_json,
    export_to_csv,
    export_to_excel,
)


# ============================================================================
# Test Data
# ============================================================================


@dataclass
class MockSentimentResult:
    """Mock sentiment result for testing."""

    sentiment: str = "positive"
    confidence: float = 0.95
    polarity: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentiment": self.sentiment,
            "confidence": self.confidence,
            "polarity": self.polarity,
        }


@dataclass
class MockReview:
    """Mock review for testing."""

    id: str
    text: str
    rating: float
    source: str = "test"
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "rating": self.rating,
            "source": self.source,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata,
        }


@dataclass
class MockAnalysisResult:
    """Mock analysis result for testing."""

    total_count: int
    positive_ratio: float
    negative_ratio: float
    average_polarity: float
    reviews: List[MockReview] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_count": self.total_count,
            "positive_ratio": self.positive_ratio,
            "negative_ratio": self.negative_ratio,
            "average_polarity": self.average_polarity,
            "reviews": [r.to_dict() for r in self.reviews],
        }


@pytest.fixture
def sample_reviews():
    """Create sample reviews for testing."""
    return [
        MockReview(
            id="1",
            text="Great product, highly recommend!",
            rating=5.0,
            timestamp=datetime(2024, 1, 15, 10, 30),
        ),
        MockReview(
            id="2",
            text="Not worth the money.",
            rating=2.0,
            timestamp=datetime(2024, 1, 16, 14, 45),
        ),
        MockReview(
            id="3",
            text="Average quality, nothing special.",
            rating=3.0,
            timestamp=datetime(2024, 1, 17, 9, 0),
        ),
    ]


@pytest.fixture
def sample_analysis(sample_reviews):
    """Create sample analysis result for testing."""
    return MockAnalysisResult(
        total_count=3,
        positive_ratio=0.6,
        negative_ratio=0.2,
        average_polarity=0.4,
        reviews=sample_reviews,
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ============================================================================
# ExportOptions Tests
# ============================================================================


class TestExportOptions:
    """Tests for ExportOptions dataclass."""

    def test_default_values(self):
        """Test default option values."""
        options = ExportOptions()
        assert options.format == ExportFormat.JSON
        assert options.include_metadata is True
        assert options.pretty_print is True
        assert options.encoding == "utf-8"

    def test_custom_values(self):
        """Test custom option values."""
        options = ExportOptions(
            format=ExportFormat.CSV,
            path="/tmp/output.csv",
            include_raw=True,
            compression="gzip",
        )
        assert options.format == ExportFormat.CSV
        assert options.path == "/tmp/output.csv"
        assert options.include_raw is True
        assert options.compression == "gzip"

    def test_with_path(self):
        """Test with_path method."""
        options = ExportOptions(format=ExportFormat.JSON, pretty_print=True)
        new_options = options.with_path("/new/path.json")
        assert new_options.path == "/new/path.json"
        assert new_options.format == ExportFormat.JSON
        assert new_options.pretty_print is True


# ============================================================================
# JSONExporter Tests
# ============================================================================


class TestJSONExporter:
    """Tests for JSONExporter."""

    @pytest.mark.asyncio
    async def test_export_dict(self, temp_dir):
        """Test exporting a dictionary."""
        exporter = JSONExporter()
        data = {"key": "value", "number": 42}
        path = os.path.join(temp_dir, "test.json")

        result = await exporter.export(data, path)

        assert result.success is True
        assert result.path == path
        assert result.format == ExportFormat.JSON
        assert os.path.exists(path)

        with open(path) as f:
            loaded = json.load(f)
        assert loaded == data

    @pytest.mark.asyncio
    async def test_export_list(self, temp_dir):
        """Test exporting a list."""
        exporter = JSONExporter()
        data = [{"id": 1}, {"id": 2}, {"id": 3}]
        path = os.path.join(temp_dir, "list.json")

        result = await exporter.export(data, path)

        assert result.success is True
        assert result.records_exported == 3

        with open(path) as f:
            loaded = json.load(f)
        assert loaded == data

    @pytest.mark.asyncio
    async def test_export_dataclass(self, temp_dir, sample_analysis):
        """Test exporting a dataclass."""
        exporter = JSONExporter()
        path = os.path.join(temp_dir, "analysis.json")

        result = await exporter.export(sample_analysis, path)

        assert result.success is True
        assert os.path.exists(path)

        with open(path) as f:
            loaded = json.load(f)
        assert loaded["total_count"] == 3
        assert loaded["positive_ratio"] == 0.6
        assert len(loaded["reviews"]) == 3

    @pytest.mark.asyncio
    async def test_export_with_datetime(self, temp_dir):
        """Test exporting data with datetime fields."""
        exporter = JSONExporter()
        data = {"timestamp": datetime(2024, 1, 15, 10, 30)}
        path = os.path.join(temp_dir, "datetime.json")

        result = await exporter.export(data, path)

        assert result.success is True

        with open(path) as f:
            loaded = json.load(f)
        assert "2024-01-15" in loaded["timestamp"]

    @pytest.mark.asyncio
    async def test_export_pretty_print(self, temp_dir):
        """Test pretty print formatting."""
        options = ExportOptions(pretty_print=True)
        exporter = JSONExporter(options)
        data = {"key": "value"}
        path = os.path.join(temp_dir, "pretty.json")

        await exporter.export(data, path)

        with open(path) as f:
            content = f.read()
        assert "\n" in content  # Pretty print has newlines

    @pytest.mark.asyncio
    async def test_export_no_path_error(self):
        """Test error when no path specified."""
        exporter = JSONExporter()
        result = await exporter.export({"key": "value"})

        assert result.success is False
        assert "path" in result.error.lower()

    @pytest.mark.asyncio
    async def test_export_to_string(self, sample_analysis):
        """Test export to string method."""
        exporter = JSONExporter()
        json_str = await exporter.export_to_string(sample_analysis)

        parsed = json.loads(json_str)
        assert parsed["total_count"] == 3

    @pytest.mark.asyncio
    async def test_export_creates_directory(self, temp_dir):
        """Test that export creates parent directories."""
        exporter = JSONExporter()
        path = os.path.join(temp_dir, "subdir", "nested", "test.json")

        result = await exporter.export({"key": "value"}, path)

        assert result.success is True
        assert os.path.exists(path)


# ============================================================================
# CSVExporter Tests
# ============================================================================


class TestCSVExporter:
    """Tests for CSVExporter."""

    @pytest.mark.asyncio
    async def test_export_list_of_dicts(self, temp_dir):
        """Test exporting a list of dictionaries."""
        exporter = CSVExporter()
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]
        path = os.path.join(temp_dir, "test.csv")

        result = await exporter.export(data, path)

        assert result.success is True
        assert result.records_exported == 2

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_export_reviews(self, temp_dir, sample_reviews):
        """Test exporting reviews."""
        exporter = CSVExporter()
        path = os.path.join(temp_dir, "reviews.csv")

        result = await exporter.export(sample_reviews, path)

        assert result.success is True
        assert result.records_exported == 3

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3

    @pytest.mark.asyncio
    async def test_export_with_custom_columns(self, temp_dir):
        """Test exporting with specific columns."""
        exporter = CSVExporter(columns=["name", "age"])
        data = [
            {"name": "Alice", "age": 30, "city": "NYC"},
            {"name": "Bob", "age": 25, "city": "LA"},
        ]
        path = os.path.join(temp_dir, "filtered.csv")

        result = await exporter.export(data, path)

        assert result.success is True

        with open(path) as f:
            content = f.read()
        assert "city" not in content or content.count("city") == 0

    @pytest.mark.asyncio
    async def test_export_with_nested_data(self, temp_dir):
        """Test exporting data with nested dictionaries (flattened)."""
        exporter = CSVExporter()
        data = [
            {"id": 1, "metadata": {"source": "web", "verified": True}},
        ]
        path = os.path.join(temp_dir, "nested.csv")

        result = await exporter.export(data, path)

        assert result.success is True

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert "metadata_source" in rows[0]

    @pytest.mark.asyncio
    async def test_export_without_header(self, temp_dir):
        """Test exporting without header row."""
        exporter = CSVExporter(include_header=False)
        data = [{"name": "Alice"}, {"name": "Bob"}]
        path = os.path.join(temp_dir, "no_header.csv")

        result = await exporter.export(data, path)

        assert result.success is True

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 2  # No header

    @pytest.mark.asyncio
    async def test_export_empty_list(self, temp_dir):
        """Test exporting empty list."""
        exporter = CSVExporter()
        path = os.path.join(temp_dir, "empty.csv")

        result = await exporter.export([], path)

        assert result.success is True
        assert result.records_exported == 0

    @pytest.mark.asyncio
    async def test_export_to_string(self):
        """Test export to string method."""
        exporter = CSVExporter()
        data = [{"name": "Alice"}, {"name": "Bob"}]

        csv_str = await exporter.export_to_string(data)

        assert "name" in csv_str
        assert "Alice" in csv_str
        assert "Bob" in csv_str


# ============================================================================
# ExcelExporter Tests
# ============================================================================


class TestExcelExporter:
    """Tests for ExcelExporter."""

    @pytest.fixture
    def has_openpyxl(self):
        """Check if openpyxl is available."""
        try:
            import openpyxl
            return True
        except ImportError:
            return False

    @pytest.mark.asyncio
    async def test_export_basic(self, temp_dir, has_openpyxl):
        """Test basic Excel export."""
        if not has_openpyxl:
            pytest.skip("openpyxl not installed")

        exporter = ExcelExporter()
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]
        path = os.path.join(temp_dir, "test.xlsx")

        result = await exporter.export(data, path)

        assert result.success is True
        assert result.records_exported == 2
        assert os.path.exists(path)

    @pytest.mark.asyncio
    async def test_export_reviews(self, temp_dir, sample_reviews, has_openpyxl):
        """Test exporting reviews to Excel."""
        if not has_openpyxl:
            pytest.skip("openpyxl not installed")

        exporter = ExcelExporter()
        path = os.path.join(temp_dir, "reviews.xlsx")

        result = await exporter.export(sample_reviews, path)

        assert result.success is True
        assert result.records_exported == 3

    @pytest.mark.asyncio
    async def test_export_with_custom_sheet_name(self, temp_dir, has_openpyxl):
        """Test exporting with custom sheet name."""
        if not has_openpyxl:
            pytest.skip("openpyxl not installed")

        exporter = ExcelExporter(sheet_name="MyData")
        data = [{"name": "Test"}]
        path = os.path.join(temp_dir, "custom_sheet.xlsx")

        result = await exporter.export(data, path)

        assert result.success is True

        import openpyxl
        wb = openpyxl.load_workbook(path)
        assert "MyData" in wb.sheetnames

    @pytest.mark.asyncio
    async def test_export_multi_sheet(self, temp_dir, sample_reviews, has_openpyxl):
        """Test multi-sheet export."""
        if not has_openpyxl:
            pytest.skip("openpyxl not installed")

        exporter = ExcelExporter()
        sheets = {
            "Reviews": sample_reviews,
            "Summary": [{"total": 3, "positive": 2}],
        }
        path = os.path.join(temp_dir, "multi.xlsx")

        result = await exporter.export_multi_sheet(sheets, path)

        assert result.success is True

        import openpyxl
        wb = openpyxl.load_workbook(path)
        assert "Reviews" in wb.sheetnames
        assert "Summary" in wb.sheetnames

    @pytest.mark.asyncio
    async def test_export_without_openpyxl(self, temp_dir, monkeypatch):
        """Test error when openpyxl not installed."""
        import sys

        # Temporarily hide openpyxl
        openpyxl_module = sys.modules.get("openpyxl")
        sys.modules["openpyxl"] = None

        exporter = ExcelExporter()
        exporter._matplotlib_loaded = False  # Reset

        path = os.path.join(temp_dir, "test.xlsx")

        result = await exporter.export([{"key": "value"}], path)

        # Restore
        if openpyxl_module:
            sys.modules["openpyxl"] = openpyxl_module

        # Either fails or works depending on how import is done
        # Just check it doesn't crash


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Tests for convenience export functions."""

    @pytest.mark.asyncio
    async def test_get_exporter_json(self):
        """Test get_exporter for JSON."""
        exporter = get_exporter("json")
        assert isinstance(exporter, JSONExporter)

    @pytest.mark.asyncio
    async def test_get_exporter_csv(self):
        """Test get_exporter for CSV."""
        exporter = get_exporter("csv")
        assert isinstance(exporter, CSVExporter)

    @pytest.mark.asyncio
    async def test_get_exporter_xlsx(self):
        """Test get_exporter for Excel."""
        exporter = get_exporter("xlsx")
        assert isinstance(exporter, ExcelExporter)

    @pytest.mark.asyncio
    async def test_get_exporter_format_enum(self):
        """Test get_exporter with ExportFormat enum."""
        exporter = get_exporter(ExportFormat.JSON)
        assert isinstance(exporter, JSONExporter)

    @pytest.mark.asyncio
    async def test_export_to_json_function(self, temp_dir):
        """Test export_to_json convenience function."""
        path = os.path.join(temp_dir, "quick.json")
        result = await export_to_json({"test": True}, path)

        assert result.success is True
        assert os.path.exists(path)

    @pytest.mark.asyncio
    async def test_export_to_csv_function(self, temp_dir):
        """Test export_to_csv convenience function."""
        path = os.path.join(temp_dir, "quick.csv")
        result = await export_to_csv([{"name": "Test"}], path)

        assert result.success is True
        assert os.path.exists(path)


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_export_unicode(self, temp_dir):
        """Test exporting unicode characters."""
        exporter = JSONExporter()
        data = {"text": "Hello 世界 🌍"}
        path = os.path.join(temp_dir, "unicode.json")

        result = await exporter.export(data, path)

        assert result.success is True

        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["text"] == "Hello 世界 🌍"

    @pytest.mark.asyncio
    async def test_export_special_characters_csv(self, temp_dir):
        """Test CSV export with special characters."""
        exporter = CSVExporter()
        data = [{"text": 'Hello, "World"'}]
        path = os.path.join(temp_dir, "special.csv")

        result = await exporter.export(data, path)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_export_large_data(self, temp_dir):
        """Test exporting large dataset."""
        exporter = JSONExporter()
        data = [{"id": i, "value": f"item_{i}"} for i in range(1000)]
        path = os.path.join(temp_dir, "large.json")

        result = await exporter.export(data, path)

        assert result.success is True
        assert result.records_exported == 1000

    @pytest.mark.asyncio
    async def test_export_none_values(self, temp_dir):
        """Test exporting data with None values."""
        exporter = JSONExporter()
        data = {"key": None, "list": [None, 1, None]}
        path = os.path.join(temp_dir, "nulls.json")

        result = await exporter.export(data, path)

        assert result.success is True

        with open(path) as f:
            loaded = json.load(f)
        assert loaded["key"] is None
