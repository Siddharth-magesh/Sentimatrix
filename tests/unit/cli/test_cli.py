"""
Unit Tests for Sentimatrix CLI.

Tests the command-line interface functionality.
"""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

import sentimatrix
from sentimatrix.cli import (
    create_parser,
    main,
    print_output,
    print_error,
    print_success,
    __version__,
)


class TestCLIParser:
    """Tests for CLI argument parser."""

    def test_parser_creation(self):
        """Test parser is created successfully."""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "sentimatrix"

    def test_analyze_command_parsing(self):
        """Test analyze command parsing."""
        parser = create_parser()
        args = parser.parse_args(["analyze", "Test text"])
        assert args.command == "analyze"
        assert args.text == "Test text"
        assert args.emotions is False

    def test_analyze_with_emotions(self):
        """Test analyze command with emotions flag."""
        parser = create_parser()
        args = parser.parse_args(["analyze", "Test text", "--emotions"])
        assert args.emotions is True

    def test_analyze_with_output(self):
        """Test analyze command with output file."""
        parser = create_parser()
        args = parser.parse_args(["analyze", "Test", "-o", "output.json"])
        assert args.output == "output.json"

    def test_analyze_file_command(self):
        """Test analyze-file command parsing."""
        parser = create_parser()
        args = parser.parse_args(["analyze-file", "input.txt"])
        assert args.command == "analyze-file"
        assert args.input == "input.txt"

    def test_analyze_file_with_options(self):
        """Test analyze-file command with options."""
        parser = create_parser()
        args = parser.parse_args([
            "analyze-file", "input.csv",
            "-o", "output.json",
            "--emotions"
        ])
        assert args.input == "input.csv"
        assert args.output == "output.json"
        assert args.emotions is True

    def test_scrape_command(self):
        """Test scrape command parsing."""
        parser = create_parser()
        args = parser.parse_args(["scrape", "amazon", "B08N5WRWNW"])
        assert args.command == "scrape"
        assert args.platform == "amazon"
        assert args.identifier == "B08N5WRWNW"
        assert args.limit == 50  # default

    def test_scrape_with_limit(self):
        """Test scrape command with limit."""
        parser = create_parser()
        args = parser.parse_args(["scrape", "steam", "730", "--limit", "100"])
        assert args.platform == "steam"
        assert args.identifier == "730"
        assert args.limit == 100

    def test_scrape_with_analyze(self):
        """Test scrape command with analyze flag."""
        parser = create_parser()
        args = parser.parse_args(["scrape", "youtube", "dQw4w9WgXcQ", "--analyze"])
        assert args.analyze is True

    def test_scrape_platforms(self):
        """Test all supported platforms."""
        parser = create_parser()
        platforms = ["amazon", "steam", "youtube", "reddit"]
        for platform in platforms:
            args = parser.parse_args(["scrape", platform, "test_id"])
            assert args.platform == platform

    def test_batch_command(self):
        """Test batch command parsing."""
        parser = create_parser()
        args = parser.parse_args(["batch", "input.csv", "-o", "output.csv"])
        assert args.command == "batch"
        assert args.input == "input.csv"
        assert args.output == "output.csv"

    def test_info_command(self):
        """Test info command parsing."""
        parser = create_parser()
        args = parser.parse_args(["info"])
        assert args.command == "info"

    def test_info_with_json(self):
        """Test info command with JSON flag."""
        parser = create_parser()
        args = parser.parse_args(["info", "--json"])
        assert args.json is True


class TestCLIPrintFunctions:
    """Tests for CLI print helper functions."""

    def test_print_output(self, capsys):
        """Test print_output function."""
        print_output("Test message")
        captured = capsys.readouterr()
        assert "Test message" in captured.out

    def test_print_error(self, capsys):
        """Test print_error function."""
        print_error("Error message")
        captured = capsys.readouterr()
        # Error goes to stderr or stdout depending on rich availability
        assert "Error" in captured.out or "Error" in captured.err

    def test_print_success(self, capsys):
        """Test print_success function."""
        print_success("Success message")
        captured = capsys.readouterr()
        assert "Success message" in captured.out


class TestCLIVersion:
    """Tests for CLI version."""

    def test_version_defined(self):
        """Test version is defined."""
        assert __version__ is not None
        assert isinstance(__version__, str)

    def test_version_format(self):
        """Test version follows semver format."""
        parts = __version__.split(".")
        assert len(parts) >= 2
        assert all(part.isdigit() or "-" in part for part in parts[:3])


class TestCLICommands:
    """Tests for CLI command handlers."""

    @pytest.mark.asyncio
    async def test_analyze_text_mocked(self):
        """Test analyze_text function with mocked Sentimatrix."""
        from sentimatrix.cli import analyze_text

        mock_sentiment_result = MagicMock()
        mock_sentiment_result.sentiment.value = "positive"
        mock_sentiment_result.confidence = 0.95
        mock_sentiment_result.scores = {"positive": 0.95, "negative": 0.03, "neutral": 0.02}

        mock_sm = AsyncMock()
        mock_sm.analyze_sentiment.return_value = mock_sentiment_result
        mock_sm.__aenter__.return_value = mock_sm
        mock_sm.__aexit__.return_value = None

        with patch.object(sentimatrix, "Sentimatrix", return_value=mock_sm):
            result = await analyze_text("Great product!")

            assert result["text"] == "Great product!"
            assert result["sentiment"]["label"] == "positive"
            assert result["sentiment"]["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_analyze_text_with_emotions(self):
        """Test analyze_text with emotion detection."""
        from sentimatrix.cli import analyze_text

        mock_sentiment = MagicMock()
        mock_sentiment.sentiment.value = "positive"
        mock_sentiment.confidence = 0.9
        mock_sentiment.scores = {}

        mock_emotion = MagicMock()
        mock_emotion.primary_emotion = "joy"
        mock_emotion.emotions = [
            MagicMock(label="joy", score=0.85),
            MagicMock(label="surprise", score=0.1),
        ]

        mock_sm = AsyncMock()
        mock_sm.analyze_sentiment.return_value = mock_sentiment
        mock_sm.detect_emotions.return_value = mock_emotion
        mock_sm.__aenter__.return_value = mock_sm
        mock_sm.__aexit__.return_value = None

        with patch.object(sentimatrix, "Sentimatrix", return_value=mock_sm):
            result = await analyze_text("Great!", include_emotions=True)

            assert "emotions" in result
            assert result["emotions"]["primary"] == "joy"


class TestCLIMain:
    """Tests for main CLI entry point."""

    def test_main_no_args(self, capsys):
        """Test main with no arguments shows help."""
        with patch.object(sys, "argv", ["sentimatrix"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_main_version(self, capsys):
        """Test --version flag."""
        with patch.object(sys, "argv", ["sentimatrix", "--version"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
            captured = capsys.readouterr()
            assert __version__ in captured.out

    def test_main_info_command(self, capsys):
        """Test info command."""
        with patch.object(sys, "argv", ["sentimatrix", "info", "--json"]):
            main()
            captured = capsys.readouterr()
            info = json.loads(captured.out)
            assert "sentimatrix_version" in info
            assert "python_version" in info


class TestCLITableCreation:
    """Tests for table creation functions."""

    def test_create_sentiment_table(self, capsys):
        """Test sentiment table creation."""
        from sentimatrix.cli import create_sentiment_table

        results = [
            {
                "text": "Great product!",
                "sentiment": {"label": "positive", "confidence": 0.95},
            },
            {
                "text": "Bad experience.",
                "sentiment": {"label": "negative", "confidence": 0.85},
            },
        ]

        create_sentiment_table(results)
        captured = capsys.readouterr()
        # Should contain some output (table or plain text)
        assert len(captured.out) > 0

    def test_create_emotion_table(self, capsys):
        """Test emotion table creation."""
        from sentimatrix.cli import create_emotion_table

        results = [
            {
                "text": "I'm so happy!",
                "emotions": {
                    "primary": "joy",
                    "emotions": [{"label": "joy", "score": 0.9}],
                },
            },
        ]

        create_emotion_table(results)
        captured = capsys.readouterr()
        assert len(captured.out) > 0


class TestCLIFileOperations:
    """Tests for file-related CLI operations."""

    @pytest.mark.asyncio
    async def test_analyze_file_txt(self, tmp_path):
        """Test analyzing a text file."""
        from sentimatrix.cli import analyze_file

        # Create input file
        input_file = tmp_path / "input.txt"
        input_file.write_text("Great product!\nBad experience.\n")

        mock_sentiment = MagicMock()
        mock_sentiment.sentiment.value = "positive"
        mock_sentiment.confidence = 0.9
        mock_sentiment.scores = {}

        mock_sm = AsyncMock()
        mock_sm.analyze_sentiment.return_value = mock_sentiment
        mock_sm.__aenter__.return_value = mock_sm
        mock_sm.__aexit__.return_value = None

        with patch.object(sentimatrix, "Sentimatrix", return_value=mock_sm):
            results = await analyze_file(input_file)

            assert len(results) == 2
            assert results[0]["text"] == "Great product!"

    @pytest.mark.asyncio
    async def test_analyze_file_json_output(self, tmp_path):
        """Test analyzing and saving to JSON."""
        from sentimatrix.cli import analyze_file

        input_file = tmp_path / "input.txt"
        input_file.write_text("Test text\n")

        output_file = tmp_path / "output.json"

        mock_sentiment = MagicMock()
        mock_sentiment.sentiment.value = "neutral"
        mock_sentiment.confidence = 0.7
        mock_sentiment.scores = {}

        mock_sm = AsyncMock()
        mock_sm.analyze_sentiment.return_value = mock_sentiment
        mock_sm.__aenter__.return_value = mock_sm
        mock_sm.__aexit__.return_value = None

        with patch.object(sentimatrix, "Sentimatrix", return_value=mock_sm):
            await analyze_file(input_file, output_file)

            assert output_file.exists()
            with open(output_file) as f:
                data = json.load(f)
            assert len(data) == 1


class TestCLIScrapeCommand:
    """Tests for scrape command."""

    @pytest.mark.asyncio
    async def test_scrape_amazon(self):
        """Test scraping Amazon reviews."""
        from sentimatrix.cli import scrape_platform

        mock_review = MagicMock()
        mock_review.id = "R123"
        mock_review.text = "Great product!"
        mock_review.rating = 5.0
        mock_review.author = "TestUser"
        mock_review.source = "amazon"

        mock_sm = AsyncMock()
        mock_sm.scrape_amazon.return_value = [mock_review]
        mock_sm.__aenter__.return_value = mock_sm
        mock_sm.__aexit__.return_value = None

        with patch.object(sentimatrix, "Sentimatrix", return_value=mock_sm):
            results = await scrape_platform("amazon", "B08N5WRWNW", limit=10)

            assert len(results) == 1
            assert results[0]["id"] == "R123"
            assert results[0]["text"] == "Great product!"

    @pytest.mark.asyncio
    async def test_scrape_with_analyze(self):
        """Test scraping with analysis enabled."""
        from sentimatrix.cli import scrape_platform

        mock_review = MagicMock()
        mock_review.id = "R123"
        mock_review.text = "Great product!"
        mock_review.rating = 5.0
        mock_review.author = "TestUser"
        mock_review.source = "steam"

        mock_sentiment = MagicMock()
        mock_sentiment.sentiment.value = "positive"
        mock_sentiment.confidence = 0.95

        mock_sm = AsyncMock()
        mock_sm.scrape_steam.return_value = [mock_review]
        mock_sm.analyze_sentiment.return_value = mock_sentiment
        mock_sm.__aenter__.return_value = mock_sm
        mock_sm.__aexit__.return_value = None

        with patch.object(sentimatrix, "Sentimatrix", return_value=mock_sm):
            results = await scrape_platform("steam", "730", analyze=True)

            assert len(results) == 1
            assert "sentiment" in results[0]
            assert results[0]["sentiment"]["label"] == "positive"

    @pytest.mark.asyncio
    async def test_scrape_unsupported_platform(self):
        """Test scraping unsupported platform raises error."""
        from sentimatrix.cli import scrape_platform

        mock_sm = AsyncMock()
        mock_sm.__aenter__.return_value = mock_sm
        mock_sm.__aexit__.return_value = None

        with patch.object(sentimatrix, "Sentimatrix", return_value=mock_sm):
            with pytest.raises(ValueError, match="Unsupported platform"):
                await scrape_platform("twitter", "123")
