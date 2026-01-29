#!/usr/bin/env python3
"""
Sentimatrix CLI - Command Line Interface

A powerful command-line tool for sentiment analysis, emotion detection,
and review scraping.

Usage:
    sentimatrix analyze "Your text here"
    sentimatrix analyze-file input.txt -o results.json
    sentimatrix scrape amazon B08N5WRWNW --limit 50
    sentimatrix scrape steam 730 --analyze
    sentimatrix batch input.csv -o output.csv

For more information, run: sentimatrix --help
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Version
__version__ = "0.2.0"

# Console for rich output
console = Console() if RICH_AVAILABLE else None


def print_output(message: str, style: str = "") -> None:
    """Print output with optional rich formatting."""
    if RICH_AVAILABLE and console:
        console.print(message, style=style)
    else:
        print(message)


def print_error(message: str) -> None:
    """Print error message."""
    if RICH_AVAILABLE and console:
        console.print(f"[red]Error:[/red] {message}")
    else:
        print(f"Error: {message}", file=sys.stderr)


def print_success(message: str) -> None:
    """Print success message."""
    if RICH_AVAILABLE and console:
        console.print(f"[green]✓[/green] {message}")
    else:
        print(f"✓ {message}")


def create_sentiment_table(results: List[Dict[str, Any]]) -> None:
    """Create and display a sentiment results table."""
    if RICH_AVAILABLE and console:
        table = Table(title="Sentiment Analysis Results")
        table.add_column("Text", style="cyan", max_width=50)
        table.add_column("Sentiment", style="magenta")
        table.add_column("Confidence", justify="right", style="green")

        for result in results:
            text = result.get("text", "")[:50] + ("..." if len(result.get("text", "")) > 50 else "")
            sentiment = result.get("sentiment", {})
            if isinstance(sentiment, dict):
                label = sentiment.get("label", "unknown")
                confidence = sentiment.get("confidence", 0.0)
            else:
                label = str(sentiment)
                confidence = 0.0

            # Color based on sentiment
            if "positive" in label.lower():
                label_style = "[green]" + label + "[/green]"
            elif "negative" in label.lower():
                label_style = "[red]" + label + "[/red]"
            else:
                label_style = "[yellow]" + label + "[/yellow]"

            table.add_row(text, label_style, f"{confidence:.2%}")

        console.print(table)
    else:
        # Plain text output
        print("\nSentiment Analysis Results:")
        print("-" * 60)
        for result in results:
            text = result.get("text", "")[:50]
            sentiment = result.get("sentiment", {})
            if isinstance(sentiment, dict):
                label = sentiment.get("label", "unknown")
                confidence = sentiment.get("confidence", 0.0)
            else:
                label = str(sentiment)
                confidence = 0.0
            print(f"  {text}: {label} ({confidence:.2%})")


def create_emotion_table(results: List[Dict[str, Any]]) -> None:
    """Create and display an emotion results table."""
    if RICH_AVAILABLE and console:
        table = Table(title="Emotion Detection Results")
        table.add_column("Text", style="cyan", max_width=40)
        table.add_column("Primary Emotion", style="magenta")
        table.add_column("Top Emotions", style="blue")

        for result in results:
            text = result.get("text", "")[:40] + ("..." if len(result.get("text", "")) > 40 else "")
            emotions = result.get("emotions", {})

            if isinstance(emotions, dict):
                primary = emotions.get("primary", "unknown")
                top_emotions = emotions.get("emotions", [])[:3]
                emotion_str = ", ".join(
                    f"{e.get('label', 'unknown')}: {e.get('score', 0):.2f}"
                    for e in top_emotions
                ) if top_emotions else "-"
            else:
                primary = str(emotions)
                emotion_str = "-"

            table.add_row(text, primary, emotion_str)

        console.print(table)
    else:
        print("\nEmotion Detection Results:")
        print("-" * 60)
        for result in results:
            text = result.get("text", "")[:40]
            emotions = result.get("emotions", {})
            if isinstance(emotions, dict):
                primary = emotions.get("primary", "unknown")
            else:
                primary = str(emotions)
            print(f"  {text}: {primary}")


async def analyze_text(
    text: str,
    include_emotions: bool = False,
    model: Optional[str] = None,
    output_format: str = "table",
) -> Dict[str, Any]:
    """Analyze sentiment (and optionally emotions) of a single text."""
    from sentimatrix import Sentimatrix

    async with Sentimatrix() as sm:
        result = {"text": text}

        # Sentiment analysis
        sentiment_result = await sm.analyze_sentiment(text)
        result["sentiment"] = {
            "label": sentiment_result.sentiment.value,
            "confidence": sentiment_result.confidence,
            "scores": sentiment_result.scores,
        }

        # Emotion detection if requested
        if include_emotions:
            emotion_result = await sm.detect_emotions(text)
            result["emotions"] = {
                "primary": emotion_result.primary_emotion,
                "emotions": [
                    {"label": e.label, "score": e.score}
                    for e in emotion_result.emotions[:5]
                ],
            }

    return result


async def analyze_file(
    input_path: Path,
    output_path: Optional[Path] = None,
    include_emotions: bool = False,
    output_format: str = "json",
) -> List[Dict[str, Any]]:
    """Analyze sentiment of texts from a file."""
    from sentimatrix import Sentimatrix

    # Read input file
    texts = []
    suffix = input_path.suffix.lower()

    if suffix == ".json":
        with open(input_path) as f:
            data = json.load(f)
            if isinstance(data, list):
                texts = [item.get("text", item) if isinstance(item, dict) else str(item) for item in data]
            elif isinstance(data, dict) and "texts" in data:
                texts = data["texts"]
            else:
                texts = [str(data)]
    elif suffix == ".csv":
        import csv
        with open(input_path) as f:
            reader = csv.DictReader(f)
            texts = [row.get("text", row.get("content", list(row.values())[0])) for row in reader]
    else:
        # Plain text file - one text per line
        with open(input_path) as f:
            texts = [line.strip() for line in f if line.strip()]

    # Analyze
    results = []
    async with Sentimatrix() as sm:
        for text in texts:
            result = {"text": text}

            sentiment_result = await sm.analyze_sentiment(text)
            result["sentiment"] = {
                "label": sentiment_result.sentiment.value,
                "confidence": sentiment_result.confidence,
            }

            if include_emotions:
                emotion_result = await sm.detect_emotions(text)
                result["emotions"] = {
                    "primary": emotion_result.primary_emotion,
                    "emotions": [
                        {"label": e.label, "score": e.score}
                        for e in emotion_result.emotions[:5]
                    ],
                }

            results.append(result)

    # Output
    if output_path:
        suffix = output_path.suffix.lower()
        if suffix == ".json":
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
        elif suffix == ".csv":
            import csv
            with open(output_path, "w", newline="") as f:
                fieldnames = ["text", "sentiment_label", "sentiment_confidence"]
                if include_emotions:
                    fieldnames.append("primary_emotion")
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in results:
                    row = {
                        "text": r["text"],
                        "sentiment_label": r["sentiment"]["label"],
                        "sentiment_confidence": r["sentiment"]["confidence"],
                    }
                    if include_emotions:
                        row["primary_emotion"] = r["emotions"]["primary"]
                    writer.writerow(row)
        else:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

        print_success(f"Results saved to {output_path}")

    return results


async def scrape_platform(
    platform: str,
    identifier: str,
    limit: int = 50,
    analyze: bool = False,
    output_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Scrape reviews from a platform."""
    from sentimatrix import Sentimatrix

    async with Sentimatrix() as sm:
        # Scrape based on platform
        platform_lower = platform.lower()

        if platform_lower == "amazon":
            reviews = await sm.scrape_amazon(identifier, limit=limit)
        elif platform_lower == "steam":
            reviews = await sm.scrape_steam(identifier, limit=limit)
        elif platform_lower == "youtube":
            reviews = await sm.scrape_youtube(identifier, limit=limit)
        elif platform_lower == "reddit":
            reviews = await sm.scrape_reddit(identifier, limit=limit)
        else:
            raise ValueError(f"Unsupported platform: {platform}")

        # Convert to dicts
        results = []
        for review in reviews:
            result = {
                "id": review.id,
                "text": review.text,
                "rating": review.rating,
                "author": review.author,
                "source": review.source,
            }

            # Analyze if requested
            if analyze and review.text:
                sentiment_result = await sm.analyze_sentiment(review.text)
                result["sentiment"] = {
                    "label": sentiment_result.sentiment.value,
                    "confidence": sentiment_result.confidence,
                }

            results.append(result)

    # Output
    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print_success(f"Results saved to {output_path}")

    return results


async def run_batch(
    input_path: Path,
    output_path: Path,
    include_emotions: bool = False,
) -> None:
    """Run batch analysis on a CSV file."""
    results = await analyze_file(
        input_path,
        output_path,
        include_emotions=include_emotions,
    )
    print_success(f"Processed {len(results)} texts")


def cmd_analyze(args: argparse.Namespace) -> None:
    """Handle analyze command."""
    result = asyncio.run(analyze_text(
        args.text,
        include_emotions=args.emotions,
        model=args.model,
    ))

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print_success(f"Results saved to {output_path}")
    elif args.json:
        print(json.dumps(result, indent=2))
    else:
        create_sentiment_table([result])
        if args.emotions:
            create_emotion_table([result])


def cmd_analyze_file(args: argparse.Namespace) -> None:
    """Handle analyze-file command."""
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    if not input_path.exists():
        print_error(f"Input file not found: {input_path}")
        sys.exit(1)

    results = asyncio.run(analyze_file(
        input_path,
        output_path,
        include_emotions=args.emotions,
    ))

    if not output_path:
        create_sentiment_table(results)
        if args.emotions:
            create_emotion_table(results)


def cmd_scrape(args: argparse.Namespace) -> None:
    """Handle scrape command."""
    output_path = Path(args.output) if args.output else None

    if RICH_AVAILABLE and console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Scraping {args.platform}...", total=None)
            results = asyncio.run(scrape_platform(
                args.platform,
                args.identifier,
                limit=args.limit,
                analyze=args.analyze,
                output_path=output_path,
            ))
            progress.update(task, completed=True)
    else:
        print(f"Scraping {args.platform}...")
        results = asyncio.run(scrape_platform(
            args.platform,
            args.identifier,
            limit=args.limit,
            analyze=args.analyze,
            output_path=output_path,
        ))

    print_success(f"Scraped {len(results)} reviews")

    if not output_path and args.analyze:
        create_sentiment_table(results)


def cmd_batch(args: argparse.Namespace) -> None:
    """Handle batch command."""
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print_error(f"Input file not found: {input_path}")
        sys.exit(1)

    asyncio.run(run_batch(
        input_path,
        output_path,
        include_emotions=args.emotions,
    ))


def cmd_version(args: argparse.Namespace) -> None:
    """Handle version command."""
    print(f"Sentimatrix v{__version__}")


def cmd_info(args: argparse.Namespace) -> None:
    """Handle info command - show system info."""
    import platform

    info = {
        "sentimatrix_version": __version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "rich_available": RICH_AVAILABLE,
    }

    # Check optional dependencies
    optional_deps = {}

    try:
        import torch
        optional_deps["torch"] = torch.__version__
    except ImportError:
        optional_deps["torch"] = "not installed"

    try:
        import transformers
        optional_deps["transformers"] = transformers.__version__
    except ImportError:
        optional_deps["transformers"] = "not installed"

    try:
        import openai
        optional_deps["openai"] = openai.__version__
    except ImportError:
        optional_deps["openai"] = "not installed"

    try:
        import playwright
        optional_deps["playwright"] = "installed"
    except ImportError:
        optional_deps["playwright"] = "not installed"

    info["optional_dependencies"] = optional_deps

    if args.json:
        print(json.dumps(info, indent=2))
    else:
        if RICH_AVAILABLE and console:
            console.print(Panel(
                f"[bold]Sentimatrix[/bold] v{__version__}\n\n"
                f"Python: {info['python_version']}\n"
                f"Platform: {info['platform']}\n\n"
                "[bold]Optional Dependencies:[/bold]\n" +
                "\n".join(f"  {k}: {v}" for k, v in optional_deps.items()),
                title="System Information"
            ))
        else:
            print(f"Sentimatrix v{__version__}")
            print(f"Python: {info['python_version']}")
            print(f"Platform: {info['platform']}")
            print("\nOptional Dependencies:")
            for k, v in optional_deps.items():
                print(f"  {k}: {v}")


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="sentimatrix",
        description="Sentimatrix - Advanced Sentiment Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sentimatrix analyze "I love this product!"
  sentimatrix analyze "Great experience!" --emotions
  sentimatrix analyze-file reviews.txt -o results.json
  sentimatrix scrape amazon B08N5WRWNW --limit 100 --analyze
  sentimatrix scrape steam 730 -o steam_reviews.json
  sentimatrix batch input.csv -o output.csv --emotions
  sentimatrix info

For more information, visit: https://github.com/sentimatrix/sentimatrix
        """,
    )

    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze sentiment of a text",
        description="Analyze the sentiment of a single text string",
    )
    analyze_parser.add_argument("text", help="Text to analyze")
    analyze_parser.add_argument(
        "-e", "--emotions",
        action="store_true",
        help="Include emotion detection",
    )
    analyze_parser.add_argument(
        "-m", "--model",
        help="Model to use for analysis",
    )
    analyze_parser.add_argument(
        "-o", "--output",
        help="Output file path",
    )
    analyze_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    analyze_parser.set_defaults(func=cmd_analyze)

    # Analyze-file command
    file_parser = subparsers.add_parser(
        "analyze-file",
        help="Analyze sentiment of texts from a file",
        description="Analyze sentiment of multiple texts from a file (txt, csv, json)",
    )
    file_parser.add_argument("input", help="Input file path")
    file_parser.add_argument(
        "-o", "--output",
        help="Output file path",
    )
    file_parser.add_argument(
        "-e", "--emotions",
        action="store_true",
        help="Include emotion detection",
    )
    file_parser.set_defaults(func=cmd_analyze_file)

    # Scrape command
    scrape_parser = subparsers.add_parser(
        "scrape",
        help="Scrape reviews from a platform",
        description="Scrape reviews from Amazon, Steam, YouTube, or Reddit",
    )
    scrape_parser.add_argument(
        "platform",
        choices=["amazon", "steam", "youtube", "reddit"],
        help="Platform to scrape from",
    )
    scrape_parser.add_argument(
        "identifier",
        help="Product/game ID or URL",
    )
    scrape_parser.add_argument(
        "-l", "--limit",
        type=int,
        default=50,
        help="Maximum number of reviews to scrape (default: 50)",
    )
    scrape_parser.add_argument(
        "-a", "--analyze",
        action="store_true",
        help="Analyze sentiment of scraped reviews",
    )
    scrape_parser.add_argument(
        "-o", "--output",
        help="Output file path (JSON)",
    )
    scrape_parser.set_defaults(func=cmd_scrape)

    # Batch command
    batch_parser = subparsers.add_parser(
        "batch",
        help="Batch process texts from CSV",
        description="Process multiple texts from a CSV file",
    )
    batch_parser.add_argument("input", help="Input CSV file")
    batch_parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output CSV file",
    )
    batch_parser.add_argument(
        "-e", "--emotions",
        action="store_true",
        help="Include emotion detection",
    )
    batch_parser.set_defaults(func=cmd_batch)

    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show system information",
        description="Display Sentimatrix version and system information",
    )
    info_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    info_parser.set_defaults(func=cmd_info)

    return parser


def main() -> None:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        args.func(args)
    except KeyboardInterrupt:
        print_output("\nOperation cancelled.", style="yellow")
        sys.exit(130)
    except Exception as e:
        print_error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
