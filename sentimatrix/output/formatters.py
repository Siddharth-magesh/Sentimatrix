"""
Sentimatrix Output Formatters

Provides formatting functionality for analysis results:
- HTML: Rich HTML reports with styling
- Text: Plain text summaries
- Markdown: Markdown-formatted reports

Example:
    >>> from sentimatrix.output.formatters import HTMLFormatter
    >>>
    >>> formatter = HTMLFormatter()
    >>> html = await formatter.format(analysis_result)
    >>> await formatter.save(html, "report.html")
"""

from __future__ import annotations

import html
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from sentimatrix.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FormatOptions:
    """
    Options for formatting operations.

    Attributes:
        title: Report title
        include_charts: Include chart placeholders/data
        include_metadata: Include metadata section
        include_raw_data: Include raw data section
        theme: Visual theme ("light", "dark", "default")
        date_format: Format for datetime fields
        max_text_length: Maximum text length before truncation
    """

    title: str = "Sentimatrix Analysis Report"
    include_charts: bool = True
    include_metadata: bool = True
    include_raw_data: bool = False
    theme: str = "default"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    max_text_length: int = 500


@dataclass
class FormatResult:
    """
    Result of a formatting operation.

    Attributes:
        success: Whether formatting succeeded
        content: Formatted content
        format_type: Format type used
        error: Error message if failed
    """

    success: bool
    content: str
    format_type: str
    error: Optional[str] = None


class BaseFormatter(ABC):
    """
    Abstract base class for all formatters.
    """

    def __init__(self, options: Optional[FormatOptions] = None) -> None:
        """
        Initialize formatter.

        Args:
            options: Format options
        """
        self.options = options or FormatOptions()

    @property
    @abstractmethod
    def format_type(self) -> str:
        """Get format type name."""
        pass

    @abstractmethod
    async def format(self, data: Any) -> str:
        """
        Format data to string.

        Args:
            data: Data to format

        Returns:
            Formatted string
        """
        pass

    async def save(self, content: str, path: str) -> bool:
        """
        Save formatted content to file.

        Args:
            content: Content to save
            path: Output file path

        Returns:
            True if saved successfully
        """
        try:
            parent = Path(path).parent
            if parent and not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"Failed to save formatted content: {e}")
            return False

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value for formatting."""
        if value is None:
            return None
        elif isinstance(value, datetime):
            return value.strftime(self.options.date_format)
        elif isinstance(value, Enum):
            return value.value
        elif is_dataclass(value) and not isinstance(value, type):
            if hasattr(value, "to_dict"):
                return value.to_dict()
            return {
                field_name: self._serialize_value(getattr(value, field_name))
                for field_name in value.__dataclass_fields__
            }
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        elif hasattr(value, "to_dict"):
            return value.to_dict()
        else:
            return value

    def _truncate_text(self, text: str, max_length: Optional[int] = None) -> str:
        """Truncate text to maximum length."""
        max_len = max_length or self.options.max_text_length
        if len(text) > max_len:
            return text[: max_len - 3] + "..."
        return text


class HTMLFormatter(BaseFormatter):
    """
    Format data as rich HTML report.

    Features:
    - Responsive layout
    - Light/dark themes
    - Summary cards
    - Data tables
    - Chart placeholders

    Example:
        >>> formatter = HTMLFormatter(FormatOptions(title="Product Analysis"))
        >>> html = await formatter.format(analysis_result)
        >>> await formatter.save(html, "report.html")
    """

    @property
    def format_type(self) -> str:
        return "html"

    def _get_css(self) -> str:
        """Get CSS styles based on theme."""
        if self.options.theme == "dark":
            return """
            :root {
                --bg-primary: #1a1a2e;
                --bg-secondary: #16213e;
                --bg-card: #0f3460;
                --text-primary: #eaeaea;
                --text-secondary: #a0a0a0;
                --accent: #e94560;
                --success: #00d26a;
                --warning: #ffc107;
                --danger: #dc3545;
                --border: #2a2a4a;
            }
            """
        else:
            return """
            :root {
                --bg-primary: #f5f7fa;
                --bg-secondary: #ffffff;
                --bg-card: #ffffff;
                --text-primary: #2c3e50;
                --text-secondary: #6c757d;
                --accent: #3498db;
                --success: #27ae60;
                --warning: #f39c12;
                --danger: #e74c3c;
                --border: #dee2e6;
            }
            """

    def _get_base_styles(self) -> str:
        """Get base CSS styles."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            padding: 30px 0;
            margin-bottom: 30px;
            border-bottom: 2px solid var(--border);
        }
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        .header .subtitle {
            color: var(--text-secondary);
            font-size: 1.1rem;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: var(--bg-card);
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border: 1px solid var(--border);
        }
        .card h3 {
            font-size: 0.9rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }
        .card .value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--accent);
        }
        .card .subvalue {
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-top: 4px;
        }
        .section {
            background: var(--bg-card);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border: 1px solid var(--border);
        }
        .section h2 {
            font-size: 1.5rem;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        th {
            background: var(--bg-secondary);
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            font-size: 0.85rem;
        }
        tr:hover {
            background: var(--bg-secondary);
        }
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        .badge-positive {
            background: rgba(39, 174, 96, 0.2);
            color: var(--success);
        }
        .badge-negative {
            background: rgba(231, 76, 60, 0.2);
            color: var(--danger);
        }
        .badge-neutral {
            background: rgba(149, 165, 166, 0.2);
            color: var(--text-secondary);
        }
        .progress-bar {
            height: 8px;
            background: var(--bg-secondary);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }
        .progress-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        .progress-fill.positive {
            background: var(--success);
        }
        .progress-fill.negative {
            background: var(--danger);
        }
        .list-group {
            list-style: none;
        }
        .list-group li {
            padding: 12px 0;
            border-bottom: 1px solid var(--border);
        }
        .list-group li:last-child {
            border-bottom: none;
        }
        .footer {
            text-align: center;
            padding: 20px;
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-top: 30px;
        }
        @media (max-width: 768px) {
            .header h1 {
                font-size: 1.8rem;
            }
            .card .value {
                font-size: 1.5rem;
            }
        }
        """

    async def format(self, data: Any) -> str:
        """
        Format data as HTML report.

        Args:
            data: Data to format (analysis result, reviews, etc.)

        Returns:
            HTML string
        """
        # Serialize data
        serialized = self._serialize_value(data)

        # Build HTML
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "  <meta charset='UTF-8'>",
            "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"  <title>{html.escape(self.options.title)}</title>",
            "  <style>",
            self._get_css(),
            self._get_base_styles(),
            "  </style>",
            "</head>",
            "<body>",
            "  <div class='container'>",
        ]

        # Header
        html_parts.extend([
            "    <div class='header'>",
            f"      <h1>{html.escape(self.options.title)}</h1>",
            f"      <p class='subtitle'>Generated on {datetime.now().strftime(self.options.date_format)}</p>",
            "    </div>",
        ])

        # Generate content based on data type
        if isinstance(serialized, dict):
            html_parts.append(self._format_dict(serialized))
        elif isinstance(serialized, list):
            html_parts.append(self._format_list(serialized))
        else:
            html_parts.append(f"<div class='section'><pre>{html.escape(str(serialized))}</pre></div>")

        # Footer
        html_parts.extend([
            "    <div class='footer'>",
            "      <p>Generated by Sentimatrix</p>",
            "    </div>",
            "  </div>",
            "</body>",
            "</html>",
        ])

        return "\n".join(html_parts)

    def _format_dict(self, data: Dict[str, Any]) -> str:
        """Format dictionary data."""
        parts = []

        # Check for common analysis result fields
        if "total_count" in data or "positive_ratio" in data:
            parts.append(self._format_summary_cards(data))

        # Sentiment summary
        if "sentiment_summary" in data and data["sentiment_summary"]:
            parts.append(self._format_sentiment_section(data["sentiment_summary"]))

        # Emotion summary
        if "emotion_summary" in data and data["emotion_summary"]:
            parts.append(self._format_emotion_section(data["emotion_summary"]))

        # Insights
        if any(k in data for k in ["summary", "pros", "cons", "themes"]):
            parts.append(self._format_insights_section(data))

        # Reviews
        if "reviews" in data and isinstance(data["reviews"], list):
            parts.append(self._format_reviews_table(data["reviews"]))

        # Generic fields
        if not parts:
            parts.append(self._format_generic_dict(data))

        return "\n".join(parts)

    def _format_summary_cards(self, data: Dict[str, Any]) -> str:
        """Format summary statistics as cards."""
        cards = []

        if "total_count" in data:
            cards.append(f"""
            <div class='card'>
                <h3>Total Reviews</h3>
                <div class='value'>{data['total_count']}</div>
            </div>
            """)

        if "positive_ratio" in data:
            positive_pct = data["positive_ratio"] * 100 if data["positive_ratio"] <= 1 else data["positive_ratio"]
            cards.append(f"""
            <div class='card'>
                <h3>Positive</h3>
                <div class='value'>{positive_pct:.1f}%</div>
                <div class='progress-bar'>
                    <div class='progress-fill positive' style='width: {positive_pct}%'></div>
                </div>
            </div>
            """)

        if "negative_ratio" in data:
            negative_pct = data["negative_ratio"] * 100 if data["negative_ratio"] <= 1 else data["negative_ratio"]
            cards.append(f"""
            <div class='card'>
                <h3>Negative</h3>
                <div class='value'>{negative_pct:.1f}%</div>
                <div class='progress-bar'>
                    <div class='progress-fill negative' style='width: {negative_pct}%'></div>
                </div>
            </div>
            """)

        if "average_polarity" in data:
            polarity = data["average_polarity"]
            cards.append(f"""
            <div class='card'>
                <h3>Avg. Polarity</h3>
                <div class='value'>{polarity:.2f}</div>
                <div class='subvalue'>Range: -1 to +1</div>
            </div>
            """)

        if cards:
            return f"<div class='summary-grid'>{''.join(cards)}</div>"
        return ""

    def _format_sentiment_section(self, data: Dict[str, Any]) -> str:
        """Format sentiment summary section."""
        rows = []
        for key, value in data.items():
            if isinstance(value, (int, float)):
                if "ratio" in key or "percentage" in key:
                    value = f"{value * 100:.1f}%" if value <= 1 else f"{value:.1f}%"
                elif isinstance(value, float):
                    value = f"{value:.3f}"
            rows.append(f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>")

        return f"""
        <div class='section'>
            <h2>Sentiment Summary</h2>
            <table>
                <thead><tr><th>Metric</th><th>Value</th></tr></thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        </div>
        """

    def _format_emotion_section(self, data: Dict[str, Any]) -> str:
        """Format emotion summary section."""
        rows = []
        for key, value in data.items():
            if isinstance(value, dict):
                # Emotion distribution
                for emotion, score in value.items():
                    score_pct = score * 100 if score <= 1 else score
                    rows.append(f"""
                    <tr>
                        <td>{html.escape(str(emotion))}</td>
                        <td>{score_pct:.1f}%</td>
                        <td>
                            <div class='progress-bar'>
                                <div class='progress-fill positive' style='width: {score_pct}%'></div>
                            </div>
                        </td>
                    </tr>
                    """)
            elif isinstance(value, (int, float)):
                rows.append(f"<tr><td>{html.escape(str(key))}</td><td colspan='2'>{value}</td></tr>")

        return f"""
        <div class='section'>
            <h2>Emotion Summary</h2>
            <table>
                <thead><tr><th>Emotion</th><th>Score</th><th>Distribution</th></tr></thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        </div>
        """

    def _format_insights_section(self, data: Dict[str, Any]) -> str:
        """Format insights section (pros, cons, themes)."""
        parts = ["<div class='section'>", "<h2>Insights</h2>"]

        if "summary" in data and data["summary"]:
            parts.append(f"<p><strong>Summary:</strong> {html.escape(str(data['summary']))}</p>")

        if "pros" in data and data["pros"]:
            pros_items = "".join(f"<li>{html.escape(str(p))}</li>" for p in data["pros"])
            parts.append(f"<h3>Pros</h3><ul class='list-group'>{pros_items}</ul>")

        if "cons" in data and data["cons"]:
            cons_items = "".join(f"<li>{html.escape(str(c))}</li>" for c in data["cons"])
            parts.append(f"<h3>Cons</h3><ul class='list-group'>{cons_items}</ul>")

        if "themes" in data and data["themes"]:
            theme_badges = "".join(f"<span class='badge badge-neutral'>{html.escape(str(t))}</span> " for t in data["themes"])
            parts.append(f"<h3>Common Themes</h3><p>{theme_badges}</p>")

        if "recommendations" in data and data["recommendations"]:
            rec_items = "".join(f"<li>{html.escape(str(r))}</li>" for r in data["recommendations"])
            parts.append(f"<h3>Recommendations</h3><ul class='list-group'>{rec_items}</ul>")

        parts.append("</div>")
        return "".join(parts)

    def _format_reviews_table(self, reviews: List[Dict[str, Any]]) -> str:
        """Format reviews as a table."""
        if not reviews:
            return ""

        rows = []
        for review in reviews[:50]:  # Limit to 50 reviews
            text = self._truncate_text(str(review.get("text", "")), 200)
            sentiment = review.get("sentiment", {})
            if isinstance(sentiment, dict):
                sentiment_label = sentiment.get("sentiment", "N/A")
                confidence = sentiment.get("confidence", 0)
            else:
                sentiment_label = "N/A"
                confidence = 0

            badge_class = "badge-positive" if sentiment_label == "positive" else (
                "badge-negative" if sentiment_label == "negative" else "badge-neutral"
            )

            rows.append(f"""
            <tr>
                <td>{html.escape(text)}</td>
                <td><span class='badge {badge_class}'>{html.escape(str(sentiment_label))}</span></td>
                <td>{confidence:.2f}</td>
            </tr>
            """)

        return f"""
        <div class='section'>
            <h2>Reviews ({len(reviews)} total)</h2>
            <table>
                <thead>
                    <tr><th>Text</th><th>Sentiment</th><th>Confidence</th></tr>
                </thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        </div>
        """

    def _format_list(self, data: List[Any]) -> str:
        """Format list data."""
        if not data:
            return "<div class='section'><p>No data available.</p></div>"

        # Check if list of reviews/dicts
        if isinstance(data[0], dict):
            return self._format_reviews_table(data)

        # Simple list
        items = "".join(f"<li>{html.escape(str(item))}</li>" for item in data)
        return f"<div class='section'><ul class='list-group'>{items}</ul></div>"

    def _format_generic_dict(self, data: Dict[str, Any]) -> str:
        """Format generic dictionary."""
        rows = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                value = str(value)[:200] + "..." if len(str(value)) > 200 else str(value)
            rows.append(f"<tr><td><strong>{html.escape(str(key))}</strong></td><td>{html.escape(str(value))}</td></tr>")

        return f"""
        <div class='section'>
            <h2>Data</h2>
            <table>
                <thead><tr><th>Field</th><th>Value</th></tr></thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        </div>
        """


class TextFormatter(BaseFormatter):
    """
    Format data as plain text.

    Example:
        >>> formatter = TextFormatter()
        >>> text = await formatter.format(analysis_result)
        >>> print(text)
    """

    @property
    def format_type(self) -> str:
        return "text"

    async def format(self, data: Any) -> str:
        """Format data as plain text."""
        serialized = self._serialize_value(data)
        lines = [
            "=" * 60,
            self.options.title.center(60),
            f"Generated: {datetime.now().strftime(self.options.date_format)}".center(60),
            "=" * 60,
            "",
        ]

        if isinstance(serialized, dict):
            lines.extend(self._format_dict(serialized))
        elif isinstance(serialized, list):
            lines.extend(self._format_list(serialized))
        else:
            lines.append(str(serialized))

        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def _format_dict(self, data: Dict[str, Any], indent: int = 0) -> List[str]:
        """Format dictionary."""
        lines = []
        prefix = "  " * indent

        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.extend(self._format_dict(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                lines.extend(self._format_list(value, indent + 1))
            else:
                lines.append(f"{prefix}{key}: {value}")

        return lines

    def _format_list(self, data: List[Any], indent: int = 0) -> List[str]:
        """Format list."""
        lines = []
        prefix = "  " * indent

        for i, item in enumerate(data[:20], 1):  # Limit to 20 items
            if isinstance(item, dict):
                lines.append(f"{prefix}{i}.")
                lines.extend(self._format_dict(item, indent + 1))
            else:
                lines.append(f"{prefix}- {item}")

        if len(data) > 20:
            lines.append(f"{prefix}... and {len(data) - 20} more items")

        return lines


class MarkdownFormatter(BaseFormatter):
    """
    Format data as Markdown.

    Example:
        >>> formatter = MarkdownFormatter()
        >>> md = await formatter.format(analysis_result)
        >>> await formatter.save(md, "report.md")
    """

    @property
    def format_type(self) -> str:
        return "markdown"

    async def format(self, data: Any) -> str:
        """Format data as Markdown."""
        serialized = self._serialize_value(data)
        lines = [
            f"# {self.options.title}",
            "",
            f"*Generated: {datetime.now().strftime(self.options.date_format)}*",
            "",
            "---",
            "",
        ]

        if isinstance(serialized, dict):
            lines.extend(self._format_dict(serialized))
        elif isinstance(serialized, list):
            lines.extend(self._format_list(serialized))
        else:
            lines.append(str(serialized))

        lines.extend([
            "",
            "---",
            "",
            "*Generated by Sentimatrix*",
        ])

        return "\n".join(lines)

    def _format_dict(self, data: Dict[str, Any], level: int = 2) -> List[str]:
        """Format dictionary as Markdown."""
        lines = []
        header_prefix = "#" * min(level, 6)

        # Summary stats
        if "total_count" in data or "positive_ratio" in data:
            lines.extend(self._format_summary(data))

        for key, value in data.items():
            if key in ["total_count", "positive_ratio", "negative_ratio", "average_polarity"]:
                continue  # Already handled in summary

            if isinstance(value, dict):
                lines.append(f"{header_prefix} {key.replace('_', ' ').title()}")
                lines.append("")
                lines.extend(self._format_dict(value, level + 1))
            elif isinstance(value, list):
                lines.append(f"{header_prefix} {key.replace('_', ' ').title()}")
                lines.append("")
                lines.extend(self._format_list(value))
            else:
                lines.append(f"**{key.replace('_', ' ').title()}:** {value}")
                lines.append("")

        return lines

    def _format_summary(self, data: Dict[str, Any]) -> List[str]:
        """Format summary statistics."""
        lines = ["## Summary", "", "| Metric | Value |", "|--------|-------|"]

        if "total_count" in data:
            lines.append(f"| Total Reviews | {data['total_count']} |")
        if "positive_ratio" in data:
            pct = data["positive_ratio"] * 100 if data["positive_ratio"] <= 1 else data["positive_ratio"]
            lines.append(f"| Positive | {pct:.1f}% |")
        if "negative_ratio" in data:
            pct = data["negative_ratio"] * 100 if data["negative_ratio"] <= 1 else data["negative_ratio"]
            lines.append(f"| Negative | {pct:.1f}% |")
        if "average_polarity" in data:
            lines.append(f"| Avg. Polarity | {data['average_polarity']:.2f} |")

        lines.append("")
        return lines

    def _format_list(self, data: List[Any]) -> List[str]:
        """Format list as Markdown."""
        lines = []

        for item in data[:30]:  # Limit
            if isinstance(item, dict):
                # Review item
                text = self._truncate_text(str(item.get("text", "")), 150)
                lines.append(f"- {text}")
            else:
                lines.append(f"- {item}")

        if len(data) > 30:
            lines.append(f"- *...and {len(data) - 30} more*")

        lines.append("")
        return lines


# Convenience functions


def get_formatter(format_type: str, **kwargs: Any) -> BaseFormatter:
    """
    Get a formatter instance for the specified format.

    Args:
        format_type: Format type ("html", "text", "markdown")
        **kwargs: Additional formatter options

    Returns:
        Formatter instance
    """
    format_type = format_type.lower()

    options = FormatOptions(**{k: v for k, v in kwargs.items() if hasattr(FormatOptions, k)})

    if format_type == "html":
        return HTMLFormatter(options)
    elif format_type == "text":
        return TextFormatter(options)
    elif format_type in ("markdown", "md"):
        return MarkdownFormatter(options)
    else:
        raise ValueError(f"Unsupported format type: {format_type}")


async def format_as_html(data: Any, title: str = "Analysis Report", **kwargs: Any) -> str:
    """
    Quick format data as HTML.

    Args:
        data: Data to format
        title: Report title
        **kwargs: Additional options

    Returns:
        HTML string
    """
    options = FormatOptions(title=title, **kwargs)
    formatter = HTMLFormatter(options)
    return await formatter.format(data)


async def format_as_text(data: Any, title: str = "Analysis Report", **kwargs: Any) -> str:
    """
    Quick format data as plain text.

    Args:
        data: Data to format
        title: Report title
        **kwargs: Additional options

    Returns:
        Text string
    """
    options = FormatOptions(title=title, **kwargs)
    formatter = TextFormatter(options)
    return await formatter.format(data)


async def format_as_markdown(data: Any, title: str = "Analysis Report", **kwargs: Any) -> str:
    """
    Quick format data as Markdown.

    Args:
        data: Data to format
        title: Report title
        **kwargs: Additional options

    Returns:
        Markdown string
    """
    options = FormatOptions(title=title, **kwargs)
    formatter = MarkdownFormatter(options)
    return await formatter.format(data)


__all__ = [
    "FormatOptions",
    "FormatResult",
    "BaseFormatter",
    "HTMLFormatter",
    "TextFormatter",
    "MarkdownFormatter",
    "get_formatter",
    "format_as_html",
    "format_as_text",
    "format_as_markdown",
]
