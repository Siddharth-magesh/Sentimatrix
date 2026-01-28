"""
Sentimatrix Visualizers

Provides visualization functionality for analysis results:
- Bar charts: Sentiment distribution, emotion scores
- Pie charts: Sentiment breakdown
- Histograms: Score distributions
- Time series: Sentiment over time (if timestamps available)

Example:
    >>> from sentimatrix.output.visualizers import ChartVisualizer
    >>>
    >>> visualizer = ChartVisualizer()
    >>> fig = await visualizer.create_sentiment_bar_chart(analysis_result)
    >>> await visualizer.save(fig, "sentiment_chart.png")
"""

from __future__ import annotations

import io
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from sentimatrix.core.logger import get_logger

logger = get_logger(__name__)


class ChartType(str, Enum):
    """Supported chart types."""

    BAR = "bar"
    HORIZONTAL_BAR = "horizontal_bar"
    PIE = "pie"
    DONUT = "donut"
    HISTOGRAM = "histogram"
    LINE = "line"
    STACKED_BAR = "stacked_bar"


class Theme(str, Enum):
    """Chart themes."""

    DEFAULT = "default"
    DARK = "dark"
    LIGHT = "light"
    COLORFUL = "colorful"


@dataclass
class VisualizationOptions:
    """
    Options for visualization operations.

    Attributes:
        width: Figure width in inches
        height: Figure height in inches
        dpi: Dots per inch for saved images
        theme: Visual theme
        title: Chart title
        show_values: Show values on chart
        show_legend: Show legend
        color_palette: Custom color palette
        save_format: Default save format
    """

    width: int = 10
    height: int = 6
    dpi: int = 150
    theme: Theme = Theme.DEFAULT
    title: Optional[str] = None
    show_values: bool = True
    show_legend: bool = True
    color_palette: Optional[List[str]] = None
    save_format: str = "png"


@dataclass
class VisualizationResult:
    """
    Result of a visualization operation.

    Attributes:
        success: Whether visualization was created
        path: Output file path (if saved)
        chart_type: Type of chart created
        error: Error message if failed
    """

    success: bool
    path: Optional[str] = None
    chart_type: Optional[ChartType] = None
    error: Optional[str] = None


class BaseVisualizer(ABC):
    """
    Abstract base class for visualizers.
    """

    def __init__(self, options: Optional[VisualizationOptions] = None) -> None:
        """
        Initialize visualizer.

        Args:
            options: Visualization options
        """
        self.options = options or VisualizationOptions()
        self._plt = None
        self._matplotlib_loaded = False

    def _get_matplotlib(self):
        """Import matplotlib (lazy import)."""
        if self._matplotlib_loaded:
            return self._plt

        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt

            self._plt = plt
            self._matplotlib_loaded = True
            return plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for visualizations. "
                "Install it with: pip install matplotlib"
            )

    def _get_colors(self) -> Dict[str, str]:
        """Get color scheme based on theme."""
        if self.options.color_palette:
            return {
                "positive": self.options.color_palette[0] if len(self.options.color_palette) > 0 else "#27ae60",
                "negative": self.options.color_palette[1] if len(self.options.color_palette) > 1 else "#e74c3c",
                "neutral": self.options.color_palette[2] if len(self.options.color_palette) > 2 else "#95a5a6",
                "primary": self.options.color_palette[0] if len(self.options.color_palette) > 0 else "#3498db",
                "secondary": self.options.color_palette[1] if len(self.options.color_palette) > 1 else "#9b59b6",
            }

        if self.options.theme == Theme.DARK:
            return {
                "positive": "#00d26a",
                "negative": "#ff4757",
                "neutral": "#a4b0be",
                "primary": "#70a1ff",
                "secondary": "#7bed9f",
                "background": "#1e1e2e",
                "text": "#ffffff",
            }
        elif self.options.theme == Theme.COLORFUL:
            return {
                "positive": "#00cec9",
                "negative": "#fd79a8",
                "neutral": "#dfe6e9",
                "primary": "#0984e3",
                "secondary": "#6c5ce7",
                "background": "#ffffff",
                "text": "#2d3436",
            }
        else:
            return {
                "positive": "#27ae60",
                "negative": "#e74c3c",
                "neutral": "#95a5a6",
                "primary": "#3498db",
                "secondary": "#9b59b6",
                "background": "#ffffff",
                "text": "#2c3e50",
            }

    def _apply_theme(self, fig, ax) -> None:
        """Apply theme to figure and axes."""
        colors = self._get_colors()
        plt = self._get_matplotlib()

        if self.options.theme == Theme.DARK:
            fig.patch.set_facecolor(colors["background"])
            ax.set_facecolor(colors["background"])
            ax.tick_params(colors=colors["text"])
            ax.xaxis.label.set_color(colors["text"])
            ax.yaxis.label.set_color(colors["text"])
            ax.title.set_color(colors["text"])
            for spine in ax.spines.values():
                spine.set_edgecolor(colors["text"])

    def _ensure_directory(self, path: str) -> None:
        """Ensure parent directory exists."""
        parent = Path(path).parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    async def create_chart(self, data: Any, chart_type: ChartType) -> Any:
        """
        Create a chart from data.

        Args:
            data: Data to visualize
            chart_type: Type of chart to create

        Returns:
            Matplotlib figure
        """
        pass

    async def save(
        self,
        figure: Any,
        path: str,
        format: Optional[str] = None,
    ) -> VisualizationResult:
        """
        Save figure to file.

        Args:
            figure: Matplotlib figure
            path: Output file path
            format: Image format (png, svg, pdf)

        Returns:
            VisualizationResult
        """
        try:
            self._ensure_directory(path)
            save_format = format or self.options.save_format
            figure.savefig(
                path,
                format=save_format,
                dpi=self.options.dpi,
                bbox_inches="tight",
                facecolor=figure.get_facecolor(),
                edgecolor="none",
            )

            # Close figure to free memory
            plt = self._get_matplotlib()
            plt.close(figure)

            logger.info(f"Saved visualization to {path}")

            return VisualizationResult(
                success=True,
                path=path,
            )
        except Exception as e:
            logger.error(f"Failed to save visualization: {e}")
            return VisualizationResult(
                success=False,
                error=str(e),
            )

    async def to_bytes(self, figure: Any, format: str = "png") -> bytes:
        """
        Convert figure to bytes.

        Args:
            figure: Matplotlib figure
            format: Image format

        Returns:
            Image bytes
        """
        buf = io.BytesIO()
        figure.savefig(
            buf,
            format=format,
            dpi=self.options.dpi,
            bbox_inches="tight",
            facecolor=figure.get_facecolor(),
        )
        buf.seek(0)

        # Close figure
        plt = self._get_matplotlib()
        plt.close(figure)

        return buf.getvalue()


class ChartVisualizer(BaseVisualizer):
    """
    Main chart visualizer for Sentimatrix.

    Supports:
    - Sentiment distribution charts
    - Emotion breakdown charts
    - Score histograms
    - Time series (sentiment over time)

    Example:
        >>> visualizer = ChartVisualizer()
        >>> fig = await visualizer.create_sentiment_bar_chart(analysis_result)
        >>> await visualizer.save(fig, "chart.png")
    """

    async def create_chart(self, data: Any, chart_type: ChartType) -> Any:
        """
        Create a chart from data.

        Args:
            data: Data to visualize
            chart_type: Type of chart

        Returns:
            Matplotlib figure
        """
        if chart_type == ChartType.BAR:
            return await self.create_bar_chart(data)
        elif chart_type == ChartType.HORIZONTAL_BAR:
            return await self.create_horizontal_bar_chart(data)
        elif chart_type == ChartType.PIE:
            return await self.create_pie_chart(data)
        elif chart_type == ChartType.DONUT:
            return await self.create_donut_chart(data)
        elif chart_type == ChartType.HISTOGRAM:
            return await self.create_histogram(data)
        elif chart_type == ChartType.LINE:
            return await self.create_line_chart(data)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

    async def create_sentiment_bar_chart(
        self,
        data: Any,
        title: Optional[str] = None,
    ) -> Any:
        """
        Create a bar chart showing sentiment distribution.

        Args:
            data: Analysis result with sentiment data
            title: Chart title

        Returns:
            Matplotlib figure
        """
        plt = self._get_matplotlib()
        colors = self._get_colors()

        fig, ax = plt.subplots(figsize=(self.options.width, self.options.height))

        # Extract sentiment counts
        if hasattr(data, "positive_ratio"):
            positive = data.positive_ratio
            negative = data.negative_ratio
            neutral = 1 - positive - negative
        elif isinstance(data, dict):
            positive = data.get("positive_ratio", 0)
            negative = data.get("negative_ratio", 0)
            neutral = data.get("neutral_ratio", 1 - positive - negative)
        else:
            positive = negative = neutral = 0.33

        labels = ["Positive", "Neutral", "Negative"]
        values = [positive * 100, neutral * 100, negative * 100]
        bar_colors = [colors["positive"], colors["neutral"], colors["negative"]]

        bars = ax.bar(labels, values, color=bar_colors, edgecolor="white", linewidth=1.5)

        # Add value labels
        if self.options.show_values:
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.annotate(
                    f"{value:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=12,
                )

        ax.set_ylabel("Percentage (%)", fontsize=12)
        ax.set_title(
            title or self.options.title or "Sentiment Distribution",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.set_ylim(0, max(values) * 1.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        self._apply_theme(fig, ax)
        plt.tight_layout()

        return fig

    async def create_emotion_bar_chart(
        self,
        data: Any,
        title: Optional[str] = None,
        top_k: int = 8,
    ) -> Any:
        """
        Create a horizontal bar chart showing emotion distribution.

        Args:
            data: Analysis result with emotion data
            title: Chart title
            top_k: Number of top emotions to show

        Returns:
            Matplotlib figure
        """
        plt = self._get_matplotlib()
        import numpy as np

        fig, ax = plt.subplots(figsize=(self.options.width, self.options.height))

        # Extract emotion data
        emotions = {}
        if hasattr(data, "emotion_summary") and data.emotion_summary:
            summary = data.emotion_summary
            if isinstance(summary, dict):
                for key, value in summary.items():
                    if isinstance(value, dict):
                        emotions.update(value)
                    elif isinstance(value, (int, float)) and key not in ["total", "count"]:
                        emotions[key] = value
        elif isinstance(data, dict):
            if "emotion_summary" in data:
                summary = data["emotion_summary"]
                if isinstance(summary, dict):
                    for key, value in summary.items():
                        if isinstance(value, dict):
                            emotions.update(value)
                        elif isinstance(value, (int, float)):
                            emotions[key] = value
            else:
                emotions = {k: v for k, v in data.items() if isinstance(v, (int, float))}

        if not emotions:
            # Default sample data
            emotions = {"joy": 0.3, "sadness": 0.1, "anger": 0.15, "fear": 0.05, "surprise": 0.1}

        # Sort and limit
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:top_k]
        labels = [e[0].title() for e in sorted_emotions]
        values = [e[1] * 100 if e[1] <= 1 else e[1] for e in sorted_emotions]

        # Create color gradient
        cmap = plt.cm.get_cmap("viridis")
        bar_colors = [cmap(i / len(labels)) for i in range(len(labels))]

        bars = ax.barh(labels[::-1], values[::-1], color=bar_colors[::-1], edgecolor="white", linewidth=1)

        # Add value labels
        if self.options.show_values:
            for bar, value in zip(bars, values[::-1]):
                width = bar.get_width()
                ax.annotate(
                    f"{value:.1f}%",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=10,
                )

        ax.set_xlabel("Score (%)", fontsize=12)
        ax.set_title(
            title or self.options.title or "Emotion Distribution",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlim(0, max(values) * 1.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        self._apply_theme(fig, ax)
        plt.tight_layout()

        return fig

    async def create_sentiment_pie_chart(
        self,
        data: Any,
        title: Optional[str] = None,
        donut: bool = False,
    ) -> Any:
        """
        Create a pie/donut chart showing sentiment breakdown.

        Args:
            data: Analysis result with sentiment data
            title: Chart title
            donut: Create donut chart instead of pie

        Returns:
            Matplotlib figure
        """
        plt = self._get_matplotlib()
        colors = self._get_colors()

        fig, ax = plt.subplots(figsize=(self.options.width, self.options.height))

        # Extract sentiment data
        if hasattr(data, "positive_ratio"):
            positive = data.positive_ratio
            negative = data.negative_ratio
            neutral = 1 - positive - negative
        elif isinstance(data, dict):
            positive = data.get("positive_ratio", 0)
            negative = data.get("negative_ratio", 0)
            neutral = data.get("neutral_ratio", 1 - positive - negative)
        else:
            positive = negative = neutral = 0.33

        labels = ["Positive", "Neutral", "Negative"]
        values = [positive, neutral, negative]
        pie_colors = [colors["positive"], colors["neutral"], colors["negative"]]

        # Filter out zero values
        filtered_data = [(l, v, c) for l, v, c in zip(labels, values, pie_colors) if v > 0]
        if filtered_data:
            labels, values, pie_colors = zip(*filtered_data)

        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            colors=pie_colors,
            autopct="%1.1f%%" if self.options.show_values else "",
            startangle=90,
            explode=[0.02] * len(values),
            shadow=False,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )

        # Style autotexts
        for autotext in autotexts:
            autotext.set_fontweight("bold")
            autotext.set_fontsize(11)

        # Create donut
        if donut:
            centre_circle = plt.Circle((0, 0), 0.60, fc="white")
            ax.add_patch(centre_circle)

            # Add center text
            total = sum(values)
            ax.text(0, 0, f"{total * 100:.0f}%\nAnalyzed", ha="center", va="center", fontsize=14, fontweight="bold")

        ax.set_title(
            title or self.options.title or "Sentiment Breakdown",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        # Legend
        if self.options.show_legend:
            ax.legend(
                wedges,
                labels,
                title="Sentiment",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1),
            )

        self._apply_theme(fig, ax)
        plt.tight_layout()

        return fig

    async def create_score_histogram(
        self,
        scores: List[float],
        title: Optional[str] = None,
        bins: int = 20,
    ) -> Any:
        """
        Create a histogram of confidence/polarity scores.

        Args:
            scores: List of scores
            title: Chart title
            bins: Number of histogram bins

        Returns:
            Matplotlib figure
        """
        plt = self._get_matplotlib()
        import numpy as np

        fig, ax = plt.subplots(figsize=(self.options.width, self.options.height))

        colors = self._get_colors()

        # Create histogram
        n, bins_arr, patches = ax.hist(
            scores,
            bins=bins,
            edgecolor="white",
            linewidth=1,
            alpha=0.7,
        )

        # Color bars based on value
        for patch, left_edge in zip(patches, bins_arr[:-1]):
            if left_edge >= 0.5:
                patch.set_facecolor(colors["positive"])
            elif left_edge >= 0:
                patch.set_facecolor(colors["neutral"])
            else:
                patch.set_facecolor(colors["negative"])

        # Add mean line
        mean_score = np.mean(scores)
        ax.axvline(mean_score, color=colors["primary"], linestyle="--", linewidth=2, label=f"Mean: {mean_score:.2f}")

        ax.set_xlabel("Score", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(
            title or self.options.title or "Score Distribution",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if self.options.show_legend:
            ax.legend()

        self._apply_theme(fig, ax)
        plt.tight_layout()

        return fig

    async def create_comparison_chart(
        self,
        data: Any,
        title: Optional[str] = None,
    ) -> Any:
        """
        Create a comparison chart for two items.

        Args:
            data: Comparison result with item_a and item_b data
            title: Chart title

        Returns:
            Matplotlib figure
        """
        plt = self._get_matplotlib()
        import numpy as np

        fig, ax = plt.subplots(figsize=(self.options.width, self.options.height))

        colors = self._get_colors()

        # Extract comparison data
        if hasattr(data, "item_a"):
            item_a_name = data.item_a
            item_b_name = data.item_b
            item_a = data.item_a_analysis
            item_b = data.item_b_analysis
        elif isinstance(data, dict):
            item_a_name = data.get("item_a", "Item A")
            item_b_name = data.get("item_b", "Item B")
            item_a = data.get("item_a_stats", {})
            item_b = data.get("item_b_stats", {})
        else:
            item_a_name = "Item A"
            item_b_name = "Item B"
            item_a = item_b = {}

        # Get metrics
        metrics = ["Positive %", "Negative %", "Avg. Polarity"]

        if hasattr(item_a, "positive_ratio"):
            a_values = [
                item_a.positive_ratio * 100,
                item_a.negative_ratio * 100,
                (item_a.average_polarity + 1) * 50,  # Scale to 0-100
            ]
            b_values = [
                item_b.positive_ratio * 100,
                item_b.negative_ratio * 100,
                (item_b.average_polarity + 1) * 50,
            ]
        elif isinstance(item_a, dict):
            a_values = [
                item_a.get("positive_ratio", 0) * 100,
                item_a.get("negative_ratio", 0) * 100,
                (item_a.get("average_polarity", 0) + 1) * 50,
            ]
            b_values = [
                item_b.get("positive_ratio", 0) * 100,
                item_b.get("negative_ratio", 0) * 100,
                (item_b.get("average_polarity", 0) + 1) * 50,
            ]
        else:
            a_values = [50, 20, 60]
            b_values = [40, 30, 55]

        x = np.arange(len(metrics))
        width = 0.35

        bars_a = ax.bar(x - width / 2, a_values, width, label=item_a_name, color=colors["primary"], edgecolor="white")
        bars_b = ax.bar(x + width / 2, b_values, width, label=item_b_name, color=colors["secondary"], edgecolor="white")

        # Add value labels
        if self.options.show_values:
            for bars in [bars_a, bars_b]:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(
                        f"{height:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )

        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(
            title or self.options.title or "Product Comparison",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if self.options.show_legend:
            ax.legend()

        self._apply_theme(fig, ax)
        plt.tight_layout()

        return fig

    async def create_bar_chart(
        self,
        data: Dict[str, float],
        title: Optional[str] = None,
    ) -> Any:
        """
        Create a generic bar chart.

        Args:
            data: Dictionary of label -> value
            title: Chart title

        Returns:
            Matplotlib figure
        """
        plt = self._get_matplotlib()

        fig, ax = plt.subplots(figsize=(self.options.width, self.options.height))

        colors = self._get_colors()

        labels = list(data.keys())
        values = list(data.values())

        bars = ax.bar(labels, values, color=colors["primary"], edgecolor="white", linewidth=1.5)

        if self.options.show_values:
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.annotate(
                    f"{value:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

        ax.set_title(title or self.options.title or "Chart", fontsize=14, fontweight="bold", pad=20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        self._apply_theme(fig, ax)
        plt.tight_layout()

        return fig

    async def create_horizontal_bar_chart(
        self,
        data: Dict[str, float],
        title: Optional[str] = None,
    ) -> Any:
        """
        Create a horizontal bar chart.

        Args:
            data: Dictionary of label -> value
            title: Chart title

        Returns:
            Matplotlib figure
        """
        plt = self._get_matplotlib()

        fig, ax = plt.subplots(figsize=(self.options.width, self.options.height))

        colors = self._get_colors()

        labels = list(data.keys())
        values = list(data.values())

        bars = ax.barh(labels, values, color=colors["primary"], edgecolor="white", linewidth=1)

        if self.options.show_values:
            for bar, value in zip(bars, values):
                width = bar.get_width()
                ax.annotate(
                    f"{value:.2f}",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                )

        ax.set_title(title or self.options.title or "Chart", fontsize=14, fontweight="bold", pad=20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        self._apply_theme(fig, ax)
        plt.tight_layout()

        return fig

    async def create_pie_chart(
        self,
        data: Dict[str, float],
        title: Optional[str] = None,
    ) -> Any:
        """
        Create a pie chart.

        Args:
            data: Dictionary of label -> value
            title: Chart title

        Returns:
            Matplotlib figure
        """
        plt = self._get_matplotlib()

        fig, ax = plt.subplots(figsize=(self.options.width, self.options.height))

        labels = list(data.keys())
        values = list(data.values())

        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%" if self.options.show_values else "",
            startangle=90,
            explode=[0.02] * len(values),
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )

        ax.set_title(title or self.options.title or "Chart", fontsize=14, fontweight="bold", pad=20)

        self._apply_theme(fig, ax)
        plt.tight_layout()

        return fig

    async def create_donut_chart(
        self,
        data: Dict[str, float],
        title: Optional[str] = None,
    ) -> Any:
        """
        Create a donut chart.

        Args:
            data: Dictionary of label -> value
            title: Chart title

        Returns:
            Matplotlib figure
        """
        plt = self._get_matplotlib()

        fig, ax = plt.subplots(figsize=(self.options.width, self.options.height))

        labels = list(data.keys())
        values = list(data.values())

        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%" if self.options.show_values else "",
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )

        # Create donut hole
        centre_circle = plt.Circle((0, 0), 0.60, fc="white")
        ax.add_patch(centre_circle)

        ax.set_title(title or self.options.title or "Chart", fontsize=14, fontweight="bold", pad=20)

        self._apply_theme(fig, ax)
        plt.tight_layout()

        return fig

    async def create_histogram(
        self,
        data: List[float],
        title: Optional[str] = None,
        bins: int = 20,
    ) -> Any:
        """
        Create a histogram.

        Args:
            data: List of values
            title: Chart title
            bins: Number of bins

        Returns:
            Matplotlib figure
        """
        return await self.create_score_histogram(data, title, bins)

    async def create_line_chart(
        self,
        data: Dict[str, List[float]],
        title: Optional[str] = None,
        x_labels: Optional[List[str]] = None,
    ) -> Any:
        """
        Create a line chart.

        Args:
            data: Dictionary of series_name -> values
            title: Chart title
            x_labels: X-axis labels

        Returns:
            Matplotlib figure
        """
        plt = self._get_matplotlib()
        import numpy as np

        fig, ax = plt.subplots(figsize=(self.options.width, self.options.height))

        colors = self._get_colors()
        color_cycle = [colors["primary"], colors["secondary"], colors["positive"], colors["negative"]]

        for i, (name, values) in enumerate(data.items()):
            x = x_labels if x_labels else list(range(len(values)))
            ax.plot(x, values, label=name, color=color_cycle[i % len(color_cycle)], linewidth=2, marker="o")

        ax.set_title(title or self.options.title or "Chart", fontsize=14, fontweight="bold", pad=20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if self.options.show_legend and len(data) > 1:
            ax.legend()

        self._apply_theme(fig, ax)
        plt.tight_layout()

        return fig


# Convenience functions


def get_visualizer(theme: str = "default", **kwargs: Any) -> ChartVisualizer:
    """
    Get a chart visualizer instance.

    Args:
        theme: Visual theme
        **kwargs: Additional options

    Returns:
        ChartVisualizer instance
    """
    options = VisualizationOptions(
        theme=Theme(theme) if isinstance(theme, str) else theme,
        **{k: v for k, v in kwargs.items() if hasattr(VisualizationOptions, k)},
    )
    return ChartVisualizer(options)


async def create_sentiment_chart(
    data: Any,
    path: Optional[str] = None,
    chart_type: str = "bar",
    **kwargs: Any,
) -> VisualizationResult:
    """
    Quick create and optionally save a sentiment chart.

    Args:
        data: Analysis result data
        path: Output file path (if None, returns figure)
        chart_type: Chart type ("bar", "pie", "donut")
        **kwargs: Additional options

    Returns:
        VisualizationResult (or figure if path is None)
    """
    visualizer = get_visualizer(**kwargs)

    if chart_type == "bar":
        fig = await visualizer.create_sentiment_bar_chart(data)
    elif chart_type == "pie":
        fig = await visualizer.create_sentiment_pie_chart(data, donut=False)
    elif chart_type == "donut":
        fig = await visualizer.create_sentiment_pie_chart(data, donut=True)
    else:
        raise ValueError(f"Unknown chart type: {chart_type}")

    if path:
        return await visualizer.save(fig, path)

    return VisualizationResult(success=True, chart_type=ChartType(chart_type))


async def create_emotion_chart(
    data: Any,
    path: Optional[str] = None,
    **kwargs: Any,
) -> VisualizationResult:
    """
    Quick create and optionally save an emotion chart.

    Args:
        data: Analysis result data
        path: Output file path
        **kwargs: Additional options

    Returns:
        VisualizationResult
    """
    visualizer = get_visualizer(**kwargs)
    fig = await visualizer.create_emotion_bar_chart(data)

    if path:
        return await visualizer.save(fig, path)

    return VisualizationResult(success=True, chart_type=ChartType.HORIZONTAL_BAR)


__all__ = [
    "ChartType",
    "Theme",
    "VisualizationOptions",
    "VisualizationResult",
    "BaseVisualizer",
    "ChartVisualizer",
    "get_visualizer",
    "create_sentiment_chart",
    "create_emotion_chart",
]
