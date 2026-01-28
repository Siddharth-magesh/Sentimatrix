"""
Sentimatrix Output Module

Provides output handlers, exporters, and visualizers for analysis results:

Exporters:
- JSONExporter: Export to JSON files
- CSVExporter: Export to CSV files
- ExcelExporter: Export to XLSX files

Formatters:
- HTMLFormatter: Format as rich HTML reports
- TextFormatter: Format as plain text
- MarkdownFormatter: Format as Markdown

Visualizers:
- ChartVisualizer: Create charts (bar, pie, histogram, line)

Example:
    >>> from sentimatrix.output import (
    ...     JSONExporter,
    ...     CSVExporter,
    ...     ExcelExporter,
    ...     HTMLFormatter,
    ...     ChartVisualizer,
    ... )
    >>>
    >>> # Export to JSON
    >>> exporter = JSONExporter()
    >>> await exporter.export(analysis_result, "results.json")
    >>>
    >>> # Create HTML report
    >>> formatter = HTMLFormatter()
    >>> html = await formatter.format(analysis_result)
    >>>
    >>> # Create visualization
    >>> visualizer = ChartVisualizer()
    >>> fig = await visualizer.create_sentiment_bar_chart(analysis_result)
    >>> await visualizer.save(fig, "chart.png")
"""

from sentimatrix.output.exporters import (
    ExportFormat,
    ExportOptions,
    ExportResult,
    BaseExporter,
    JSONExporter,
    CSVExporter,
    ExcelExporter,
    get_exporter,
    export_to_json,
    export_to_csv,
    export_to_excel,
)

from sentimatrix.output.formatters import (
    FormatOptions,
    FormatResult,
    BaseFormatter,
    HTMLFormatter,
    TextFormatter,
    MarkdownFormatter,
    get_formatter,
    format_as_html,
    format_as_text,
    format_as_markdown,
)

from sentimatrix.output.visualizers import (
    ChartType,
    Theme,
    VisualizationOptions,
    VisualizationResult,
    BaseVisualizer,
    ChartVisualizer,
    get_visualizer,
    create_sentiment_chart,
    create_emotion_chart,
)


__all__ = [
    # Exporters
    "ExportFormat",
    "ExportOptions",
    "ExportResult",
    "BaseExporter",
    "JSONExporter",
    "CSVExporter",
    "ExcelExporter",
    "get_exporter",
    "export_to_json",
    "export_to_csv",
    "export_to_excel",
    # Formatters
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
    # Visualizers
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
