"""
Sentimatrix Output Exporters

Provides export functionality for analysis results to various formats:
- JSON: Structured JSON files with full detail
- CSV: Tabular format for spreadsheet compatibility
- Excel: Rich XLSX files with multiple sheets

Example:
    >>> from sentimatrix.output.exporters import JSONExporter, CSVExporter, ExcelExporter
    >>>
    >>> # Export analysis results
    >>> exporter = JSONExporter()
    >>> await exporter.export(analysis_result, "output.json")
    >>>
    >>> # Export with options
    >>> exporter = CSVExporter(include_metadata=True)
    >>> await exporter.export_reviews(reviews, "reviews.csv")
"""

from __future__ import annotations

import csv
import gzip
import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from sentimatrix.core.logger import get_logger

logger = get_logger(__name__)


class ExportFormat(str, Enum):
    """Supported export formats."""

    JSON = "json"
    CSV = "csv"
    XLSX = "xlsx"
    HTML = "html"


@dataclass
class ExportOptions:
    """
    Options for export operations.

    Attributes:
        format: Export format
        path: Output file path
        include_raw: Include raw/original data
        include_metadata: Include metadata fields
        compression: Compression type ("gzip", "zip", None)
        pretty_print: Pretty print JSON output
        date_format: Format for datetime fields
        encoding: File encoding
    """

    format: ExportFormat = ExportFormat.JSON
    path: Optional[str] = None
    include_raw: bool = False
    include_metadata: bool = True
    compression: Optional[str] = None
    pretty_print: bool = True
    date_format: str = "%Y-%m-%d %H:%M:%S"
    encoding: str = "utf-8"

    def with_path(self, path: str) -> "ExportOptions":
        """Create new options with different path."""
        return ExportOptions(
            format=self.format,
            path=path,
            include_raw=self.include_raw,
            include_metadata=self.include_metadata,
            compression=self.compression,
            pretty_print=self.pretty_print,
            date_format=self.date_format,
            encoding=self.encoding,
        )


@dataclass
class ExportResult:
    """
    Result of an export operation.

    Attributes:
        success: Whether export succeeded
        path: Output file path
        format: Export format used
        records_exported: Number of records exported
        file_size_bytes: Size of output file
        duration_ms: Time taken for export
        error: Error message if failed
    """

    success: bool
    path: str
    format: ExportFormat
    records_exported: int = 0
    file_size_bytes: int = 0
    duration_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "path": self.path,
            "format": self.format.value,
            "records_exported": self.records_exported,
            "file_size_bytes": self.file_size_bytes,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


class BaseExporter(ABC):
    """
    Abstract base class for all exporters.

    Provides common functionality for export operations.
    """

    def __init__(self, options: Optional[ExportOptions] = None) -> None:
        """
        Initialize exporter.

        Args:
            options: Export options
        """
        self.options = options or ExportOptions()

    @property
    @abstractmethod
    def format(self) -> ExportFormat:
        """Get export format."""
        pass

    @abstractmethod
    async def export(
        self,
        data: Any,
        path: Optional[str] = None,
        **kwargs: Any,
    ) -> ExportResult:
        """
        Export data to file.

        Args:
            data: Data to export
            path: Output file path (overrides options)
            **kwargs: Additional options

        Returns:
            ExportResult with operation details
        """
        pass

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value for export."""
        if value is None:
            return None
        elif isinstance(value, datetime):
            return value.strftime(self.options.date_format)
        elif isinstance(value, Enum):
            return value.value
        elif is_dataclass(value) and not isinstance(value, type):
            return self._serialize_dataclass(value)
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        elif hasattr(value, "to_dict"):
            return value.to_dict()
        else:
            return value

    def _serialize_dataclass(self, obj: Any) -> Dict[str, Any]:
        """Serialize a dataclass to dictionary."""
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        result = {}
        for field_name in obj.__dataclass_fields__:
            value = getattr(obj, field_name)
            result[field_name] = self._serialize_value(value)
        return result

    def _ensure_directory(self, path: str) -> None:
        """Ensure parent directory exists."""
        parent = Path(path).parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

    def _get_file_size(self, path: str) -> int:
        """Get file size in bytes."""
        try:
            return os.path.getsize(path)
        except OSError:
            return 0


class JSONExporter(BaseExporter):
    """
    Export data to JSON format.

    Supports:
    - Single objects and lists
    - Dataclass serialization
    - Pretty printing
    - Gzip compression

    Example:
        >>> exporter = JSONExporter()
        >>> result = await exporter.export(analysis, "results.json")
        >>> print(f"Exported to {result.path}")
    """

    @property
    def format(self) -> ExportFormat:
        return ExportFormat.JSON

    async def export(
        self,
        data: Any,
        path: Optional[str] = None,
        **kwargs: Any,
    ) -> ExportResult:
        """
        Export data to JSON file.

        Args:
            data: Data to export (dict, list, or dataclass)
            path: Output file path
            **kwargs: Additional options (indent, sort_keys)

        Returns:
            ExportResult
        """
        import time

        start_time = time.time()
        output_path = path or self.options.path

        if not output_path:
            return ExportResult(
                success=False,
                path="",
                format=self.format,
                error="No output path specified",
            )

        try:
            # Serialize data
            serialized = self._serialize_value(data)

            # Determine formatting
            indent = kwargs.get("indent", 2 if self.options.pretty_print else None)
            sort_keys = kwargs.get("sort_keys", False)

            # Convert to JSON string
            json_str = json.dumps(
                serialized,
                indent=indent,
                sort_keys=sort_keys,
                ensure_ascii=False,
                default=str,
            )

            # Ensure directory exists
            self._ensure_directory(output_path)

            # Write file (with optional compression)
            if self.options.compression == "gzip":
                if not output_path.endswith(".gz"):
                    output_path += ".gz"
                with gzip.open(output_path, "wt", encoding=self.options.encoding) as f:
                    f.write(json_str)
            else:
                with open(output_path, "w", encoding=self.options.encoding) as f:
                    f.write(json_str)

            # Count records
            record_count = 1
            if isinstance(serialized, list):
                record_count = len(serialized)
            elif isinstance(serialized, dict) and "reviews" in serialized:
                record_count = len(serialized.get("reviews", []))

            duration_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Exported {record_count} records to JSON",
                path=output_path,
                size=self._get_file_size(output_path),
            )

            return ExportResult(
                success=True,
                path=output_path,
                format=self.format,
                records_exported=record_count,
                file_size_bytes=self._get_file_size(output_path),
                duration_ms=duration_ms,
            )

        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            return ExportResult(
                success=False,
                path=output_path,
                format=self.format,
                error=str(e),
            )

    async def export_to_string(self, data: Any, **kwargs: Any) -> str:
        """
        Export data to JSON string (without writing to file).

        Args:
            data: Data to export
            **kwargs: Additional options

        Returns:
            JSON string
        """
        serialized = self._serialize_value(data)
        indent = kwargs.get("indent", 2 if self.options.pretty_print else None)
        return json.dumps(
            serialized,
            indent=indent,
            ensure_ascii=False,
            default=str,
        )


class CSVExporter(BaseExporter):
    """
    Export data to CSV format.

    Supports:
    - Lists of dictionaries/dataclasses
    - Automatic column detection
    - Custom column ordering
    - Nested field flattening

    Example:
        >>> exporter = CSVExporter()
        >>> result = await exporter.export(reviews, "reviews.csv")
    """

    def __init__(
        self,
        options: Optional[ExportOptions] = None,
        columns: Optional[List[str]] = None,
        include_header: bool = True,
        delimiter: str = ",",
        quotechar: str = '"',
    ) -> None:
        """
        Initialize CSV exporter.

        Args:
            options: Export options
            columns: Column names to include (None = auto-detect)
            include_header: Include header row
            delimiter: Field delimiter
            quotechar: Quote character
        """
        super().__init__(options)
        self.columns = columns
        self.include_header = include_header
        self.delimiter = delimiter
        self.quotechar = quotechar

    @property
    def format(self) -> ExportFormat:
        return ExportFormat.CSV

    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = "",
        sep: str = "_",
    ) -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items: List[tuple] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, (list, tuple)):
                # Convert lists to string representation
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)

    def _prepare_row(self, item: Any) -> Dict[str, Any]:
        """Prepare a single row for CSV export."""
        if isinstance(item, dict):
            flat = self._flatten_dict(item)
        elif hasattr(item, "to_dict"):
            flat = self._flatten_dict(item.to_dict())
        elif is_dataclass(item) and not isinstance(item, type):
            flat = self._flatten_dict(self._serialize_dataclass(item))
        else:
            flat = {"value": item}

        # Serialize remaining values
        return {k: self._serialize_value(v) for k, v in flat.items()}

    async def export(
        self,
        data: Any,
        path: Optional[str] = None,
        **kwargs: Any,
    ) -> ExportResult:
        """
        Export data to CSV file.

        Args:
            data: Data to export (list of dicts/dataclasses)
            path: Output file path
            **kwargs: Additional options

        Returns:
            ExportResult
        """
        import time

        start_time = time.time()
        output_path = path or self.options.path

        if not output_path:
            return ExportResult(
                success=False,
                path="",
                format=self.format,
                error="No output path specified",
            )

        try:
            # Ensure data is a list
            if not isinstance(data, list):
                if hasattr(data, "reviews"):
                    data = data.reviews
                elif hasattr(data, "results"):
                    data = data.results
                else:
                    data = [data]

            if not data:
                return ExportResult(
                    success=True,
                    path=output_path,
                    format=self.format,
                    records_exported=0,
                )

            # Prepare rows
            rows = [self._prepare_row(item) for item in data]

            # Determine columns
            columns = self.columns
            if not columns:
                # Auto-detect from first row
                all_keys: set = set()
                for row in rows:
                    all_keys.update(row.keys())
                columns = sorted(all_keys)

            # Ensure directory exists
            self._ensure_directory(output_path)

            # Write CSV
            with open(output_path, "w", newline="", encoding=self.options.encoding) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=columns,
                    delimiter=self.delimiter,
                    quotechar=self.quotechar,
                    extrasaction="ignore",
                )

                if self.include_header:
                    writer.writeheader()

                for row in rows:
                    writer.writerow(row)

            duration_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Exported {len(rows)} records to CSV",
                path=output_path,
                columns=len(columns),
            )

            return ExportResult(
                success=True,
                path=output_path,
                format=self.format,
                records_exported=len(rows),
                file_size_bytes=self._get_file_size(output_path),
                duration_ms=duration_ms,
            )

        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            return ExportResult(
                success=False,
                path=output_path,
                format=self.format,
                error=str(e),
            )

    async def export_to_string(self, data: Any, **kwargs: Any) -> str:
        """
        Export data to CSV string.

        Args:
            data: Data to export
            **kwargs: Additional options

        Returns:
            CSV string
        """
        # Ensure data is a list
        if not isinstance(data, list):
            if hasattr(data, "reviews"):
                data = data.reviews
            elif hasattr(data, "results"):
                data = data.results
            else:
                data = [data]

        if not data:
            return ""

        # Prepare rows
        rows = [self._prepare_row(item) for item in data]

        # Determine columns
        columns = self.columns
        if not columns:
            all_keys: set = set()
            for row in rows:
                all_keys.update(row.keys())
            columns = sorted(all_keys)

        # Write to string
        output = StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=columns,
            delimiter=self.delimiter,
            quotechar=self.quotechar,
            extrasaction="ignore",
        )

        if self.include_header:
            writer.writeheader()

        for row in rows:
            writer.writerow(row)

        return output.getvalue()


class ExcelExporter(BaseExporter):
    """
    Export data to Excel (XLSX) format.

    Supports:
    - Multiple sheets (reviews, summary, insights)
    - Automatic column sizing
    - Header styling
    - Number/date formatting

    Requires: openpyxl

    Example:
        >>> exporter = ExcelExporter()
        >>> result = await exporter.export(analysis_result, "report.xlsx")
    """

    def __init__(
        self,
        options: Optional[ExportOptions] = None,
        sheet_name: str = "Data",
        auto_size_columns: bool = True,
        header_style: bool = True,
    ) -> None:
        """
        Initialize Excel exporter.

        Args:
            options: Export options
            sheet_name: Default sheet name
            auto_size_columns: Auto-size column widths
            header_style: Apply header styling
        """
        super().__init__(options)
        self.sheet_name = sheet_name
        self.auto_size_columns = auto_size_columns
        self.header_style = header_style

    @property
    def format(self) -> ExportFormat:
        return ExportFormat.XLSX

    def _get_openpyxl(self):
        """Import openpyxl (lazy import)."""
        try:
            import openpyxl
            from openpyxl.styles import Font, PatternFill, Alignment
            from openpyxl.utils import get_column_letter

            return openpyxl, Font, PatternFill, Alignment, get_column_letter
        except ImportError:
            raise ImportError(
                "openpyxl is required for Excel export. "
                "Install it with: pip install openpyxl"
            )

    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = "",
        sep: str = "_",
    ) -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items: List[tuple] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, (list, tuple)):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)

    def _prepare_row(self, item: Any) -> Dict[str, Any]:
        """Prepare a single row for Excel export."""
        if isinstance(item, dict):
            flat = self._flatten_dict(item)
        elif hasattr(item, "to_dict"):
            flat = self._flatten_dict(item.to_dict())
        elif is_dataclass(item) and not isinstance(item, type):
            flat = self._flatten_dict(self._serialize_dataclass(item))
        else:
            flat = {"value": item}

        return {k: self._serialize_value(v) for k, v in flat.items()}

    async def export(
        self,
        data: Any,
        path: Optional[str] = None,
        **kwargs: Any,
    ) -> ExportResult:
        """
        Export data to Excel file.

        Args:
            data: Data to export
            path: Output file path
            **kwargs: Additional options (sheet_name)

        Returns:
            ExportResult
        """
        import time

        start_time = time.time()
        output_path = path or self.options.path

        if not output_path:
            return ExportResult(
                success=False,
                path="",
                format=self.format,
                error="No output path specified",
            )

        try:
            openpyxl, Font, PatternFill, Alignment, get_column_letter = self._get_openpyxl()

            # Create workbook
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = kwargs.get("sheet_name", self.sheet_name)

            # Ensure data is a list
            if not isinstance(data, list):
                if hasattr(data, "reviews"):
                    data = data.reviews
                elif hasattr(data, "results"):
                    data = data.results
                else:
                    data = [data]

            if not data:
                # Save empty workbook
                self._ensure_directory(output_path)
                wb.save(output_path)
                return ExportResult(
                    success=True,
                    path=output_path,
                    format=self.format,
                    records_exported=0,
                    file_size_bytes=self._get_file_size(output_path),
                )

            # Prepare rows
            rows = [self._prepare_row(item) for item in data]

            # Get all columns
            all_keys: set = set()
            for row in rows:
                all_keys.update(row.keys())
            columns = sorted(all_keys)

            # Write header
            for col_idx, col_name in enumerate(columns, 1):
                cell = ws.cell(row=1, column=col_idx, value=col_name)
                if self.header_style:
                    cell.font = Font(bold=True)
                    cell.fill = PatternFill(
                        start_color="4472C4",
                        end_color="4472C4",
                        fill_type="solid",
                    )
                    cell.font = Font(bold=True, color="FFFFFF")
                    cell.alignment = Alignment(horizontal="center")

            # Write data rows
            for row_idx, row_data in enumerate(rows, 2):
                for col_idx, col_name in enumerate(columns, 1):
                    value = row_data.get(col_name, "")
                    ws.cell(row=row_idx, column=col_idx, value=value)

            # Auto-size columns
            if self.auto_size_columns:
                for col_idx, col_name in enumerate(columns, 1):
                    max_length = len(str(col_name))
                    for row_data in rows:
                        value = row_data.get(col_name, "")
                        max_length = max(max_length, len(str(value)))
                    # Limit column width
                    adjusted_width = min(max_length + 2, 50)
                    ws.column_dimensions[get_column_letter(col_idx)].width = adjusted_width

            # Save workbook
            self._ensure_directory(output_path)
            wb.save(output_path)

            duration_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Exported {len(rows)} records to Excel",
                path=output_path,
                sheets=1,
            )

            return ExportResult(
                success=True,
                path=output_path,
                format=self.format,
                records_exported=len(rows),
                file_size_bytes=self._get_file_size(output_path),
                duration_ms=duration_ms,
            )

        except Exception as e:
            logger.error(f"Excel export failed: {e}")
            return ExportResult(
                success=False,
                path=output_path,
                format=self.format,
                error=str(e),
            )

    async def export_multi_sheet(
        self,
        sheets: Dict[str, Any],
        path: Optional[str] = None,
    ) -> ExportResult:
        """
        Export multiple sheets to a single Excel file.

        Args:
            sheets: Dictionary of sheet_name -> data
            path: Output file path

        Returns:
            ExportResult

        Example:
            >>> await exporter.export_multi_sheet({
            ...     "Reviews": reviews,
            ...     "Summary": summary_data,
            ...     "Insights": insights_data,
            ... }, "report.xlsx")
        """
        import time

        start_time = time.time()
        output_path = path or self.options.path

        if not output_path:
            return ExportResult(
                success=False,
                path="",
                format=self.format,
                error="No output path specified",
            )

        try:
            openpyxl, Font, PatternFill, Alignment, get_column_letter = self._get_openpyxl()

            wb = openpyxl.Workbook()
            # Remove default sheet
            wb.remove(wb.active)

            total_records = 0

            for sheet_name, data in sheets.items():
                ws = wb.create_sheet(title=sheet_name[:31])  # Excel sheet name limit

                # Ensure data is a list
                if not isinstance(data, list):
                    if hasattr(data, "reviews"):
                        data = data.reviews
                    elif hasattr(data, "results"):
                        data = data.results
                    elif hasattr(data, "to_dict"):
                        data = [data]
                    else:
                        data = [data]

                if not data:
                    continue

                # Prepare rows
                rows = [self._prepare_row(item) for item in data]
                total_records += len(rows)

                # Get columns
                all_keys: set = set()
                for row in rows:
                    all_keys.update(row.keys())
                columns = sorted(all_keys)

                # Write header
                for col_idx, col_name in enumerate(columns, 1):
                    cell = ws.cell(row=1, column=col_idx, value=col_name)
                    if self.header_style:
                        cell.font = Font(bold=True, color="FFFFFF")
                        cell.fill = PatternFill(
                            start_color="4472C4",
                            end_color="4472C4",
                            fill_type="solid",
                        )
                        cell.alignment = Alignment(horizontal="center")

                # Write data
                for row_idx, row_data in enumerate(rows, 2):
                    for col_idx, col_name in enumerate(columns, 1):
                        value = row_data.get(col_name, "")
                        ws.cell(row=row_idx, column=col_idx, value=value)

                # Auto-size
                if self.auto_size_columns:
                    for col_idx, col_name in enumerate(columns, 1):
                        max_length = len(str(col_name))
                        for row_data in rows:
                            value = row_data.get(col_name, "")
                            max_length = max(max_length, len(str(value)))
                        adjusted_width = min(max_length + 2, 50)
                        ws.column_dimensions[get_column_letter(col_idx)].width = adjusted_width

            # Save
            self._ensure_directory(output_path)
            wb.save(output_path)

            duration_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Exported {total_records} records to Excel",
                path=output_path,
                sheets=len(sheets),
            )

            return ExportResult(
                success=True,
                path=output_path,
                format=self.format,
                records_exported=total_records,
                file_size_bytes=self._get_file_size(output_path),
                duration_ms=duration_ms,
            )

        except Exception as e:
            logger.error(f"Multi-sheet Excel export failed: {e}")
            return ExportResult(
                success=False,
                path=output_path,
                format=self.format,
                error=str(e),
            )


# Convenience functions


def get_exporter(format: Union[str, ExportFormat], **kwargs: Any) -> BaseExporter:
    """
    Get an exporter instance for the specified format.

    Args:
        format: Export format ("json", "csv", "xlsx")
        **kwargs: Additional exporter options

    Returns:
        Exporter instance

    Example:
        >>> exporter = get_exporter("json", pretty_print=True)
        >>> await exporter.export(data, "output.json")
    """
    if isinstance(format, str):
        format = ExportFormat(format.lower())

    if format == ExportFormat.JSON:
        return JSONExporter(**kwargs)
    elif format == ExportFormat.CSV:
        return CSVExporter(**kwargs)
    elif format == ExportFormat.XLSX:
        return ExcelExporter(**kwargs)
    else:
        raise ValueError(f"Unsupported export format: {format}")


async def export_to_json(
    data: Any,
    path: str,
    pretty_print: bool = True,
    **kwargs: Any,
) -> ExportResult:
    """
    Quick export to JSON file.

    Args:
        data: Data to export
        path: Output file path
        pretty_print: Pretty print output
        **kwargs: Additional options

    Returns:
        ExportResult
    """
    options = ExportOptions(pretty_print=pretty_print)
    exporter = JSONExporter(options)
    return await exporter.export(data, path, **kwargs)


async def export_to_csv(
    data: Any,
    path: str,
    columns: Optional[List[str]] = None,
    **kwargs: Any,
) -> ExportResult:
    """
    Quick export to CSV file.

    Args:
        data: Data to export
        path: Output file path
        columns: Column names (None = auto-detect)
        **kwargs: Additional options

    Returns:
        ExportResult
    """
    exporter = CSVExporter(columns=columns)
    return await exporter.export(data, path, **kwargs)


async def export_to_excel(
    data: Any,
    path: str,
    sheet_name: str = "Data",
    **kwargs: Any,
) -> ExportResult:
    """
    Quick export to Excel file.

    Args:
        data: Data to export
        path: Output file path
        sheet_name: Sheet name
        **kwargs: Additional options

    Returns:
        ExportResult
    """
    exporter = ExcelExporter(sheet_name=sheet_name)
    return await exporter.export(data, path, **kwargs)


__all__ = [
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
]
