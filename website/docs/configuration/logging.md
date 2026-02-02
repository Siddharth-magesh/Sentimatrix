---
title: Logging Configuration
description: Configure logging output and format
---

# Logging Configuration

Configure how Sentimatrix logs information.

## Basic Configuration

```python
LogConfig(
    level="INFO",
    format="json",
    console_output=True
)
```

```yaml
logging:
  level: INFO
  format: json
  console_output: true
```

## Log Levels

| Level | Description |
|-------|-------------|
| `DEBUG` | Detailed diagnostic info |
| `INFO` | General operational info |
| `WARNING` | Warning messages |
| `ERROR` | Error messages |
| `CRITICAL` | Critical failures |

## File Logging

```python
LogConfig(
    level="INFO",
    file_path="logs/sentimatrix.log",
    max_file_size_mb=10,
    backup_count=5
)
```

```yaml
logging:
  level: INFO
  file_path: logs/sentimatrix.log
  max_file_size_mb: 10
  backup_count: 5
```

## Log Formats

### JSON Format (Default)

```json
{"timestamp": "2024-01-15T10:30:00", "level": "INFO", "message": "Analyzing text", "module": "sentimatrix.core"}
```

### Text Format

```
2024-01-15 10:30:00 INFO [sentimatrix.core] Analyzing text
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `level` | Log level | `INFO` |
| `format` | Output format | `json` |
| `file_path` | Log file path | `None` |
| `max_file_size_mb` | Max file size | `10` |
| `backup_count` | Backup files | `5` |
| `console_output` | Log to console | `True` |
| `colorize` | Colorized output | `True` |

## Related

- [Configuration Overview](index.md)
