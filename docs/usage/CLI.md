# Sentimatrix CLI Reference

The Sentimatrix command-line interface provides quick access to sentiment analysis, emotion detection, and web scraping without writing code.

## Installation

The CLI is automatically installed with Sentimatrix:

```bash
pip install sentimatrix
```

Verify installation:

```bash
sentimatrix --version
```

## Commands Overview

| Command | Description |
|---------|-------------|
| `analyze` | Analyze sentiment of a single text |
| `analyze-file` | Batch analyze texts from a file |
| `scrape` | Scrape reviews from web platforms |
| `batch` | Process CSV files with sentiment analysis |
| `info` | Display system information |

---

## `analyze` - Single Text Analysis

Analyze the sentiment of a single text string.

### Usage

```bash
sentimatrix analyze TEXT [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `TEXT` | The text to analyze (required) |

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--emotions` | `-e` | Include emotion detection |
| `--model` | `-m` | Model to use for analysis |
| `--output` | `-o` | Output file path |
| `--json` | | Output as JSON to stdout |

### Examples

```bash
# Basic sentiment analysis
sentimatrix analyze "I love this product!"

# With emotion detection
sentimatrix analyze "I'm so frustrated with this!" --emotions

# Output as JSON
sentimatrix analyze "Great experience!" --json

# Save to file
sentimatrix analyze "Amazing product!" -o result.json
```

### Output

**Table format (default):**
```
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Text                 ┃ Sentiment┃ Confidence ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━┩
│ I love this product! │ positive │     95.00% │
└──────────────────────┴──────────┴────────────┘
```

**JSON format:**
```json
{
  "text": "I love this product!",
  "sentiment": {
    "label": "positive",
    "confidence": 0.95,
    "scores": {
      "positive": 0.95,
      "negative": 0.02,
      "neutral": 0.03
    }
  }
}
```

---

## `analyze-file` - Batch File Analysis

Analyze sentiment of multiple texts from a file.

### Usage

```bash
sentimatrix analyze-file INPUT [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `INPUT` | Input file path (txt, csv, or json) |

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--output` | `-o` | Output file path |
| `--emotions` | `-e` | Include emotion detection |

### Supported Input Formats

**Text file (.txt)** - One text per line:
```
Great product!
Terrible experience.
It's okay.
```

**CSV file (.csv)** - Must have a `text` column:
```csv
text,rating
Great product!,5
Terrible experience.,1
```

**JSON file (.json)** - Array of objects or texts:
```json
[
  {"text": "Great product!"},
  {"text": "Terrible experience."}
]
```

### Examples

```bash
# Analyze text file
sentimatrix analyze-file reviews.txt

# Analyze CSV and save results
sentimatrix analyze-file data.csv -o results.json

# Include emotions
sentimatrix analyze-file feedback.txt --emotions -o analysis.csv
```

---

## `scrape` - Web Scraping

Scrape reviews from supported platforms.

### Usage

```bash
sentimatrix scrape PLATFORM IDENTIFIER [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `PLATFORM` | Platform to scrape: `amazon`, `steam`, `youtube`, `reddit` |
| `IDENTIFIER` | Product ID, game ID, video ID, or post ID |

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--limit` | `-l` | Maximum reviews to scrape (default: 50) |
| `--analyze` | `-a` | Analyze sentiment of scraped reviews |
| `--output` | `-o` | Output file path (JSON) |

### Supported Platforms

| Platform | Identifier | Example |
|----------|------------|---------|
| Amazon | ASIN (10-char product ID) | `B08N5WRWNW` |
| Steam | App ID (numeric) | `730` (CS:GO) |
| YouTube | Video ID (11 chars) | `dQw4w9WgXcQ` |
| Reddit | Post ID | `abc123` |

### Examples

```bash
# Scrape Amazon product reviews
sentimatrix scrape amazon B08N5WRWNW --limit 100

# Scrape and analyze Steam reviews
sentimatrix scrape steam 730 --limit 50 --analyze

# Scrape YouTube comments and save
sentimatrix scrape youtube dQw4w9WgXcQ -o comments.json

# Scrape Reddit comments with analysis
sentimatrix scrape reddit t3_abc123 --analyze -o reddit_analysis.json
```

### Output

```json
[
  {
    "id": "R123ABC456",
    "text": "Great product!",
    "rating": 5.0,
    "author": "JohnDoe",
    "source": "amazon",
    "sentiment": {
      "label": "positive",
      "confidence": 0.95
    }
  }
]
```

---

## `batch` - CSV Batch Processing

Process a CSV file and add sentiment analysis columns.

### Usage

```bash
sentimatrix batch INPUT -o OUTPUT [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `INPUT` | Input CSV file |

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--output` | `-o` | Output CSV file (required) |
| `--emotions` | `-e` | Include emotion detection |

### Input Format

CSV must have a `text` or `content` column:

```csv
id,text,date
1,Great product!,2024-01-15
2,Terrible experience.,2024-01-16
```

### Output Format

Adds sentiment columns:

```csv
id,text,date,sentiment_label,sentiment_confidence
1,Great product!,2024-01-15,positive,0.95
2,Terrible experience.,2024-01-16,negative,0.88
```

With `--emotions`:

```csv
id,text,date,sentiment_label,sentiment_confidence,primary_emotion
1,Great product!,2024-01-15,positive,0.95,joy
2,Terrible experience.,2024-01-16,negative,0.88,anger
```

### Examples

```bash
# Basic batch processing
sentimatrix batch input.csv -o output.csv

# With emotions
sentimatrix batch reviews.csv -o analyzed.csv --emotions
```

---

## `info` - System Information

Display Sentimatrix version and system information.

### Usage

```bash
sentimatrix info [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--json` | Output as JSON |

### Examples

```bash
# Display info
sentimatrix info

# JSON output
sentimatrix info --json
```

### Output

```
╭──────────────────── System Information ─────────────────────╮
│ Sentimatrix v0.2.0                                          │
│                                                             │
│ Python: 3.11.6                                              │
│ Platform: Linux-5.15.0-x86_64-with-glibc2.35                │
│                                                             │
│ Optional Dependencies:                                      │
│   torch: 2.1.0                                              │
│   transformers: 4.36.0                                      │
│   openai: 1.6.0                                             │
│   playwright: installed                                     │
╰─────────────────────────────────────────────────────────────╯
```

---

## Global Options

These options are available for all commands:

| Option | Description |
|--------|-------------|
| `--help` | Show help message |
| `--version` | Show version |

---

## Environment Variables

The CLI respects these environment variables:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (for LLM features) |
| `GROQ_API_KEY` | Groq API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `YOUTUBE_API_KEY` | YouTube Data API key |

---

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | General error |
| 130 | Interrupted (Ctrl+C) |

---

## Examples: Full Workflows

### Analyze Customer Feedback

```bash
# 1. Export feedback to text file (one per line)
# 2. Run analysis
sentimatrix analyze-file customer_feedback.txt \
    --emotions \
    -o feedback_analysis.json

# 3. View results
cat feedback_analysis.json | jq '.[] | {text: .text[:50], sentiment: .sentiment.label}'
```

### Product Review Analysis

```bash
# Scrape and analyze Amazon reviews
sentimatrix scrape amazon B08N5WRWNW \
    --limit 200 \
    --analyze \
    -o amazon_reviews.json

# Count sentiments
cat amazon_reviews.json | jq '[.[] | .sentiment.label] | group_by(.) | map({(.[0]): length})'
```

### Compare Two Products

```bash
# Scrape both products
sentimatrix scrape amazon ASIN_A --analyze -o product_a.json
sentimatrix scrape amazon ASIN_B --analyze -o product_b.json

# Compare positive ratios
echo "Product A:"
cat product_a.json | jq '[.[] | select(.sentiment.label == "positive")] | length'

echo "Product B:"
cat product_b.json | jq '[.[] | select(.sentiment.label == "positive")] | length'
```

---

## Troubleshooting

### Command Not Found

If `sentimatrix` is not found:

```bash
# Ensure pip installed correctly
pip install --upgrade sentimatrix

# Or use Python module directly
python -m sentimatrix.cli --help
```

### Rich Not Installed

For beautiful terminal output, install Rich:

```bash
pip install rich
```

Without Rich, output falls back to plain text.

### Slow First Run

The first analysis may be slow as models are downloaded. Subsequent runs use cached models.

---

## See Also

- [Quick Start Guide](./QUICKSTART.md)
- [Configuration Reference](./CONFIGURATION.md)
- [API Reference](../api/REFERENCE.md)
