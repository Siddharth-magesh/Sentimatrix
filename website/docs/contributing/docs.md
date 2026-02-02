---
title: Documentation Guide
description: How to contribute to documentation
---

# Documentation Guide

How to contribute to Sentimatrix documentation.

## Documentation Structure

```
website/
├── mkdocs.yml           # MkDocs configuration
├── docs/
│   ├── index.md         # Homepage
│   ├── getting-started/ # Getting started guides
│   ├── providers/       # LLM provider docs
│   ├── scrapers/        # Scraper docs
│   ├── features/        # Feature documentation
│   ├── configuration/   # Config docs
│   ├── api/             # API reference
│   ├── examples/        # Example code
│   └── contributing/    # Contributing guides
└── requirements.txt     # MkDocs dependencies
```

## Local Development

```bash
cd website
pip install -r requirements.txt
mkdocs serve
```

Open http://localhost:8000 to preview.

## Writing Documentation

### Frontmatter

Every page needs frontmatter:

```markdown
---
title: Page Title
description: Brief description for SEO
---

# Page Title

Content here...
```

### Code Examples

Use fenced code blocks with language:

```python
async with Sentimatrix() as sm:
    result = await sm.analyze("Hello!")
```

### Admonitions

```markdown
!!! note
    This is a note.

!!! warning
    This is a warning.

!!! tip
    This is a tip.
```

### Tabs

```markdown
=== "Python"

    ```python
    config = SentimatrixConfig()
    ```

=== "YAML"

    ```yaml
    llm:
      provider: groq
    ```
```

## Style Guide

1. **Clear and concise**: Get to the point
2. **Code first**: Show code examples early
3. **Practical**: Focus on real use cases
4. **Consistent**: Follow existing patterns
5. **Complete**: Include all parameters

## Adding New Pages

1. Create markdown file in appropriate directory
2. Add to `nav` in `mkdocs.yml`
3. Add links from related pages
4. Preview locally before submitting

## Building for Production

```bash
mkdocs build
```

Output in `site/` directory.

## Related

- [Development Setup](setup.md)
- [Contributing Overview](index.md)
