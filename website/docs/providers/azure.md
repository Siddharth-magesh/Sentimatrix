---
title: Azure OpenAI
description: Integration with Azure OpenAI Service
---

# Azure OpenAI

Azure OpenAI provides OpenAI models with enterprise security, compliance, and regional deployment options.

## Quick Start

```python
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

config = SentimatrixConfig(
    llm=LLMConfig(
        provider="azure_openai",
        model="gpt-4o",  # Your deployment name
        api_key="your-azure-key",
        api_base="https://your-resource.openai.azure.com/",
    )
)

async with Sentimatrix(config) as sm:
    summary = await sm.summarize_reviews(reviews)
```

## Configuration

```python
LLMConfig(
    provider="azure_openai",
    model="your-deployment-name",     # Deployment name, not model name
    api_key="your-azure-key",         # Or AZURE_OPENAI_API_KEY
    api_base="https://your-resource.openai.azure.com/",
    temperature=0.7,
    max_tokens=4096,
    timeout=30,
)
```

## Environment Variables

```bash
export AZURE_OPENAI_API_KEY="your-azure-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT="your-deployment-name"
```

## Available Models

| Model | Azure Deployment | Context |
|-------|-----------------|---------|
| GPT-4o | gpt-4o | 128K |
| GPT-4o mini | gpt-4o-mini | 128K |
| GPT-4 Turbo | gpt-4-turbo | 128K |
| GPT-4 | gpt-4 | 8K/32K |
| GPT-3.5 Turbo | gpt-35-turbo | 16K |

## Features

- **Enterprise Security**: Private endpoints, VNET integration
- **Compliance**: SOC 2, HIPAA, GDPR ready
- **Regional Deployment**: Data residency control
- **Content Filtering**: Built-in safety filters
- **SLA**: 99.9% uptime guarantee

## Setup Steps

1. Create Azure OpenAI resource in Azure Portal
2. Deploy a model (creates a deployment)
3. Get API key and endpoint from resource
4. Configure Sentimatrix with deployment name

## Example: Enterprise Deployment

```python
# For enterprise/compliance requirements
config = SentimatrixConfig(
    llm=LLMConfig(
        provider="azure_openai",
        model="gpt-4o-production",  # Your deployment name
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_base="https://mycompany-openai.openai.azure.com/",
    )
)

async with Sentimatrix(config) as sm:
    # All data stays within Azure
    results = await sm.analyze_batch(sensitive_reviews)
```

## Pricing

Same as OpenAI models, billed through Azure:

| Model | Input | Output |
|-------|-------|--------|
| GPT-4o | $2.50/1M | $10/1M |
| GPT-4o mini | $0.15/1M | $0.60/1M |
| GPT-4 Turbo | $10/1M | $30/1M |

## Related

- [Provider Overview](index.md)
- [OpenAI](openai.md)
- [AWS Bedrock](bedrock.md)
- [Selection Guide](selection-guide.md)
