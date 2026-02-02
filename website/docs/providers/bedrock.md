---
title: AWS Bedrock
description: Use AWS Bedrock's foundation models with Sentimatrix
---

# AWS Bedrock

AWS Bedrock provides access to multiple foundation models (Claude, Llama, Titan, Mistral, Cohere) through AWS infrastructure with enterprise security and compliance.

<span class="status stable">Stable</span>

## Quick Facts

| Property | Value |
|----------|-------|
| **Models** | Claude, Llama, Titan, Mistral, Cohere |
| **Security** | AWS IAM, VPC, encryption |
| **Compliance** | SOC, HIPAA, FedRAMP |
| **Streaming** | :material-check: Supported |
| **Functions** | :material-check: Supported |
| **Vision** | :material-check: Supported (Claude) |

## Setup

### Prerequisites

1. AWS Account with Bedrock access
2. IAM credentials with Bedrock permissions
3. Enable models in Bedrock console

### Configure

=== "Environment Variables"

    ```bash
    export AWS_ACCESS_KEY_ID="..."
    export AWS_SECRET_ACCESS_KEY="..."
    export AWS_REGION="us-east-1"
    ```

=== "Python"

    ```python
    from sentimatrix.config import SentimatrixConfig, LLMConfig

    config = SentimatrixConfig(
        llm=LLMConfig(
            provider="bedrock",
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            aws_region="us-east-1"
        )
    )
    ```

=== "YAML"

    ```yaml
    llm:
      provider: bedrock
      model: anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_region: us-east-1
    ```

## Available Models

| Provider | Model ID | Best For |
|----------|----------|----------|
| **Anthropic** | `anthropic.claude-3-5-sonnet-20241022-v2:0` | General, reasoning |
| **Anthropic** | `anthropic.claude-3-haiku-20240307-v1:0` | Fast, cost-effective |
| **Meta** | `meta.llama3-70b-instruct-v1:0` | Open-source quality |
| **Amazon** | `amazon.titan-text-premier-v1:0` | AWS-native |
| **Mistral** | `mistral.mistral-large-2407-v1:0` | European compliance |
| **Cohere** | `cohere.command-r-plus-v1:0` | RAG, embeddings |

## Usage Examples

### Basic Usage

```python
import asyncio
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

config = SentimatrixConfig(
    llm=LLMConfig(
        provider="bedrock",
        model="anthropic.claude-3-5-sonnet-20241022-v2:0"
    )
)

async def main():
    async with Sentimatrix(config) as sm:
        insights = await sm.generate_insights(reviews)
        print(insights)

asyncio.run(main())
```

### With IAM Role

```python
config = SentimatrixConfig(
    llm=LLMConfig(
        provider="bedrock",
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        aws_region="us-east-1",
        # Uses default credential chain (IAM role, env vars, etc.)
    )
)
```

## Configuration Options

```python
LLMConfig(
    provider="bedrock",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",

    # AWS settings
    aws_region="us-east-1",
    aws_access_key_id="...",  # Optional if using IAM role
    aws_secret_access_key="...",

    # Generation settings
    temperature=0.7,
    max_tokens=4096,

    # Reliability
    timeout=60,
    max_retries=3,
)
```

## Best Practices

1. **Use IAM Roles in Production**
    - Avoid hardcoded credentials
    - Use EC2 instance roles or EKS service accounts

2. **Choose Region Carefully**
    - Not all models available in all regions
    - Consider latency and data residency

3. **Enable Model Access**
    - Models must be enabled in Bedrock console
    - Some require approval

## Related

- [Provider Overview](index.md)
- [Azure OpenAI](azure.md)
- [Selection Guide](selection-guide.md)
