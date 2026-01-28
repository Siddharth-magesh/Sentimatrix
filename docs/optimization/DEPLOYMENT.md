# Sentimatrix V2 - Deployment Guide

## Deployment Options

### 1. Local Development

```bash
# Install
pip install sentimatrix

# Or with all extras
pip install sentimatrix[all]

# Run
python -c "from sentimatrix import Sentimatrix; ..."
```

### 2. Docker

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PLAYWRIGHT_BROWSERS_PATH=/app/browsers

# Install Playwright browsers
RUN playwright install chromium

CMD ["python", "-m", "sentimatrix.server"]
```

**Docker Compose:**
```yaml
version: '3.8'

services:
  sentimatrix:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - GROQ_API_KEY=${GROQ_API_KEY}
    volumes:
      - ./config:/app/config
      - ./cache:/app/cache
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  redis_data:
  ollama_data:
```

### 3. Kubernetes

**Deployment:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentimatrix
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sentimatrix
  template:
    metadata:
      labels:
        app: sentimatrix
    spec:
      containers:
      - name: sentimatrix
        image: sentimatrix:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: sentimatrix-secrets
              key: openai-api-key
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

**Service:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: sentimatrix
spec:
  selector:
    app: sentimatrix
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

**HPA:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sentimatrix-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sentimatrix
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## Cloud Deployments

### AWS

**EC2:**
- Recommended: g4dn.xlarge (GPU) or c5.2xlarge (CPU)
- AMI: Deep Learning AMI

**ECS/Fargate:**
```json
{
  "containerDefinitions": [{
    "name": "sentimatrix",
    "image": "sentimatrix:latest",
    "cpu": 2048,
    "memory": 4096,
    "portMappings": [{"containerPort": 8000}]
  }]
}
```

**Lambda (for simple tasks):**
- Package as Lambda container
- Limited to 15 min timeout
- Best for event-driven processing

### Google Cloud

**Cloud Run:**
```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: sentimatrix
spec:
  template:
    spec:
      containers:
      - image: gcr.io/project/sentimatrix
        resources:
          limits:
            memory: 4Gi
            cpu: "2"
```

**GKE:**
- Use standard Kubernetes deployment
- Enable GPU node pools for inference

### Azure

**Container Apps:**
```yaml
properties:
  template:
    containers:
    - name: sentimatrix
      image: sentimatrix:latest
      resources:
        cpu: 2
        memory: 4Gi
```

---

## Configuration by Environment

### Development

```yaml
# config/development.yaml
debug: true
logging:
  level: DEBUG
cache:
  backend: memory
llm:
  provider: ollama
scrapers:
  headless: false  # See browser for debugging
```

### Staging

```yaml
# config/staging.yaml
debug: false
logging:
  level: INFO
cache:
  backend: redis
  redis_url: redis://redis:6379
llm:
  provider: groq  # Cost-effective
```

### Production

```yaml
# config/production.yaml
debug: false
logging:
  level: WARNING
  format: json
cache:
  backend: redis
  redis_url: ${REDIS_URL}
  ttl: 3600
llm:
  provider: openai
  fallback: [anthropic, groq]
scrapers:
  headless: true
  proxy:
    enabled: true
    provider: brightdata
monitoring:
  enabled: true
  provider: datadog
```

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SENTIMATRIX_CONFIG` | Config file path | No |
| `OPENAI_API_KEY` | OpenAI API key | Conditional |
| `ANTHROPIC_API_KEY` | Anthropic API key | Conditional |
| `GROQ_API_KEY` | Groq API key | Conditional |
| `REDIS_URL` | Redis connection URL | No |
| `DATABASE_URL` | Database connection URL | No |
| `LOG_LEVEL` | Logging level | No |

---

## Health Checks

**Endpoints:**
- `/health` - Basic health check
- `/ready` - Readiness (all dependencies up)
- `/metrics` - Prometheus metrics

**Health Check Implementation:**
```python
@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/ready")
async def ready():
    checks = {
        "redis": await check_redis(),
        "models": await check_models_loaded(),
        "llm": await check_llm_provider()
    }
    all_ready = all(checks.values())
    return {"ready": all_ready, "checks": checks}
```

---

## Monitoring

### Prometheus Metrics

```python
# Exposed metrics
sentimatrix_requests_total{method, endpoint}
sentimatrix_request_duration_seconds{method, endpoint}
sentimatrix_sentiment_predictions_total{label}
sentimatrix_scraper_requests_total{platform, status}
sentimatrix_llm_requests_total{provider, model}
sentimatrix_cache_hits_total
sentimatrix_cache_misses_total
```

### Logging

```yaml
logging:
  level: INFO
  format: json
  outputs:
    - type: stdout
    - type: file
      path: /var/log/sentimatrix/app.log
      rotation: daily
      retention: 7
```

---

## Security

### API Authentication

```yaml
security:
  api_key:
    enabled: true
    header: X-API-Key
  rate_limiting:
    enabled: true
    requests_per_minute: 60
```

### Secrets Management

- Use environment variables or secrets manager
- Never commit secrets to code
- Rotate API keys regularly

### Network Security

- Use HTTPS in production
- Implement CORS properly
- Use VPC/private networks for internal services
