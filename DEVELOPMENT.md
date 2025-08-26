# 🛠️ AskData Development Guide

This guide is for developers contributing to the AskData system. It covers the development workflow, architecture patterns, and best practices.

## 🏗️ Architecture Principles

### 1. Service Isolation
- Each service is completely independent
- No shared state between services
- Communication only via HTTP MCP tools
- Strong isolation per `connection_id`

### 2. MCP Tool Contract
- Fixed tool names and schemas
- Input/output validation via Pydantic
- Versioned API contracts
- Backward compatibility requirements

### 3. Run Envelope Pattern
- Every operation includes `RunEnvelope`
- Contains `run_id`, `step_id`, `connection_id`
- Enables tracing and debugging
- Supports multi-tenant scenarios

## 🔧 Development Setup

### Prerequisites
```bash
# Python 3.11+
python --version

# Docker Desktop
docker --version

# Git
git --version
```

### Local Development
```bash
# Clone and setup
git clone <repo-url>
cd askdata
./ops/setup.sh  # or ops\setup.bat on Windows

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
./ops/test.sh
```

### 🎯 Minimal Development Stack
For development and testing, start with just the essential services:

```bash
# Start minimal stack (orchestrator + registry + ChromaDB)
docker-compose -f ops/docker-compose.minimal.yml up -d

# Check health endpoints
curl http://localhost:8000/health  # Orchestrator
curl http://localhost:8001/health  # Connection Registry

# View logs
docker-compose -f ops/docker-compose.minimal.yml logs -f
```

**Minimal Stack Services:**
- ✅ **API Orchestrator** (port 8000) - Main workflow coordinator
- ✅ **Connection Registry** (port 8001) - Database connection management  
- ✅ **ChromaDB** (internal) - Vector database for embeddings

### 🚀 Full Development Stack
Once minimal stack is working, start all services:

```bash
# Start all services (requires all service implementations)
docker-compose -f ops/docker-compose.yml up -d

# Check all service health
./ops/test.sh  # Unix/Linux/macOS
# or
ops\test.bat   # Windows
```

## 📁 Project Structure

```
askdata/
├── services/                 # Microservices
│   ├── api-orchestrator/    # Main orchestrator
│   ├── connection-registry/ # DB connection management
│   ├── introspect/         # Schema introspection
│   ├── indexer/            # Vector indexing
│   ├── table-retriever/    # Semantic search
│   ├── micro-profiler/     # Data profiling
│   ├── column-pruner/      # Column selection
│   ├── join-graph/         # Join optimization
│   ├── metric-resolver/    # Business term mapping
│   ├── sql-generator/      # SQL generation
│   ├── sql-validator/      # SQL validation
│   ├── query-executor/     # Query execution
│   └── result-explainer/   # Result explanation
├── contracts/               # MCP tool definitions
├── frontend/               # React frontend
├── ops/                    # Operations and deployment
├── data/                   # Persistent data (mounted)
├── logs/                   # Logs and traces (mounted)
└── exports/                # CSV exports (mounted)
```

## 🚀 Service Development

### Creating a New Service

1. **Create Service Directory**
```bash
mkdir services/my-service
cd services/my-service
```

2. **Create Requirements**
```python
# requirements.txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
# Add service-specific dependencies
```

3. **Create Main Service**
```python
# main.py
from fastapi import FastAPI
from contracts.mcp_tools import MCP_TOOLS

app = FastAPI(title="My Service")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "my-service"}

@app.post("/mcp/my_tool")
async def my_mcp_tool(input_data: dict):
    # Implement MCP tool logic
    return {"result": "success"}
```

4. **Create Dockerfile**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "main.py"]
```

5. **Add to Docker Compose**
```yaml
# ops/docker-compose.yml
my-service:
  build:
    context: ../services/my-service
    dockerfile: Dockerfile
  container_name: askdata-my-service
  ports:
    - "800X:8000"
  networks:
    - askdata-network
```

### MCP Tool Implementation

```python
from contracts.mcp_tools import MyToolInput, MyToolOutput

@app.post("/mcp/my_tool")
async def my_mcp_tool(input_data: MyToolInput) -> MyToolOutput:
    # Validate input
    # Process request
    # Return structured output
    return MyToolOutput(
        result="success",
        data=processed_data
    )
```

## 🧪 Testing

### Unit Tests
```bash
# Run all tests
pytest

# Run specific service tests
pytest services/my-service/

# Run with coverage
pytest --cov=services/my-service
```

### Integration Tests
```bash
# Start test environment
docker-compose -f ops/docker-compose.test.yml up -d

# Run integration tests
pytest tests/integration/

# Cleanup
docker-compose -f ops/docker-compose.test.yml down
```

### Test Structure
```
tests/
├── unit/                    # Unit tests
│   ├── services/           # Service-specific tests
│   └── contracts/          # Contract tests
├── integration/            # Integration tests
├── fixtures/               # Test data and mocks
└── conftest.py            # Test configuration
```

## 🔍 Debugging

### Local Development
```bash
# Run service locally
cd services/my-service
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Debug with breakpoints
python -m pdb main.py
```

### Docker Debugging
```bash
# View logs
docker-compose -f ops/docker-compose.yml logs -f my-service

# Exec into container
docker exec -it askdata-my-service bash

# Check container health
docker inspect askdata-my-service
```

### Tracing and Logging
```python
import structlog
import logging

# Structured logging
logger = structlog.get_logger()
logger.info("Processing request", 
           run_id=run_id, 
           connection_id=connection_id)

# JSON logging to files
logging.basicConfig(
    filename=f"logs/{service_name}.log",
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## 📊 Monitoring

### Health Checks
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "my-service",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0"
    }
```

### Metrics
```python
from prometheus_client import Counter, Histogram

# Define metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests')
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')

# Use in endpoints
@app.middleware("http")
async def metrics_middleware(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    REQUEST_COUNT.inc()
    REQUEST_DURATION.observe(duration)
    
    return response
```

### Logging
```python
# Structured logging with context
logger = structlog.get_logger()

def log_operation(operation: str, **kwargs):
    logger.info(f"Operation: {operation}", 
               operation=operation,
               timestamp=datetime.utcnow(),
               **kwargs)

# Usage
log_operation("table_search", 
             connection_id="conn_123",
             query="sales data",
             results_count=5)
```

## 🔒 Security

### Input Validation
```python
from pydantic import BaseModel, validator

class MyInput(BaseModel):
    connection_id: str
    query: str
    
    @validator('connection_id')
    def validate_connection_id(cls, v):
        if not v.startswith('conn_'):
            raise ValueError('Invalid connection ID format')
        return v
```

### Authentication
```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def get_current_user(token: str = Security(security)):
    # Validate token
    if not is_valid_token(token):
        raise HTTPException(status_code=401, detail="Invalid token")
    return get_user_from_token(token)

@app.post("/protected")
async def protected_endpoint(user = Depends(get_current_user)):
    return {"message": f"Hello {user.name}"}
```

## 🚀 Deployment

### Environment Configuration
```bash
# Production environment
export ENVIRONMENT=production
export LOG_LEVEL=WARNING
export DEBUG=false

# Development environment
export ENVIRONMENT=development
export LOG_LEVEL=DEBUG
export DEBUG=true
```

### Docker Build
```bash
# Build service
docker build -t askdata-my-service:latest services/my-service/

# Tag for registry
docker tag askdata-my-service:latest registry.example.com/askdata-my-service:v1.0.0

# Push to registry
docker push registry.example.com/askdata-my-service:v1.0.0
```

### Kubernetes Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: askdata-my-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: askdata-my-service
  template:
    metadata:
      labels:
        app: askdata-my-service
    spec:
      containers:
      - name: my-service
        image: registry.example.com/askdata-my-service:v1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
```

## 📚 Best Practices

### 1. Error Handling
```python
from fastapi import HTTPException
import logging

@app.post("/mcp/my_tool")
async def my_mcp_tool(input_data: dict):
    try:
        result = await process_request(input_data)
        return result
    except ValidationError as e:
        logging.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except DatabaseError as e:
        logging.error(f"Database error: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

### 2. Async Operations
```python
import asyncio
from typing import List

async def process_batch(items: List[dict]) -> List[dict]:
    # Process items concurrently
    tasks = [process_item(item) for item in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle exceptions
    processed = []
    for result in results:
        if isinstance(result, Exception):
            logging.error(f"Item processing failed: {result}")
        else:
            processed.append(result)
    
    return processed
```

### 3. Configuration Management
```python
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    environment: str = "development"
    log_level: str = "INFO"
    database_url: Optional[str] = None
    redis_url: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

### 4. Testing Patterns
```python
import pytest
from unittest.mock import Mock, patch

@pytest.fixture
def mock_database():
    return Mock()

@pytest.fixture
def sample_input():
    return {
        "connection_id": "conn_123",
        "query": "test query"
    }

async def test_my_tool_success(mock_database, sample_input):
    with patch('my_service.get_database', return_value=mock_database):
        result = await my_mcp_tool(sample_input)
        assert result["status"] == "success"
```

## 🎯 Performance Optimization

### 1. Caching
```python
import asyncio
from functools import lru_cache

# In-memory caching
@lru_cache(maxsize=1000)
def get_cached_data(key: str):
    return expensive_operation(key)

# Async caching with TTL
cache = {}

async def get_cached_async(key: str, ttl: int = 3600):
    if key in cache:
        data, timestamp = cache[key]
        if time.time() - timestamp < ttl:
            return data
    
    data = await expensive_async_operation(key)
    cache[key] = (data, time.time())
    return data
```

### 2. Batch Processing
```python
async def process_batch_efficient(items: List[dict], batch_size: int = 100):
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await asyncio.gather(
            *[process_item(item) for item in batch]
        )
        results.extend(batch_results)
    
    return results
```

### 3. Connection Pooling
```python
import aiohttp
import asyncio

class ServiceClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = None
    
    async def get_session(self):
        if self.session is None:
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
            self.session = aiohttp.ClientSession(connector=connector)
        return self.session
    
    async def close(self):
        if self.session:
            await self.session.close()
```

## 🔄 CI/CD Pipeline

### GitHub Actions
```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        pip install -r requirements-dev.txt
    - name: Run tests
      run: |
        pytest --cov=services/
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    - name: Build and push
      run: |
        docker build -t askdata:latest .
        docker push askdata:latest
```

## ⚠️ Known Issues & Development Notes

### Current System Status
- **Frontend**: React application not yet implemented (Phase 5)
- **Mock Data**: Some services return sample data for development
- **Database Drivers**: Limited to basic adapter implementations
- **Performance**: No production-level optimization or caching

### Development Limitations
- **Port Conflicts**: Ensure ports 8000-8013 are available
- **Missing Agents**: Some advanced features require additional service implementation
- **Service Dependencies**: Some services depend on others being healthy
- **Resource Usage**: Full stack may require 6-8 GB RAM

### Development Workarounds
- Use `docker-compose.minimal.yml` for initial testing
- Check service health endpoints before proceeding
- Monitor Docker resource usage and adjust as needed
- Refer to `ops/config.local` for configuration options
- Use stub mode in orchestrator for unimplemented services

### Testing Strategy
- Start with minimal stack to verify basic functionality
- Test individual services before full integration
- Use health endpoints for service validation
- Check logs for detailed error information
- Validate MCP tool contracts and schemas

## 🎉 Contributing

### Development Workflow
1. **Fork** the repository
2. **Create** a feature branch
3. **Implement** your changes
4. **Add** tests for new functionality
5. **Update** documentation
6. **Submit** a pull request

### Code Standards
- **Python**: Follow PEP 8, use type hints
- **Testing**: Maintain >90% coverage
- **Documentation**: Update README and docstrings
- **Commits**: Use conventional commit messages

### Review Process
- All PRs require review
- Tests must pass
- Code coverage must not decrease
- Documentation must be updated

---

**Happy coding! 🚀**

*For more information, check the main README.md and QUICKSTART.md files.* 