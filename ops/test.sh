#!/bin/bash

# AskData Test Script
# This script runs basic tests to verify the system setup

set -e

echo "🧪 Running AskData system tests..."

# =============================================================================
# ENVIRONMENT CHECKS
# =============================================================================

echo "🔍 Checking environment..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please run setup.sh first."
    exit 1
fi

# Check if config.local exists
if [ ! -f ops/config.local ]; then
    echo "❌ config.local not found. Please run setup.sh first."
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running."
    exit 1
fi

echo "✅ Environment check passed"

# =============================================================================
# DOCKER NETWORK CHECK
# =============================================================================

echo "🐳 Checking Docker network..."

if ! docker network ls | grep -q askdata-network; then
    echo "❌ askdata-network not found. Please run setup.sh first."
    exit 1
fi

echo "✅ Docker network check passed"

# =============================================================================
# DIRECTORY STRUCTURE CHECK
# =============================================================================

echo "📁 Checking directory structure..."

REQUIRED_DIRS=(
    "data/chroma"
    "data/opensearch"
    "data/redis"
    "data/datasets"
    "data/feedback"
    "data/models"
    "logs/traces"
    "exports"
    "services/api-orchestrator"
    "services/connection-registry"
    "services/rlhf-service"
    "contracts"
    "ops"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "❌ Missing required directory: $dir"
        exit 1
    fi
done

echo "✅ Directory structure check passed"

# =============================================================================
# SERVICE FILES CHECK
# =============================================================================

echo "🔧 Checking service files..."

REQUIRED_FILES=(
    "services/api-orchestrator/main.py"
    "services/api-orchestrator/requirements.txt"
    "services/api-orchestrator/Dockerfile"
    "services/connection-registry/main.py"
    "services/connection-registry/requirements.txt"
    "services/connection-registry/Dockerfile"
    "services/introspect-service/main.py"
    "services/introspect-service/requirements.txt"
    "services/introspect-service/Dockerfile"
    "services/vector-store-service/main.py"
    "services/vector-store-service/requirements.txt"
    "services/vector-store-service/Dockerfile"
    "services/table-retriever-service/main.py"
    "services/table-retriever-service/requirements.txt"
    "services/table-retriever-service/Dockerfile"
    "services/micro-profiler-service/main.py"
    "services/micro-profiler-service/requirements.txt"
    "services/micro-profiler-service/Dockerfile"
    "services/rlhf-service/main.py"
    "services/rlhf-service/requirements.txt"
    "services/rlhf-service/Dockerfile"
    "contracts/mcp_tools.py"
    "ops/docker-compose.yml"
    "ops/config.local"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing required file: $file"
        exit 1
    fi
done

echo "✅ Service files check passed"

# =============================================================================
# DOCKER IMAGE BUILD TEST
# =============================================================================

echo "🔨 Testing Docker image builds..."

# Test API Orchestrator build
echo "Building api-orchestrator..."
if ! docker build -t askdata-api-orchestrator:test services/api-orchestrator/ > /dev/null 2>&1; then
    echo "❌ Failed to build api-orchestrator image"
    exit 1
fi

# Test Connection Registry build
echo "Building connection-registry..."
if ! docker build -t askdata-connection-registry:test services/connection-registry/ > /dev/null 2>&1; then
    echo "❌ Failed to build connection-registry image"
    exit 1
fi

# Clean up test images
docker rmi askdata-api-orchestrator:test > /dev/null 2>&1 || true
docker rmi askdata-connection-registry:test > /dev/null 2>&1 || true

echo "✅ Docker image build test passed"

# =============================================================================
# PYTHON SYNTAX CHECK
# =============================================================================

echo "🐍 Checking Python syntax..."

# Check if Python 3.11+ is available
if command -v python3.11 > /dev/null 2>&1; then
    PYTHON_CMD="python3.11"
elif command -v python3 > /dev/null 2>&1; then
    PYTHON_CMD="python3"
else
    echo "⚠️  Python 3 not found, skipping syntax check"
    PYTHON_CMD=""
fi

if [ -n "$PYTHON_CMD" ]; then
    # Check main.py files
    for service in api-orchestrator connection-registry rlhf-service; do
        if [ -f "services/$service/main.py" ]; then
            echo "Checking services/$service/main.py..."
            if ! $PYTHON_CMD -m py_compile "services/$service/main.py"; then
                echo "❌ Python syntax error in services/$service/main.py"
                exit 1
            fi
        fi
    done
    
    # Check contracts
    if [ -f "contracts/mcp_tools.py" ]; then
        echo "Checking contracts/mcp_tools.py..."
        if ! $PYTHON_CMD -m py_compile "contracts/mcp_tools.py"; then
            echo "❌ Python syntax error in contracts/mcp_tools.py"
            exit 1
        fi
    fi
    
    echo "✅ Python syntax check passed"
fi

# =============================================================================
# DOCKER COMPOSE VALIDATION
# =============================================================================

echo "🐳 Validating Docker Compose configuration..."

# Check if docker-compose.yml is valid
if ! docker-compose -f ops/docker-compose.yml config > /dev/null 2>&1; then
    echo "❌ Invalid docker-compose.yml configuration"
    exit 1
fi

echo "✅ Docker Compose validation passed"

# =============================================================================
# VOLUME PERMISSIONS CHECK
# =============================================================================

echo "🔒 Checking volume permissions..."

# Check if data directories are writable
for dir in data logs exports; do
    if [ ! -w "$dir" ]; then
        echo "❌ Directory $dir is not writable"
        exit 1
    fi
done

echo "✅ Volume permissions check passed"

# =============================================================================
# SERVICE HEALTH CHECK (if running)
# =============================================================================

echo "🏥 Checking service health (if running)..."

# Check if services are running and healthy
if docker-compose -f ops/docker-compose.yml ps | grep -q "Up"; then
    echo "📡 Services are running, checking health..."
    
    # Check orchestrator health
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ API Orchestrator is healthy"
    else
        echo "⚠️  API Orchestrator health check failed"
    fi
    
    # Check connection registry health
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo "✅ Connection Registry is healthy"
    else
        echo "⚠️  Connection Registry health check failed"
    fi
    
    # Check introspect service health
    if curl -s http://localhost:8002/health > /dev/null 2>&1; then
        echo "✅ Introspect Service is healthy"
    else
        echo "⚠️  Introspect Service health check failed"
    fi
    
    # Check vector store service health
    if curl -s http://localhost:8003/health > /dev/null 2>&1; then
        echo "✅ Vector Store Service is healthy"
    else
        echo "⚠️  Vector Store Service health check failed"
    fi
    
    # Check table retriever service health
    if curl -s http://localhost:8004/health > /dev/null 2>&1; then
        echo "✅ Table Retriever Service is healthy"
    else
        echo "⚠️  Table Retriever Service health check failed"
    fi
    
    # Check micro profiler service health
    if curl -s http://localhost:8005/health > /dev/null 2>&1; then
        echo "✅ Micro Profiler Service is healthy"
    else
        echo "⚠️  Micro Profiler Service health check failed"
    fi
    
    # Check SQL validator service health
    if curl -s http://localhost:8011/health > /dev/null 2>&1; then
        echo "✅ SQL Validator Service is healthy"
    else
        echo "⚠️  SQL Validator Service health check failed"
    fi
    
    # Check query executor service health
    if curl -s http://localhost:8012/health > /dev/null 2>&1; then
        echo "✅ Query Executor Service is healthy"
    else
        echo "⚠️  Query Executor Service health check failed"
    fi
    
    # Check result explainer service health
    if curl -s http://localhost:8013/health > /dev/null 2>&1; then
        echo "✅ Result Explainer Service is healthy"
    else
        echo "⚠️  Result Explainer Service health check failed"
    fi

    # Check RLHF service health
    if curl -s http://localhost:8014/health > /dev/null 2>&1; then
        echo "✅ RLHF Service is healthy"
    else
        echo "⚠️  RLHF Service health check failed"
    fi
else
    echo "ℹ️  Services are not running. Start them with 'docker-compose -f ops/docker-compose.yml up -d'"
fi

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "🎉 All tests passed successfully!"
echo ""
echo "✅ Environment configuration"
echo "✅ Directory structure"
echo "✅ Service files"
echo "✅ Docker network"
echo "✅ Docker image builds"
echo "✅ Python syntax"
echo "✅ Docker Compose configuration"
echo "✅ Volume permissions"
echo ""
echo "🚀 Your AskData system is ready to run!"
echo ""
echo "Next steps:"
echo "1. Edit .env with your database credentials"
echo "2. Run 'docker-compose -f ops/docker-compose.yml up -d'"
echo "3. Check service health with 'docker-compose -f ops/docker-compose.yml ps'"
echo ""
echo "Happy querying! 🎯" 