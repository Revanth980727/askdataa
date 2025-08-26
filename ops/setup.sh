#!/bin/bash

# AskData Setup Script
# This script sets up the local development environment

set -e

echo "🚀 Setting up AskData development environment..."

# =============================================================================
# CHECK PREREQUISITES
# =============================================================================

echo "📋 Checking prerequisites..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop and try again."
    exit 1
fi

# Check Docker version
DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | cut -d'.' -f1)
if [ "$DOCKER_VERSION" -lt 20 ]; then
    echo "❌ Docker version 20.0 or higher is required. Current version: $(docker --version)"
    exit 1
fi

# Check available memory (approximate)
if command -v free > /dev/null 2>&1; then
    MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$MEMORY_GB" -lt 6 ]; then
        echo "⚠️  Warning: Less than 6GB RAM available. Performance may be affected."
    fi
fi

echo "✅ Prerequisites check passed"

# =============================================================================
# CREATE DIRECTORY STRUCTURE
# =============================================================================

echo "📁 Creating directory structure..."

# Create main directories
mkdir -p data/chroma
mkdir -p data/opensearch
mkdir -p data/redis
mkdir -p data/datasets
mkdir -p data/feedback
mkdir -p data/models
mkdir -p logs/traces
mkdir -p exports

# Create service directories (only implemented services)
mkdir -p services/api-orchestrator
mkdir -p services/connection-registry
mkdir -p services/introspect-service
mkdir -p services/vector-store-service
mkdir -p services/table-retriever-service
mkdir -p services/micro-profiler-service
mkdir -p services/sql-validator-service
mkdir -p services/query-executor-service
mkdir -p services/result-explainer-service
mkdir -p services/rlhf-service

# Create frontend directory
mkdir -p frontend

# Create contracts directory
mkdir -p contracts

echo "✅ Directory structure created"

# =============================================================================
# SET UP ENVIRONMENT FILES
# =============================================================================

echo "🔐 Setting up environment configuration..."

# Copy environment template if .env doesn't exist
if [ ! -f .env ]; then
    if [ -f env.example ]; then
        cp env.example .env
        echo "📝 Created .env from template. Please edit with your configuration."
    else
        echo "⚠️  No env.example found. Please create .env manually."
    fi
else
    echo "✅ .env file already exists"
fi

# Create config.local if it doesn't exist
if [ ! -f ops/config.local ]; then
    echo "📝 Creating default config.local..."
    # This will be created by the Python script
else
    echo "✅ config.local already exists"
fi

# =============================================================================
# SET UP DOCKER NETWORK
# =============================================================================

echo "🐳 Setting up Docker network..."

# Create custom network if it doesn't exist
if ! docker network ls | grep -q askdata-network; then
    docker network create askdata-network
    echo "✅ Created askdata-network"
else
    echo "✅ askdata-network already exists"
fi

# =============================================================================
# BUILD BASE IMAGES
# =============================================================================

echo "🔨 Building base service images..."

# Build API Orchestrator
if [ -f services/api-orchestrator/Dockerfile ]; then
    echo "Building api-orchestrator..."
    docker build -t askdata-api-orchestrator:latest services/api-orchestrator/
fi

# Build Connection Registry
if [ -f services/connection-registry/Dockerfile ]; then
    echo "Building connection-registry..."
    docker build -t askdata-connection-registry:latest services/connection-registry/
fi

echo "✅ Base images built"

# =============================================================================
# SET UP VOLUME PERMISSIONS
# =============================================================================

echo "🔒 Setting up volume permissions..."

# Ensure data directories have correct permissions
chmod 755 data/
chmod 755 logs/
chmod 755 exports/

# Create .gitkeep files to preserve empty directories
find data logs exports -type d -empty -exec touch {}/.gitkeep \;

echo "✅ Volume permissions set"

# =============================================================================
# VERIFY SETUP
# =============================================================================

echo "🔍 Verifying setup..."

# Check if all required directories exist
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
    "services/introspect"
    "services/indexer"
    "services/table-retriever"
    "services/micro-profiler"
    "services/column-pruner"
    "services/join-graph"
    "services/metric-resolver"
    "services/sql-generator"
    "services/sql-validator"
    "services/query-executor"
    "services/result-explainer"
    "services/rlhf-service"
    "frontend"
    "contracts"
    "ops"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "❌ Missing required directory: $dir"
        exit 1
    fi
done

# Check if Docker network exists
if ! docker network ls | grep -q askdata-network; then
    echo "❌ Docker network askdata-network not found"
    exit 1
fi

echo "✅ Setup verification passed"

# =============================================================================
# NEXT STEPS
# =============================================================================

echo ""
echo "🎉 AskData setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file with your database credentials and API keys"
echo "2. Complete the service implementations in the services/ directory"
echo "3. Build the frontend React application"
echo "4. Run 'docker-compose up -d' to start all services"
echo ""
echo "📚 Documentation:"
echo "- README.md: Project overview and architecture"
echo "- ops/config.local: Configuration options"
echo "- contracts/mcp_tools.py: Service interface definitions"
echo ""
echo "🔧 Development commands:"
echo "- Start services: docker-compose up -d"
echo "- View logs: docker-compose logs -f"
echo "- Stop services: docker-compose down"
echo "- Rebuild: docker-compose build --no-cache"
echo ""
echo "🚀 Happy coding!"

# =============================================================================
# SERVICE HEALTH CHECK (if running)
# =============================================================================

echo ""
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
else
    echo "ℹ️  Services are not running. Start them with 'docker-compose -f ops/docker-compose.yml up -d'"
fi 