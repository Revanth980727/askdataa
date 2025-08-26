@echo off
REM AskData Setup Script for Windows
REM This script sets up the local development environment

echo 🚀 Setting up AskData development environment...

REM =============================================================================
REM CHECK PREREQUISITES
REM =============================================================================

echo 📋 Checking prerequisites...

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running. Please start Docker Desktop and try again.
    exit /b 1
)

echo ✅ Prerequisites check passed

REM =============================================================================
REM CREATE DIRECTORY STRUCTURE
REM =============================================================================

echo 📁 Creating directory structure...

REM Create main directories
if not exist "data\chroma" mkdir "data\chroma"
if not exist "data\opensearch" mkdir "data\opensearch"
if not exist "data\redis" mkdir "data\redis"
if not exist "data\datasets" mkdir "data\datasets"
if not exist "data\feedback" mkdir "data\feedback"
if not exist "data\models" mkdir "data\models"
if not exist "logs\traces" mkdir "logs\traces"
if not exist "exports" mkdir "exports"

REM Create service directories
if not exist "services\api-orchestrator" mkdir "services\api-orchestrator"
if not exist "services\connection-registry" mkdir "services\connection-registry"
if not exist "services\introspect" mkdir "services\introspect"
if not exist "services\indexer" mkdir "services\indexer"
if not exist "services\table-retriever" mkdir "services\table-retriever"
if not exist "services\micro-profiler" mkdir "services\micro-profiler"
if not exist "services\column-pruner" mkdir "services\column-pruner"
if not exist "services\join-graph" mkdir "services\join-graph"
if not exist "services\metric-resolver" mkdir "services\metric-resolver"
if not exist "services\sql-generator" mkdir "services\sql-generator"
if not exist "services\sql-validator-service" mkdir "services\sql-validator-service"
if not exist "services\query-executor-service" mkdir "services\query-executor-service"
if not exist "services\result-explainer-service" mkdir "services\result-explainer-service"
if not exist "services\rlhf-service" mkdir "services\rlhf-service"

REM Create frontend directory
if not exist "frontend" mkdir "frontend"

REM Create contracts directory
if not exist "contracts" mkdir "contracts"

echo ✅ Directory structure created

REM =============================================================================
REM SET UP ENVIRONMENT FILES
REM =============================================================================

echo 🔐 Setting up environment configuration...

REM Copy environment template if .env doesn't exist
if not exist ".env" (
    if exist "env.example" (
        copy "env.example" ".env"
        echo 📝 Created .env from template. Please edit with your configuration.
    ) else (
        echo ⚠️  No env.example found. Please create .env manually.
    )
) else (
    echo ✅ .env file already exists
)

echo ✅ Environment setup completed

REM =============================================================================
REM SET UP DOCKER NETWORK
REM =============================================================================

echo 🐳 Setting up Docker network...

REM Create custom network if it doesn't exist
docker network ls | findstr "askdata-network" >nul 2>&1
if %errorlevel% neq 0 (
    docker network create askdata-network
    echo ✅ Created askdata-network
) else (
    echo ✅ askdata-network already exists
)

REM =============================================================================
REM BUILD BASE IMAGES
REM =============================================================================

echo 🔨 Building base service images...

REM Build API Orchestrator
if exist "services\api-orchestrator\Dockerfile" (
    echo Building api-orchestrator...
    docker build -t askdata-api-orchestrator:latest services\api-orchestrator\
)

REM Build Connection Registry
if exist "services\connection-registry\Dockerfile" (
    echo Building connection-registry...
    docker build -t askdata-connection-registry:latest services\connection-registry\
)

echo ✅ Base images built

REM =============================================================================
REM VERIFY SETUP
REM =============================================================================

echo 🔍 Verifying setup...

REM Check if all required directories exist
set REQUIRED_DIRS=data\chroma data\opensearch data\redis data\datasets data\feedback data\models logs\traces exports services\api-orchestrator services\connection-registry services\introspect services\indexer services\table-retriever services\micro-profiler services\column-pruner services\join-graph services\metric-resolver services\sql-generator services\sql-validator services\query-executor services\result-explainer services\rlhf-service frontend contracts ops

for %%d in (%REQUIRED_DIRS%) do (
    if not exist "%%d" (
        echo ❌ Missing required directory: %%d
        exit /b 1
    )
)

REM Check if Docker network exists
docker network ls | findstr "askdata-network" >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker network askdata-network not found
    exit /b 1
)

echo ✅ Setup verification passed

REM =============================================================================
REM SERVICE HEALTH CHECK (if running)
REM =============================================================================

echo 🏥 Checking service health (if running)...

REM Check if services are running and healthy
docker-compose -f ops\docker-compose.yml ps | findstr "Up" >nul 2>&1
if %errorlevel% equ 0 (
    echo 📡 Services are running, checking health...
    
    REM Check orchestrator health
    curl -s http://localhost:8000/health >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ API Orchestrator is healthy
    ) else (
        echo ⚠️  API Orchestrator health check failed
    )
    
    REM Check connection registry health
    curl -s http://localhost:8001/health >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ Connection Registry is healthy
    ) else (
        echo ⚠️  Connection Registry health check failed
    )
    
    REM Check introspect service health
    curl -s http://localhost:8002/health >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ Introspect Service is healthy
    ) else (
        echo ⚠️  Introspect Service health check failed
    )
    
    REM Check vector store service health
    curl -s http://localhost:8003/health >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ Vector Store Service is healthy
    ) else (
        echo ⚠️  Vector Store Service health check failed
    )
    
    REM Check table retriever service health
    curl -s http://localhost:8004/health >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ Table Retriever Service is healthy
    ) else (
        echo ⚠️  Table Retriever Service health check failed
    )
    
    REM Check micro profiler service health
    curl -s http://localhost:8005/health >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ Micro Profiler Service is healthy
    ) else (
        echo ⚠️  Micro Profiler Service health check failed
    )
    
    REM Check SQL validator service health
    curl -s http://localhost:8011/health >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ SQL Validator Service is healthy
    ) else (
        echo ⚠️  SQL Validator Service health check failed
    )
    
    REM Check query executor service health
    curl -s http://localhost:8012/health >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ Query Executor Service is healthy
    ) else (
        echo ⚠️  Query Executor Service health check failed
    )
    
    REM Check result explainer service health
    curl -s http://localhost:8013/health >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ Result Explainer Service is healthy
    ) else (
        echo ⚠️  Result Explainer Service health check failed
    )
) else (
    echo ℹ️  Services are not running. Start them with 'docker-compose -f ops\docker-compose.yml up -d'
)

REM =============================================================================
REM NEXT STEPS
REM =============================================================================

echo.
echo 🎉 AskData setup completed successfully!
echo.
echo 📋 Next steps:
echo 1. Edit .env file with your database credentials and API keys
echo 2. Complete the service implementations in the services\ directory
echo 3. Build the frontend React application
echo 4. Run 'docker-compose -f ops\docker-compose.yml up -d' to start all services
echo.
echo 📚 Documentation:
echo - README.md: Project overview and architecture
echo - ops\config.local: Configuration options
echo - contracts\mcp_tools.py: Service interface definitions
echo.
echo 🔧 Development commands:
echo - Start services: docker-compose -f ops\docker-compose.yml up -d
echo - View logs: docker-compose -f ops\docker-compose.yml logs -f
echo - Stop services: docker-compose -f ops\docker-compose.yml down
echo - Rebuild: docker-compose -f ops\docker-compose.yml build --no-cache
echo.
echo 🚀 Happy coding!
pause 