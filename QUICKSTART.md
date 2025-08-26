# 🚀 AskData Quick Start Guide

Get up and running with AskData in minutes! This guide will walk you through setting up the system and running your first natural language database query.

## ⚡ Prerequisites

- **Docker Desktop** running with 6-8 GB RAM allocated
- **Git** for cloning the repository
- **Text editor** for configuration files

## 🎯 Quick Start (5 minutes)

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd askdata

# Run the setup script
# On Windows:
ops\setup.bat

# On macOS/Linux:
./ops/setup.sh
```

### 2. Configure Your Environment

```bash
# Edit the environment file with your credentials
# Copy from template if needed
cp env.example .env

# Edit .env with your database and LLM credentials
notepad .env  # Windows
# or
nano .env     # macOS/Linux
```

**Required configuration:**
- Database connection details (host, port, credentials)
- LLM API key (OpenAI, Anthropic, or local model)
- Vector store settings

### 3. Start the System

#### 🎯 Minimal Bring-Up (Recommended for First Run)
Start with just the essential services to verify your setup:

```bash
# Start minimal stack (orchestrator + registry + ChromaDB)
docker-compose -f ops/docker-compose.minimal.yml up -d

# Check service health
curl http://localhost:8000/health  # Orchestrator
curl http://localhost:8001/health  # Connection Registry

# View running services
docker-compose -f ops/docker-compose.minimal.yml ps

# Check logs
docker-compose -f ops/docker-compose.minimal.yml logs -f
```

**Minimal Stack Services:**
- ✅ **API Orchestrator** (port 8000) - Main workflow coordinator
- ✅ **Connection Registry** (port 8001) - Database connection management  
- ✅ **ChromaDB** (internal) - Vector database for embeddings

#### 🚀 Full System Start
Once minimal stack is working, start all services:

```bash
# Start all services (requires all service implementations)
docker-compose -f ops/docker-compose.yml up -d

# Check all service health
./ops/test.sh  # Unix/Linux/macOS
# or
ops\test.bat   # Windows
```

**Full Stack Services:**
- All minimal services plus:
- Introspect, Vector Store, Table Retriever, Micro Profiler
- SQL Validator, Query Executor, Result Explainer
- Additional agent services for advanced features

### 4. Access the System

- **Frontend**: http://localhost:3000
- **API**: http://localhost:8000
- **Health Check**: http://localhost:8000/health

## 🔧 First Connection

### 1. Add Database Connection

Navigate to the Connections page and add your first database:

- **Database Type**: PostgreSQL, MySQL, Snowflake, etc.
- **Host**: Your database server
- **Port**: Database port (5432 for PostgreSQL, 3306 for MySQL)
- **Database**: Database name
- **Username/Password**: Your credentials

### 2. Test Connection

Click "Test Connection" to verify:
- Credentials are correct
- Network connectivity works
- Database is accessible

### 3. Start Ingestion

Click "Start Ingestion" to:
- Introspect database schema
- Generate vector embeddings
- Build search indexes

## 🎯 First Query

### 1. Ask a Question

Go to the Ask page and type a natural language question:

```
"Show me total sales by month for the last year"
"Which customers have the highest order values?"
"What's the average order size by region?"
```

### 2. Watch the Magic

The system will:
1. **Route** to the right database
2. **Search** for relevant tables and columns
3. **Profile** data characteristics
4. **Generate** optimized SQL
5. **Execute** the query
6. **Explain** the results

### 3. Review Results

- **Table View**: Raw data results
- **Chart View**: Visual representations
- **SQL View**: Generated SQL code
- **Explain View**: Decision reasoning

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │ API Orchestrator │    │ Database       │
│   (React)       │◄──►│   (LangGraph)    │◄──►│   Connections   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Agent Services │
                       │   (MCP Tools)    │
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Vector Store   │
                       │   (ChromaDB)     │
                       └──────────────────┘
```

## 🔍 Key Services

- **API Orchestrator**: Main workflow coordinator
- **Connection Registry**: Database connection management
- **Table Retriever**: Semantic table search
- **Micro Profiler**: Data analysis and fingerprinting
- **SQL Generator**: Natural language to SQL conversion
- **Query Executor**: SQL execution and result streaming

## 🚨 Troubleshooting

### Common Issues

**Docker not running**
```bash
# Start Docker Desktop
# Check with:
docker info
```

**Port conflicts**
```bash
# Check what's using the ports:
netstat -an | findstr :8000  # Windows
lsof -i :8000                # macOS/Linux
```

**Service won't start**
```bash
# Check logs:
docker-compose -f ops/docker-compose.yml logs <service-name>

# Rebuild:
docker-compose -f ops/docker-compose.yml build --no-cache
```

**Memory issues**
- Increase Docker Desktop memory allocation
- Close other applications
- Check system memory usage

### Health Checks

```bash
# Check all services:
docker-compose -f ops/docker-compose.yml ps

# Individual service health:
curl http://localhost:8000/health
curl http://localhost:8001/health
```

## 🗑️ What Forget Does

When you delete a database connection, AskData performs a **complete purge** of all related data:

**Immediately Removed:**
- Database connection configuration and credentials
- All active handles and session data
- Vector embeddings and search indexes
- Table profiles and data fingerprints
- Fine-tuned models and adapters
- Query history and cached results
- User feedback and training data
- All logs and traces (optionally)

**Audit Trail:**
- Generates detailed offboard report
- Logs all purged artifacts
- Complies with data privacy requirements
- Provides deletion confirmation

This ensures **complete data isolation** and **strong privacy guarantees** when switching between databases.

## ⚠️ Known Issues & Limitations

### Current System Status
- **Frontend**: React application not yet implemented (Phase 5)
- **Mock Data**: Some services return sample data for development
- **Database Drivers**: Limited to basic adapter implementations
- **Performance**: No production-level optimization

### Development Notes
- **Port Conflicts**: Ensure ports 8000-8013 are available
- **Missing Agents**: Some advanced features require additional services
- **Resource Usage**: Full stack may require 6-8 GB RAM
- **Service Dependencies**: Some services depend on others being healthy

### Troubleshooting Tips
- Start with minimal stack first to verify basic functionality
- Check service health endpoints before proceeding
- Monitor Docker resource usage and adjust as needed
- Use `ops/config.local` for configuration options
- Check service logs for detailed error information

## 📚 Next Steps

### 1. Explore Features

- **Fine-tuning**: Train custom models for your database
- **RLHF Loop**: Improve results with feedback
- **Advanced Profiling**: Deep data analysis
- **Hybrid Search**: Combine vector and lexical search

### 2. Customize

- **Prompts**: Modify SQL generation prompts
- **Scoring**: Adjust table/column relevance weights
- **Caching**: Configure TTL and storage policies
- **Monitoring**: Set up logging and metrics

### 3. Scale

- **Multiple Databases**: Connect to different data sources
- **Load Balancing**: Distribute queries across instances
- **High Availability**: Set up service redundancy
- **Performance**: Optimize for your workload

## 🆘 Getting Help

- **Documentation**: Check README.md for detailed information
- **Issues**: Report bugs in the repository
- **Community**: Join discussions and ask questions
- **Support**: Contact the development team

## 🎉 You're Ready!

Congratulations! You now have a fully functional AskData system running locally. 

**Try it out:**
1. Connect to your database
2. Ask questions in plain English
3. Explore the generated SQL and results
4. Provide feedback to improve the system

**Happy querying! 🎯**

---

*Need help? Check the troubleshooting section above or refer to the main README.md for comprehensive documentation.* 