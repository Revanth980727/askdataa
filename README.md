# AskData - Intelligent Database Query System

A local, containerized system that allows you to plug in any database with credentials or API, ask questions in natural language, and get SQL, results, and clear explanations.

## 🎯 North Star

**Goal**: Plug in any database with creds or API. Ask any question. Get SQL, results, and a clear reason why.

**Key Principles**:
- No cross-DB bleed - switching DBs resets memory and adapters
- Everything runs locally in Docker
- Complete isolation per connection_id
- Strong forget capabilities

## 🏗️ Architecture

### Tech Stack
- **Backend**: Python 3.11, FastAPI, Pydantic, LangGraph
- **Agents**: Small FastAPI apps exposing MCP tools over HTTP/WebSocket
- **Vector Store**: ChromaDB (local) + optional OpenSearch (hybrid search)
- **Frontend**: ReactJS, Vite, React Query, Zustand, Radix UI
- **ML**: Transformers, sentence embeddings, LoRA via PEFT

### Core Services
- **api-orchestrator**: Main orchestrator using LangGraph
- **connection-registry**: Manages DB connections and handles
- **introspect**: Database schema introspection
- **indexer**: Vector indexing and search
- **table-retriever**: Semantic table retrieval
- **micro-profiler**: Table profiling and fingerprinting
- **column-pruner**: Intelligent column selection
- **join-graph**: Join path optimization
- **metric-resolver**: Business term mapping
- **sql-generator**: LLM-powered SQL generation
- **sql-validator**: SQL validation and dialect rewriting
- **query-executor**: Query execution and result streaming
- **result-explainer**: Result explanation and trace analysis

## 🚀 Quick Start

### Prerequisites
- Docker Desktop (allocate 6-8 GB RAM)
- Git

### Setup
1. Clone this repository
2. Copy `.env.example` to `.env` and configure your database credentials
3. Run the setup script:
   ```bash
   ./ops/setup.sh
   ```

### Minimal Bring-Up (Recommended for First Run)
Start with just the essential services to verify your setup:

```bash
# Start minimal stack (orchestrator + registry + ChromaDB)
docker-compose -f ops/docker-compose.minimal.yml up -d

# Check health endpoints
curl http://localhost:8000/health  # Orchestrator
curl http://localhost:8001/health  # Connection Registry
curl http://localhost:8002/health  # ChromaDB (internal)
```

### Full System Start
Once minimal stack is working:

```bash
# Start all services
docker-compose -f ops/docker-compose.yml up -d

# Access the frontend at http://localhost:3000 (when implemented)
```

### Directory Structure
```
askdata/
├── services/           # Backend microservices
├── frontend/          # React frontend
├── contracts/         # MCP tool specs and schemas
├── ops/              # Docker compose and deployment
├── data/             # Vector store and data (mounted)
├── logs/             # JSON traces and logs (mounted)
└── exports/          # CSV exports (mounted)
```

## 🔐 Configuration

### Environment Variables (.env)
- Database credentials for each connection
- LLM API configuration
- Vector store settings
- Service URLs and ports

### Local Config (config.local)
- Budgets and limits
- Retry policies
- Feature toggles
- Performance settings

## 🔒 Security & Isolation

- **Connection Isolation**: Each database connection is completely isolated
- **Handle-based Access**: Services receive short-lived handles, never raw credentials
- **Namespace Separation**: All operations are scoped by connection_id
- **Strong Forget**: Complete data removal when connections are deleted

### What Purge Deletes
When a connection is deleted, the system completely removes all associated data:

- **Secrets & Credentials**: Database passwords, API keys, connection strings
- **Embeddings**: All vector store collections for the connection
- **Caches**: Profiling results, query results, execution traces
- **Adapters**: Fine-tuned LoRA models specific to the connection
- **Datasets**: Training data, feedback data, evaluation metrics
- **Traces**: Complete execution logs and debugging information
- **Metadata**: Schema information, table profiles, join graphs
- **Temporary Files**: Any cached or temporary data files

This ensures complete data isolation and compliance with data privacy requirements.

## 📊 Features

- **Multi-Database Support**: Connect to any database with proper adapters
- **Semantic Search**: Find relevant tables and columns using embeddings
- **Intelligent Profiling**: Automatic table analysis and fingerprinting
- **Smart Column Selection**: AI-powered column pruning and join optimization
- **SQL Generation**: Natural language to SQL with LLM assistance
- **Result Explanation**: Clear reasoning for every query decision
- **Fine-tuning**: Per-database LoRA adapters for improved performance
- **RLHF Loop**: Continuous improvement through user feedback

## 🧪 Testing

Run the test suite:
```bash
./ops/test.sh
```

## ⚠️ Known Issues

### Current Limitations
- **Frontend**: React application is not yet implemented (Phase 5)
- **Mock Endpoints**: Some services return mock data for development
- **Database Drivers**: Limited to basic database adapter implementations
- **Performance**: No production-level optimization or caching

### Development Notes
- **Port Conflicts**: Ensure no other services are using ports 8000-8013
- **Missing Agents**: Some advanced features require additional service implementation
- **Mock Data**: Services may return sample data instead of real database results
- **Resource Usage**: Docker containers may require 6-8 GB RAM for full stack

### Workarounds
- Use `docker-compose.minimal.yml` for initial testing
- Check service health endpoints before proceeding
- Monitor Docker resource usage and adjust as needed
- Refer to `ops/config.local` for configuration options

## 📈 Monitoring

- JSON logs in `/logs` directory
- Metrics endpoints on each service
- Performance tracing and profiling
- Health checks and status monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

[Add your license here]

## 🆘 Support

For issues and questions, please check the documentation or create an issue in the repository. 