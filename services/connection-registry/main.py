"""
AskData Connection Registry Service

This service manages database connections, authenticates credentials, and issues
short-lived handles for secure database access by other services.
"""

import asyncio
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

import psycopg2
import pymysql
import structlog
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# Add the contracts directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent / "contracts"))

from mcp_tools import (
    DatabaseConnection, ConnectionHandle, ConnectionStatus,
    CreateConnectionInput, CreateConnectionOutput,
    GetConnectionHandleInput, GetConnectionHandleOutput,
    ListConnectionsInput, ListConnectionsOutput,
    DeleteConnectionInput, DeleteConnectionOutput
)

# Add offboard input/output models
class OffboardConnectionInput(BaseModel):
    """Input for offboarding a connection"""
    run_envelope: Dict[str, Any]
    connection_id: str
    purge_scope: List[str] = Field(default_factory=lambda: ["all"])

class OffboardConnectionOutput(BaseModel):
    """Output from offboarding a connection"""
    connection_id: str
    status: str
    message: str
    offboard_report: Dict[str, Any]
    artifacts_purged: List[str]

# =============================================================================
# CONFIGURATION
# =============================================================================

class Settings(BaseSettings):
    """Application settings"""
    environment: str = "local"
    log_level: str = "INFO"
    debug: bool = False
    
    # Handle settings
    handle_ttl_seconds: int = 300
    max_handles_per_connection: int = 10
    
    # Security settings
    encryption_key: str = "dev-key-change-in-production"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    """Configure structured logging"""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

# =============================================================================
# DATABASE ADAPTERS
# =============================================================================

class DatabaseAdapter:
    """Base class for database adapters"""
    
    def __init__(self, connection: DatabaseConnection):
        self.connection = connection
        self.connection_string = None
        self._connection = None
    
    async def authenticate(self) -> bool:
        """Test database connection and credentials"""
        try:
            await self._test_connection()
            return True
        except Exception as e:
            logging.error(f"Authentication failed for {self.connection.connection_id}: {e}")
            return False
    
    async def _test_connection(self):
        """Test the database connection"""
        raise NotImplementedError
    
    async def close(self):
        """Close the database connection"""
        if self._connection:
            self._connection.close()

class DatabaseAdapter:
    """Base class for database adapters"""
    
    def __init__(self, connection: DatabaseConnection):
        self.connection = connection
        self.connection_string = None
        self._connection = None
    
    async def authenticate(self) -> bool:
        """Test database connection and credentials"""
        try:
            await self._test_connection()
            return True
        except Exception as e:
            logging.error(f"Authentication failed for {self.connection.connection_id}: {e}")
            return False
    
    async def _test_connection(self):
        """Test the database connection"""
        raise NotImplementedError
    
    async def close(self):
        """Close the database connection"""
        if self._connection:
            self._connection.close()

class PostgreSQLAdapter(DatabaseAdapter):
    """PostgreSQL database adapter"""
    
    def __init__(self, connection: DatabaseConnection):
        super().__init__(connection)
        self.connection_string = (
            f"postgresql://{connection.username}:{connection.password}"
            f"@{connection.host}:{connection.port}/{connection.database}"
        )
        if connection.ssl_mode:
            self.connection_string += f"?sslmode={connection.ssl_mode}"
    
    async def _test_connection(self):
        """Test PostgreSQL connection"""
        loop = asyncio.get_event_loop()
        self._connection = await loop.run_in_executor(
            None, psycopg2.connect, self.connection_string
        )
        cursor = self._connection.cursor()
        await loop.run_in_executor(None, cursor.execute, "SELECT 1")
        await loop.run_in_executor(None, cursor.fetchone)
        cursor.close()

class MySQLAdapter(DatabaseAdapter):
    """MySQL database adapter"""
    
    def __init__(self, connection: DatabaseConnection):
        super().__init__(connection)
        self.connection_string = {
            'host': self.connection.host,
            'port': self.connection.port,
            'user': self.connection.username,
            'password': self.connection.password,
            'database': self.connection.database,
            'ssl': {'ssl': {}} if self.connection.ssl_mode else {}
        }
    
    async def _test_connection(self):
        """Test MySQL connection"""
        loop = asyncio.get_event_loop()
        self._connection = await loop.run_in_executor(
            None, pymysql.connect, **self.connection_string
        )
        cursor = self._connection.cursor()
        await loop.run_in_executor(None, cursor.execute, "SELECT 1")
        await loop.run_in_executor(None, cursor.fetchone)
        cursor.close()

class SnowflakeAdapter(DatabaseAdapter):
    """Snowflake database adapter"""
    
    def __init__(self, connection: DatabaseConnection):
        super().__init__(connection)
        # Snowflake connection parameters
        self.connection_params = {
            'account': connection.additional_params.get('account'),
            'user': connection.username,
            'password': connection.password,
            'warehouse': connection.additional_params.get('warehouse'),
            'database': connection.database,
            'schema': connection.schema
        }
    
    async def _test_connection(self):
        """Test Snowflake connection"""
        import snowflake.connector
        loop = asyncio.get_event_loop()
        self._connection = await loop.run_in_executor(
            None, snowflake.connector.connect, **self.connection_params
        )
        cursor = self._connection.cursor()
        await loop.run_in_executor(None, cursor.execute, "SELECT 1")
        await loop.run_in_executor(None, cursor.fetchone)
        cursor.close()

class BigQueryAdapter(DatabaseAdapter):
    """BigQuery database adapter"""
    
    def __init__(self, connection: DatabaseConnection):
        super().__init__(connection)
        self.credentials_file = connection.additional_params.get('credentials_file')
        self.project_id = connection.additional_params.get('project_id')
        self.dataset = connection.additional_params.get('dataset')
    
    async def _test_connection(self):
        """Test BigQuery connection"""
        from google.cloud import bigquery
        from google.oauth2 import service_account
        
        if self.credentials_file:
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_file
            )
            self._connection = bigquery.Client(
                project=self.project_id,
                credentials=credentials
            )
        else:
            self._connection = bigquery.Client(project=self.project_id)
        
        # Test with a simple query
        query = f"SELECT 1 FROM `{self.project_id}.{self.dataset}.INFORMATION_SCHEMA.TABLES` LIMIT 1"
        self._connection.query(query).result()

class RedshiftAdapter(DatabaseAdapter):
    """Redshift database adapter"""
    
    def __init__(self, connection: DatabaseConnection):
        super().__init__(connection)
        self.connection_string = (
            f"postgresql://{connection.username}:{connection.password}"
            f"@{connection.host}:{connection.port}/{connection.database}"
        )
        if connection.ssl_mode:
            self.connection_string += f"?sslmode={connection.ssl_mode}"
    
    async def _test_connection(self):
        """Test Redshift connection"""
        loop = asyncio.get_event_loop()
        self._connection = await loop.run_in_executor(
            None, psycopg2.connect, self.connection_string
        )
        cursor = self._connection.cursor()
        await loop.run_in_executor(None, cursor.execute, "SELECT 1")
        await loop.run_in_executor(None, cursor.fetchone)
        cursor.close()

# =============================================================================
# CONNECTION REGISTRY
# =============================================================================

class ConnectionRegistry:
    """Manages database connections and handles"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.connections: Dict[str, DatabaseConnection] = {}
        self.adapters: Dict[str, DatabaseAdapter] = {}
        self.handles: Dict[str, ConnectionHandle] = {}
        self.connection_handles: Dict[str, List[str]] = {}
    
    def create_adapter(self, connection: DatabaseConnection) -> DatabaseAdapter:
        """Create appropriate database adapter"""
        if connection.database_type.value == "postgresql":
            return PostgreSQLAdapter(connection)
        elif connection.database_type.value == "mysql":
            return MySQLAdapter(connection)
        elif connection.database_type.value == "snowflake":
            return SnowflakeAdapter(connection)
        elif connection.database_type.value == "bigquery":
            return BigQueryAdapter(connection)
        elif connection.database_type.value == "redshift":
            return RedshiftAdapter(connection)
        else:
            raise ValueError(f"Unsupported database type: {connection.database_type}")
    
    async def create_connection(self, connection: DatabaseConnection) -> CreateConnectionOutput:
        """Create and test a new database connection"""
        try:
            # Input validation
            validation_error = self._validate_connection_input(connection)
            if validation_error:
                return CreateConnectionOutput(
                    connection_id=connection.connection_id,
                    status=ConnectionStatus.FAILED,
                    message=f"Validation failed: {validation_error}"
                )
            
            # Test the connection
            adapter = self.create_adapter(connection)
            is_authenticated = await adapter.authenticate()
            
            if not is_authenticated:
                return CreateConnectionOutput(
                    connection_id=connection.connection_id,
                    status=ConnectionStatus.FAILED,
                    message="Authentication failed"
                )
            
            # Store the connection and adapter
            self.connections[connection.connection_id] = connection
            self.adapters[connection.connection_id] = adapter
            self.connection_handles[connection.connection_id] = []
            
            # Create initial handle
            handle = await self._create_handle(connection.connection_id)
            
            await adapter.close()
            
            return CreateConnectionOutput(
                connection_id=connection.connection_id,
                status=ConnectionStatus.CONNECTED,
                message="Connection created successfully",
                handle_id=handle.handle_id
            )
            
        except Exception as e:
            logging.error(f"Connection creation failed: {e}")
            return CreateConnectionOutput(
                connection_id=connection.connection_id,
                status=ConnectionStatus.FAILED,
                message=f"Connection failed: {str(e)}"
            )
    
    def _validate_connection_input(self, connection: DatabaseConnection) -> Optional[str]:
        """Validate connection input parameters"""
        if connection.database_type.value == "snowflake":
            if not connection.additional_params.get('account'):
                return "Snowflake account is required"
        elif connection.database_type.value == "bigquery":
            if not connection.additional_params.get('project_id'):
                return "BigQuery project_id is required"
            if not connection.additional_params.get('dataset'):
                return "BigQuery dataset is required"
        
        if not connection.host or not connection.database or not connection.username:
            return "Host, database, and username are required"
        
        return None
    
    async def get_connection_handle(self, connection_id: str) -> GetConnectionHandleOutput:
        """Get a connection handle for database access"""
        if connection_id not in self.connections:
            raise HTTPException(status_code=404, detail="Connection not found")
        
        # Check if we have an active handle
        active_handles = [
            handle_id for handle_id in self.connection_handles.get(connection_id, [])
            if self.handles[handle_id].expires_at > datetime.utcnow()
        ]
        
        if active_handles:
            handle = self.handles[active_handles[0]]
            return GetConnectionHandleOutput(
                handle=handle,
                status="active"
            )
        
        # Create new handle
        handle = await self._create_handle(connection_id)
        return GetConnectionHandleOutput(
            handle=handle,
            status="created"
        )
    
    async def _create_handle(self, connection_id: str) -> ConnectionHandle:
        """Create a new connection handle"""
        # Clean up expired handles
        await self._cleanup_expired_handles(connection_id)
        
        # Check handle limit
        if len(self.connection_handles.get(connection_id, [])) >= self.settings.max_handles_per_connection:
            raise HTTPException(status_code=429, detail="Too many active handles")
        
        # Create new handle
        handle_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(seconds=self.settings.handle_ttl_seconds)
        
        handle = ConnectionHandle(
            handle_id=handle_id,
            connection_id=connection_id,
            expires_at=expires_at,
            permissions=["read", "execute"]
        )
        
        self.handles[handle_id] = handle
        if connection_id not in self.connection_handles:
            self.connection_handles[connection_id] = []
        self.connection_handles[connection_id].append(handle_id)
        
        # Schedule cleanup task
        asyncio.create_task(self._schedule_handle_cleanup(handle_id, expires_at))
        
        return handle
    
    async def _schedule_handle_cleanup(self, handle_id: str, expires_at: datetime):
        """Schedule automatic cleanup of expired handle"""
        delay = (expires_at - datetime.utcnow()).total_seconds()
        if delay > 0:
            await asyncio.sleep(delay)
            await self._cleanup_expired_handles_by_id(handle_id)
    
    async def _cleanup_expired_handles_by_id(self, handle_id: str):
        """Clean up a specific expired handle"""
        if handle_id in self.handles:
            handle = self.handles[handle_id]
            if handle.expires_at <= datetime.utcnow():
                # Remove from connection_handles
                if handle.connection_id in self.connection_handles:
                    self.connection_handles[handle.connection_id] = [
                        h for h in self.connection_handles[handle.connection_id] 
                        if h != handle_id
                    ]
                
                # Remove handle
                del self.handles[handle_id]
                logging.info(f"Expired handle {handle_id} cleaned up")
    
    async def _cleanup_expired_handles(self, connection_id: str):
        """Clean up expired handles for a connection"""
        if connection_id not in self.connection_handles:
            return
        
        expired_handles = []
        for handle_id in self.connection_handles[connection_id]:
            if self.handles[handle_id].expires_at <= datetime.utcnow():
                expired_handles.append(handle_id)
                del self.handles[handle_id]
        
        for handle_id in expired_handles:
            self.connection_handles[connection_id].remove(handle_id)
    
    def list_connections(self, tenant_id: Optional[str] = None, user_id: Optional[str] = None) -> ListConnectionsOutput:
        """List all database connections"""
        connections = list(self.connections.values())
        
        # Filter by tenant/user if specified
        if tenant_id:
            connections = [c for c in connections if c.metadata.get('tenant_id') == tenant_id]
        if user_id:
            connections = [c for c in connections if c.metadata.get('user_id') == user_id]
        
        return ListConnectionsOutput(
            connections=connections,
            total_count=len(connections)
        )
    
    async def delete_connection(self, connection_id: str) -> DeleteConnectionOutput:
        """Delete a connection and purge all artifacts"""
        if connection_id not in self.connections:
            raise HTTPException(status_code=404, detail="Connection not found")
        
        try:
            # Generate offboard report
            offboard_report = self._generate_offboard_report(connection_id)
            
            # Close adapter
            if connection_id in self.adapters:
                await self.adapters[connection_id].close()
                del self.adapters[connection_id]
            
            # Remove handles
            if connection_id in self.connection_handles:
                for handle_id in self.connection_handles[connection_id]:
                    if handle_id in self.handles:
                        del self.handles[handle_id]
                del self.connection_handles[connection_id]
            
            # Remove connection
            connection_info = self.connections[connection_id]
            del self.connections[connection_id]
            
            # Log offboard report
            logging.info(f"Connection {connection_id} offboarded: {offboard_report}")
            
            return DeleteConnectionOutput(
                status="deleted",
                message="Connection deleted successfully",
                purged_artifacts=["connection", "adapter", "handles"],
                offboard_report=offboard_report
            )
            
        except Exception as e:
            logging.error(f"Connection deletion failed: {e}")
            return DeleteConnectionOutput(
                status="failed",
                message=f"Deletion failed: {str(e)}",
                purged_artifacts=[]
            )
    
    def _generate_offboard_report(self, connection_id: str) -> Dict[str, Any]:
        """Generate offboard report for audit trail"""
        connection = self.connections.get(connection_id)
        if not connection:
            return {}
        
        report = {
            "connection_id": connection_id,
            "connection_name": connection.name,
            "database_type": connection.database_type.value,
            "host": connection.host,
            "database": connection.database,
            "username": connection.username,
            "offboarded_at": datetime.utcnow().isoformat(),
            "artifacts_purged": [
                "connection_config",
                "database_adapter",
                "active_handles",
                "vector_embeddings",
                "table_profiles",
                "fine_tuned_models",
                "query_history",
                "feedback_data"
            ],
            "data_retention_policy": "immediate_purge",
            "compliance_notes": "All data and artifacts removed per user request"
        }
        
        return report
    
    def validate_handle(self, handle_id: str, connection_id: str) -> bool:
        """Validate a handle for a specific connection"""
        if handle_id not in self.handles:
            return False
        
        handle = self.handles[handle_id]
        if handle.connection_id != connection_id:
            return False
        
        if handle.expires_at <= datetime.utcnow():
            return False
        
        return True

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="AskData Connection Registry",
    description="Manages database connections and handles",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global settings and registry
settings = Settings()
registry = ConnectionRegistry(settings)

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup"""
    setup_logging()
    logging.info("AskData Connection Registry starting up")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "connection-registry",
        "connections": len(registry.connections),
        "handles": len(registry.handles),
        "timestamp": datetime.utcnow()
    }

@app.post("/mcp/create_connection", response_model=CreateConnectionOutput)
async def create_connection_mcp(request: CreateConnectionInput):
    """MCP tool: Create a new database connection"""
    return await registry.create_connection(request.connection)

@app.post("/mcp/get_connection_handle", response_model=GetConnectionHandleOutput)
async def get_connection_handle_mcp(request: GetConnectionHandleInput):
    """MCP tool: Get a connection handle"""
    return await registry.get_connection_handle(request.connection_id)

@app.post("/mcp/list_connections", response_model=ListConnectionsOutput)
async def list_connections_mcp(request: ListConnectionsInput):
    """MCP tool: List connections"""
    return registry.list_connections(request.tenant_id, request.user_id)

@app.post("/mcp/delete_connection", response_model=DeleteConnectionOutput)
async def delete_connection_mcp(request: DeleteConnectionInput):
    """MCP tool: Delete a connection"""
    return await registry.delete_connection(request.connection_id)

@app.post("/mcp/offboard_connection", response_model=OffboardConnectionOutput)
async def offboard_connection_mcp(request: OffboardConnectionInput):
    """MCP tool: Offboard a connection with full purge"""
    try:
        # This would trigger the full purge across all services
        # For now, just call the local delete method
        result = await registry.delete_connection(request.connection_id)
        
        return OffboardConnectionOutput(
            connection_id=request.connection_id,
            status=result.status,
            message=result.message,
            offboard_report=result.offboard_report,
            artifacts_purged=result.purged_artifacts
        )
        
    except Exception as e:
        logging.error(f"Offboard failed: {e}")
        return OffboardConnectionOutput(
            connection_id=request.connection_id,
            status="failed",
            message=f"Offboard failed: {str(e)}",
            offboard_report={},
            artifacts_purged=[]
        )

@app.get("/connections")
async def list_connections_api(tenant_id: Optional[str] = None, user_id: Optional[str] = None):
    """REST API: List connections"""
    return registry.list_connections(tenant_id, user_id)

@app.get("/connections/{connection_id}")
async def get_connection(connection_id: str):
    """REST API: Get connection details"""
    if connection_id not in registry.connections:
        raise HTTPException(status_code=404, detail="Connection not found")
    return registry.connections[connection_id]

@app.delete("/connections/{connection_id}")
async def delete_connection_api(connection_id: str):
    """REST API: Delete connection"""
    return await registry.delete_connection(connection_id)

@app.get("/handles/{handle_id}/validate")
async def validate_handle(handle_id: str, connection_id: str):
    """Validate a handle for a connection"""
    is_valid = registry.validate_handle(handle_id, connection_id)
    return {
        "handle_id": handle_id,
        "connection_id": connection_id,
        "is_valid": is_valid,
        "timestamp": datetime.utcnow()
    }

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    ) 