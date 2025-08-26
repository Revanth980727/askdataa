"""
AskData Query Executor Service

This service executes validated SQL queries against databases and returns results.
It manages query execution lifecycle, result pagination, and provides execution metadata.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import json
import hashlib

import structlog
import httpx
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# Add the contracts directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent / "contracts"))

from mcp_tools import (
    ExecuteQueryInput, ExecuteQueryOutput,
    QueryResult, QueryMetadata, ExecutionStatus
)

# =============================================================================
# CONFIGURATION
# =============================================================================

class Settings(BaseSettings):
    """Application settings"""
    environment: str = "local"
    log_level: str = "INFO"
    debug: bool = False
    
    # Connection registry settings
    connection_registry_url: str = "http://connection-registry:8000"
    connection_registry_timeout: int = 30
    
    # Execution settings
    default_timeout: int = 300  # 5 minutes
    max_result_rows: int = 10000
    preview_rows: int = 100
    chunk_size: int = 1000
    
    # Cache settings
    cache_ttl_minutes: int = 30
    max_cache_size: int = 100
    
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
# DATA MODELS
# =============================================================================

class QueryExecutionRequest(BaseModel):
    """Request to execute a query"""
    sql: str
    connection_id: str
    run_envelope: Dict[str, Any]
    timeout: Optional[int] = None
    max_rows: Optional[int] = None
    preview_only: bool = False

class QueryExecutionResponse(BaseModel):
    """Response containing query execution results"""
    success: bool
    execution_id: Optional[str] = None
    result: Optional[QueryResult] = None
    error: Optional[str] = None
    metadata: Optional[QueryMetadata] = None

class ExecutionCache:
    """In-memory cache for query results"""
    
    def __init__(self, ttl_minutes: int, max_size: int):
        self.ttl_minutes = ttl_minutes
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, datetime] = {}
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if not expired"""
        if key not in self.cache:
            return None
        
        # Check if expired
        if datetime.utcnow() - self.access_times[key] > timedelta(minutes=self.ttl_minutes):
            self._remove(key)
            return None
        
        # Update access time
        self.access_times[key] = datetime.utcnow()
        return self.cache[key]
    
    def set(self, key: str, value: Dict[str, Any]):
        """Set cached result"""
        # Evict oldest if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times, key=lambda k: self.access_times[k])
            self._remove(oldest_key)
        
        self.cache[key] = value
        self.access_times[key] = datetime.utcnow()
    
    def _remove(self, key: str):
        """Remove item from cache"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]

# =============================================================================
# SERVICE CLIENTS
# =============================================================================

class ServiceClient:
    """HTTP client for calling other services"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def call_mcp_tool(self, tool_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool on a service"""
        url = f"{self.base_url}/mcp/{tool_name}"
        
        try:
            response = await self.client.post(url, json=input_data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error calling {tool_name}: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Service error: {e.response.text}")
        except httpx.RequestError as e:
            logging.error(f"Request error calling {tool_name}: {e}")
            raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

# =============================================================================
# QUERY EXECUTOR SERVICE
# =============================================================================

class QueryExecutorService:
    """Service for executing SQL queries against databases"""
    
    def __init__(self, connection_registry_client: ServiceClient, settings: Settings):
        self.connection_registry = connection_registry_client
        self.settings = settings
        self.cache = ExecutionCache(settings.cache_ttl_minutes, settings.max_cache_size)
        self.active_executions: Dict[str, Dict[str, Any]] = {}
    
    async def execute_query(self, request: QueryExecutionRequest) -> ExecuteQueryOutput:
        """Execute a SQL query and return results"""
        try:
            connection_id = request.connection_id
            sql = request.sql
            timeout = request.timeout or self.settings.default_timeout
            max_rows = request.max_rows or self.settings.max_result_rows
            
            # Generate execution ID
            execution_id = self._generate_execution_id(connection_id, sql)
            
            # Check cache first
            cache_key = self._generate_cache_key(connection_id, sql, max_rows)
            cached_result = self.cache.get(cache_key)
            if cached_result and not request.preview_only:
                logging.info(f"Returning cached result for execution {execution_id}")
                return ExecuteQueryOutput(
                    execution_id=execution_id,
                    status=ExecutionStatus.COMPLETED,
                    result=cached_result["result"],
                    metadata=cached_result["metadata"],
                    execution_time=datetime.utcnow().isoformat()
                )
            
            # Get connection handle
            handle_result = await self._get_connection_handle(connection_id, request.run_envelope)
            if not handle_result:
                raise HTTPException(status_code=400, detail="Failed to get connection handle")
            
            # Track execution
            self.active_executions[execution_id] = {
                "status": ExecutionStatus.RUNNING,
                "start_time": datetime.utcnow(),
                "connection_id": connection_id,
                "sql": sql,
                "max_rows": max_rows
            }
            
            # Execute query
            if request.preview_only:
                result = await self._execute_preview_query(handle_result, sql, execution_id)
            else:
                result = await self._execute_full_query(handle_result, sql, execution_id, max_rows, timeout)
            
            # Cache result if successful
            if result and result.status == ExecutionStatus.COMPLETED:
                self.cache.set(cache_key, {
                    "result": result.result,
                    "metadata": result.metadata
                })
            
            # Clean up execution tracking
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
            
            return result
            
        except Exception as e:
            logging.error(f"Query execution failed: {e}")
            # Clean up execution tracking
            if 'execution_id' in locals() and execution_id in self.active_executions:
                del self.active_executions[execution_id]
            
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """Get the status of a query execution"""
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id]
            duration = (datetime.utcnow() - execution["start_time"]).total_seconds()
            
            return {
                "execution_id": execution_id,
                "status": execution["status"],
                "start_time": execution["start_time"].isoformat(),
                "duration_seconds": duration,
                "connection_id": execution["connection_id"]
            }
        
        return {"execution_id": execution_id, "status": "not_found"}
    
    async def cancel_execution(self, execution_id: str) -> Dict[str, Any]:
        """Cancel a running query execution"""
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id]
            execution["status"] = ExecutionStatus.CANCELLED
            
            return {
                "execution_id": execution_id,
                "status": "cancelled",
                "message": "Execution cancelled successfully"
            }
        
        return {"execution_id": execution_id, "status": "not_found"}
    
    async def _get_connection_handle(self, connection_id: str, run_envelope: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get a connection handle from the registry"""
        try:
            result = await self.connection_registry.call_mcp_tool("get_connection_handle", {
                "run_envelope": run_envelope,
                "connection_id": connection_id
            })
            return result
        except Exception as e:
            logging.error(f"Failed to get connection handle: {e}")
            return None
    
    async def _execute_preview_query(self, handle: Dict[str, Any], sql: str, execution_id: str) -> ExecuteQueryOutput:
        """Execute a preview query (limited rows)"""
        try:
            # Add LIMIT clause for preview
            preview_sql = self._add_limit_clause(sql, self.settings.preview_rows)
            
            # Execute with connection handle
            result_data = await self._execute_with_handle(handle, preview_sql)
            
            # Create result object
            result = QueryResult(
                data=result_data["rows"],
                columns=result_data["columns"],
                total_rows=result_data["total_rows"],
                preview_rows=len(result_data["rows"]),
                is_preview=True
            )
            
            metadata = QueryMetadata(
                execution_id=execution_id,
                sql_executed=preview_sql,
                execution_time_ms=result_data["execution_time_ms"],
                rows_returned=len(result_data["rows"]),
                total_rows_available=result_data["total_rows"],
                is_preview=True
            )
            
            return ExecuteQueryOutput(
                execution_id=execution_id,
                status=ExecutionStatus.COMPLETED,
                result=result,
                metadata=metadata,
                execution_time=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            logging.error(f"Preview query execution failed: {e}")
            return ExecuteQueryOutput(
                execution_id=execution_id,
                status=ExecutionStatus.FAILED,
                error=str(e),
                execution_time=datetime.utcnow().isoformat()
            )
    
    async def _execute_full_query(self, handle: Dict[str, Any], sql: str, execution_id: str, 
                                 max_rows: int, timeout: int) -> ExecuteQueryOutput:
        """Execute a full query with pagination if needed"""
        try:
            # Execute with connection handle
            result_data = await self._execute_with_handle(handle, sql, timeout)
            
            # Handle large result sets
            if result_data["total_rows"] > max_rows:
                # Truncate to max_rows
                result_data["rows"] = result_data["rows"][:max_rows]
                result_data["total_rows"] = max_rows
            
            # Create result object
            result = QueryResult(
                data=result_data["rows"],
                columns=result_data["columns"],
                total_rows=result_data["total_rows"],
                preview_rows=len(result_data["rows"]),
                is_preview=False
            )
            
            metadata = QueryMetadata(
                execution_id=execution_id,
                sql_executed=sql,
                execution_time_ms=result_data["execution_time_ms"],
                rows_returned=len(result_data["rows"]),
                total_rows_available=result_data["total_rows"],
                is_preview=False
            )
            
            return ExecuteQueryOutput(
                execution_id=execution_id,
                status=ExecutionStatus.COMPLETED,
                result=result,
                metadata=metadata,
                execution_time=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            logging.error(f"Full query execution failed: {e}")
            return ExecuteQueryOutput(
                execution_id=execution_id,
                status=ExecutionStatus.FAILED,
                error=str(e),
                execution_time=datetime.utcnow().isoformat()
            )
    
    async def _execute_with_handle(self, handle: Dict[str, Any], sql: str, timeout: int = None) -> Dict[str, Any]:
        """Execute SQL using the connection handle"""
        # In a real implementation, this would:
        # 1. Use the handle to connect to the database
        # 2. Execute the SQL
        # 3. Return results
        
        # For now, return mock data
        return self._generate_mock_result(sql)
    
    def _generate_mock_result(self, sql: str) -> Dict[str, Any]:
        """Generate mock query result for demonstration"""
        # Parse SQL to determine result structure
        sql_lower = sql.lower()
        
        if "users" in sql_lower:
            columns = ["id", "username", "email", "created_at", "is_active"]
            rows = [
                [1, "john_doe", "john@example.com", "2024-01-15", True],
                [2, "jane_smith", "jane@example.com", "2024-01-16", True],
                [3, "bob_wilson", "bob@example.com", "2024-01-17", False]
            ]
        elif "orders" in sql_lower:
            columns = ["order_id", "user_id", "order_date", "total_amount", "status"]
            rows = [
                [1001, 1, "2024-01-15", 99.99, "completed"],
                [1002, 2, "2024-01-16", 149.99, "pending"],
                [1003, 1, "2024-01-17", 79.99, "completed"]
            ]
        else:
            columns = ["id", "name", "value"]
            rows = [
                [1, "Item 1", 100],
                [2, "Item 2", 200],
                [3, "Item 3", 300]
            ]
        
        return {
            "rows": rows,
            "columns": columns,
            "total_rows": len(rows),
            "execution_time_ms": 150
        }
    
    def _add_limit_clause(self, sql: str, limit: int) -> str:
        """Add LIMIT clause to SQL if not present"""
        sql_upper = sql.upper().strip()
        if "LIMIT" not in sql_upper:
            return f"{sql} LIMIT {limit}"
        return sql
    
    def _generate_execution_id(self, connection_id: str, sql: str) -> str:
        """Generate unique execution ID"""
        content = f"{connection_id}:{sql}:{datetime.utcnow().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _generate_cache_key(self, connection_id: str, sql: str, max_rows: int) -> str:
        """Generate cache key for query results"""
        content = f"{connection_id}:{sql}:{max_rows}"
        return hashlib.sha256(content.encode()).hexdigest()

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="AskData Query Executor Service",
    description="Executes validated SQL queries and returns results",
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

# Global settings and service
settings = Settings()
connection_registry_client = ServiceClient(settings.connection_registry_url, settings.connection_registry_timeout)
query_executor_service = QueryExecutorService(connection_registry_client, settings)

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup"""
    setup_logging()
    logging.info("AskData Query Executor Service starting up")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    await connection_registry_client.close()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "query-executor",
        "active_executions": len(query_executor_service.active_executions),
        "cache_size": len(query_executor_service.cache.cache),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/mcp/execute_query", response_model=ExecuteQueryOutput)
async def execute_query_mcp(request: ExecuteQueryInput):
    """MCP tool: Execute SQL query"""
    return await query_executor_service.execute_query(QueryExecutionRequest(
        sql=request.sql,
        connection_id=request.connection_id,
        run_envelope=request.run_envelope,
        timeout=request.timeout,
        max_rows=request.max_rows,
        preview_only=request.preview_only
    ))

@app.get("/execution/{execution_id}/status")
async def get_execution_status(execution_id: str):
    """Get execution status"""
    return await query_executor_service.get_execution_status(execution_id)

@app.post("/execution/{execution_id}/cancel")
async def cancel_execution(execution_id: str):
    """Cancel execution"""
    return await query_executor_service.cancel_execution(execution_id)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"service": "query-executor", "status": "running"}

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
