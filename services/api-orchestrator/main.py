"""
AskData API Orchestrator Service

This service coordinates the execution of natural language queries through a LangGraph-based
workflow, calling other microservices via MCP tools over HTTP.
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from enum import Enum

import httpx
import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Add the contracts directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent / "contracts"))

from mcp_tools import (
    RunEnvelope, RunStatus, NodeStatus, DatabaseConnection, Budget,
    CreateConnectionInput, CreateConnectionOutput,
    IntrospectDatabaseInput, IntrospectDatabaseOutput,
    IndexConnectionInput, IndexConnectionOutput,
    SearchTablesInput, SearchTablesOutput,
    ProfileTablesInput, ProfileTablesOutput,
    PruneColumnsInput, PruneColumnsOutput,
    BuildJoinGraphInput, BuildJoinGraphOutput,
    ResolveMetricsInput, ResolveMetricsOutput,
    GenerateSQLInput, GenerateSQLOutput,
    ValidateSQLInput, ValidateSQLOutput,
    ExecuteQueryInput, ExecuteQueryOutput,
    ExplainResultsInput, ExplainResultsOutput
)

# SQL Safety Gate
import re
from typing import Tuple
import functools

def validate_sql_safety(sql: str) -> Tuple[bool, str]:
    """
    Validate SQL for safety - only allow SELECT statements
    Returns (is_safe, error_message)
    """
    # Remove comments and normalize whitespace
    sql_clean = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
    sql_clean = re.sub(r'/\*.*?\*/', '', sql_clean, flags=re.DOTALL)
    sql_clean = re.sub(r'\s+', ' ', sql_clean).strip()
    
    # Check for multiple statements
    if ';' in sql_clean:
        return False, "Multiple statements not allowed"
    
    # Check for non-SELECT statements
    sql_upper = sql_clean.upper()
    if not sql_upper.startswith('SELECT'):
        return False, "Only SELECT statements are allowed"
    
    # Check for dangerous keywords
    dangerous_keywords = [
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE',
        'GRANT', 'REVOKE', 'EXECUTE', 'EXEC', 'UNLOAD', 'EXPORT', 'COPY'
    ]
    
    for keyword in dangerous_keywords:
        if keyword in sql_upper:
            return False, f"Dangerous keyword '{keyword}' not allowed"
    
    # Check for external functions (basic check)
    if re.search(r'\b[A-Z_][A-Z0-9_]*\s*\(', sql_upper):
        # Allow common SQL functions
        allowed_functions = [
            'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'COALESCE', 'NULLIF',
            'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'CAST', 'CONVERT',
            'DATE', 'YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND'
        ]
        
        # This is a simplified check - in production you'd want a more robust parser
        if not any(func in sql_upper for func in allowed_functions):
            return False, "External function calls not allowed"
    
    return True, "SQL is safe"

def track_node_execution(func):
    """Decorator to track node execution timing and retries"""
    @functools.wraps(func)
    async def wrapper(self, state: OrchestratorState, *args, **kwargs):
        start_time = time.time()
        node_name = func.__name__.replace('_', ' ')
        
        # Update current node
        state.current_node = node_name
        state.step_id = str(uuid.uuid4())
        
        # Track retry count
        if node_name not in state.retry_counts:
            state.retry_counts[node_name] = 0
        
        try:
            # Execute the node
            result = await func(self, state, *args, **kwargs)
            
            # Record timing
            execution_time = time.time() - start_time
            state.step_timings[node_name] = execution_time
            
            return result
            
        except Exception as e:
            # Increment retry count
            state.retry_counts[node_name] += 1
            
            # Check if we should retry
            if state.retry_counts[node_name] <= self.settings.max_retries:
                logging.warning(f"Node {node_name} failed, retrying ({state.retry_counts[node_name]}/{self.settings.max_retries}): {e}")
                await asyncio.sleep(self.settings.retry_backoff_seconds)
                return await wrapper(self, state, *args, **kwargs)
            else:
                # Max retries exceeded
                state.status = RunStatus.FAILED
                state.errors.append(f"Node {node_name} failed after {self.settings.max_retries} retries: {str(e)}")
                logging.error(f"Node {node_name} failed after {self.settings.max_retries} retries: {e}")
                return state
    
    return wrapper

class ModelRegistry:
    """Simple local model registry for per-connection adapters"""
    
    def __init__(self, registry_path: str = "./data/models/model_registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.models = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry from file"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load model registry: {e}")
                return {}
        
        # Create sample registry if it doesn't exist
        sample_registry = {
            "default": {
                "base_model": "gpt-4",
                "adapter_model": None,
                "description": "Default model for all connections",
                "created_at": datetime.utcnow().isoformat()
            },
            "example_connection": {
                "base_model": "gpt-4",
                "adapter_model": "lora:example_connection_v1",
                "description": "Example connection with fine-tuned adapter",
                "created_at": datetime.utcnow().isoformat()
            }
        }
        
        try:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.registry_path, 'w') as f:
                json.dump(sample_registry, f, indent=2, default=str)
            logging.info(f"Created sample model registry at {self.registry_path}")
            return sample_registry
        except Exception as e:
            logging.error(f"Failed to create sample model registry: {e}")
            return {}
    
    def _save_registry(self):
        """Save model registry to file"""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.models, f, indent=2, default=str)
        except Exception as e:
            logging.error(f"Failed to save model registry: {e}")
    
    def get_adapter_for_connection(self, connection_id: str) -> Optional[str]:
        """Get adapter model for a connection"""
        return self.models.get(connection_id, {}).get('adapter_model')
    
    def get_base_model_for_connection(self, connection_id: str) -> str:
        """Get base model for a connection, fallback to default"""
        connection_config = self.models.get(connection_id, {})
        return connection_config.get('base_model', self.models.get('default', {}).get('base_model', 'gpt-4'))
    
    def get_model_info_for_connection(self, connection_id: str) -> Dict[str, Any]:
        """Get complete model information for a connection"""
        connection_config = self.models.get(connection_id, {})
        default_config = self.models.get('default', {})
        
        return {
            "base_model": connection_config.get('base_model', default_config.get('base_model', 'gpt-4')),
            "adapter_model": connection_config.get('adapter_model'),
            "description": connection_config.get('description', default_config.get('description', 'Default model')),
            "model_type": "adapter" if connection_config.get('adapter_model') else "base"
        }
    
    def set_adapter_for_connection(self, connection_id: str, adapter_model: str):
        """Set adapter model for a connection"""
        if connection_id not in self.models:
            self.models[connection_id] = {}
        self.models[connection_id]['adapter_model'] = adapter_model
        self.models[connection_id]['updated_at'] = datetime.utcnow().isoformat()
        self._save_registry()
    
    def remove_connection(self, connection_id: str):
        """Remove connection from registry"""
        if connection_id in self.models:
            del self.models[connection_id]
            self._save_registry()

# =============================================================================
# CONFIGURATION
# =============================================================================

class Settings(BaseSettings):
    """Application settings"""
    environment: str = "local"
    log_level: str = "INFO"
    debug: bool = False
    
    # Service URLs
    connection_registry_url: str = "http://connection-registry:8000"
    introspect_url: str = "http://introspect:8000"
    vector_store_url: str = "http://vector-store:8000"
    table_retriever_url: str = "http://table-retriever:8000"
    micro_profiler_url: str = "http://micro-profiler:8000"
    column_pruner_url: str = "http://column-pruner:8000"
    join_graph_url: str = "http://join-graph:8000"
    metric_resolver_url: str = "http://metric-resolver:8000"
    sql_generator_url: str = "http://sql-generator:8000"
    sql_validator_url: str = "http://sql-validator:8000"
    query_executor_url: str = "http://query-executor:8000"
    result_explainer_url: str = "http://result-explainer:8000"
    
    # Performance settings
    max_tables_per_query: int = 20
    max_columns_per_table: int = 15
    default_sample_size: int = 500
    max_prompt_tokens: int = 8000
    query_timeout_seconds: int = 300
    
    # Retry settings
    max_retries: int = 3
    retry_backoff_seconds: int = 1
    
    # Stub mode for development
    stub_mode: bool = Field(False, description="Enable stub mode for unimplemented services")
    
    # Vector store settings
    vector_backend: str = "chroma"  # chroma or opensearch
    embedding_version: str = "1.0"
    
    # Model registry settings
    model_registry_path: str = "./data/models/model_registry.json"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

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
# SERVICE CLIENT
# =============================================================================

class ServiceClient:
    """HTTP client for calling other microservices"""
    
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
# ORCHESTRATOR STATE
# =============================================================================

class OrchestratorState(BaseModel):
    """State maintained during query execution"""
    run_id: str
    step_id: str
    active_connection_id: str
    question: str
    status: RunStatus = RunStatus.PENDING
    current_node: Optional[str] = None
    node_results: Dict[str, Any] = {}
    errors: List[str] = []
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    retry_counts: Dict[str, int] = Field(default_factory=dict)
    step_timings: Dict[str, float] = Field(default_factory=dict)
    budget_exceeded: bool = Field(False, description="Whether budget limits were exceeded")
    budget: Optional[Budget] = Field(None, description="Execution budget constraints")

class RunRegistry:
    """In-memory registry for tracking query runs"""
    
    def __init__(self):
        self.runs: Dict[str, OrchestratorState] = {}
        self.lock = asyncio.Lock()
        self.trace_dir = Path("logs/runs")
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure the trace directory exists and is writable
        try:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            # Test write access
            test_file = self.trace_dir / ".test"
            test_file.write_text("test")
            test_file.unlink()
            logging.info(f"Trace directory ready: {self.trace_dir}")
        except Exception as e:
            logging.error(f"Failed to initialize trace directory: {e}")
            # Fallback to current directory
            self.trace_dir = Path(".")
    
    async def create_run(self, state: OrchestratorState) -> str:
        """Create a new run"""
        async with self.lock:
            self.runs[state.run_id] = state
            # Write initial trace file
            try:
                await self._write_trace_file(state)
            except Exception as e:
                logging.error(f"Failed to write initial trace file: {e}")
                # Don't fail the run creation if trace writing fails
            
            return state.run_id
    
    async def get_run(self, run_id: str) -> Optional[OrchestratorState]:
        """Get a run by ID"""
        async with self.lock:
            return self.runs.get(run_id)
    
    async def update_run(self, run_id: str, updates: Dict[str, Any]) -> bool:
        """Update a run"""
        async with self.lock:
            if run_id in self.runs:
                for key, value in updates.items():
                    if hasattr(self.runs[run_id], key):
                        setattr(self.runs[run_id], key, value)
                # Write updated trace file
                try:
                    await self._write_trace_file(self.runs[run_id])
                except Exception as e:
                    logging.error(f"Failed to write updated trace file: {e}")
                    # Don't fail the update if trace writing fails
                
                return True
            return False
    
    async def _write_trace_file(self, state: OrchestratorState):
        """Write trace file for a run"""
        try:
            trace_file = self.trace_dir / f"{state.run_id}.json"
            trace_data = {
                "run_id": state.run_id,
                "step_id": state.step_id,
                "active_connection_id": state.active_connection_id,
                "question": state.question,
                "status": state.status.value,
                "current_node": state.current_node,
                "start_time": state.start_time.isoformat() if state.start_time else None,
                "end_time": state.end_time.isoformat() if state.end_time else None,
                "errors": state.errors,
                "node_results": state.node_results,
                "budget": state.budget.model_dump() if state.budget else None,
                "budget_exceeded": state.budget_exceeded,
                "retry_counts": state.retry_counts,
                "step_timings": state.step_timings,
                "metadata": state.metadata,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Ensure the trace directory exists
            trace_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(trace_file, 'w') as f:
                json.dump(trace_data, f, indent=2, default=str)
                
            logging.debug(f"Trace file updated: {trace_file}")
                
        except Exception as e:
            logging.error(f"Failed to write trace file: {e}")
            # Don't fail the operation if trace writing fails
    
    async def list_runs(self, limit: int = 100) -> List[OrchestratorState]:
        """List recent runs"""
        async with self.lock:
            return list(self.runs.values())[-limit:]
    
    async def cleanup_old_runs(self, max_age_hours: int = 24):
        """Clean up old runs"""
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        async with self.lock:
            old_runs = [
                run_id for run_id, run in self.runs.items()
                if run.start_time < cutoff
            ]
            for run_id in old_runs:
                del self.runs[run_id]

class QueryRequest(BaseModel):
    """Request to execute a natural language query"""
    question: str = Field(..., description="Natural language question")
    connection_id: str = Field(..., description="Database connection ID")
    session_id: Optional[str] = Field(None, description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")
    options: Optional[Dict[str, Any]] = Field(None, description="Query options")

class QueryResponse(BaseModel):
    """Response from query execution"""
    run_id: str
    status: RunStatus
    results: Optional[Dict[str, Any]] = None
    sql: Optional[str] = None
    explanation: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time: float
    timeline: List[Dict[str, Any]]

# =============================================================================
# LANGGRAPH WORKFLOW
# =============================================================================

class AskDataWorkflow:
    """LangGraph-based workflow for query execution"""
    
    def __init__(self, settings: Settings, model_registry: ModelRegistry):
        self.settings = settings
        self.model_registry = model_registry
        self.clients = {}
        self._setup_clients()
        self.graph = self._build_graph()
        self.checkpointer = MemorySaver()
    
    def _setup_clients(self):
        """Initialize service clients"""
        self.clients = {
            "connection_registry": ServiceClient(self.settings.connection_registry_url),
            "introspect": ServiceClient(self.settings.introspect_url),
            "vector_store": ServiceClient(self.settings.vector_store_url),
            "table_retriever": ServiceClient(self.settings.table_retriever_url),
            "micro_profiler": ServiceClient(self.settings.micro_profiler_url),
            "column_pruner": ServiceClient(self.settings.column_pruner_url),
            "join_graph": ServiceClient(self.settings.join_graph_url),
            "metric_resolver": ServiceClient(self.settings.metric_resolver_url),
            "sql_generator": ServiceClient(self.settings.sql_generator_url),
            "sql_validator": ServiceClient(self.settings.sql_validator_url),
            "query_executor": ServiceClient(self.settings.query_executor_url),
            "result_explainer": ServiceClient(self.settings.result_explainer_url),
        }
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(OrchestratorState)
        
        # Add nodes
        workflow.add_node("route_connection", self._route_connection)
        workflow.add_node("retrieve_tables", self._retrieve_tables)
        workflow.add_node("profile_tables", self._profile_tables)
        workflow.add_node("prune_columns", self._prune_columns)
        workflow.add_node("build_join_graph", self._build_join_graph)
        workflow.add_node("resolve_metrics", self._resolve_metrics)
        workflow.add_node("generate_sql", self._generate_sql)
        workflow.add_node("validate_sql", self._validate_sql)
        workflow.add_node("execute_query", self._execute_query)
        workflow.add_node("explain_results", self._explain_results)
        
        # Add edges
        workflow.set_entry_point("route_connection")
        workflow.add_edge("route_connection", "retrieve_tables")
        workflow.add_edge("retrieve_tables", "profile_tables")
        workflow.add_edge("profile_tables", "prune_columns")
        workflow.add_edge("prune_columns", "build_join_graph")
        workflow.add_edge("build_join_graph", "resolve_metrics")
        workflow.add_edge("resolve_metrics", "generate_sql")
        workflow.add_edge("generate_sql", "validate_sql")
        workflow.add_edge("validate_sql", "execute_query")
        workflow.add_edge("execute_query", "explain_results")
        workflow.add_edge("explain_results", END)
        
        # Add conditional edges for error handling
        workflow.add_conditional_edges(
            "route_connection",
            self._should_continue,
            {
                "continue": "retrieve_tables",
                "fail": END
            }
        )
        
        workflow.add_conditional_edges(
            "retrieve_tables",
            self._should_continue,
            {
                "continue": "profile_tables",
                "fail": END
            }
        )
        
        workflow.add_conditional_edges(
            "profile_tables",
            self._should_continue,
            {
                "continue": "prune_columns",
                "fail": END
            }
        )
        
        # Add similar conditionals for other nodes...
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _should_continue(self, state: OrchestratorState) -> str:
        """Determine if workflow should continue or fail"""
        if state.status == RunStatus.FAILED:
            return "fail"
        
        # Check budget constraints
        if state.budget_exceeded:
            return "fail"
        
        return "continue"
    
    def _check_budget(self, state: OrchestratorState, operation: str) -> bool:
        """Check if operation is within budget constraints"""
        if not state.budget:
            return True
        
        # If budget is exceeded, don't allow further operations
        if state.budget_exceeded:
            return False
        
        # Check table count
        if operation == "retrieve_tables":
            current_tables = len(state.node_results.get("table_retriever", {}).get("results", []))
            if current_tables > state.budget.max_tables:
                state.budget_exceeded = True
                state.errors.append(f"Table count {current_tables} exceeds budget limit {state.budget.max_tables}")
                return False
        
        # Check sample size
        if operation == "profile_tables":
            sample_size = state.node_results.get("micro_profiler", {}).get("profiles", [])
            if sample_size and any(profile.get("sample_size", 0) > state.budget.max_sample_rows for profile in sample_size):
                state.budget_exceeded = True
                state.errors.append(f"Sample size exceeds budget limit {state.budget.max_sample_rows}")
                return False
        
        # Check prompt tokens
        if operation == "generate_sql":
            token_usage = state.node_results.get("sql_generator", {}).get("token_usage", {})
            if token_usage.get("total", 0) > state.budget.max_prompt_tokens:
                state.budget_exceeded = True
                state.errors.append(f"Token usage {token_usage.get('total', 0)} exceeds budget limit {state.budget.max_prompt_tokens}")
                return False
        
        # Check result rows
        if operation == "execute_query":
            row_count = state.node_results.get("query_executor", {}).get("row_count", 0)
            if row_count > state.budget.max_result_rows:
                state.budget_exceeded = True
                state.errors.append(f"Result rows {row_count} exceeds budget limit {state.budget.max_result_rows}")
                return False
        
        return True
    
    def _decrement_budget(self, state: OrchestratorState, operation: str, amount: int = 1):
        """Decrement budget counters for an operation"""
        if not state.budget:
            return
        
        if operation == "tables":
            # This would be handled by the actual table count
            pass
        elif operation == "sample_rows":
            # This would be handled by the actual sample size
            pass
        elif operation == "prompt_tokens":
            # This would be handled by the actual token usage
            pass
        elif operation == "result_rows":
            # This would be handled by the actual result count
            pass
        
        # Log budget usage
        logging.info(f"Budget usage for {operation}: {amount}")
    
    async def _get_connection_dialect(self, state: OrchestratorState) -> str:
        """Get the database dialect for the active connection"""
        try:
            connection_details = await self.clients["connection_registry"].call_mcp_tool(
                "list_connections",
                {
                    "run_envelope": {
                        "run_id": state.run_id,
                        "step_id": state.step_id,
                        "active_connection_id": state.active_connection_id
                    }
                }
            )
            
            active_connection = next(
                (conn for conn in connection_details.get("connections", []) 
                 if conn["connection_id"] == state.active_connection_id),
                None
            )
            
            if not active_connection:
                raise ValueError(f"Connection {state.active_connection_id} not found")
            
            return active_connection.get("database_type", "postgresql")
            
        except Exception as e:
            logging.error(f"Failed to get connection dialect: {e}")
            return "postgresql"  # Default fallback
    
    async def execute_workflow(self, state: OrchestratorState) -> OrchestratorState:
        """Execute the workflow using LangGraph"""
        try:
            # Execute the graph with retries
            config = {"configurable": {"thread_id": state.run_id}}
            
            # Try to execute the workflow
            for attempt in range(self.settings.max_retries + 1):
                try:
                    result = await self.graph.ainvoke(state, config)
                    
                    # Update final state
                    result.status = RunStatus.COMPLETED
                    result.end_time = datetime.utcnow()
                    
                    # Write final trace file
                    try:
                        await self._write_final_trace(result)
                    except Exception as e:
                        logging.error(f"Failed to write final trace file: {e}")
                        # Don't fail the workflow if trace writing fails
                    
                    return result
                    
                except Exception as e:
                    if attempt < self.settings.max_retries:
                        logging.warning(f"Workflow attempt {attempt + 1} failed, retrying: {e}")
                        await asyncio.sleep(self.settings.retry_backoff_seconds * (2 ** attempt))
                        continue
                    else:
                        raise e
                        
        except Exception as e:
            logging.error(f"Workflow execution failed after {self.settings.max_retries + 1} attempts: {e}")
            state.status = RunStatus.FAILED
            state.errors.append(str(e))
            state.end_time = datetime.utcnow()
            
            # Write error trace file
            try:
                await self._write_final_trace(state)
            except Exception as e:
                logging.error(f"Failed to write error trace file: {e}")
                # Don't fail the workflow if trace writing fails
            
            return state
    
    async def _write_final_trace(self, state: OrchestratorState):
        """Write final trace file with complete execution details"""
        try:
            trace_file = Path("logs/runs") / f"{state.run_id}.json"
            trace_file.parent.mkdir(parents=True, exist_ok=True)
            
            trace_data = {
                "run_id": state.run_id,
                "step_id": state.step_id,
                "active_connection_id": state.active_connection_id,
                "question": state.question,
                "status": state.status.value,
                "current_node": state.current_node,
                "start_time": state.start_time.isoformat() if state.start_time else None,
                "end_time": state.end_time.isoformat() if state.end_time else None,
                "errors": state.errors,
                "node_results": state.node_results,
                "budget": state.budget.model_dump() if state.budget else None,
                "budget_exceeded": state.budget_exceeded,
                "retry_counts": state.retry_counts,
                "step_timings": state.step_timings,
                "metadata": state.metadata,
                "execution_summary": {
                    "total_nodes": len(state.node_results),
                    "successful_nodes": len([k for k, v in state.node_results.items() if v]),
                    "total_execution_time": (state.end_time - state.start_time).total_seconds() if state.end_time and state.start_time else 0,
                    "average_node_time": sum(state.step_timings.values()) / len(state.step_timings) if state.step_timings else 0,
                    "total_retries": sum(state.retry_counts.values())
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            with open(trace_file, 'w') as f:
                json.dump(trace_data, f, indent=2, default=str)
                
            logging.info(f"Final trace file written: {trace_file}")
                
        except Exception as e:
            logging.error(f"Failed to write final trace file: {e}")
    
    @track_node_execution
    async def _route_connection(self, state: OrchestratorState) -> OrchestratorState:
        """Route to the appropriate database connection"""
        
        try:
            # If connection_id is provided, validate it exists
            if state.active_connection_id:
                # Validate connection exists and is accessible
                connections = await self.clients["connection_registry"].call_mcp_tool("list_connections", {
                    "run_envelope": {
                        "run_id": state.run_id,
                        "step_id": state.step_id,
                        "active_connection_id": state.active_connection_id
                    }
                })
                
                # Check if connection exists
                connection_exists = any(
                    conn["connection_id"] == state.active_connection_id 
                    for conn in connections.get("connections", [])
                )
                
                if not connection_exists:
                    state.status = RunStatus.FAILED
                    state.errors.append(f"Connection {state.active_connection_id} not found")
                    return state
                
                state.status = RunStatus.RUNNING
                return state
            else:
                # Auto-routing: score available connections based on query relevance
                connections = await self.clients["connection_registry"].call_mcp_tool("list_connections", {
                    "run_envelope": {
                        "run_id": state.run_id,
                        "step_id": state.step_id,
                        "active_connection_id": "auto"
                    }
                })
                
                if connections.get("connections"):
                    # Simple scoring: prefer connections with recent activity
                    # In a real implementation, you'd score based on query relevance
                    scored_connections = []
                    for conn in connections["connections"]:
                        score = 0
                        # Prefer connections with recent activity
                        if conn.get("updated_at"):
                            try:
                                updated = datetime.fromisoformat(conn["updated_at"].replace('Z', '+00:00'))
                                days_old = (datetime.utcnow() - updated).days
                                if days_old < 7:
                                    score += 10
                                elif days_old < 30:
                                    score += 5
                            except:
                                pass
                        
                        # Prefer certain database types for certain queries
                        db_type = conn.get("database_type", "").lower()
                        if "user" in state.question.lower() and "postgresql" in db_type:
                            score += 5
                        elif "order" in state.question.lower() and "mysql" in db_type:
                            score += 3
                        
                        scored_connections.append((conn, score))
                    
                    # Sort by score and pick the best
                    scored_connections.sort(key=lambda x: x[1], reverse=True)
                    best_connection = scored_connections[0][0]
                    
                    state.active_connection_id = best_connection["connection_id"]
                    state.metadata["auto_routed"] = True
                    state.metadata["routing_score"] = scored_connections[0][1]
                    state.metadata["available_connections"] = len(connections["connections"])
                    
                    logging.info(f"Auto-routed to connection {best_connection['connection_id']} with score {scored_connections[0][1]}")
                    state.status = RunStatus.RUNNING
                else:
                    state.status = RunStatus.FAILED
                    state.errors.append("No database connections available")
                
                return state
                
        except Exception as e:
            state.status = RunStatus.FAILED
            state.errors.append(f"Connection routing failed: {str(e)}")
            return state

    @track_node_execution
    async def _retrieve_tables(self, state: OrchestratorState) -> OrchestratorState:
        """Retrieve relevant tables for the query"""
        
        try:
            if self.settings.stub_mode:
                # Stub mode: return mock data
                mock_result = {
                    "connection_id": state.active_connection_id,
                    "query": state.question,
                    "results": [
                        {
                            "id": "public.users",
                            "score": 0.95,
                            "table": {
                                "table_id": "public.users",
                                "schema_name": "public",
                                "table_name": "users",
                                "full_name": "public.users",
                                "description": "User accounts table"
                            },
                            "relevance_signals": {"name_match": 0.9, "semantic": 0.8}
                        },
                        {
                            "id": "public.orders", 
                            "score": 0.85,
                            "table": {
                                "table_id": "public.orders",
                                "schema_name": "public",
                                "table_name": "orders", 
                                "full_name": "public.orders",
                                "description": "Customer orders table"
                            },
                            "relevance_signals": {"name_match": 0.7, "semantic": 0.8}
                        }
                    ],
                    "total_found": 2,
                    "search_time": "2024-01-01T00:00:00Z"
                }
                state.node_results["table_retriever"] = mock_result
                logging.info("Stub mode: Retrieved 2 mock tables")
            else:
                # Real mode: call the service
                input_data = SearchTablesInput(
                    run_envelope=RunEnvelope(
                        run_id=state.run_id,
                        step_id=state.step_id,
                        active_connection_id=state.active_connection_id
                    ),
                    connection_id=state.active_connection_id,
                    query=state.question,
                    top_k=self.settings.max_tables_per_query
                ).model_dump()
                
                result = await self.clients["table_retriever"].call_mcp_tool("search_tables", input_data)
                state.node_results["table_retriever"] = result
                logging.info(f"Retrieved {len(result.get('results', []))} tables")
            
            # Check budget constraints
            if not self._check_budget(state, "retrieve_tables"):
                state.status = RunStatus.FAILED
                return state
            
        except Exception as e:
            state.status = RunStatus.FAILED
            state.errors.append(f"Table retrieval failed: {str(e)}")
            logging.error(f"Table retrieval failed: {e}")
        
        return state
    
    @track_node_execution
    async def _profile_tables(self, state: OrchestratorState) -> OrchestratorState:
        """Profile the selected tables"""
        
        try:
            if self.settings.stub_mode:
                # Stub mode: return mock profiling data
                mock_result = {
                    "connection_id": state.active_connection_id,
                    "profiles": [
                        {
                            "table_id": "public.users",
                            "sample_size": 500,
                            "null_rates": {"id": 0.0, "email": 0.0, "name": 0.05},
                            "distinct_counts": {"id": 1000, "email": 1000, "name": 950},
                            "date_ranges": {},
                            "numeric_stats": {"id": {"min": 1, "max": 1000, "avg": 500.5}},
                            "categorical_values": {"email": ["user@example.com", "admin@example.com"]},
                            "fingerprints": {"data_quality": "high", "completeness": 0.95},
                            "generated_at": "2024-01-01T00:00:00Z",
                            "schema_revision": "1.0"
                        }
                    ],
                    "profiling_time": "2024-01-01T00:00:00Z",
                    "total_tables_profiled": 1,
                    "versioning": {
                        "api_version": "1.0",
                        "schema_revision": "1.0",
                        "embedding_version": "1.0",
                        "timestamp": "2024-01-01T00:00:00Z"
                    }
                }
                state.node_results["micro_profiler"] = mock_result
                logging.info("Stub mode: Profiled 1 mock table")
            else:
                # Real mode: call the service
                # Get connection handle first
                handle_result = await self.clients["connection_registry"].call_mcp_tool(
                    "get_connection_handle",
                    {
                        "run_envelope": RunEnvelope(
                            run_id=state.run_id,
                            step_id=state.step_id,
                            active_connection_id=state.active_connection_id
                        ).model_dump(),
                        "connection_id": state.active_connection_id
                    }
                )
                
                table_ids = [r["table"]["table_id"] for r in state.node_results["table_retriever"]["results"]]
                
                input_data = ProfileTablesInput(
                    run_envelope=RunEnvelope(
                        run_id=state.run_id,
                        step_id=state.step_id,
                        active_connection_id=state.active_connection_id
                    ),
                    handle_id=handle_result["handle"]["handle_id"],
                    table_ids=table_ids,
                    sample_size=self.settings.default_sample_size
                ).model_dump()
                
                result = await self.clients["micro_profiler"].call_mcp_tool("profile_tables", input_data)
                state.node_results["micro_profiler"] = result
                logging.info(f"Profiled {len(result.get('profiles', []))} tables")
            
            # Check budget constraints
            if not self._check_budget(state, "profile_tables"):
                state.status = RunStatus.FAILED
                return state
            
        except Exception as e:
            state.status = RunStatus.FAILED
            state.errors.append(f"Table profiling failed: {str(e)}")
            logging.error(f"Table profiling failed: {e}")
        
        return state
    
    @track_node_execution
    async def _prune_columns(self, state: OrchestratorState) -> OrchestratorState:
        """Prune columns to keep only relevant ones"""
        
        try:
            # Get schema information
            schema_result = await self.clients["introspect"].call_mcp_tool(
                "introspect_database",
                {
                    "run_envelope": RunEnvelope(
                        run_id=state.run_id,
                        step_id=state.step_id,
                        active_connection_id=state.active_connection_id
                    ).model_dump(),
                    "connection_id": state.active_connection_id
                }
            )
            
            input_data = PruneColumnsInput(
                run_envelope=RunEnvelope(
                    run_id=state.run_id,
                    step_id=state.step_id,
                    active_connection_id=state.active_connection_id
                ),
                connection_id=state.active_connection_id,
                question=state.question,
                selected_tables=state.node_results["table_retriever"]["results"],
                table_profiles=state.node_results["micro_profiler"]["profiles"],
                primary_keys=schema_result["primary_keys"],
                foreign_keys=schema_result["foreign_keys"],
                max_columns_per_table=self.settings.max_columns_per_table
            ).model_dump()
            
            result = await self.clients["column_pruner"].call_mcp_tool("prune_columns", input_data)
            state.node_results["column_pruner"] = result
            
            logging.info(f"Pruned columns for {len(result.get('pruned_tables', []))} tables")
            
        except Exception as e:
            state.status = RunStatus.FAILED
            state.errors.append(f"Column pruning failed: {str(e)}")
            logging.error(f"Column pruning failed: {e}")
        
        return state
    
    @track_node_execution
    async def _build_join_graph(self, state: OrchestratorState) -> OrchestratorState:
        """Build optimal join paths"""
        
        try:
            schema_result = await self.clients["introspect"].call_mcp_tool(
                "introspect_database",
                {
                    "run_envelope": RunEnvelope(
                        run_id=state.run_id,
                        step_id=state.step_id,
                        active_connection_id=state.active_connection_id
                    ).model_dump(),
                    "connection_id": state.active_connection_id
                }
            )
            
            input_data = BuildJoinGraphInput(
                run_envelope=RunEnvelope(
                    run_id=state.run_id,
                    step_id=state.step_id,
                    active_connection_id=state.active_connection_id
                ),
                connection_id=state.active_connection_id,
                selected_tables=state.node_results["table_retriever"]["results"],
                primary_keys=schema_result["primary_keys"],
                foreign_keys=schema_result["foreign_keys"],
                table_profiles=state.node_results["micro_profiler"]["profiles"]
            ).model_dump()
            
            result = await self.clients["join_graph"].call_mcp_tool("build_join_graph", input_data)
            state.node_results["join_graph"] = result
            
            logging.info(f"Built join graph with {len(result.get('join_tree', []))} joins")
            
        except Exception as e:
            state.status = RunStatus.FAILED
            state.errors.append(f"Join graph building failed: {str(e)}")
            logging.error(f"Join graph building failed: {e}")
        
        return state
    
    @track_node_execution
    async def _resolve_metrics(self, state: OrchestratorState) -> OrchestratorState:
        """Resolve business terms to database columns"""
        
        try:
            input_data = ResolveMetricsInput(
                run_envelope=RunEnvelope(
                    run_id=state.run_id,
                    step_id=state.step_id,
                    active_connection_id=state.active_connection_id
                ),
                connection_id=state.active_connection_id,
                business_terms=[state.question],  # Extract terms from question
                pruned_schema=state.node_results["column_pruner"],
                table_profiles=state.node_results["micro_profiler"]["profiles"]
            ).model_dump()
            
            result = await self.clients["metric_resolver"].call_mcp_tool("resolve_metrics", input_data)
            state.node_results["metric_resolver"] = result
            
            logging.info(f"Resolved {len(result.get('metric_bindings', {}))} metrics")
            
        except Exception as e:
            state.status = RunStatus.FAILED
            state.errors.append(f"Metric resolution failed: {str(e)}")
            logging.error(f"Metric resolution failed: {e}")
        
        return state
    
    @track_node_execution
    async def _generate_sql(self, state: OrchestratorState) -> OrchestratorState:
        """Generate SQL from the question and schema"""
        
        try:
            # Get adapter model for this connection
            adapter_model = self.model_registry.get_adapter_for_connection(
                state.active_connection_id
            )
            
            # Get dialect from the connection registry
            dialect_id = await self._get_connection_dialect(state)
            
            input_data = GenerateSQLInput(
                run_envelope=RunEnvelope(
                    run_id=state.run_id,
                    step_id=state.step_id,
                    active_connection_id=state.active_connection_id,
                    budget=state.budget
                ),
                connection_id=state.active_connection_id,
                question=state.question,
                skinny_schema=state.node_results["column_pruner"],
                join_graph=state.node_results["join_graph"],
                metric_bindings=state.node_results["metric_resolver"],
                dialect_id=dialect_id,
                prompt_cap=self.settings.max_prompt_tokens
            ).model_dump()
            
            result = await self.clients["sql_generator"].call_mcp_tool("generate_sql", input_data)
            state.node_results["sql_generator"] = result
            
            # SQL Safety Gate - Check generated SQL before proceeding
            sql = result.get("sql", "")
            is_safe, error_message = validate_sql_safety(sql)
            
            if not is_safe:
                state.status = RunStatus.FAILED
                state.errors.append(f"SQL safety check failed: {error_message}")
                logging.error(f"SQL safety check failed: {error_message}")
                return state
            
            # Check budget constraints
            if not self._check_budget(state, "generate_sql"):
                state.status = RunStatus.FAILED
                return state
            
            logging.info(f"Generated SQL with confidence {result.get('confidence_score', 0)}")
            
        except Exception as e:
            state.status = RunStatus.FAILED
            state.errors.append(f"SQL generation failed: {str(e)}")
            logging.error(f"SQL generation failed: {e}")
        
        return state
    
    @track_node_execution
    async def _validate_sql(self, state: OrchestratorState) -> OrchestratorState:
        """Validate the generated SQL"""
        
        try:
            handle_result = await self.clients["connection_registry"].call_mcp_tool(
                "get_connection_handle",
                {
                    "run_envelope": RunEnvelope(
                        run_id=state.run_id,
                        step_id=state.step_id,
                        active_connection_id=state.active_connection_id
                    ).model_dump(),
                    "connection_id": state.active_connection_id
                }
            )
            
            # Get dialect from the connection registry
            dialect_id = await self._get_connection_dialect(state)
            
            input_data = ValidateSQLInput(
                run_envelope=RunEnvelope(
                    run_id=state.run_id,
                    step_id=state.step_id,
                    active_connection_id=state.active_connection_id,
                    budget=state.budget
                ),
                handle_id=handle_result["handle"]["handle_id"],
                sql=state.node_results["sql_generator"]["sql"],
                dialect_id=dialect_id
            ).model_dump()
            
            result = await self.clients["sql_validator"].call_mcp_tool("validate_sql", input_data)
            state.node_results["sql_validator"] = result
            
            if not result.get("is_valid", False):
                state.status = RunStatus.FAILED
                state.errors.append(f"SQL validation failed: {result.get('errors', [])}")
                return state
            
            logging.info("SQL validation passed")
            
        except Exception as e:
            state.status = RunStatus.FAILED
            state.errors.append(f"SQL validation failed: {str(e)}")
            logging.error(f"SQL validation failed: {e}")
        
        return state
    
    @track_node_execution
    async def _execute_query(self, state: OrchestratorState) -> OrchestratorState:
        """Execute the validated SQL"""
        
        try:
            handle_result = await self.clients["connection_registry"].call_mcp_tool(
                "get_connection_handle",
                {
                    "run_envelope": RunEnvelope(
                        run_id=state.run_id,
                        step_id=state.step_id,
                        active_connection_id=state.active_connection_id
                    ).model_dump(),
                    "connection_id": state.active_connection_id
                }
            )
            
            input_data = ExecuteQueryInput(
                run_envelope=RunEnvelope(
                    run_id=state.run_id,
                    step_id=state.step_id,
                    active_connection_id=state.active_connection_id
                ),
                handle_id=handle_result["handle"]["handle_id"],
                sql=state.node_results["sql_generator"]["sql"],
                row_limit=10000,
                timeout_seconds=self.settings.query_timeout_seconds
            ).model_dump()
            
            result = await self.clients["query_executor"].call_mcp_tool("execute_query", input_data)
            state.node_results["query_executor"] = result
            
            # Check budget constraints
            if not self._check_budget(state, "execute_query"):
                state.status = RunStatus.FAILED
                return state
            
            logging.info(f"Query executed successfully, returned {result.get('row_count', 0)} rows")
            
        except Exception as e:
            state.status = RunStatus.FAILED
            state.errors.append(f"Query execution failed: {str(e)}")
            logging.error(f"Query execution failed: {e}")
        
        return state
    
    @track_node_execution
    async def _explain_results(self, state: OrchestratorState) -> OrchestratorState:
        """Explain the results and decision process"""
        
        try:
            # Get complete model information for this connection
            model_info = self.model_registry.get_model_info_for_connection(
                state.active_connection_id
            )
            
            input_data = ExplainResultsInput(
                run_envelope=RunEnvelope(
                    run_id=state.run_id,
                    step_id=state.step_id,
                    active_connection_id=state.active_connection_id
                ),
                connection_id=state.active_connection_id,
                trace_decisions=state.node_results,
                question=state.question,
                final_sql=state.node_results["sql_generator"]["sql"],
                results_summary=state.node_results["query_executor"]
            ).model_dump()
            
            # Add model information to the input
            input_data["model_info"] = model_info
            
            result = await self.clients["result_explainer"].call_mcp_tool("explain_results", input_data)
            state.node_results["result_explainer"] = result
            
            logging.info("Results explanation generated")
            
        except Exception as e:
            state.status = RunStatus.FAILED
            state.errors.append(f"Result explanation failed: {str(e)}")
            logging.error(f"Result explanation failed: {e}")
        
        return state
    
    async def close(self):
        """Close all service clients"""
        for client in self.clients.values():
            await client.close()

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="AskData API Orchestrator",
    description="Main orchestrator service for natural language database queries",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global settings and workflow
settings = Settings()
model_registry = ModelRegistry(settings.model_registry_path)
workflow = AskDataWorkflow(settings, model_registry)
run_registry = RunRegistry()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup"""
    setup_logging()
    logging.info("AskData API Orchestrator starting up")
    
    # Start cleanup task
    asyncio.create_task(cleanup_old_runs())

async def cleanup_old_runs():
    """Periodically clean up old runs"""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            await run_registry.cleanup_old_runs(max_age_hours=24)
            logging.info("Cleaned up old runs")
        except Exception as e:
            logging.error(f"Cleanup task failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    await workflow.close()
    logging.info("AskData API Orchestrator shutting down")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "api-orchestrator", "timestamp": datetime.utcnow()}

@app.post("/query", response_model=QueryResponse)
async def execute_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Execute a natural language query"""
    start_time = time.time()
    
    # Handle Auto connection routing
    connection_id = request.connection_id
    if connection_id.lower() == "auto":
        connection_id = "Auto"  # Special value for auto-routing
    
    # Create run state with budget from config
    run_id = str(uuid.uuid4())
    budget = Budget(
        max_tables=settings.max_tables_per_query,
        max_columns_per_table=settings.max_columns_per_table,
        max_sample_rows=settings.default_sample_size,
        max_prompt_tokens=settings.max_prompt_tokens,
        max_result_rows=10000,
        query_timeout_seconds=settings.query_timeout_seconds,
        max_retries=settings.max_retries
    )
    
    state = OrchestratorState(
        run_id=run_id,
        step_id=str(uuid.uuid4()),
        active_connection_id=connection_id,
        question=request.question,
        budget=budget,
        metadata={
            "session_id": request.session_id,
            "user_id": request.user_id,
            "tenant_id": request.tenant_id,
            "options": request.options or {},
            "auto_routing": connection_id == "Auto"
        }
    )
    
    # Register the run
    await run_registry.create_run(state)
    
    # Execute workflow in background
    background_tasks.add_task(workflow.execute_workflow, state)
    
    # Return immediate response
    return QueryResponse(
        run_id=run_id,
        status=RunStatus.RUNNING,
        execution_time=time.time() - start_time,
        timeline=[]
    )

@app.get("/query/{run_id}/status")
async def get_query_status(run_id: str):
    """Get the status of a query execution"""
    run = await run_registry.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    # Calculate progress based on completed nodes
    total_nodes = 9  # Total number of workflow nodes
    completed_nodes = len([k for k, v in run.node_results.items() if v])
    progress = completed_nodes / total_nodes if total_nodes > 0 else 0
    
    # Build timeline from node results and timings
    timeline = []
    for node_name, result in run.node_results.items():
        if result:
            node_timing = run.step_timings.get(node_name, 0)
            retry_count = run.retry_counts.get(node_name, 0)
            
            timeline.append({
                "node": node_name,
                "status": "completed",
                "duration": round(node_timing, 2),
                "retries": retry_count,
                "timestamp": run.start_time.isoformat() if run.start_time else None
            })
    
    # Add current running node if any
    if run.current_node and run.status == RunStatus.RUNNING:
        timeline.append({
            "node": run.current_node,
            "status": "running",
            "duration": 0,
            "retries": run.retry_counts.get(run.current_node, 0),
            "timestamp": datetime.utcnow().isoformat()
        })
    
    return {
        "run_id": run_id,
        "status": run.status.value,
        "current_node": run.current_node,
        "progress": round(progress, 2),
        "errors": run.errors,
        "start_time": run.start_time,
        "execution_time": (datetime.utcnow() - run.start_time).total_seconds() if run.start_time else 0,
        "timeline": timeline,
        "budget_exceeded": run.budget_exceeded,
        "total_retries": sum(run.retry_counts.values())
    }

@app.get("/query/{run_id}/results")
async def get_query_results(run_id: str):
    """Get the results of a completed query"""
    run = await run_registry.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    if run.status != RunStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Run not completed. Status: {run.status.value}")
    
    # Extract results from the run
    sql_generator_result = run.node_results.get("sql_generator", {})
    query_executor_result = run.node_results.get("query_executor", {})
    result_explainer_result = run.node_results.get("result_explainer", {})
    
    return {
        "run_id": run_id,
        "status": run.status.value,
        "results": query_executor_result.get("results", []),
        "sql": sql_generator_result.get("sql", ""),
        "explanation": result_explainer_result.get("explanation", {}),
        "execution_time": (run.end_time - run.start_time).total_seconds() if run.end_time and run.start_time else 0,
        "row_count": query_executor_result.get("row_count", 0),
        "confidence_score": sql_generator_result.get("confidence_score", 0)
    }

@app.post("/connections", response_model=CreateConnectionOutput)
async def create_connection(request: CreateConnectionInput):
    """Create a new database connection"""
    try:
        result = await workflow.clients["connection_registry"].call_mcp_tool(
            "create_connection",
            request.model_dump()
        )
        return CreateConnectionOutput(**result)
    except Exception as e:
        logging.error(f"Connection creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/connections")
async def list_connections(tenant_id: Optional[str] = None, user_id: Optional[str] = None):
    """List all database connections"""
    try:
        result = await workflow.clients["connection_registry"].call_mcp_tool(
            "list_connections",
            {
                "run_envelope": RunEnvelope(
                    run_id=str(uuid.uuid4()),
                    step_id=str(uuid.uuid4()),
                    active_connection_id=""
                ).model_dump(),
                "tenant_id": tenant_id,
                "user_id": user_id
            }
        )
        return result
    except Exception as e:
        logging.error(f"Connection listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/query/{run_id}/trace")
async def get_query_trace(run_id: str):
    """Get the complete trace file for a query execution"""
    try:
        trace_file = Path("logs/runs") / f"{run_id}.json"
        if not trace_file.exists():
            # Check if it's a live run first
            run = await run_registry.get_run(run_id)
            if run:
                # Return live state as trace
                return {
                    "run_id": run.run_id,
                    "step_id": run.step_id,
                    "active_connection_id": run.active_connection_id,
                    "question": run.question,
                    "status": run.status.value,
                    "current_node": run.current_node,
                    "start_time": run.start_time.isoformat() if run.start_time else None,
                    "end_time": run.end_time.isoformat() if run.end_time else None,
                    "errors": run.errors,
                    "node_results": run.node_results,
                    "budget": run.budget.model_dump() if run.budget else None,
                    "budget_exceeded": run.budget_exceeded,
                    "retry_counts": run.retry_counts,
                    "step_timings": run.step_timings,
                    "metadata": run.metadata,
                    "timestamp": datetime.utcnow().isoformat(),
                    "note": "Live run state (trace file not yet written)"
                }
            else:
                raise HTTPException(status_code=404, detail="Run not found")
        
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)
        
        return trace_data
        
    except Exception as e:
        logging.error(f"Failed to read trace file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read trace file: {str(e)}")

@app.get("/runs")
async def list_runs(limit: int = 100):
    """List recent query runs"""
    try:
        runs = await run_registry.list_runs(limit)
        return {
            "runs": [
                {
                    "run_id": run.run_id,
                    "status": run.status.value,
                    "question": run.question,
                    "active_connection_id": run.active_connection_id,
                    "start_time": run.start_time,
                    "current_node": run.current_node,
                    "errors": run.errors,
                    "budget_exceeded": run.budget_exceeded,
                    "total_retries": sum(run.retry_counts.values())
                }
                for run in runs
            ],
            "total": len(runs)
        }
    except Exception as e:
        logging.error(f"Failed to list runs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list runs: {str(e)}")

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