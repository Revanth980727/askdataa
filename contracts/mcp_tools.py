"""
Model Context Protocol (MCP) Tool Specifications for AskData

This module defines the standardized MCP tools that all agent services must implement.
Each tool has a fixed name, input/output schema, and behavior contract.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
import uuid
from datetime import datetime

# =============================================================================
# COMMON TYPES AND ENUMS
# =============================================================================

class DatabaseType(str, Enum):
    """Supported database types"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SNOWFLAKE = "snowflake"
    BIGQUERY = "bigquery"
    REDSHIFT = "redshift"
    SQLITE = "sqlite"
    ORACLE = "oracle"
    SQLSERVER = "sqlserver"

class ConnectionStatus(str, Enum):
    """Connection status enumeration"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    FAILED = "failed"
    DISCONNECTED = "disconnected"

class RunStatus(str, Enum):
    """Run execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class NodeStatus(str, Enum):
    """Node execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class FeedbackType(str, Enum):
    """Feedback event types for RLHF"""
    APPROVAL = "approval"
    REJECTION = "rejection"
    EDIT = "edit"

class ErrorCode(str, Enum):
    """Standard error codes for MCP tools"""
    # Connection errors
    CONNECTION_FAILED = "connection_failed"
    INVALID_CREDENTIALS = "invalid_credentials"
    NETWORK_TIMEOUT = "network_timeout"
    
    # Authentication errors
    INVALID_HANDLE = "invalid_handle"
    HANDLE_EXPIRED = "handle_expired"
    INSUFFICIENT_PERMISSIONS = "insufficient_permissions"
    
    # Validation errors
    INVALID_INPUT = "invalid_input"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    INVALID_FORMAT = "invalid_format"
    
    # Execution errors
    QUERY_TIMEOUT = "query_timeout"
    INSUFFICIENT_RESOURCES = "insufficient_resources"
    DATABASE_ERROR = "database_error"
    
    # System errors
    SERVICE_UNAVAILABLE = "service_unavailable"
    INTERNAL_ERROR = "internal_error"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"

class ToolVersion(str, Enum):
    """MCP tool versions for backward compatibility"""
    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"

class Versioning(BaseModel):
    """Version information for data consistency"""
    api_version: str = Field("1.0", description="API version")
    schema_revision: str = Field(..., description="Database schema revision")
    embedding_version: str = Field(..., description="Embedding model version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# =============================================================================
# CORE DATA MODELS
# =============================================================================

class Budget(BaseModel):
    """Budget constraints for query execution"""
    max_tables: int = Field(20, description="Maximum tables to consider")
    max_columns_per_table: int = Field(15, description="Maximum columns per table")
    max_sample_rows: int = Field(1000, description="Maximum rows to sample")
    max_prompt_tokens: int = Field(8000, description="Maximum prompt tokens")
    max_result_rows: int = Field(10000, description="Maximum result rows")
    query_timeout_seconds: int = Field(300, description="Query timeout in seconds")
    max_retries: int = Field(3, description="Maximum retries per step")
    reranker_enabled: bool = Field(True, description="Enable reranker for search")

class RunEnvelope(BaseModel):
    """Standard envelope for all operations"""
    run_id: str = Field(..., description="Unique run identifier")
    step_id: str = Field(..., description="Current step identifier")
    active_connection_id: str = Field(..., description="Active database connection ID")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    budget: Budget = Field(default_factory=Budget, description="Execution budget")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DatabaseConnection(BaseModel):
    """Database connection configuration"""
    connection_id: str = Field(..., description="Unique connection identifier")
    name: str = Field(..., description="Connection display name")
    database_type: DatabaseType = Field(..., description="Database type")
    host: str = Field(..., description="Database host")
    port: int = Field(..., description="Database port")
    database: str = Field(..., description="Database name")
    schema: Optional[str] = Field(None, description="Default schema")
    username: str = Field(..., description="Database username")
    password: Optional[str] = Field(None, description="Database password")
    ssl_mode: Optional[str] = Field(None, description="SSL mode")
    additional_params: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class ConnectionHandle(BaseModel):
    """Short-lived connection handle for secure access"""
    handle_id: str = Field(..., description="Unique handle identifier")
    connection_id: str = Field(..., description="Associated connection ID")
    expires_at: datetime = Field(..., description="Handle expiration time")
    permissions: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class TableMetadata(BaseModel):
    """Table metadata information"""
    table_id: str = Field(..., description="Unique table identifier")
    schema_name: str = Field(..., description="Schema name")
    table_name: str = Field(..., description="Table name")
    full_name: str = Field(..., description="Full table name (schema.table)")
    row_count: Optional[int] = Field(None, description="Approximate row count")
    last_modified: Optional[datetime] = Field(None, description="Last modification time")
    description: Optional[str] = Field(None, description="Table description")
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ColumnMetadata(BaseModel):
    """Column metadata information"""
    column_id: str = Field(..., description="Unique column identifier")
    table_id: str = Field(..., description="Parent table ID")
    column_name: str = Field(..., description="Column name")
    data_type: str = Field(..., description="Database data type")
    is_nullable: bool = Field(..., description="Whether column can be null")
    is_primary_key: bool = Field(False, description="Whether column is primary key")
    is_foreign_key: bool = Field(False, description="Whether column is foreign key")
    referenced_table: Optional[str] = Field(None, description="Referenced table for FK")
    referenced_column: Optional[str] = Field(None, description="Referenced column for FK")
    default_value: Optional[str] = Field(None, description="Default value")
    description: Optional[str] = Field(None, description="Column description")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TableProfile(BaseModel):
    """Table profiling results"""
    table_id: str = Field(..., description="Table identifier")
    sample_size: int = Field(..., description="Number of rows sampled")
    null_rates: Dict[str, float] = Field(..., description="Null rate per column")
    distinct_counts: Dict[str, int] = Field(..., description="Distinct count per column")
    date_ranges: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    numeric_stats: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    categorical_values: Dict[str, List[str]] = Field(default_factory=dict)
    fingerprints: Dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    schema_revision: str = Field(..., description="Schema revision when profiled")

class SearchResult(BaseModel):
    """Generic search result"""
    id: str = Field(..., description="Result identifier")
    score: float = Field(..., description="Relevance score")
    metadata: Dict[str, Any] = Field(..., description="Result metadata")
    explanation: Optional[str] = Field(None, description="Score explanation")

class TableSearchResult(SearchResult):
    """Table search result"""
    table: TableMetadata = Field(..., description="Table metadata")
    relevance_signals: Dict[str, float] = Field(..., description="Individual signal scores")

class ColumnSearchResult(SearchResult):
    """Column search result"""
    column: ColumnMetadata = Field(..., description="Column metadata")
    table: TableMetadata = Field(..., description="Parent table metadata")
    relevance_signals: Dict[str, float] = Field(..., description="Individual signal scores")

# =============================================================================
# MCP TOOL DEFINITIONS
# =============================================================================

class MCPTool(BaseModel):
    """Base MCP tool definition"""
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    version: ToolVersion = Field(ToolVersion.V1_0, description="Tool version")
    input_schema: Dict[str, Any] = Field(..., description="Input JSON schema")
    output_schema: Dict[str, Any] = Field(..., description="Output JSON schema")
    error_codes: List[ErrorCode] = Field(default_factory=list, description="Possible error codes")
    deprecated: bool = Field(False, description="Whether tool is deprecated")
    replacement_tool: Optional[str] = Field(None, description="Replacement tool if deprecated")

# =============================================================================
# CONNECTION REGISTRY TOOLS
# =============================================================================

class CreateConnectionInput(BaseModel):
    """Input for creating a new database connection"""
    run_envelope: RunEnvelope
    connection: DatabaseConnection
    schema_revision: Optional[str] = Field(None, description="Schema revision for consistency")

class CreateConnectionOutput(BaseModel):
    """Output from creating a database connection"""
    connection_id: str
    status: ConnectionStatus
    message: str
    handle_id: Optional[str] = None
    versioning: Versioning = Field(..., description="Version information for consistency")

class GetConnectionHandleInput(BaseModel):
    """Input for getting a connection handle"""
    run_envelope: RunEnvelope
    connection_id: str
    schema_revision: Optional[str] = Field(None, description="Schema revision for consistency")

class GetConnectionHandleOutput(BaseModel):
    """Output from getting a connection handle"""
    handle: ConnectionHandle
    status: str
    versioning: Versioning = Field(..., description="Version information for consistency")

class ListConnectionsInput(BaseModel):
    """Input for listing connections"""
    run_envelope: RunEnvelope
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    schema_revision: Optional[str] = Field(None, description="Schema revision for consistency")

class ListConnectionsOutput(BaseModel):
    """Output from listing connections"""
    connections: List[DatabaseConnection]
    total_count: int
    versioning: Versioning = Field(..., description="Version information for consistency")

class DeleteConnectionInput(BaseModel):
    """Input for deleting a connection"""
    run_envelope: RunEnvelope
    connection_id: str
    schema_revision: Optional[str] = Field(None, description="Schema revision for consistency")

class DeleteConnectionOutput(BaseModel):
    """Output from deleting a connection"""
    status: str
    message: str
    purged_artifacts: List[str]
    offboard_report: Optional[Dict[str, Any]] = None
    versioning: Versioning = Field(..., description="Version information for consistency")

class OffboardConnectionInput(BaseModel):
    """Input for offboarding a connection with full purge"""
    run_envelope: RunEnvelope
    connection_id: str
    schema_revision: Optional[str] = Field(None, description="Schema revision for consistency")

class OffboardConnectionOutput(BaseModel):
    """Output from offboarding a connection"""
    connection_id: str
    status: str
    message: str
    offboard_report: Dict[str, Any]
    artifacts_purged: List[str]
    versioning: Versioning = Field(..., description="Version information for consistency")

# =============================================================================
# INTROSPECTION TOOLS
# =============================================================================

class IntrospectDatabaseInput(BaseModel):
    """Input for database introspection"""
    run_envelope: RunEnvelope
    connection_id: str
    schemas: Optional[List[str]] = None
    include_row_counts: bool = True
    include_descriptions: bool = True
    schema_revision: Optional[str] = Field(None, description="Schema revision for consistency")

class IntrospectDatabaseOutput(BaseModel):
    """Output from database introspection"""
    connection_id: str
    tables: List[TableMetadata]
    columns: List[ColumnMetadata]
    primary_keys: List[Dict[str, str]]
    foreign_keys: List[Dict[str, str]]
    schema_revision: str
    introspection_time: datetime
    total_tables: int
    total_columns: int
    versioning: Versioning

# =============================================================================
# INDEXING TOOLS
# =============================================================================

class IndexConnectionInput(BaseModel):
    """Input for indexing a connection"""
    run_envelope: RunEnvelope
    connection_id: str
    tables: List[TableMetadata]
    columns: List[ColumnMetadata]
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    schema_revision: str = Field(..., description="Schema revision for drift tracking")
    embedding_version: str = Field(..., description="Embedding model version for consistency")
    force_reindex: bool = Field(False, description="Force reindexing even if version exists")

class IndexConnectionOutput(BaseModel):
    """Output from indexing a connection"""
    connection_id: str
    status: str
    message: str
    tables_indexed: int
    columns_indexed: int
    collections_created: List[str]
    embedding_version: str
    indexing_time: datetime
    versioning: Optional[Versioning] = None

class SearchTablesInput(BaseModel):
    """Input for table search"""
    run_envelope: RunEnvelope
    connection_id: str
    query: str
    top_k: int = 10
    filters: Optional[Dict[str, Any]] = None
    schema_revision: Optional[str] = Field(None, description="Schema revision for consistency")
    embedding_version: Optional[str] = Field(None, description="Embedding model version for consistency")

class SearchTablesOutput(BaseModel):
    """Output from table search"""
    connection_id: str
    query: str
    results: List[TableSearchResult]
    total_found: int
    search_time: datetime
    search_method: Optional[str] = None
    semantic_weight: Optional[float] = None
    lexical_weight: Optional[float] = None
    versioning: Versioning = Field(..., description="Version information for consistency")

class SearchColumnsInput(BaseModel):
    """Input for column search"""
    run_envelope: RunEnvelope
    connection_id: str
    table_ids: List[str]
    query: str
    top_k: int = 15
    filters: Optional[Dict[str, Any]] = None
    schema_revision: Optional[str] = Field(None, description="Schema revision for consistency")
    embedding_version: Optional[str] = Field(None, description="Embedding model version for consistency")

class SearchColumnsOutput(BaseModel):
    """Output from column search"""
    connection_id: str
    query: str
    results: List[ColumnSearchResult]
    total_found: int
    search_time: datetime
    search_method: Optional[str] = None
    semantic_weight: Optional[float] = None
    lexical_weight: Optional[float] = None
    versioning: Versioning = Field(..., description="Version information for consistency")

class DeleteConnectionIndexInput(BaseModel):
    """Input for deleting connection index"""
    run_envelope: RunEnvelope
    connection_id: str

class DeleteConnectionIndexOutput(BaseModel):
    """Output from deleting connection index"""
    connection_id: str
    status: str
    message: str
    versioning: Versioning = Field(..., description="Version information for consistency")

# =============================================================================
# PROFILING TOOLS
# =============================================================================

class ProfileTablesInput(BaseModel):
    """Input for table profiling"""
    run_envelope: RunEnvelope
    handle_id: str
    table_ids: List[str]
    sample_size: int = 500
    include_fingerprints: bool = True
    schema_revision: Optional[str] = Field(None, description="Schema revision for consistency")

class ProfileTablesOutput(BaseModel):
    """Output from table profiling"""
    connection_id: str
    profiles: List[TableProfile]
    profiling_time: datetime
    total_tables_profiled: int
    versioning: Versioning

class GetTableProfileInput(BaseModel):
    """Input for getting table profile"""
    run_envelope: RunEnvelope
    connection_id: str
    table_id: str
    schema_revision: Optional[str] = Field(None, description="Schema revision for consistency")

class GetTableProfileOutput(BaseModel):
    """Output from getting table profile"""
    profile: Optional[TableProfile]
    status: str
    versioning: Versioning = Field(..., description="Version information for consistency")

# =============================================================================
# COLUMN PRUNING TOOLS
# =============================================================================

class PruneColumnsInput(BaseModel):
    """Input for column pruning"""
    run_envelope: RunEnvelope
    connection_id: str
    question: str
    selected_tables: List[TableMetadata]
    table_profiles: List[TableProfile]
    primary_keys: List[Dict[str, str]]
    foreign_keys: List[Dict[str, str]]
    max_columns_per_table: int = 15
    schema_revision: Optional[str] = Field(None, description="Schema revision for consistency")

class PruneColumnsOutput(BaseModel):
    """Output from column pruning"""
    connection_id: str
    pruned_tables: List[Dict[str, Any]]
    kept_columns: Dict[str, List[str]]
    column_scores: Dict[str, Dict[str, float]]
    pruning_reasons: Dict[str, List[str]]
    versioning: Versioning = Field(..., description="Version information for consistency")

# =============================================================================
# JOIN GRAPH TOOLS
# =============================================================================

class BuildJoinGraphInput(BaseModel):
    """Input for building join graph"""
    run_envelope: RunEnvelope
    connection_id: str
    selected_tables: List[TableMetadata]
    primary_keys: List[Dict[str, str]]
    foreign_keys: List[Dict[str, str]]
    table_profiles: List[TableProfile]
    schema_revision: Optional[str] = Field(None, description="Schema revision for consistency")

class BuildJoinGraphOutput(BaseModel):
    """Output from building join graph"""
    connection_id: str
    join_tree: List[Dict[str, Any]]
    join_keys: Dict[str, Dict[str, str]]
    fan_out_warnings: List[str]
    join_optimization: Dict[str, Any]
    versioning: Versioning = Field(..., description="Version information for consistency")

# =============================================================================
# METRIC RESOLUTION TOOLS
# =============================================================================

class ResolveMetricsInput(BaseModel):
    """Input for metric resolution"""
    run_envelope: RunEnvelope
    connection_id: str
    business_terms: List[str]
    pruned_schema: Dict[str, Any]
    table_profiles: List[TableProfile]
    schema_revision: Optional[str] = Field(None, description="Schema revision for consistency")

class ResolveMetricsOutput(BaseModel):
    """Output from metric resolution"""
    connection_id: str
    metric_bindings: Dict[str, Dict[str, Any]]
    filter_bindings: Dict[str, Dict[str, Any]]
    confidence_scores: Dict[str, float]
    unresolved_terms: List[str]
    versioning: Versioning = Field(..., description="Version information for consistency")

# =============================================================================
# SQL GENERATION TOOLS
# =============================================================================

class GenerateSQLInput(BaseModel):
    """Input for SQL generation"""
    run_envelope: RunEnvelope
    connection_id: str
    question: str
    skinny_schema: Dict[str, Any]
    join_graph: Dict[str, Any]
    metric_bindings: Dict[str, Any]
    dialect_id: str
    prompt_cap: int = 8000
    schema_revision: Optional[str] = Field(None, description="Schema revision for consistency")

class GenerateSQLOutput(BaseModel):
    """Output from SQL generation"""
    connection_id: str
    sql: str
    rationale: str
    token_usage: Dict[str, int]
    generation_time: float
    model_used: str
    confidence_score: float
    versioning: Versioning = Field(..., description="Version information for consistency")

# =============================================================================
# SQL VALIDATION TOOLS
# =============================================================================

class ValidateSQLInput(BaseModel):
    """Input for SQL validation"""
    run_envelope: RunEnvelope
    handle_id: str
    sql: str
    dialect_id: str
    schema_revision: Optional[str] = Field(None, description="Schema revision for consistency")

class ValidateSQLOutput(BaseModel):
    """Output from SQL validation"""
    connection_id: str
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    repaired_sql: Optional[str] = None
    validation_time: float
    versioning: Versioning = Field(..., description="Version information for consistency")

# =============================================================================
# QUERY EXECUTION TOOLS
# =============================================================================

class ExecuteQueryInput(BaseModel):
    """Input for query execution"""
    run_envelope: RunEnvelope
    handle_id: str
    sql: str
    row_limit: int = 10000
    timeout_seconds: int = 300
    schema_revision: Optional[str] = Field(None, description="Schema revision for consistency")

class ExecuteQueryOutput(BaseModel):
    """Output from query execution"""
    connection_id: str
    results: List[Dict[str, Any]]
    row_count: int
    execution_time: float
    cost_hints: Dict[str, Any]
    is_partial: bool
    error_message: Optional[str] = None
    versioning: Versioning = Field(..., description="Version information for consistency")

# =============================================================================
# RESULT EXPLANATION TOOLS
# =============================================================================

class ExplainResultsInput(BaseModel):
    """Input for result explanation"""
    run_envelope: RunEnvelope
    connection_id: str
    trace_decisions: Dict[str, Any]
    question: str
    final_sql: str
    results_summary: Dict[str, Any]
    schema_revision: Optional[str] = Field(None, description="Schema revision for consistency")

class ExplainResultsOutput(BaseModel):
    """Output from result explanation"""
    connection_id: str
    explanation: Dict[str, Any]
    decision_summary: List[str]
    tables_used: List[str]
    columns_kept: Dict[str, List[str]]
    joins_applied: List[Dict[str, Any]]
    filters_applied: List[Dict[str, Any]]
    metric_mappings: Dict[str, Any]
    model_used: str
    confidence_score: float
    versioning: Versioning = Field(..., description="Version information for consistency")


class CaptureFeedbackInput(RunEnvelope):
    """Input for capturing user feedback"""
    event_type: FeedbackType = Field(..., description="Feedback event type")
    prompt: str = Field(..., description="Original prompt or query")
    response: str = Field(..., description="Model response")
    edited_response: Optional[str] = Field(None, description="User-edited response")
    comment: Optional[str] = Field(None, description="Additional comments")


class CaptureFeedbackOutput(BaseModel):
    """Output for feedback capture"""
    status: str = Field(..., description="Operation status")


class PromoteAdapterInput(BaseModel):
    """Input for training and promoting adapter from feedback"""
    connection_id: str = Field(..., description="Connection identifier")
    description: Optional[str] = Field(None, description="Description for the adapter")


class PromoteAdapterOutput(BaseModel):
    """Output for adapter promotion"""
    adapter_model: str = Field(..., description="Registered adapter model identifier")
    message: str = Field(..., description="Result message")

# =============================================================================
# TOOL REGISTRY
# =============================================================================

MCP_TOOLS = {
    # Connection Registry
    "create_connection": MCPTool(
        name="create_connection",
        description="Create a new database connection",
        input_schema=CreateConnectionInput.model_json_schema(),
        output_schema=CreateConnectionOutput.model_json_schema(),
        error_codes=[
            ErrorCode.INVALID_INPUT,
            ErrorCode.CONNECTION_FAILED,
            ErrorCode.INVALID_CREDENTIALS,
            ErrorCode.NETWORK_TIMEOUT,
            ErrorCode.INTERNAL_ERROR
        ]
    ),
    "get_connection_handle": MCPTool(
        name="get_connection_handle",
        description="Get a short-lived handle for database access",
        input_schema=GetConnectionHandleInput.model_json_schema(),
        output_schema=GetConnectionHandleOutput.model_json_schema(),
        error_codes=[
            ErrorCode.INVALID_INPUT,
            ErrorCode.CONNECTION_FAILED,
            ErrorCode.INVALID_CREDENTIALS,
            ErrorCode.INSUFFICIENT_PERMISSIONS,
            ErrorCode.INTERNAL_ERROR
        ]
    ),
    "list_connections": MCPTool(
        name="list_connections",
        description="List all database connections",
        input_schema=ListConnectionsInput.model_json_schema(),
        output_schema=ListConnectionsOutput.model_json_schema(),
        error_codes=[
            ErrorCode.INVALID_INPUT,
            ErrorCode.INSUFFICIENT_PERMISSIONS,
            ErrorCode.INTERNAL_ERROR
        ]
    ),
    "delete_connection": MCPTool(
        name="delete_connection",
        description="Delete a database connection and purge all artifacts",
        input_schema=DeleteConnectionInput.model_json_schema(),
        output_schema=DeleteConnectionOutput.model_json_schema(),
        error_codes=[
            ErrorCode.CONNECTION_FAILED,
            ErrorCode.INVALID_HANDLE,
            ErrorCode.INSUFFICIENT_PERMISSIONS,
            ErrorCode.INTERNAL_ERROR
        ]
    ),
    "offboard_connection": MCPTool(
        name="offboard_connection",
        description="Offboard a connection with complete data purge and audit trail",
        input_schema=OffboardConnectionInput.model_json_schema(),
        output_schema=OffboardConnectionOutput.model_json_schema(),
        error_codes=[
            ErrorCode.CONNECTION_FAILED,
            ErrorCode.INVALID_HANDLE,
            ErrorCode.INSUFFICIENT_PERMISSIONS,
            ErrorCode.INTERNAL_ERROR
        ]
    ),
    
    # Introspection
    "introspect_database": MCPTool(
        name="introspect_database",
        description="Introspect database schema and metadata",
        input_schema=IntrospectDatabaseInput.model_json_schema(),
        output_schema=IntrospectDatabaseOutput.model_json_schema(),
        error_codes=[
            ErrorCode.CONNECTION_FAILED,
            ErrorCode.INVALID_HANDLE,
            ErrorCode.NETWORK_TIMEOUT,
            ErrorCode.DATABASE_ERROR,
            ErrorCode.INSUFFICIENT_PERMISSIONS,
            ErrorCode.INTERNAL_ERROR
        ]
    ),
    
    # Indexing
    "index_connection": MCPTool(
        name="index_connection",
        description="Index connection tables and columns for semantic search",
        input_schema=IndexConnectionInput.model_json_schema(),
        output_schema=IndexConnectionOutput.model_json_schema(),
        error_codes=[
            ErrorCode.INVALID_INPUT,
            ErrorCode.INSUFFICIENT_RESOURCES,
            ErrorCode.SERVICE_UNAVAILABLE,
            ErrorCode.INTERNAL_ERROR
        ]
    ),
    "search_tables": MCPTool(
        name="search_tables",
        description="Search for relevant tables using semantic similarity",
        input_schema=SearchTablesInput.model_json_schema(),
        output_schema=SearchTablesOutput.model_json_schema(),
        error_codes=[
            ErrorCode.INVALID_INPUT,
            ErrorCode.SERVICE_UNAVAILABLE,
            ErrorCode.INTERNAL_ERROR
        ]
    ),
    "search_columns": MCPTool(
        name="search_columns",
        description="Search for relevant columns within specified tables",
        input_schema=SearchColumnsInput.model_json_schema(),
        output_schema=SearchColumnsOutput.model_json_schema(),
        error_codes=[
            ErrorCode.INVALID_INPUT,
            ErrorCode.SERVICE_UNAVAILABLE,
            ErrorCode.INTERNAL_ERROR
        ]
    ),
    "delete_connection_index": MCPTool(
        name="delete_connection_index",
        description="Delete all vector embeddings for a connection",
        input_schema=DeleteConnectionIndexInput.model_json_schema(),
        output_schema=DeleteConnectionIndexOutput.model_json_schema(),
        error_codes=[
            ErrorCode.INVALID_INPUT,
            ErrorCode.SERVICE_UNAVAILABLE,
            ErrorCode.INTERNAL_ERROR
        ]
    ),
    
    # Profiling
    "profile_tables": MCPTool(
        name="profile_tables",
        description="Profile tables to understand data characteristics",
        input_schema=ProfileTablesInput.model_json_schema(),
        output_schema=ProfileTablesOutput.model_json_schema(),
        error_codes=[
            ErrorCode.INVALID_INPUT,
            ErrorCode.INVALID_HANDLE,
            ErrorCode.QUERY_TIMEOUT,
            ErrorCode.INSUFFICIENT_RESOURCES,
            ErrorCode.INTERNAL_ERROR
        ]
    ),
    "get_table_profile": MCPTool(
        name="get_table_profile",
        description="Get cached table profile if available",
        input_schema=GetTableProfileInput.model_json_schema(),
        output_schema=GetTableProfileOutput.model_json_schema(),
        error_codes=[
            ErrorCode.INVALID_INPUT,
            ErrorCode.INVALID_HANDLE,
            ErrorCode.INTERNAL_ERROR
        ]
    ),
    
    # Column Pruning
    "prune_columns": MCPTool(
        name="prune_columns",
        description="Intelligently select relevant columns for the query",
        input_schema=PruneColumnsInput.model_json_schema(),
        output_schema=PruneColumnsOutput.model_json_schema(),
        error_codes=[
            ErrorCode.INVALID_INPUT,
            ErrorCode.INTERNAL_ERROR
        ]
    ),
    
    # Join Graph
    "build_join_graph": MCPTool(
        name="build_join_graph",
        description="Build optimal join paths between selected tables",
        input_schema=BuildJoinGraphInput.model_json_schema(),
        output_schema=BuildJoinGraphOutput.model_json_schema(),
        error_codes=[
            ErrorCode.INVALID_INPUT,
            ErrorCode.INTERNAL_ERROR
        ]
    ),
    
    # Metric Resolution
    "resolve_metrics": MCPTool(
        name="resolve_metrics",
        description="Map business terms to database columns and metrics",
        input_schema=ResolveMetricsInput.model_json_schema(),
        output_schema=ResolveMetricsOutput.model_json_schema(),
        error_codes=[
            ErrorCode.INVALID_INPUT,
            ErrorCode.INTERNAL_ERROR
        ]
    ),
    
    # SQL Generation
    "generate_sql": MCPTool(
        name="generate_sql",
        description="Generate SQL from natural language question",
        input_schema=GenerateSQLInput.model_json_schema(),
        output_schema=GenerateSQLOutput.model_json_schema(),
        error_codes=[
            ErrorCode.INVALID_INPUT,
            ErrorCode.INSUFFICIENT_RESOURCES,
            ErrorCode.RATE_LIMIT_EXCEEDED,
            ErrorCode.INTERNAL_ERROR
        ]
    ),
    
    # SQL Validation
    "validate_sql": MCPTool(
        name="validate_sql",
        description="Validate and optionally repair generated SQL",
        input_schema=ValidateSQLInput.model_json_schema(),
        output_schema=ValidateSQLOutput.model_json_schema(),
        error_codes=[
            ErrorCode.INVALID_INPUT,
            ErrorCode.INVALID_FORMAT,
            ErrorCode.INTERNAL_ERROR
        ]
    ),
    
    # Query Execution
    "execute_query": MCPTool(
        name="execute_query",
        description="Execute SQL query and return results",
        input_schema=ExecuteQueryInput.model_json_schema(),
        output_schema=ExecuteQueryOutput.model_json_schema(),
        error_codes=[
            ErrorCode.INVALID_INPUT,
            ErrorCode.INVALID_HANDLE,
            ErrorCode.QUERY_TIMEOUT,
            ErrorCode.INSUFFICIENT_RESOURCES,
            ErrorCode.DATABASE_ERROR,
            ErrorCode.INTERNAL_ERROR
        ]
    ),
    
    # Result Explanation
    "explain_results": MCPTool(
        name="explain_results",
        description="Explain query results and decision process",
        input_schema=ExplainResultsInput.model_json_schema(),
        output_schema=ExplainResultsOutput.model_json_schema(),
        error_codes=[
            ErrorCode.INVALID_INPUT,
            ErrorCode.INTERNAL_ERROR
        ]
    ),

    # RLHF Feedback Capture
    "capture_feedback": MCPTool(
        name="capture_feedback",
        description="Capture user feedback events for RLHF",
        input_schema=CaptureFeedbackInput.model_json_schema(),
        output_schema=CaptureFeedbackOutput.model_json_schema(),
        error_codes=[
            ErrorCode.INVALID_INPUT,
            ErrorCode.INTERNAL_ERROR
        ]
    ),

    # Adapter Promotion
    "promote_adapter": MCPTool(
        name="promote_adapter",
        description="Train and promote adapters from feedback",
        input_schema=PromoteAdapterInput.model_json_schema(),
        output_schema=PromoteAdapterOutput.model_json_schema(),
        error_codes=[
            ErrorCode.INVALID_INPUT,
            ErrorCode.INTERNAL_ERROR
        ]
    ),
}