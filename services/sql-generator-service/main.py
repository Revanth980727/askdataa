"""
SQL Generator Service

This service converts natural language queries and resolved metrics into executable
SQL statements using LLM-based generation and template-based approaches.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SQL Generator Service", version="1.0.0")

# =============================================================================
# Data Models
# =============================================================================

class SQLDialect(str, Enum):
    """Supported SQL dialects"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SNOWFLAKE = "snowflake"
    BIGQUERY = "bigquery"
    REDSHIFT = "redshift"
    GENERIC = "generic"

class QueryType(str, Enum):
    """Types of SQL queries"""
    SELECT = "select"
    COUNT = "count"
    AGGREGATE = "aggregate"
    TIME_SERIES = "time_series"
    COMPARISON = "comparison"
    TREND = "trend"
    DISTRIBUTION = "distribution"

class JoinType(str, Enum):
    """Types of SQL joins"""
    INNER = "INNER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    FULL = "FULL"
    CROSS = "CROSS"

class AggregationType(str, Enum):
    """Types of aggregations"""
    COUNT = "COUNT"
    SUM = "SUM"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"
    PERCENTILE = "PERCENTILE"
    CUSTOM = "CUSTOM"

class TimeGranularity(str, Enum):
    """Time granularities for time series queries"""
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"

class ColumnReference(BaseModel):
    """Reference to a database column"""
    table_name: str
    column_name: str
    schema_name: str = "public"
    alias: Optional[str] = None
    data_type: Optional[str] = None
    is_aggregated: bool = False
    aggregation_function: Optional[str] = None

class JoinCondition(BaseModel):
    """SQL join condition"""
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    join_type: JoinType = JoinType.INNER
    condition: Optional[str] = None

class WhereCondition(BaseModel):
    """SQL WHERE condition"""
    column: ColumnReference
    operator: str  # =, !=, >, <, >=, <=, LIKE, IN, etc.
    value: Any
    logical_operator: Optional[str] = None  # AND, OR
    parentheses: bool = False

class OrderByClause(BaseModel):
    """SQL ORDER BY clause"""
    column: ColumnReference
    direction: str = "ASC"  # ASC, DESC
    nulls_position: Optional[str] = None  # NULLS FIRST, NULLS LAST

class GroupByClause(BaseModel):
    """SQL GROUP BY clause"""
    columns: List[ColumnReference]
    having_conditions: List[WhereCondition] = []

class GeneratedSQL(BaseModel):
    """Generated SQL statement"""
    sql: str
    query_type: QueryType
    dialect: SQLDialect
    tables: List[str] = []
    columns: List[ColumnReference] = []
    joins: List[JoinCondition] = []
    where_conditions: List[WhereCondition] = []
    group_by: Optional[GroupByClause] = None
    order_by: List[OrderByClause] = []
    limit: Optional[int] = None
    offset: Optional[int] = None
    confidence: float = 0.0
    explanation: str = ""
    warnings: List[str] = []
    estimated_cost: Optional[float] = None

class SQLGenerationRequest(BaseModel):
    """Request to generate SQL from natural language"""
    query: str
    connection_id: str
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    resolved_metrics: List[Dict[str, Any]] = []
    table_metadata: List[Dict[str, Any]] = []
    join_graph: Optional[Dict[str, Any]] = None
    dialect: SQLDialect = SQLDialect.POSTGRESQL
    options: Dict[str, Any] = {}

class SQLGenerationResponse(BaseModel):
    """Response containing generated SQL"""
    success: bool
    query: str
    generated_sql: Optional[GeneratedSQL] = None
    alternatives: List[GeneratedSQL] = []
    confidence: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}

class SQLTemplateRequest(BaseModel):
    """Request to generate SQL from a template"""
    template_name: str
    parameters: Dict[str, Any]
    connection_id: str
    dialect: SQLDialect = SQLDialect.POSTGRESQL
    options: Dict[str, Any] = {}

class SQLTemplateResponse(BaseModel):
    """Response containing templated SQL"""
    success: bool
    template_name: str
    generated_sql: Optional[GeneratedSQL] = None
    error: Optional[str] = None

# =============================================================================
# Core Components
# =============================================================================

class QueryAnalyzer:
    """Analyzes natural language queries to determine intent and structure"""
    
    def __init__(self):
        self.query_patterns = self._build_query_patterns()
    
    def _build_query_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build patterns for query analysis"""
        return {
            "count": [
                {"pattern": r"\b(count|how many|total number|number of)\b", "confidence": 0.9},
                {"pattern": r"\b(users|customers|orders|products)\b", "confidence": 0.8}
            ],
            "aggregate": [
                {"pattern": r"\b(sum|total|average|avg|mean|median|min|max)\b", "confidence": 0.9},
                {"pattern": r"\b(revenue|sales|amount|cost|price)\b", "confidence": 0.8}
            ],
            "time_series": [
                {"pattern": r"\b(over time|trend|monthly|yearly|daily|weekly)\b", "confidence": 0.9},
                {"pattern": r"\b(last|past|previous|this|next)\b", "confidence": 0.8}
            ],
            "comparison": [
                {"pattern": r"\b(compare|vs|versus|difference|higher|lower)\b", "confidence": 0.9},
                {"pattern": r"\b(between|among|across|by)\b", "confidence": 0.8}
            ]
        }
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze a natural language query"""
        analysis = {
            "query_type": None,
            "confidence": 0.0,
            "entities": [],
            "time_references": [],
            "comparison_indicators": [],
            "aggregation_indicators": []
        }
        
        query_lower = query.lower()
        
        # Determine query type
        for query_type, patterns in self.query_patterns.items():
            for pattern_info in patterns:
                if re.search(pattern_info["pattern"], query_lower):
                    analysis["query_type"] = query_type
                    analysis["confidence"] = max(analysis["confidence"], pattern_info["confidence"])
                    break
        
        # Extract time references
        time_patterns = [
            r"\b(last|past|previous)\s+(\d+)\s+(day|week|month|year)s?\b",
            r"\b(this|next)\s+(day|week|month|quarter|year)\b",
            r"\b(\d{4})-(\d{2})-(\d{2})\b",  # YYYY-MM-DD
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b"
        ]
        
        for pattern in time_patterns:
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                analysis["time_references"].append({
                    "text": match.group(),
                    "span": match.span()
                })
        
        # Extract comparison indicators
        comparison_patterns = [
            r"\b(compare|vs|versus|difference|higher|lower|better|worse)\b",
            r"\b(between|among|across|by)\b"
        ]
        
        for pattern in comparison_patterns:
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                analysis["comparison_indicators"].append({
                    "text": match.group(),
                    "span": match.span()
                })
        
        # Extract aggregation indicators
        aggregation_patterns = [
            r"\b(sum|total|average|avg|mean|median|min|max|count)\b",
            r"\b(per|each|per user|per order)\b"
        ]
        
        for pattern in aggregation_patterns:
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                analysis["aggregation_indicators"].append({
                    "text": match.group(),
                    "span": match.span()
                })
        
        return analysis

class SQLTemplateEngine:
    """Generates SQL from predefined templates"""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load SQL templates"""
        return {
            "user_count": {
                "description": "Count total users",
                "sql": "SELECT COUNT(*) as user_count FROM {table_name}",
                "parameters": ["table_name"],
                "query_type": QueryType.COUNT
            },
            "revenue_sum": {
                "description": "Sum of revenue",
                "sql": "SELECT SUM({amount_column}) as total_revenue FROM {table_name}",
                "parameters": ["table_name", "amount_column"],
                "query_type": QueryType.AGGREGATE
            },
            "average_order_value": {
                "description": "Average order value",
                "sql": "SELECT AVG({amount_column}) as avg_order_value FROM {table_name}",
                "query_type": QueryType.AGGREGATE,
                "parameters": ["table_name", "amount_column"]
            },
            "time_series_revenue": {
                "description": "Revenue over time",
                "sql": """
                SELECT 
                    DATE_TRUNC('{time_granularity}', {date_column}) as time_period,
                    SUM({amount_column}) as revenue
                FROM {table_name}
                GROUP BY DATE_TRUNC('{time_granularity}', {date_column})
                ORDER BY time_period
                """,
                "query_type": QueryType.TIME_SERIES,
                "parameters": ["table_name", "date_column", "amount_column", "time_granularity"]
            }
        }
    
    def generate_from_template(self, template_name: str, parameters: Dict[str, Any], 
                             dialect: SQLDialect) -> Optional[str]:
        """Generate SQL from a template"""
        if template_name not in self.templates:
            return None
        
        template = self.templates[template_name]
        sql = template["sql"]
        
        # Replace parameters
        for param_name, param_value in parameters.items():
            placeholder = "{" + param_name + "}"
            if placeholder in sql:
                sql = sql.replace(placeholder, str(param_value))
        
        # Apply dialect-specific adjustments
        sql = self._apply_dialect_adjustments(sql, dialect)
        
        return sql
    
    def _apply_dialect_adjustments(self, sql: str, dialect: SQLDialect) -> str:
        """Apply dialect-specific SQL adjustments"""
        if dialect == SQLDialect.MYSQL:
            # MySQL uses DATE_FORMAT instead of DATE_TRUNC
            sql = sql.replace("DATE_TRUNC", "DATE_FORMAT")
            sql = sql.replace("'day'", "'%Y-%m-%d'")
            sql = sql.replace("'month'", "'%Y-%m'")
            sql = sql.replace("'year'", "'%Y'")
        elif dialect == SQLDialect.SNOWFLAKE:
            # Snowflake uses DATE_TRUNC but with different syntax
            pass  # Default syntax works
        elif dialect == SQLDialect.BIGQUERY:
            # BigQuery uses DATE_TRUNC but with different syntax
            pass  # Default syntax works
        
        return sql

class SQLGenerator:
    """Generates SQL from natural language queries"""
    
    def __init__(self, connection_registry_url: str = "http://connection-registry:8000"):
        self.connection_registry_url = connection_registry_url
        self.http_client = httpx.AsyncClient(timeout=60.0)
        self.query_analyzer = QueryAnalyzer()
        self.template_engine = SQLTemplateEngine()
    
    async def generate_sql(self, request: SQLGenerationRequest) -> SQLGenerationResponse:
        """Generate SQL from a natural language query"""
        try:
            # Analyze the query
            query_analysis = self.query_analyzer.analyze_query(request.query)
            
            # Determine query type
            query_type = self._determine_query_type(query_analysis, request.resolved_metrics)
            
            # Generate SQL based on type
            if query_type == QueryType.COUNT:
                generated_sql = await self._generate_count_sql(request, query_analysis)
            elif query_type == QueryType.AGGREGATE:
                generated_sql = await self._generate_aggregate_sql(request, query_analysis)
            elif query_type == QueryType.TIME_SERIES:
                generated_sql = await self._generate_time_series_sql(request, query_analysis)
            elif query_type == QueryType.COMPARISON:
                generated_sql = await self._generate_comparison_sql(request, query_analysis)
            else:
                generated_sql = await self._generate_generic_sql(request, query_analysis)
            
            if generated_sql:
                return SQLGenerationResponse(
                    success=True,
                    query=request.query,
                    generated_sql=generated_sql,
                    confidence=generated_sql.confidence,
                    metadata={
                        "query_type": query_type,
                        "analysis": query_analysis,
                        "dialect": request.dialect
                    }
                )
            else:
                return SQLGenerationResponse(
                    success=False,
                    query=request.query,
                    error="Failed to generate SQL"
                )
                
        except Exception as e:
            logger.error(f"Error generating SQL: {str(e)}")
            return SQLGenerationResponse(
                success=False,
                query=request.query,
                error=str(e)
            )
    
    def _determine_query_type(self, analysis: Dict[str, Any], 
                            resolved_metrics: List[Dict[str, Any]]) -> QueryType:
        """Determine the type of query to generate"""
        if analysis["query_type"] == "count":
            return QueryType.COUNT
        elif analysis["query_type"] == "aggregate":
            return QueryType.AGGREGATE
        elif analysis["query_type"] == "time_series":
            return QueryType.TIME_SERIES
        elif analysis["query_type"] == "comparison":
            return QueryType.COMPARISON
        else:
            # Default based on metrics
            if resolved_metrics:
                metric_types = [m.get("metric_type", "") for m in resolved_metrics]
                if "count" in metric_types:
                    return QueryType.COUNT
                elif "sum" in metric_types or "average" in metric_types:
                    return QueryType.AGGREGATE
                else:
                    return QueryType.SELECT
            else:
                return QueryType.SELECT
    
    async def _generate_count_sql(self, request: SQLGenerationRequest, 
                                analysis: Dict[str, Any]) -> Optional[GeneratedSQL]:
        """Generate COUNT SQL"""
        try:
            # Find the main table to count
            main_table = self._find_main_table(request.table_metadata, request.query)
            if not main_table:
                return None
            
            # Build basic COUNT query
            sql = f"SELECT COUNT(*) as total_count FROM {main_table['schema_name']}.{main_table['name']}"
            
            # Add WHERE conditions if any
            where_conditions = self._extract_where_conditions(request.query, request.table_metadata)
            if where_conditions:
                sql += " WHERE " + " AND ".join(where_conditions)
            
            # Add LIMIT if specified
            if "limit" in request.options:
                sql += f" LIMIT {request.options['limit']}"
            
            return GeneratedSQL(
                sql=sql,
                query_type=QueryType.COUNT,
                dialect=request.dialect,
                tables=[main_table['name']],
                confidence=0.8,
                explanation="Generated COUNT query based on natural language request"
            )
            
        except Exception as e:
            logger.error(f"Error generating COUNT SQL: {str(e)}")
            return None
    
    async def _generate_aggregate_sql(self, request: SQLGenerationRequest, 
                                    analysis: Dict[str, Any]) -> Optional[GeneratedSQL]:
        """Generate AGGREGATE SQL"""
        try:
            # Find the main table and aggregation columns
            main_table = self._find_main_table(request.table_metadata, request.query)
            if not main_table:
                return None
            
            # Find aggregation columns
            agg_columns = self._find_aggregation_columns(request.resolved_metrics, main_table)
            if not agg_columns:
                return None
            
            # Build SELECT clause
            select_parts = []
            for col in agg_columns:
                if col.aggregation_function:
                    select_parts.append(f"{col.aggregation_function}({col.table_name}.{col.column_name}) as {col.alias or col.column_name}")
                else:
                    select_parts.append(f"{col.table_name}.{col.column_name}")
            
            sql = f"SELECT {', '.join(select_parts)} FROM {main_table['schema_name']}.{main_table['name']}"
            
            # Add WHERE conditions
            where_conditions = self._extract_where_conditions(request.query, request.table_metadata)
            if where_conditions:
                sql += " WHERE " + " AND ".join(where_conditions)
            
            # Add GROUP BY if needed
            group_by_cols = [col for col in agg_columns if not col.is_aggregated]
            if group_by_cols:
                group_by_parts = [f"{col.table_name}.{col.column_name}" for col in group_by_cols]
                sql += f" GROUP BY {', '.join(group_by_parts)}"
            
            # Add ORDER BY
            if agg_columns:
                first_col = agg_columns[0]
                sql += f" ORDER BY {first_col.table_name}.{first_col.column_name} DESC"
            
            return GeneratedSQL(
                sql=sql,
                query_type=QueryType.AGGREGATE,
                dialect=request.dialect,
                tables=[main_table['name']],
                columns=agg_columns,
                confidence=0.8,
                explanation="Generated AGGREGATE query based on natural language request"
            )
            
        except Exception as e:
            logger.error(f"Error generating AGGREGATE SQL: {str(e)}")
            return None
    
    async def _generate_time_series_sql(self, request: SQLGenerationRequest, 
                                      analysis: Dict[str, Any]) -> Optional[GeneratedSQL]:
        """Generate TIME_SERIES SQL"""
        try:
            # Find the main table and time column
            main_table = self._find_main_table(request.table_metadata, request.query)
            if not main_table:
                return None
            
            # Find time column
            time_column = self._find_time_column(main_table)
            if not time_column:
                return None
            
            # Find value column to aggregate
            value_column = self._find_value_column(request.resolved_metrics, main_table)
            if not value_column:
                return None
            
            # Determine time granularity
            granularity = self._determine_time_granularity(analysis["time_references"])
            
            # Build time series query
            if request.dialect == SQLDialect.POSTGRESQL:
                sql = f"""
                SELECT 
                    DATE_TRUNC('{granularity}', {time_column.column_name}) as time_period,
                    COUNT(*) as count,
                    SUM({value_column.column_name}) as total_value
                FROM {main_table['schema_name']}.{main_table['name']}
                GROUP BY DATE_TRUNC('{granularity}', {time_column.column_name})
                ORDER BY time_period
                """
            else:
                # Generic SQL
                sql = f"""
                SELECT 
                    {time_column.column_name} as time_period,
                    COUNT(*) as count,
                    SUM({value_column.column_name}) as total_value
                FROM {main_table['schema_name']}.{main_table['name']}
                GROUP BY {time_column.column_name}
                ORDER BY time_period
                """
            
            return GeneratedSQL(
                sql=sql,
                query_type=QueryType.TIME_SERIES,
                dialect=request.dialect,
                tables=[main_table['name']],
                columns=[time_column, value_column],
                confidence=0.8,
                explanation="Generated TIME_SERIES query based on natural language request"
            )
            
        except Exception as e:
            logger.error(f"Error generating TIME_SERIES SQL: {str(e)}")
            return None
    
    async def _generate_comparison_sql(self, request: SQLGenerationRequest, 
                                     analysis: Dict[str, Any]) -> Optional[GeneratedSQL]:
        """Generate COMPARISON SQL"""
        try:
            # Find the main table
            main_table = self._find_main_table(request.table_metadata, request.query)
            if not main_table:
                return None
            
            # Find comparison columns
            comparison_columns = self._find_comparison_columns(request.resolved_metrics, main_table)
            if not comparison_columns:
                return None
            
            # Build comparison query
            select_parts = []
            for col in comparison_columns:
                if col.aggregation_function:
                    select_parts.append(f"{col.aggregation_function}({col.column_name}) as {col.alias or col.column_name}")
                else:
                    select_parts.append(f"{col.column_name}")
            
            sql = f"SELECT {', '.join(select_parts)} FROM {main_table['schema_name']}.{main_table['name']}"
            
            # Add WHERE conditions
            where_conditions = self._extract_where_conditions(request.query, request.table_metadata)
            if where_conditions:
                sql += " WHERE " + " AND ".join(where_conditions)
            
            return GeneratedSQL(
                sql=sql,
                query_type=QueryType.COMPARISON,
                dialect=request.dialect,
                tables=[main_table['name']],
                columns=comparison_columns,
                confidence=0.7,
                explanation="Generated COMPARISON query based on natural language request"
            )
            
        except Exception as e:
            logger.error(f"Error generating COMPARISON SQL: {str(e)}")
            return None
    
    async def _generate_generic_sql(self, request: SQLGenerationRequest, 
                                  analysis: Dict[str, Any]) -> Optional[GeneratedSQL]:
        """Generate generic SELECT SQL"""
        try:
            # Find the main table
            main_table = self._find_main_table(request.table_metadata, request.query)
            if not main_table:
                return None
            
            # Find relevant columns
            relevant_columns = self._find_relevant_columns(request.resolved_metrics, main_table)
            if not relevant_columns:
                # Select all columns
                relevant_columns = [ColumnReference(
                    table_name=main_table['name'],
                    column_name="*",
                    schema_name=main_table.get('schema_name', 'public')
                )]
            
            # Build SELECT clause
            select_parts = []
            for col in relevant_columns:
                if col.column_name == "*":
                    select_parts.append("*")
                else:
                    select_parts.append(f"{col.table_name}.{col.column_name}")
            
            sql = f"SELECT {', '.join(select_parts)} FROM {main_table['schema_name']}.{main_table['name']}"
            
            # Add WHERE conditions
            where_conditions = self._extract_where_conditions(request.query, request.table_metadata)
            if where_conditions:
                sql += " WHERE " + " AND ".join(where_conditions)
            
            # Add LIMIT
            sql += " LIMIT 100"
            
            return GeneratedSQL(
                sql=sql,
                query_type=QueryType.SELECT,
                dialect=request.dialect,
                tables=[main_table['name']],
                columns=relevant_columns,
                confidence=0.6,
                explanation="Generated generic SELECT query based on natural language request"
            )
            
        except Exception as e:
            logger.error(f"Error generating generic SQL: {str(e)}")
            return None
    
    def _find_main_table(self, table_metadata: List[Dict[str, Any]], 
                        query: str) -> Optional[Dict[str, Any]]:
        """Find the main table for the query"""
        query_lower = query.lower()
        
        # Look for table names mentioned in the query
        for table in table_metadata:
            table_name = table['name'].lower()
            if table_name in query_lower:
                return table
        
        # If no match, return the first table
        return table_metadata[0] if table_metadata else None
    
    def _find_aggregation_columns(self, resolved_metrics: List[Dict[str, Any]], 
                                 main_table: Dict[str, Any]) -> List[ColumnReference]:
        """Find columns for aggregation"""
        columns = []
        
        for metric in resolved_metrics:
            metric_type = metric.get("metric_type", "")
            if metric_type in ["count", "sum", "average"]:
                # Find corresponding column
                for col in main_table.get("columns", []):
                    if self._is_metric_column(col, metric):
                        columns.append(ColumnReference(
                            table_name=main_table['name'],
                            column_name=col['name'],
                            schema_name=main_table.get('schema_name', 'public'),
                            data_type=col.get('data_type'),
                            is_aggregated=True,
                            aggregation_function=self._get_aggregation_function(metric_type)
                        ))
                        break
        
        return columns
    
    def _find_time_column(self, main_table: Dict[str, Any]) -> Optional[ColumnReference]:
        """Find time column in the table"""
        time_patterns = ["date", "time", "created", "updated", "timestamp"]
        
        for col in main_table.get("columns", []):
            col_name = col['name'].lower()
            col_type = col.get('data_type', '').lower()
            
            if any(pattern in col_name for pattern in time_patterns) or 'time' in col_type:
                return ColumnReference(
                    table_name=main_table['name'],
                    column_name=col['name'],
                    schema_name=main_table.get('schema_name', 'public'),
                    data_type=col.get('data_type')
                )
        
        return None
    
    def _find_value_column(self, resolved_metrics: List[Dict[str, Any]], 
                          main_table: Dict[str, Any]) -> Optional[ColumnReference]:
        """Find value column for aggregation"""
        value_patterns = ["amount", "price", "cost", "revenue", "sales", "value"]
        
        for col in main_table.get("columns", []):
            col_name = col['name'].lower()
            col_type = col.get('data_type', '').lower()
            
            if any(pattern in col_name for pattern in value_patterns) or 'numeric' in col_type:
                return ColumnReference(
                    table_name=main_table['name'],
                    column_name=col['name'],
                    schema_name=main_table.get('schema_name', 'public'),
                    data_type=col.get('data_type')
                )
        
        return None
    
    def _find_comparison_columns(self, resolved_metrics: List[Dict[str, Any]], 
                                main_table: Dict[str, Any]) -> List[ColumnReference]:
        """Find columns for comparison"""
        columns = []
        
        for metric in resolved_metrics:
            for col in main_table.get("columns", []):
                if self._is_metric_column(col, metric):
                    columns.append(ColumnReference(
                        table_name=main_table['name'],
                        column_name=col['name'],
                        schema_name=main_table.get('schema_name', 'public'),
                        data_type=col.get('data_type')
                    ))
                    break
        
        return columns
    
    def _find_relevant_columns(self, resolved_metrics: List[Dict[str, Any]], 
                             main_table: Dict[str, Any]) -> List[ColumnReference]:
        """Find relevant columns for the query"""
        columns = []
        
        for metric in resolved_metrics:
            for col in main_table.get("columns", []):
                if self._is_metric_column(col, metric):
                    columns.append(ColumnReference(
                        table_name=main_table['name'],
                        column_name=col['name'],
                        schema_name=main_table.get('schema_name', 'public'),
                        data_type=col.get('data_type')
                    ))
                    break
        
        return columns
    
    def _is_metric_column(self, column: Dict[str, Any], metric: Dict[str, Any]) -> bool:
        """Check if a column matches a metric"""
        # Simple matching logic - can be enhanced
        col_name = column['name'].lower()
        metric_name = metric.get("metric_name", "").lower()
        
        return metric_name in col_name or col_name in metric_name
    
    def _get_aggregation_function(self, metric_type: str) -> str:
        """Get SQL aggregation function for metric type"""
        mapping = {
            "count": "COUNT",
            "sum": "SUM",
            "average": "AVG",
            "min": "MIN",
            "max": "MAX"
        }
        return mapping.get(metric_type, "COUNT")
    
    def _determine_time_granularity(self, time_references: List[Dict[str, Any]]) -> str:
        """Determine time granularity from time references"""
        if not time_references:
            return "day"
        
        # Analyze time references to determine granularity
        # This is a simplified implementation
        return "day"
    
    def _extract_where_conditions(self, query: str, 
                                table_metadata: List[Dict[str, Any]]) -> List[str]:
        """Extract WHERE conditions from natural language query"""
        # This is a simplified implementation
        # In a real system, this would use NLP to extract conditions
        conditions = []
        
        # Example: extract date ranges
        if "last" in query.lower() and "days" in query.lower():
            conditions.append("created_at >= CURRENT_DATE - INTERVAL '7 days'")
        
        return conditions

# =============================================================================
# Service Instance
# =============================================================================

sql_generator = SQLGenerator()

# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "SQL Generator Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/generate - Generate SQL from natural language",
            "/template - Generate SQL from template",
            "/health - Health check"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "sql-generator-service"
    }

@app.post("/generate", response_model=SQLGenerationResponse)
async def generate_sql(request: SQLGenerationRequest):
    """Generate SQL from natural language query"""
    return await sql_generator.generate_sql(request)

@app.post("/template", response_model=SQLTemplateResponse)
async def generate_from_template(request: SQLTemplateRequest):
    """Generate SQL from a template"""
    sql = sql_generator.template_engine.generate_from_template(
        request.template_name, 
        request.parameters, 
        request.dialect
    )
    
    if sql:
        return SQLTemplateResponse(
            success=True,
            template_name=request.template_name,
            generated_sql=GeneratedSQL(
                sql=sql,
                query_type=QueryType.SELECT,
                dialect=request.dialect,
                confidence=0.9,
                explanation=f"Generated from template: {request.template_name}"
            )
        )
    else:
        return SQLTemplateResponse(
            success=False,
            template_name=request.template_name,
            error="Template not found or invalid parameters"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
