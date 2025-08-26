"""
Metric Resolver Service

This service maps business metrics, KPIs, and calculations mentioned in natural language
queries to the appropriate database columns, functions, and aggregation methods.
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

app = FastAPI(title="Metric Resolver Service", version="1.0.0")

# =============================================================================
# Data Models
# =============================================================================

class MetricType(str, Enum):
    """Types of business metrics"""
    COUNT = "count"
    SUM = "sum"
    AVERAGE = "average"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    PERCENTAGE = "percentage"
    RATIO = "ratio"
    GROWTH_RATE = "growth_rate"
    CUSTOM = "custom"

class AggregationFunction(str, Enum):
    """Database aggregation functions"""
    COUNT = "COUNT"
    SUM = "SUM"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"
    PERCENTILE_CONT = "PERCENTILE_CONT"
    STDDEV = "STDDEV"
    VARIANCE = "VARIANCE"

class MetricCategory(str, Enum):
    """Categories of business metrics"""
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    CUSTOMER = "customer"
    PRODUCT = "product"
    TIME_BASED = "time_based"
    QUALITY = "quality"
    EFFICIENCY = "efficiency"

class MetricDefinition(BaseModel):
    """Definition of a business metric"""
    metric_name: str
    display_name: str
    description: str
    metric_type: MetricType
    category: MetricCategory
    aggregation_function: AggregationFunction
    unit: Optional[str] = None
    formula: Optional[str] = None
    business_context: Dict[str, Any] = {}
    synonyms: List[str] = []
    confidence: float = 0.0

class ColumnMapping(BaseModel):
    """Mapping of a metric to database columns"""
    table_name: str
    column_name: str
    schema_name: str = "public"
    data_type: str
    is_primary_key: bool = False
    is_foreign_key: bool = False
    nullable: bool = True
    default_value: Optional[Any] = None
    confidence: float = 0.0

class MetricMapping(BaseModel):
    """Complete mapping of a metric to database implementation"""
    metric_definition: MetricDefinition
    primary_column: ColumnMapping
    supporting_columns: List[ColumnMapping] = []
    filters: List[Dict[str, Any]] = []
    group_by_columns: List[ColumnMapping] = []
    order_by_columns: List[ColumnMapping] = []
    time_granularity: Optional[str] = None
    confidence: float = 0.0

class MetricResolutionRequest(BaseModel):
    """Request to resolve metrics in a query"""
    query: str
    connection_id: str
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    context: Dict[str, Any] = {}
    options: Dict[str, Any] = {}

class MetricResolutionResponse(BaseModel):
    """Response containing resolved metrics"""
    success: bool
    query: str
    resolved_metrics: List[MetricMapping] = []
    unresolved_terms: List[str] = []
    suggestions: List[str] = []
    confidence: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}

class MetricDiscoveryRequest(BaseModel):
    """Request to discover available metrics"""
    connection_id: str
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    categories: Optional[List[MetricCategory]] = None
    search_term: Optional[str] = None

class MetricDiscoveryResponse(BaseModel):
    """Response containing discovered metrics"""
    success: bool
    available_metrics: List[MetricDefinition] = []
    total_count: int = 0
    categories: List[MetricCategory] = []
    error: Optional[str] = None

# =============================================================================
# Core Components
# =============================================================================

class MetricPatternMatcher:
    """Matches natural language patterns to metric definitions"""
    
    def __init__(self):
        self.patterns = self._build_patterns()
    
    def _build_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build regex patterns for metric detection"""
        return {
            "count": [
                {"pattern": r"\b(count|number|total|how many)\b", "confidence": 0.9},
                {"pattern": r"\b(users|customers|orders|products)\b", "confidence": 0.8},
                {"pattern": r"\b(active|total|unique)\b", "confidence": 0.7}
            ],
            "sum": [
                {"pattern": r"\b(sum|total|revenue|sales|amount|cost)\b", "confidence": 0.9},
                {"pattern": r"\b(earnings|income|profit|loss)\b", "confidence": 0.8}
            ],
            "average": [
                {"pattern": r"\b(average|avg|mean|typical|usual)\b", "confidence": 0.9},
                {"pattern": r"\b(per|each|per user|per order)\b", "confidence": 0.8}
            ],
            "percentage": [
                {"pattern": r"\b(percentage|percent|%|rate|ratio)\b", "confidence": 0.9},
                {"pattern": r"\b(conversion|success|failure|churn)\b", "confidence": 0.8}
            ],
            "growth": [
                {"pattern": r"\b(growth|increase|decrease|change|trend)\b", "confidence": 0.9},
                {"pattern": r"\b(monthly|yearly|quarterly|weekly)\b", "confidence": 0.8}
            ]
        }
    
    def extract_metric_terms(self, query: str) -> List[Dict[str, Any]]:
        """Extract potential metric terms from a query"""
        terms = []
        query_lower = query.lower()
        
        for metric_type, patterns in self.patterns.items():
            for pattern_info in patterns:
                matches = re.finditer(pattern_info["pattern"], query_lower)
                for match in matches:
                    terms.append({
                        "term": match.group(),
                        "metric_type": metric_type,
                        "confidence": pattern_info["confidence"],
                        "position": match.span()
                    })
        
        return terms

class MetricResolver:
    """Resolves metrics to database columns and functions"""
    
    def __init__(self, connection_registry_url: str = "http://connection-registry:8000"):
        self.connection_registry_url = connection_registry_url
        self.http_client = httpx.AsyncClient(timeout=60.0)
        self.metric_definitions = self._load_metric_definitions()
    
    def _load_metric_definitions(self) -> List[MetricDefinition]:
        """Load predefined metric definitions"""
        return [
            MetricDefinition(
                metric_name="user_count",
                display_name="User Count",
                description="Total number of users",
                metric_type=MetricType.COUNT,
                category=MetricCategory.CUSTOMER,
                aggregation_function=AggregationFunction.COUNT,
                unit="users",
                synonyms=["users", "customers", "total users", "user base"],
                confidence=0.9
            ),
            MetricDefinition(
                metric_name="revenue",
                display_name="Revenue",
                description="Total revenue amount",
                metric_type=MetricType.SUM,
                category=MetricCategory.FINANCIAL,
                aggregation_function=AggregationFunction.SUM,
                unit="currency",
                synonyms=["sales", "income", "earnings", "total revenue"],
                confidence=0.9
            ),
            MetricDefinition(
                metric_name="average_order_value",
                display_name="Average Order Value",
                description="Average amount per order",
                metric_type=MetricType.AVERAGE,
                category=MetricCategory.FINANCIAL,
                aggregation_function=AggregationFunction.AVG,
                unit="currency",
                synonyms=["aov", "avg order", "typical order"],
                confidence=0.8
            ),
            MetricDefinition(
                metric_name="conversion_rate",
                display_name="Conversion Rate",
                description="Percentage of visitors who convert",
                metric_type=MetricType.PERCENTAGE,
                category=MetricCategory.OPERATIONAL,
                aggregation_function=AggregationFunction.AVG,
                unit="percentage",
                formula="(conversions / visitors) * 100",
                synonyms=["conversion", "success rate", "conversion %"],
                confidence=0.8
            )
        ]
    
    async def resolve_metrics(self, request: MetricResolutionRequest) -> MetricResolutionResponse:
        """Resolve metrics mentioned in a query"""
        try:
            # Extract metric terms
            pattern_matcher = MetricPatternMatcher()
            metric_terms = pattern_matcher.extract_metric_terms(request.query)
            
            # Get table metadata
            table_metadata = await self._get_table_metadata(request.connection_id)
            
            # Resolve each metric term
            resolved_metrics = []
            unresolved_terms = []
            
            for term_info in metric_terms:
                metric_mapping = await self._resolve_single_metric(
                    term_info, table_metadata, request
                )
                if metric_mapping:
                    resolved_metrics.append(metric_mapping)
                else:
                    unresolved_terms.append(term_info["term"])
            
            # Calculate overall confidence
            confidence = self._calculate_overall_confidence(resolved_metrics)
            
            return MetricResolutionResponse(
                success=True,
                query=request.query,
                resolved_metrics=resolved_metrics,
                unresolved_terms=unresolved_terms,
                confidence=confidence,
                metadata={
                    "total_terms": len(metric_terms),
                    "resolved_count": len(resolved_metrics),
                    "unresolved_count": len(unresolved_terms)
                }
            )
            
        except Exception as e:
            logger.error(f"Error resolving metrics: {str(e)}")
            return MetricResolutionResponse(
                success=False,
                query=request.query,
                error=str(e)
            )
    
    async def _resolve_single_metric(self, term_info: Dict[str, Any], 
                                   table_metadata: List[Dict[str, Any]], 
                                   request: MetricResolutionRequest) -> Optional[MetricMapping]:
        """Resolve a single metric term to database columns"""
        try:
            # Find matching metric definition
            metric_def = self._find_metric_definition(term_info["term"])
            if not metric_def:
                return None
            
            # Find matching columns
            matching_columns = self._find_matching_columns(
                term_info, metric_def, table_metadata
            )
            
            if not matching_columns:
                return None
            
            # Create metric mapping
            primary_column = matching_columns[0]
            supporting_columns = matching_columns[1:] if len(matching_columns) > 1 else []
            
            return MetricMapping(
                metric_definition=metric_def,
                primary_column=primary_column,
                supporting_columns=supporting_columns,
                confidence=term_info["confidence"] * metric_def.confidence
            )
            
        except Exception as e:
            logger.error(f"Error resolving single metric: {str(e)}")
            return None
    
    def _find_metric_definition(self, term: str) -> Optional[MetricDefinition]:
        """Find a metric definition that matches the term"""
        term_lower = term.lower()
        
        for metric_def in self.metric_definitions:
            if (term_lower in metric_def.synonyms or 
                term_lower in metric_def.display_name.lower() or
                term_lower in metric_def.description.lower()):
                return metric_def
        
        return None
    
    def _find_matching_columns(self, term_info: Dict[str, Any], 
                             metric_def: MetricDefinition, 
                             table_metadata: List[Dict[str, Any]]) -> List[ColumnMapping]:
        """Find database columns that match the metric"""
        matching_columns = []
        
        for table in table_metadata:
            for column in table.get("columns", []):
                column_mapping = ColumnMapping(
                    table_name=table["name"],
                    column_name=column["name"],
                    schema_name=table.get("schema_name", "public"),
                    data_type=column["data_type"],
                    is_primary_key=column.get("is_primary_key", False),
                    is_foreign_key=column.get("is_foreign_key", False),
                    nullable=column.get("nullable", True),
                    default_value=column.get("default_value")
                )
                
                # Score the column match
                score = self._score_column_match(term_info, metric_def, column_mapping)
                if score > 0.5:  # Threshold for matching
                    column_mapping.confidence = score
                    matching_columns.append(column_mapping)
        
        # Sort by confidence and return top matches
        matching_columns.sort(key=lambda x: x.confidence, reverse=True)
        return matching_columns[:3]  # Return top 3 matches
    
    def _score_column_match(self, term_info: Dict[str, Any], 
                          metric_def: MetricDefinition, 
                          column_mapping: ColumnMapping) -> float:
        """Score how well a column matches a metric"""
        score = 0.0
        
        # Name similarity
        term_lower = term_info["term"].lower()
        column_lower = column_mapping.column_name.lower()
        table_lower = column_mapping.table_name.lower()
        
        if term_lower in column_lower:
            score += 0.4
        elif term_lower in table_lower:
            score += 0.3
        
        # Data type compatibility
        if self._is_compatible_data_type(metric_def.metric_type, column_mapping.data_type):
            score += 0.3
        
        # Business context
        if self._matches_business_context(metric_def.category, column_mapping):
            score += 0.2
        
        return min(score, 1.0)
    
    def _is_compatible_data_type(self, metric_type: MetricType, data_type: str) -> bool:
        """Check if data type is compatible with metric type"""
        data_type_lower = data_type.lower()
        
        if metric_type in [MetricType.COUNT, MetricType.SUM, MetricType.AVERAGE]:
            return any(numeric in data_type_lower for numeric in ["int", "bigint", "decimal", "numeric", "float", "double"])
        elif metric_type == MetricType.PERCENTAGE:
            return any(numeric in data_type_lower for numeric in ["decimal", "numeric", "float", "double"])
        elif metric_type == MetricType.MIN or metric_type == MetricType.MAX:
            return True  # Most data types support min/max
        
        return False
    
    def _matches_business_context(self, category: MetricCategory, 
                                column_mapping: ColumnMapping) -> bool:
        """Check if column matches business context"""
        column_lower = column_mapping.column_name.lower()
        table_lower = column_mapping.table_name.lower()
        
        if category == MetricCategory.FINANCIAL:
            return any(financial in column_lower for financial in ["amount", "price", "cost", "revenue", "sales"])
        elif category == MetricCategory.CUSTOMER:
            return any(customer in column_lower for customer in ["user", "customer", "client"]) or "user" in table_lower
        elif category == MetricCategory.OPERATIONAL:
            return any(operational in column_lower for operational in ["status", "state", "flag", "active"])
        
        return False
    
    def _calculate_overall_confidence(self, resolved_metrics: List[MetricMapping]) -> float:
        """Calculate overall confidence score"""
        if not resolved_metrics:
            return 0.0
        
        total_confidence = sum(metric.confidence for metric in resolved_metrics)
        return total_confidence / len(resolved_metrics)
    
    async def _get_table_metadata(self, connection_id: str) -> List[Dict[str, Any]]:
        """Get table metadata from the introspect service"""
        try:
            # Placeholder for introspect service call
            return [
                {
                    "name": "users",
                    "schema_name": "public",
                    "columns": [
                        {"name": "id", "data_type": "integer", "is_primary_key": True},
                        {"name": "email", "data_type": "varchar", "nullable": False},
                        {"name": "created_at", "data_type": "timestamp", "nullable": False}
                    ]
                },
                {
                    "name": "orders",
                    "schema_name": "public",
                    "columns": [
                        {"name": "id", "data_type": "integer", "is_primary_key": True},
                        {"name": "user_id", "data_type": "integer", "is_foreign_key": True},
                        {"name": "amount", "data_type": "decimal", "nullable": False},
                        {"name": "status", "data_type": "varchar", "nullable": False},
                        {"name": "created_at", "data_type": "timestamp", "nullable": False}
                    ]
                }
            ]
        except Exception as e:
            logger.error(f"Error getting table metadata: {str(e)}")
            return []

class MetricDiscoveryEngine:
    """Discovers available metrics in a database"""
    
    def __init__(self):
        self.metric_definitions = self._load_metric_definitions()
    
    def _load_metric_definitions(self) -> List[MetricDefinition]:
        """Load all available metric definitions"""
        # This would typically load from a database or configuration
        return [
            MetricDefinition(
                metric_name="user_count",
                display_name="User Count",
                description="Total number of users",
                metric_type=MetricType.COUNT,
                category=MetricCategory.CUSTOMER,
                aggregation_function=AggregationFunction.COUNT,
                unit="users",
                synonyms=["users", "customers", "total users", "user base"],
                confidence=0.9
            ),
            MetricDefinition(
                metric_name="revenue",
                display_name="Revenue",
                description="Total revenue amount",
                metric_type=MetricType.SUM,
                category=MetricCategory.FINANCIAL,
                aggregation_function=AggregationFunction.SUM,
                unit="currency",
                synonyms=["sales", "income", "earnings", "total revenue"],
                confidence=0.9
            ),
            MetricDefinition(
                metric_name="average_order_value",
                display_name="Average Order Value",
                description="Average amount per order",
                metric_type=MetricType.AVERAGE,
                category=MetricCategory.FINANCIAL,
                aggregation_function=AggregationFunction.AVG,
                unit="currency",
                synonyms=["aov", "avg order", "typical order"],
                confidence=0.8
            ),
            MetricDefinition(
                metric_name="conversion_rate",
                display_name="Conversion Rate",
                description="Percentage of visitors who convert",
                metric_type=MetricType.PERCENTAGE,
                category=MetricCategory.OPERATIONAL,
                aggregation_function=AggregationFunction.AVG,
                unit="percentage",
                formula="(conversions / visitors) * 100",
                synonyms=["conversion", "success rate", "conversion %"],
                confidence=0.8
            )
        ]
    
    async def discover_metrics(self, request: MetricDiscoveryRequest) -> MetricDiscoveryResponse:
        """Discover available metrics"""
        try:
            available_metrics = self.metric_definitions.copy()
            
            # Apply filters
            if request.categories:
                available_metrics = [
                    metric for metric in available_metrics 
                    if metric.category in request.categories
                ]
            
            if request.search_term:
                search_lower = request.search_term.lower()
                available_metrics = [
                    metric for metric in available_metrics
                    if (search_lower in metric.display_name.lower() or
                        search_lower in metric.description.lower() or
                        any(search_lower in synonym.lower() for synonym in metric.synonyms))
                ]
            
            # Get unique categories
            categories = list(set(metric.category for metric in available_metrics))
            
            return MetricDiscoveryResponse(
                success=True,
                available_metrics=available_metrics,
                total_count=len(available_metrics),
                categories=categories
            )
            
        except Exception as e:
            logger.error(f"Error discovering metrics: {str(e)}")
            return MetricDiscoveryResponse(
                success=False,
                error=str(e)
            )

# =============================================================================
# Service Instance
# =============================================================================

metric_resolver = MetricResolver()
metric_discovery_engine = MetricDiscoveryEngine()

# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Metric Resolver Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/resolve - Resolve metrics in a query",
            "/discover - Discover available metrics",
            "/health - Health check"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "metric-resolver-service"
    }

@app.post("/resolve", response_model=MetricResolutionResponse)
async def resolve_metrics(request: MetricResolutionRequest):
    """Resolve metrics mentioned in a natural language query"""
    return await metric_resolver.resolve_metrics(request)

@app.post("/discover", response_model=MetricDiscoveryResponse)
async def discover_metrics(request: MetricDiscoveryRequest):
    """Discover available metrics in a database"""
    return await metric_discovery_engine.discover_metrics(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
