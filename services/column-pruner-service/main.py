"""
Column Pruner Service - Intelligent Column Selection

This service is responsible for:
1. Analyzing query intent and selecting relevant columns
2. Filtering out low-quality or irrelevant columns
3. Optimizing column selection for performance
4. Providing column recommendations for queries
5. Supporting query planning and optimization
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum

import httpx
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Data Models
# ============================================================================

class ColumnRelevance(str, Enum):
    """Column relevance levels"""
    CRITICAL = "critical"      # Essential for the query
    HIGH = "high"              # Very important
    MEDIUM = "medium"          # Moderately important
    LOW = "low"                # Low importance
    EXCLUDE = "exclude"        # Should be excluded

class ColumnCategory(str, Enum):
    """Column categories for classification"""
    IDENTIFIER = "identifier"      # Primary keys, unique identifiers
    MEASURE = "measure"            # Metrics, counts, amounts
    DIMENSION = "dimension"        # Categories, attributes
    TIMESTAMP = "timestamp"        # Date/time fields
    METADATA = "metadata"          # System fields, audit trails
    RELATIONSHIP = "relationship"  # Foreign keys, references

class ColumnScore(BaseModel):
    """Column scoring information"""
    column_name: str
    relevance_score: float  # 0.0 to 1.0
    relevance_level: ColumnRelevance
    category: ColumnCategory
    confidence: float = 0.0
    
    # Scoring factors
    query_match_score: float = 0.0
    data_quality_score: float = 0.0
    business_relevance_score: float = 0.0
    performance_score: float = 0.0
    
    # Reasoning
    reasoning: str = ""
    matched_terms: List[str] = []
    quality_issues: List[str] = []

class PruningRequest(BaseModel):
    """Request to prune columns for a query"""
    connection_id: str
    tenant_id: str
    user_id: str
    query: str
    table_names: List[str] = []
    available_columns: List[Dict[str, Any]] = []
    
    # Pruning options
    max_columns: int = 20
    min_relevance_score: float = 0.3
    include_quality_analysis: bool = True
    include_business_context: bool = True
    exclude_categories: List[ColumnCategory] = []

class PruningResponse(BaseModel):
    """Response from column pruning operation"""
    success: bool
    selected_columns: List[ColumnScore] = []
    excluded_columns: List[ColumnScore] = []
    total_columns_analyzed: int = 0
    pruning_time_ms: float = 0.0
    error: Optional[str] = None
    recommendations: List[str] = []

# ============================================================================
# Column Analysis Engine
# ============================================================================

class ColumnAnalyzer:
    """Core column analysis engine"""
    
    def __init__(self):
        # Common business terms and their relevance
        self.business_terms = {
            "revenue": ["amount", "total", "sum", "price", "cost", "value"],
            "customers": ["user", "client", "customer", "account", "member"],
            "orders": ["order", "transaction", "purchase", "sale", "booking"],
            "products": ["product", "item", "goods", "service", "offering"],
            "time": ["date", "time", "created", "updated", "timestamp", "period"],
            "location": ["address", "city", "state", "country", "region", "zip"],
            "status": ["status", "state", "active", "inactive", "enabled", "disabled"]
        }
        
        # Column name patterns and their categories
        self.column_patterns = {
            "identifier": [r"_id$", r"^id$", r"uuid", r"guid", r"key"],
            "measure": [r"count", r"sum", r"total", r"amount", r"quantity", r"price"],
            "dimension": [r"name", r"type", r"category", r"brand", r"color", r"size"],
            "timestamp": [r"date", r"time", r"created", r"updated", r"modified"],
            "metadata": [r"version", r"hash", r"checksum", r"signature"],
            "relationship": [r"_id$", r"ref_", r"fk_", r"parent_", r"child_"]
        }
    
    def analyze_column_relevance(self, column_info: Dict[str, Any], 
                                query: str, business_context: Dict[str, Any]) -> ColumnScore:
        """Analyze column relevance for a given query"""
        column_name = column_info.get('name', '')
        data_type = column_info.get('data_type', '')
        
        # Initialize column score
        score = ColumnScore(
            column_name=column_name,
            relevance_score=0.0,
            relevance_level=ColumnRelevance.LOW,
            category=self._categorize_column(column_name, data_type),
            confidence=0.0
        )
        
        # Calculate query match score
        query_match_score = self._calculate_query_match_score(column_name, query)
        score.query_match_score = query_match_score
        
        # Calculate data quality score
        quality_score = self._calculate_quality_score(column_info)
        score.data_quality_score = quality_score
        
        # Calculate business relevance score
        business_score = self._calculate_business_relevance(column_name, business_context)
        score.business_relevance_score = business_score
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(column_info)
        score.performance_score = performance_score
        
        # Calculate overall relevance score
        overall_score = self._calculate_overall_score(score)
        score.relevance_score = overall_score
        
        # Determine relevance level
        score.relevance_level = self._determine_relevance_level(overall_score)
        
        # Generate reasoning
        score.reasoning = self._generate_reasoning(score)
        
        # Extract matched terms
        score.matched_terms = self._extract_matched_terms(column_name, query)
        
        # Identify quality issues
        score.quality_issues = self._identify_quality_issues(column_info)
        
        return score
    
    def _categorize_column(self, column_name: str, data_type: str) -> ColumnCategory:
        """Categorize column based on name and data type"""
        column_lower = column_name.lower()
        data_type_lower = data_type.lower()
        
        # Check for identifier patterns
        if any(pattern in column_lower for pattern in ['_id', 'id', 'uuid', 'guid', 'key']):
            return ColumnCategory.IDENTIFIER
        
        # Check for measure patterns
        if any(pattern in column_lower for pattern in ['count', 'sum', 'total', 'amount', 'quantity', 'price']):
            return ColumnCategory.MEASURE
        
        # Check for timestamp patterns
        if any(pattern in column_lower for pattern in ['date', 'time', 'created', 'updated', 'modified']):
            return ColumnCategory.TIMESTAMP
        
        # Check for metadata patterns
        if any(pattern in column_lower for pattern in ['version', 'hash', 'checksum', 'signature']):
            return ColumnCategory.METADATA
        
        # Check for relationship patterns
        if any(pattern in column_lower for pattern in ['ref_', 'fk_', 'parent_', 'child_']):
            return ColumnCategory.RELATIONSHIP
        
        # Default to dimension
        return ColumnCategory.DIMENSION
    
    def _calculate_query_match_score(self, column_name: str, query: str) -> float:
        """Calculate how well a column matches the query"""
        query_lower = query.lower()
        column_lower = column_name.lower()
        
        score = 0.0
        
        # Exact column name match
        if column_name.lower() in query_lower:
            score += 0.4
        
        # Partial column name match
        if any(word in column_lower for word in query_lower.split()):
            score += 0.3
        
        # Synonym matching
        synonyms = self._get_column_synonyms(column_name)
        for synonym in synonyms:
            if synonym in query_lower:
                score += 0.2
                break
        
        # Business term matching
        for business_term, related_terms in self.business_terms.items():
            if business_term in query_lower:
                if any(term in column_lower for term in related_terms):
                    score += 0.3
                    break
        
        return min(score, 1.0)
    
    def _calculate_quality_score(self, column_info: Dict[str, Any]) -> float:
        """Calculate data quality score for a column"""
        score = 1.0
        
        # Check for null rate
        null_rate = column_info.get('null_rate', 0.0)
        if null_rate > 0.5:
            score -= 0.3
        elif null_rate > 0.2:
            score -= 0.1
        
        # Check for distinct rate
        distinct_rate = column_info.get('distinct_rate', 1.0)
        if distinct_rate < 0.1:
            score -= 0.2
        elif distinct_rate < 0.3:
            score -= 0.1
        
        # Check for data type appropriateness
        data_type = column_info.get('data_type', '')
        if data_type and 'unknown' in data_type.lower():
            score -= 0.2
        
        return max(score, 0.0)
    
    def _calculate_business_relevance(self, column_name: str, 
                                    business_context: Dict[str, Any]) -> float:
        """Calculate business relevance score"""
        score = 0.5  # Base score
        
        # Check if column is in business context
        if business_context.get('business_domain'):
            domain = business_context['business_domain'].lower()
            column_lower = column_name.lower()
            
            # Domain-specific relevance
            if domain == 'users' and any(word in column_lower for word in ['name', 'email', 'user']):
                score += 0.3
            elif domain == 'orders' and any(word in column_lower for word in ['order', 'amount', 'status']):
                score += 0.3
            elif domain == 'products' and any(word in column_lower for word in ['product', 'price', 'category']):
                score += 0.3
        
        # Check for common business fields
        business_fields = ['name', 'description', 'status', 'type', 'category']
        if any(field in column_name.lower() for field in business_fields):
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_performance_score(self, column_info: Dict[str, Any]) -> float:
        """Calculate performance score for a column"""
        score = 1.0
        
        # Check data type size
        data_type = column_info.get('data_type', '')
        if 'text' in data_type.lower() or 'varchar' in data_type.lower():
            max_length = column_info.get('max_length', 0)
            if max_length > 1000:
                score -= 0.2
            elif max_length > 500:
                score -= 0.1
        
        # Check if column is indexed
        if column_info.get('is_indexed', False):
            score += 0.2
        
        # Check if column is primary key
        if column_info.get('is_primary_key', False):
            score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_overall_score(self, column_score: ColumnScore) -> float:
        """Calculate overall relevance score"""
        # Weighted combination of individual scores
        weights = {
            'query_match': 0.4,
            'quality': 0.2,
            'business': 0.2,
            'performance': 0.2
        }
        
        overall_score = (
            column_score.query_match_score * weights['query_match'] +
            column_score.data_quality_score * weights['quality'] +
            column_score.business_relevance_score * weights['business'] +
            column_score.performance_score * weights['performance']
        )
        
        return min(overall_score, 1.0)
    
    def _determine_relevance_level(self, score: float) -> ColumnRelevance:
        """Determine relevance level based on score"""
        if score >= 0.8:
            return ColumnRelevance.CRITICAL
        elif score >= 0.6:
            return ColumnRelevance.HIGH
        elif score >= 0.4:
            return ColumnRelevance.MEDIUM
        elif score >= 0.2:
            return ColumnRelevance.LOW
        else:
            return ColumnRelevance.EXCLUDE
    
    def _generate_reasoning(self, column_score: ColumnScore) -> str:
        """Generate reasoning for column score"""
        reasons = []
        
        if column_score.query_match_score > 0.5:
            reasons.append("High query relevance")
        
        if column_score.data_quality_score > 0.8:
            reasons.append("Good data quality")
        elif column_score.data_quality_score < 0.5:
            reasons.append("Data quality concerns")
        
        if column_score.business_relevance_score > 0.7:
            reasons.append("High business relevance")
        
        if column_score.performance_score > 0.8:
            reasons.append("Good performance characteristics")
        
        if column_score.category == ColumnCategory.IDENTIFIER:
            reasons.append("Identifier column")
        elif column_score.category == ColumnCategory.MEASURE:
            reasons.append("Measure/metric column")
        
        return "; ".join(reasons) if reasons else "General relevance"
    
    def _extract_matched_terms(self, column_name: str, query: str) -> List[str]:
        """Extract terms that matched between column and query"""
        query_terms = set(query.lower().split())
        column_terms = set(column_name.lower().split('_'))
        
        # Find common terms
        matched = query_terms.intersection(column_terms)
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        matched = matched - common_words
        
        return list(matched)
    
    def _identify_quality_issues(self, column_info: Dict[str, Any]) -> List[str]:
        """Identify data quality issues for a column"""
        issues = []
        
        null_rate = column_info.get('null_rate', 0.0)
        if null_rate > 0.5:
            issues.append("High null rate")
        
        distinct_rate = column_info.get('distinct_rate', 1.0)
        if distinct_rate < 0.1:
            issues.append("Low distinct rate")
        
        if column_info.get('data_type') == 'unknown':
            issues.append("Unknown data type")
        
        return issues
    
    def _get_column_synonyms(self, column_name: str) -> List[str]:
        """Get synonyms for a column name"""
        synonyms = []
        column_lower = column_name.lower()
        
        # Common synonyms
        if 'id' in column_lower:
            synonyms.extend(['identifier', 'key', 'primary'])
        if 'name' in column_lower:
            synonyms.extend(['title', 'label', 'description'])
        if 'date' in column_lower:
            synonyms.extend(['time', 'timestamp', 'created', 'updated'])
        if 'amount' in column_lower:
            synonyms.extend(['total', 'sum', 'value', 'price'])
        
        return synonyms

# ============================================================================
# Column Pruning Engine
# ============================================================================

class ColumnPruner:
    """Main column pruning engine"""
    
    def __init__(self):
        self.analyzer = ColumnAnalyzer()
    
    async def prune_columns(self, request: PruningRequest) -> PruningResponse:
        """Prune columns based on query relevance"""
        start_time = datetime.now()
        
        try:
            # Analyze each column
            column_scores = []
            for column_info in request.available_columns:
                # Get business context (would come from table retriever service)
                business_context = self._get_business_context(column_info, request.table_names)
                
                # Analyze column relevance
                score = self.analyzer.analyze_column_relevance(
                    column_info, request.query, business_context
                )
                
                column_scores.append(score)
            
            # Sort columns by relevance score
            column_scores.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Apply pruning rules
            selected_columns = []
            excluded_columns = []
            
            for score in column_scores:
                # Check exclusion criteria
                if score.relevance_level == ColumnRelevance.EXCLUDE:
                    excluded_columns.append(score)
                    continue
                
                # Check minimum relevance score
                if score.relevance_score < request.min_relevance_score:
                    excluded_columns.append(score)
                    continue
                
                # Check excluded categories
                if score.category in request.exclude_categories:
                    excluded_columns.append(score)
                    continue
                
                # Add to selected columns if within limit
                if len(selected_columns) < request.max_columns:
                    selected_columns.append(score)
                else:
                    excluded_columns.append(score)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(selected_columns, excluded_columns)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return PruningResponse(
                success=True,
                selected_columns=selected_columns,
                excluded_columns=excluded_columns,
                total_columns_analyzed=len(request.available_columns),
                pruning_time_ms=processing_time,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Column pruning failed: {e}")
            return PruningResponse(
                success=False,
                error=str(e)
            )
    
    def _get_business_context(self, column_info: Dict[str, Any], 
                            table_names: List[str]) -> Dict[str, Any]:
        """Get business context for a column"""
        # This would typically call the table retriever service
        # For now, return placeholder context
        
        table_name = column_info.get('table_name', '')
        if 'user' in table_name.lower():
            return {'business_domain': 'users'}
        elif 'order' in table_name.lower():
            return {'business_domain': 'orders'}
        elif 'product' in table_name.lower():
            return {'business_domain': 'products'}
        else:
            return {'business_domain': 'general'}
    
    def _generate_recommendations(self, selected_columns: List[ColumnScore], 
                                excluded_columns: List[ColumnScore]) -> List[str]:
        """Generate recommendations based on pruning results"""
        recommendations = []
        
        # Check if we have enough columns
        if len(selected_columns) < 5:
            recommendations.append("Consider lowering minimum relevance score for more columns")
        
        # Check for missing critical columns
        critical_columns = [col for col in selected_columns if col.relevance_level == ColumnRelevance.CRITICAL]
        if len(critical_columns) == 0:
            recommendations.append("No critical columns found - review query intent")
        
        # Check for quality issues
        low_quality_columns = [col for col in selected_columns if col.data_quality_score < 0.5]
        if low_quality_columns:
            recommendations.append(f"Review data quality for {len(low_quality_columns)} selected columns")
        
        # Check for performance concerns
        low_performance_columns = [col for col in selected_columns if col.performance_score < 0.5]
        if low_performance_columns:
            recommendations.append(f"Consider performance impact of {len(low_performance_columns)} columns")
        
        return recommendations

# ============================================================================
# Main Service
# ============================================================================

class ColumnPrunerService:
    """Main column pruner service"""
    
    def __init__(self):
        self.pruner = ColumnPruner()
    
    async def prune_columns(self, request: PruningRequest) -> PruningResponse:
        """Prune columns for a query"""
        return await self.pruner.prune_columns(request)

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="AskData Column Pruner Service",
    description="Intelligent column selection and pruning for queries",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service
pruner_service = ColumnPrunerService()

@app.post("/prune", response_model=PruningResponse)
async def prune_columns(request: PruningRequest):
    """Prune columns for a query"""
    return await pruner_service.prune_columns(request)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "column-pruner-service"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AskData Column Pruner Service",
        "version": "1.0.0",
        "endpoints": {
            "prune": "/prune",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
