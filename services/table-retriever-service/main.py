"""
AskData Table Retriever Service

This service provides hybrid search (lexical + embedding) for tables within a connection scope.
It combines semantic search from the vector store with lexical matching for optimal results.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import structlog
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# Add the contracts directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent / "contracts"))

from mcp_tools import (
    SearchTablesInput, SearchTablesOutput,
    SearchColumnsInput, SearchColumnsOutput
)

# =============================================================================
# CONFIGURATION
# =============================================================================

class Settings(BaseSettings):
    """Application settings"""
    environment: str = "local"
    log_level: str = "INFO"
    debug: bool = False
    
    # Vector store settings
    vector_store_url: str = "http://vector-store:8000"
    vector_store_timeout: int = 30
    
    # Introspect service settings
    introspect_url: str = "http://introspect:8000"
    introspect_timeout: int = 30
    
    # Search settings
    default_top_k: int = 10
    semantic_weight: float = 0.7
    lexical_weight: float = 0.3
    min_score_threshold: float = 0.1
    
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
# HYBRID SEARCH ENGINE
# =============================================================================

class HybridSearchEngine:
    """Engine for combining semantic and lexical search"""
    
    def __init__(self, vector_store_client: ServiceClient, introspect_client: ServiceClient, settings: Settings):
        self.vector_store = vector_store_client
        self.introspect = introspect_client
        self.settings = settings
    
    async def search_tables(self, input_data: SearchTablesInput) -> SearchTablesOutput:
        """Search for relevant tables using hybrid approach"""
        try:
            connection_id = input_data.connection_id
            query = input_data.query
            top_k = input_data.top_k or self.settings.default_top_k
            
            # Get schema information for lexical search
            schema_result = await self.introspect.call_mcp_tool("introspect_database", {
                "run_envelope": input_data.run_envelope,
                "connection_id": connection_id
            })
            
            # Perform semantic search
            semantic_results = await self._semantic_search_tables(connection_id, query, top_k * 2)
            
            # Perform lexical search
            lexical_results = await self._lexical_search_tables(schema_result, query, top_k * 2)
            
            # Combine and rank results
            combined_results = await self._combine_search_results(
                semantic_results, lexical_results, query, top_k
            )
            
            return SearchTablesOutput(
                connection_id=connection_id,
                query=query,
                results=combined_results,
                total_found=len(combined_results),
                search_time=datetime.utcnow().isoformat(),
                search_method="hybrid",
                semantic_weight=self.settings.semantic_weight,
                lexical_weight=self.settings.lexical_weight
            )
            
        except Exception as e:
            logging.error(f"Table search failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def search_columns(self, input_data: SearchColumnsInput) -> SearchColumnsOutput:
        """Search for relevant columns using hybrid approach"""
        try:
            connection_id = input_data.connection_id
            query = input_data.query
            top_k = input_data.top_k or self.settings.default_top_k
            table_filter = input_data.table_filter
            
            # Get schema information for lexical search
            schema_result = await self.introspect.call_mcp_tool("introspect_database", {
                "run_envelope": input_data.run_envelope,
                "connection_id": connection_id
            })
            
            # Perform semantic search
            semantic_results = await self._semantic_search_columns(
                connection_id, query, top_k * 2, table_filter
            )
            
            # Perform lexical search
            lexical_results = await self._lexical_search_columns(
                schema_result, query, top_k * 2, table_filter
            )
            
            # Combine and rank results
            combined_results = await self._combine_column_results(
                semantic_results, lexical_results, query, top_k
            )
            
            return SearchColumnsOutput(
                connection_id=connection_id,
                query=query,
                results=combined_results,
                total_found=len(combined_results),
                search_time=datetime.utcnow().isoformat(),
                search_method="hybrid",
                semantic_weight=self.settings.semantic_weight,
                lexical_weight=self.settings.lexical_weight
            )
            
        except Exception as e:
            logging.error(f"Column search failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _semantic_search_tables(self, connection_id: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform semantic search for tables"""
        try:
            result = await self.vector_store.call_mcp_tool("search_tables", {
                "run_envelope": {"connection_id": connection_id},
                "connection_id": connection_id,
                "query": query,
                "top_k": top_k
            })
            return result.get("results", [])
        except Exception as e:
            logging.warning(f"Semantic search failed, falling back to lexical: {e}")
            return []
    
    async def _semantic_search_columns(self, connection_id: str, query: str, top_k: int, table_filter: Optional[str]) -> List[Dict[str, Any]]:
        """Perform semantic search for columns"""
        try:
            result = await self.vector_store.call_mcp_tool("search_columns", {
                "run_envelope": {"connection_id": connection_id},
                "connection_id": connection_id,
                "query": query,
                "top_k": top_k,
                "table_filter": table_filter
            })
            return result.get("results", [])
        except Exception as e:
            logging.warning(f"Semantic search failed, falling back to lexical: {e}")
            return []
    
    async def _lexical_search_tables(self, schema_result: Dict[str, Any], query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform lexical search for tables"""
        tables = schema_result.get("tables", [])
        query_lower = query.lower()
        
        results = []
        for table in tables:
            score = self._calculate_lexical_score(query_lower, table)
            if score > self.settings.min_score_threshold:
                results.append({
                    "id": table["table_id"],
                    "score": score,
                    "table": table,
                    "relevance_signals": {
                        "lexical": score,
                        "name_match": self._calculate_name_match(query_lower, table["table_name"]),
                        "description_match": self._calculate_description_match(query_lower, table.get("description", ""))
                    }
                })
        
        # Sort by score and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    async def _lexical_search_columns(self, schema_result: Dict[str, Any], query: str, top_k: int, table_filter: Optional[str]) -> List[Dict[str, Any]]:
        """Perform lexical search for columns"""
        columns = schema_result.get("columns", [])
        query_lower = query.lower()
        
        # Filter by table if specified
        if table_filter:
            columns = [col for col in columns if col["table_id"] == table_filter]
        
        results = []
        for column in columns:
            score = self._calculate_lexical_score(query_lower, column)
            if score > self.settings.min_score_threshold:
                results.append({
                    "id": column["column_id"],
                    "score": score,
                    "column": column,
                    "relevance_signals": {
                        "lexical": score,
                        "name_match": self._calculate_name_match(query_lower, column["column_name"]),
                        "type_match": self._calculate_type_match(query_lower, column["data_type"])
                    }
                })
        
        # Sort by score and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    async def _combine_search_results(self, semantic_results: List[Dict], lexical_results: List[Dict], 
                                    query: str, top_k: int) -> List[Dict]:
        """Combine semantic and lexical search results"""
        # Create a map of results by ID
        combined_map = {}
        
        # Add semantic results
        for result in semantic_results:
            result_id = result["id"]
            combined_map[result_id] = {
                **result,
                "semantic_score": result["score"],
                "lexical_score": 0.0,
                "combined_score": result["score"] * self.settings.semantic_weight
            }
        
        # Add lexical results
        for result in lexical_results:
            result_id = result["id"]
            if result_id in combined_map:
                # Update existing result
                combined_map[result_id]["lexical_score"] = result["score"]
                combined_map[result_id]["combined_score"] += result["score"] * self.settings.lexical_weight
            else:
                # Add new result
                combined_map[result_id] = {
                    **result,
                    "semantic_score": 0.0,
                    "lexical_score": result["score"],
                    "combined_score": result["score"] * self.settings.lexical_weight
                }
        
        # Convert to list and sort by combined score
        combined_list = list(combined_map.values())
        combined_list.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Update final scores
        for result in combined_list:
            result["score"] = result["combined_score"]
        
        return combined_list[:top_k]
    
    async def _combine_column_results(self, semantic_results: List[Dict], lexical_results: List[Dict], 
                                    query: str, top_k: int) -> List[Dict]:
        """Combine semantic and lexical search results for columns"""
        # Same logic as table results
        return await self._combine_search_results(semantic_results, lexical_results, query, top_k)
    
    def _calculate_lexical_score(self, query: str, item: Dict[str, Any]) -> float:
        """Calculate lexical search score for an item"""
        name = item.get("table_name", item.get("column_name", ""))
        description = item.get("description", "")
        
        name_score = self._calculate_name_match(query, name)
        desc_score = self._calculate_description_match(query, description)
        
        # Weight name matches higher than description matches
        return (name_score * 0.7) + (desc_score * 0.3)
    
    def _calculate_name_match(self, query: str, name: str) -> float:
        """Calculate name matching score"""
        if not name:
            return 0.0
        
        name_lower = name.lower()
        
        # Exact match
        if query == name_lower:
            return 1.0
        
        # Contains match
        if query in name_lower or name_lower in query:
            return 0.8
        
        # Word boundary match
        query_words = set(query.split())
        name_words = set(name_lower.split())
        if query_words & name_words:
            return 0.6
        
        # Partial word match
        for query_word in query_words:
            for name_word in name_words:
                if query_word in name_word or name_word in query_word:
                    return 0.4
        
        return 0.0
    
    def _calculate_description_match(self, query: str, description: str) -> float:
        """Calculate description matching score"""
        if not description:
            return 0.0
        
        desc_lower = description.lower()
        
        # Contains match
        if query in desc_lower:
            return 0.7
        
        # Word overlap
        query_words = set(query.split())
        desc_words = set(desc_lower.split())
        if query_words & desc_words:
            return 0.5
        
        # Partial word match
        for query_word in query_words:
            for desc_word in desc_words:
                if query_word in desc_word or desc_word in query_word:
                    return 0.3
        
        return 0.0
    
    def _calculate_type_match(self, query: str, data_type: str) -> float:
        """Calculate data type matching score"""
        if not data_type:
            return 0.0
        
        type_lower = data_type.lower()
        query_lower = query.lower()
        
        # Type-specific keywords
        type_keywords = {
            "date": ["date", "time", "timestamp", "when", "created", "updated", "modified"],
            "numeric": ["number", "count", "sum", "average", "price", "amount", "quantity", "total"],
            "text": ["name", "description", "title", "comment", "text", "string"],
            "boolean": ["flag", "active", "enabled", "status", "boolean", "is_"]
        }
        
        for type_name, keywords in type_keywords.items():
            if type_name in type_lower:
                for keyword in keywords:
                    if keyword in query_lower:
                        return 0.6
        
        return 0.0

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="AskData Table Retriever Service",
    description="Provides hybrid search for tables and columns",
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
vector_store_client = ServiceClient(settings.vector_store_url, settings.vector_store_timeout)
introspect_client = ServiceClient(settings.introspect_url, settings.introspect_timeout)
search_engine = HybridSearchEngine(vector_store_client, introspect_client, settings)

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup"""
    setup_logging()
    logging.info("AskData Table Retriever Service starting up")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    await vector_store_client.close()
    await introspect_client.close()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "table-retriever",
        "vector_store_connected": True,
        "introspect_connected": True,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/mcp/search_tables", response_model=SearchTablesOutput)
async def search_tables_mcp(request: SearchTablesInput):
    """MCP tool: Search for relevant tables using hybrid approach"""
    return await search_engine.search_tables(request)

@app.post("/mcp/search_columns", response_model=SearchColumnsOutput)
async def search_columns_mcp(request: SearchColumnsInput):
    """MCP tool: Search for relevant columns using hybrid approach"""
    return await search_engine.search_columns(request)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"service": "table-retriever", "status": "running"}

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
