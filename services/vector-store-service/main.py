"""
AskData Vector Store Service

This service provides MCP endpoints for ChromaDB operations including:
- Indexing tables and columns with embeddings
- Semantic search for tables and columns
- Connection-scoped collections
- Embedding version management
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
    IndexConnectionInput, IndexConnectionOutput,
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
    
    # ChromaDB settings
    chromadb_host: str = "http://chromadb:8000"
    chromadb_timeout: int = 30
    
    # Embedding settings
    embedding_model: str = "text-embedding-ada-002"  # Default OpenAI model
    embedding_dimension: int = 1536
    embedding_version: str = "1.0"
    
    # Search settings
    default_top_k: int = 10
    hybrid_search_weight: float = 0.7  # Weight for semantic vs lexical search
    
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
# CHROMADB CLIENT
# =============================================================================

class ChromaDBClient:
    """Client for ChromaDB operations"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def create_collection(self, collection_name: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a new collection"""
        url = f"{self.base_url}/api/v1/collections"
        payload = {
            "name": collection_name,
            "metadata": metadata or {},
            "get_or_create": True
        }
        
        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Failed to create collection {collection_name}: {e}")
            raise
    
    async def add_documents(self, collection_name: str, documents: List[str], 
                           metadatas: List[Dict[str, Any]], ids: List[str]) -> Dict[str, Any]:
        """Add documents to a collection"""
        url = f"{self.base_url}/api/v1/collections/{collection_name}/add"
        payload = {
            "documents": documents,
            "metadatas": metadatas,
            "ids": ids
        }
        
        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Failed to add documents to {collection_name}: {e}")
            raise
    
    async def search(self, collection_name: str, query_texts: List[str], 
                    n_results: int = 10, where: Dict[str, Any] = None) -> Dict[str, Any]:
        """Search documents in a collection"""
        url = f"{self.base_url}/api/v1/collections/{collection_name}/query"
        payload = {
            "query_texts": query_texts,
            "n_results": n_results
        }
        if where:
            payload["where"] = where
        
        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Failed to search in {collection_name}: {e}")
            raise
    
    async def delete_collection(self, collection_name: str) -> Dict[str, Any]:
        """Delete a collection"""
        url = f"{self.base_url}/api/v1/collections/{collection_name}"
        
        try:
            response = await self.client.delete(url)
            response.raise_for_status()
            return {"status": "deleted", "collection": collection_name}
        except Exception as e:
            logging.error(f"Failed to delete collection {collection_name}: {e}")
            raise
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

# =============================================================================
# VECTOR STORE SERVICE
# =============================================================================

class VectorStoreService:
    """Service for vector store operations"""
    
    def __init__(self, chroma_client: ChromaDBClient, settings: Settings):
        self.chroma = chroma_client
        self.settings = settings
        self.collections_cache = {}  # Cache collection info
    
    def _get_collection_name(self, connection_id: str, content_type: str = "tables") -> str:
        """Generate collection name for a connection"""
        return f"{connection_id}_{content_type}_{self.settings.embedding_version}"
    
    async def index_connection(self, input_data: IndexConnectionInput) -> IndexConnectionOutput:
        """Index tables and columns for a connection"""
        try:
            connection_id = input_data.connection_id
            tables = input_data.tables
            columns = input_data.columns
            
            # Create collections
            tables_collection = self._get_collection_name(connection_id, "tables")
            columns_collection = self._get_collection_name(connection_id, "columns")
            
            # Create collections if they don't exist
            await self.chroma.create_collection(tables_collection, {
                "connection_id": connection_id,
                "content_type": "tables",
                "embedding_version": self.settings.embedding_version,
                "created_at": datetime.utcnow().isoformat()
            })
            
            await self.chroma.create_collection(columns_collection, {
                "connection_id": connection_id,
                "content_type": "columns",
                "embedding_version": self.settings.embedding_version,
                "created_at": datetime.utcnow().isoformat()
            })
            
            # Index tables
            if tables:
                table_docs = []
                table_metadatas = []
                table_ids = []
                
                for table in tables:
                    # Create document text from table info
                    doc_text = f"{table.full_name} {table.description or ''}"
                    
                    table_docs.append(doc_text)
                    table_metadatas.append({
                        "table_id": table.table_id,
                        "schema_name": table.schema_name,
                        "table_name": table.table_name,
                        "full_name": table.full_name,
                        "description": table.description,
                        "row_count": table.row_count,
                        "connection_id": connection_id,
                        "embedding_version": self.settings.embedding_version
                    })
                    table_ids.append(table.table_id)
                
                await self.chroma.add_documents(tables_collection, table_docs, table_metadatas, table_ids)
            
            # Index columns
            if columns:
                column_docs = []
                column_metadatas = []
                column_ids = []
                
                for column in columns:
                    # Create document text from column info
                    doc_text = f"{column.table_id}.{column.column_name} {column.description or ''} {column.data_type}"
                    
                    column_docs.append(doc_text)
                    column_metadatas.append({
                        "column_id": column.column_id,
                        "table_id": column.table_id,
                        "column_name": column.column_name,
                        "data_type": column.data_type,
                        "is_nullable": column.is_nullable,
                        "is_primary_key": column.is_primary_key,
                        "description": column.description,
                        "connection_id": connection_id,
                        "embedding_version": self.settings.embedding_version
                    })
                    column_ids.append(column.column_id)
                
                await self.chroma.add_documents(columns_collection, column_docs, column_metadatas, column_ids)
            
            # Cache collection info
            self.collections_cache[connection_id] = {
                "tables_collection": tables_collection,
                "columns_collection": columns_collection,
                "indexed_at": datetime.utcnow().isoformat()
            }
            
            return IndexConnectionOutput(
                connection_id=connection_id,
                status="indexed",
                message="Connection indexed successfully",
                tables_indexed=len(tables) if tables else 0,
                columns_indexed=len(columns) if columns else 0,
                collections_created=[tables_collection, columns_collection],
                embedding_version=self.settings.embedding_version,
                indexing_time=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            logging.error(f"Indexing failed for connection {input_data.connection_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def search_tables(self, input_data: SearchTablesInput) -> SearchTablesOutput:
        """Search for relevant tables"""
        try:
            connection_id = input_data.connection_id
            query = input_data.query
            top_k = input_data.top_k or self.settings.default_top_k
            
            # Get collection name
            collection_name = self._get_collection_name(connection_id, "tables")
            
            # Search in ChromaDB
            search_result = await self.chroma.search(
                collection_name=collection_name,
                query_texts=[query],
                n_results=top_k,
                where={"connection_id": connection_id}
            )
            
            # Format results
            results = []
            if "results" in search_result and search_result["results"]:
                for i, doc_id in enumerate(search_result["results"]["ids"][0]):
                    metadata = search_result["results"]["metadatas"][0][i]
                    distance = search_result["results"]["distances"][0][i] if "distances" in search_result["results"] else 0.0
                    
                    # Convert distance to similarity score (0-1)
                    similarity_score = max(0.0, 1.0 - distance)
                    
                    results.append({
                        "id": doc_id,
                        "score": similarity_score,
                        "table": {
                            "table_id": metadata["table_id"],
                            "schema_name": metadata["schema_name"],
                            "table_name": metadata["table_name"],
                            "full_name": metadata["full_name"],
                            "description": metadata["description"],
                            "row_count": metadata.get("row_count", 0)
                        },
                        "relevance_signals": {
                            "semantic": similarity_score,
                            "name_match": self._calculate_name_match(query, metadata["table_name"]),
                            "description_match": self._calculate_description_match(query, metadata.get("description", ""))
                        }
                    })
            
            return SearchTablesOutput(
                connection_id=connection_id,
                query=query,
                results=results,
                total_found=len(results),
                search_time=datetime.utcnow().isoformat(),
                collection_used=collection_name
            )
            
        except Exception as e:
            logging.error(f"Table search failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def search_columns(self, input_data: SearchColumnsInput) -> SearchColumnsOutput:
        """Search for relevant columns"""
        try:
            connection_id = input_data.connection_id
            query = input_data.query
            top_k = input_data.top_k or self.settings.default_top_k
            table_filter = input_data.table_filter
            
            # Get collection name
            collection_name = self._get_collection_name(connection_id, "columns")
            
            # Build where clause
            where_clause = {"connection_id": connection_id}
            if table_filter:
                where_clause["table_id"] = table_filter
            
            # Search in ChromaDB
            search_result = await self.chroma.search(
                collection_name=collection_name,
                query_texts=[query],
                n_results=top_k,
                where=where_clause
            )
            
            # Format results
            results = []
            if "results" in search_result and search_result["results"]:
                for i, doc_id in enumerate(search_result["results"]["ids"][0]):
                    metadata = search_result["results"]["metadatas"][0][i]
                    distance = search_result["results"]["distances"][0][i] if "distances" in search_result["results"] else 0.0
                    
                    # Convert distance to similarity score (0-1)
                    similarity_score = max(0.0, 1.0 - distance)
                    
                    results.append({
                        "id": doc_id,
                        "score": similarity_score,
                        "column": {
                            "column_id": metadata["column_id"],
                            "table_id": metadata["table_id"],
                            "column_name": metadata["column_name"],
                            "data_type": metadata["data_type"],
                            "is_nullable": metadata["is_nullable"],
                            "is_primary_key": metadata["is_primary_key"],
                            "description": metadata.get("description", "")
                        },
                        "relevance_signals": {
                            "semantic": similarity_score,
                            "name_match": self._calculate_name_match(query, metadata["column_name"]),
                            "type_match": self._calculate_type_match(query, metadata["data_type"])
                        }
                    })
            
            return SearchColumnsOutput(
                connection_id=connection_id,
                query=query,
                results=results,
                total_found=len(results),
                search_time=datetime.utcnow().isoformat(),
                collection_used=collection_name
            )
            
        except Exception as e:
            logging.error(f"Column search failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def delete_connection_index(self, connection_id: str) -> Dict[str, Any]:
        """Delete all collections for a connection"""
        try:
            # Get collection names
            tables_collection = self._get_collection_name(connection_id, "tables")
            columns_collection = self._get_collection_name(connection_id, "columns")
            
            # Delete collections
            deleted_collections = []
            
            try:
                await self.chroma.delete_collection(tables_collection)
                deleted_collections.append(tables_collection)
            except Exception as e:
                logging.warning(f"Failed to delete tables collection: {e}")
            
            try:
                await self.chroma.delete_collection(columns_collection)
                deleted_collections.append(columns_collection)
            except Exception as e:
                logging.warning(f"Failed to delete columns collection: {e}")
            
            # Remove from cache
            if connection_id in self.collections_cache:
                del self.collections_cache[connection_id]
            
            return {
                "connection_id": connection_id,
                "status": "deleted",
                "deleted_collections": deleted_collections,
                "deletion_time": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Failed to delete connection index: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _calculate_name_match(self, query: str, name: str) -> float:
        """Calculate name matching score"""
        query_lower = query.lower()
        name_lower = name.lower()
        
        # Exact match
        if query_lower == name_lower:
            return 1.0
        
        # Contains match
        if query_lower in name_lower or name_lower in query_lower:
            return 0.8
        
        # Word boundary match
        query_words = set(query_lower.split())
        name_words = set(name_lower.split())
        if query_words & name_words:
            return 0.6
        
        return 0.0
    
    def _calculate_description_match(self, query: str, description: str) -> float:
        """Calculate description matching score"""
        if not description:
            return 0.0
        
        query_lower = query.lower()
        desc_lower = description.lower()
        
        # Contains match
        if query_lower in desc_lower:
            return 0.7
        
        # Word overlap
        query_words = set(query_lower.split())
        desc_words = set(desc_lower.split())
        if query_words & desc_words:
            return 0.5
        
        return 0.0
    
    def _calculate_type_match(self, query: str, data_type: str) -> float:
        """Calculate data type matching score"""
        query_lower = query.lower()
        type_lower = data_type.lower()
        
        # Type-specific keywords
        type_keywords = {
            "date": ["date", "time", "timestamp", "when", "created", "updated"],
            "numeric": ["number", "count", "sum", "average", "price", "amount", "quantity"],
            "text": ["name", "description", "title", "comment", "text"],
            "boolean": ["flag", "active", "enabled", "status", "boolean"]
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
    title="AskData Vector Store Service",
    description="Provides MCP endpoints for ChromaDB operations",
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
chroma_client = ChromaDBClient(settings.chromadb_host, settings.chromadb_timeout)
vector_store_service = VectorStoreService(chroma_client, settings)

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup"""
    setup_logging()
    logging.info("AskData Vector Store Service starting up")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    await chroma_client.close()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "vector-store",
        "chromadb_connected": True,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/mcp/index_connection", response_model=IndexConnectionOutput)
async def index_connection_mcp(request: IndexConnectionInput):
    """MCP tool: Index connection tables and columns"""
    return await vector_store_service.index_connection(request)

@app.post("/mcp/search_tables", response_model=SearchTablesOutput)
async def search_tables_mcp(request: SearchTablesInput):
    """MCP tool: Search for relevant tables"""
    return await vector_store_service.search_tables(request)

@app.post("/mcp/search_columns", response_model=SearchColumnsOutput)
async def search_columns_mcp(request: SearchColumnsInput):
    """MCP tool: Search for relevant columns"""
    return await vector_store_service.search_columns(request)

@app.delete("/mcp/delete_connection_index/{connection_id}")
async def delete_connection_index_mcp(connection_id: str):
    """MCP tool: Delete connection index"""
    return await vector_store_service.delete_connection_index(connection_id)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"service": "vector-store", "status": "running"}

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
