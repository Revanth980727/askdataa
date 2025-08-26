"""
Indexer Service - Vector Indexing for Database Metadata

This service is responsible for:
1. Converting database metadata to vector embeddings
2. Storing embeddings in ChromaDB/OpenSearch
3. Building semantic search indexes
4. Managing index versions and schema drift
5. Providing similarity search capabilities
"""

import asyncio
import json
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings
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

class IndexRequest(BaseModel):
    """Request to index database metadata"""
    connection_id: str
    tenant_id: str
    user_id: str
    metadata: Dict[str, Any]  # Schema metadata from introspect service
    embedding_model: str = "text-embedding-ada-002"
    force_reindex: bool = False
    schema_revision: str = ""

class IndexResponse(BaseModel):
    """Response from indexing operation"""
    success: bool
    index_id: Optional[str] = None
    total_vectors: int = 0
    index_size_mb: float = 0.0
    processing_time_ms: float = 0.0
    error: Optional[str] = None
    warnings: List[str] = []

class SearchRequest(BaseModel):
    """Request to search the index"""
    connection_id: str
    tenant_id: str
    user_id: str
    query: str
    top_k: int = 10
    similarity_threshold: float = 0.7
    filters: Optional[Dict[str, Any]] = None

class SearchResult(BaseModel):
    """Individual search result"""
    table_name: str
    schema_name: str
    full_name: str
    similarity_score: float
    business_domain: Optional[str] = None
    column_matches: List[Dict[str, Any]] = []
    table_summary: str = ""

class SearchResponse(BaseModel):
    """Response from search operation"""
    success: bool
    results: List[SearchResult] = []
    total_results: int = 0
    search_time_ms: float = 0.0
    error: Optional[str] = None

class IndexMetadata(BaseModel):
    """Metadata about an index"""
    index_id: str
    connection_id: str
    tenant_id: str
    schema_revision: str
    embedding_model: str
    total_vectors: int
    index_size_mb: float
    created_at: datetime
    last_updated: datetime
    status: str  # active, building, error

# ============================================================================
# Vector Store Interface
# ============================================================================

class VectorStore:
    """Abstract interface for vector stores"""
    
    async def create_collection(self, name: str, metadata: Dict[str, Any]) -> bool:
        """Create a new collection"""
        raise NotImplementedError
    
    async def add_documents(self, collection_name: str, documents: List[str], 
                           metadatas: List[Dict[str, Any]], ids: List[str]) -> bool:
        """Add documents to collection"""
        raise NotImplementedError
    
    async def search(self, collection_name: str, query: str, top_k: int, 
                    filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        raise NotImplementedError
    
    async def delete_collection(self, name: str) -> bool:
        """Delete a collection"""
        raise NotImplementedError
    
    async def get_collection_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get collection information"""
        raise NotImplementedError

class ChromaDBStore(VectorStore):
    """ChromaDB implementation of vector store"""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.client = chromadb.HttpClient(
            host=host,
            port=port,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
    
    async def create_collection(self, name: str, metadata: Dict[str, Any]) -> bool:
        """Create a new collection"""
        try:
            # Check if collection exists
            try:
                collection = self.client.get_collection(name)
                logger.info(f"Collection {name} already exists")
                return True
            except:
                pass
            
            # Create new collection
            collection = self.client.create_collection(
                name=name,
                metadata=metadata
            )
            logger.info(f"Created collection {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection {name}: {e}")
            return False
    
    async def add_documents(self, collection_name: str, documents: List[str], 
                           metadatas: List[Dict[str, Any]], ids: List[str]) -> bool:
        """Add documents to collection"""
        try:
            collection = self.client.get_collection(collection_name)
            
            # Add documents in batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_metas = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                
                collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
            
            logger.info(f"Added {len(documents)} documents to collection {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add documents to collection {collection_name}: {e}")
            return False
    
    async def search(self, collection_name: str, query: str, top_k: int, 
                    filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            collection = self.client.get_collection(collection_name)
            
            # Convert filters to ChromaDB format
            where = None
            if filters:
                where = {}
                for key, value in filters.items():
                    if isinstance(value, list):
                        where[key] = {"$in": value}
                    else:
                        where[key] = value
            
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        'document': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                        'distance': results['distances'][0][i] if results['distances'] and results['distances'][0] else 0.0,
                        'id': results['ids'][0][i] if results['ids'] and results['ids'][0] else ""
                    })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Search failed for collection {collection_name}: {e}")
            return []
    
    async def delete_collection(self, name: str) -> bool:
        """Delete a collection"""
        try:
            self.client.delete_collection(name)
            logger.info(f"Deleted collection {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection {name}: {e}")
            return False
    
    async def get_collection_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get collection information"""
        try:
            collection = self.client.get_collection(name)
            count = collection.count()
            return {
                "name": name,
                "count": count,
                "metadata": collection.metadata
            }
        except Exception as e:
            logger.error(f"Failed to get collection info for {name}: {e}")
            return None

# ============================================================================
# Document Processing
# ============================================================================

class DocumentProcessor:
    """Processes database metadata into searchable documents"""
    
    @staticmethod
    def create_table_document(table_profile: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Create a document for a table"""
        table_name = table_profile.get('name', '')
        schema_name = table_profile.get('schema_name', '')
        full_name = table_profile.get('full_name', '')
        
        # Create document text
        document_parts = [
            f"Table: {full_name}",
            f"Business domain: {table_profile.get('business_domain', 'unknown')}",
            f"Comment: {table_profile.get('comment', 'No description')}",
            f"Size: {table_profile.get('size_mb', 0)} MB",
            f"Estimated rows: {table_profile.get('estimated_rows', 0)}"
        ]
        
        # Add column information
        columns = table_profile.get('columns', [])
        if columns:
            document_parts.append("Columns:")
            for col in columns:
                col_info = [
                    f"- {col.get('name', '')} ({col.get('data_type', '')})",
                    f"  Type: {col.get('column_type', 'unknown')}",
                    f"  Nullable: {col.get('is_nullable', False)}",
                    f"  Primary key: {col.get('is_primary_key', False)}",
                    f"  Foreign key: {col.get('is_foreign_key', False)}"
                ]
                
                if col.get('semantic_type'):
                    col_info.append(f"  Semantic type: {col.get('semantic_type')}")
                
                if col.get('comment'):
                    col_info.append(f"  Comment: {col.get('comment')}")
                
                document_parts.extend(col_info)
        
        # Add relationships
        primary_keys = table_profile.get('primary_keys', [])
        if primary_keys:
            document_parts.append(f"Primary keys: {', '.join(primary_keys)}")
        
        foreign_keys = table_profile.get('foreign_keys', [])
        if foreign_keys:
            fk_info = []
            for fk in foreign_keys:
                fk_info.append(f"{fk.get('column', '')} -> {fk.get('references', '')}")
            document_parts.append(f"Foreign keys: {', '.join(fk_info)}")
        
        document_text = "\n".join(document_parts)
        
        # Create metadata
        metadata = {
            "type": "table",
            "table_name": table_name,
            "schema_name": schema_name,
            "full_name": full_name,
            "business_domain": table_profile.get('business_domain'),
            "size_mb": table_profile.get('size_mb', 0),
            "estimated_rows": table_profile.get('estimated_rows', 0),
            "column_count": len(columns),
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys
        }
        
        return document_text, metadata
    
    @staticmethod
    def create_column_documents(table_profile: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """Create documents for individual columns"""
        documents = []
        table_name = table_profile.get('name', '')
        schema_name = table_profile.get('schema_name', '')
        full_name = table_profile.get('full_name', '')
        
        columns = table_profile.get('columns', [])
        for col in columns:
            col_name = col.get('name', '')
            
            # Create document text
            document_parts = [
                f"Column: {full_name}.{col_name}",
                f"Data type: {col.get('data_type', '')}",
                f"Column type: {col.get('column_type', 'unknown')}",
                f"Nullable: {col.get('is_nullable', False)}",
                f"Primary key: {col.get('is_primary_key', False)}",
                f"Foreign key: {col.get('is_foreign_key', False)}"
            ]
            
            if col.get('semantic_type'):
                document_parts.append(f"Semantic type: {col.get('semantic_type')}")
            
            if col.get('comment'):
                document_parts.append(f"Comment: {col.get('comment')}")
            
            if col.get('default_value'):
                document_parts.append(f"Default: {col.get('default_value')}")
            
            # Add statistics if available
            if col.get('total_rows', 0) > 0:
                document_parts.append(f"Total rows: {col.get('total_rows')}")
                document_parts.append(f"Null count: {col.get('null_count')}")
                document_parts.append(f"Distinct count: {col.get('distinct_count')}")
            
            document_text = "\n".join(document_parts)
            
            # Create metadata
            metadata = {
                "type": "column",
                "table_name": table_name,
                "schema_name": schema_name,
                "full_name": full_name,
                "column_name": col_name,
                "data_type": col.get('data_type', ''),
                "column_type": col.get('column_type', 'unknown'),
                "semantic_type": col.get('semantic_type'),
                "is_nullable": col.get('is_nullable', False),
                "is_primary_key": col.get('is_primary_key', False),
                "is_foreign_key": col.get('is_foreign_key', False),
                "business_domain": table_profile.get('business_domain')
            }
            
            documents.append((document_text, metadata))
        
        return documents

# ============================================================================
# Main Indexer Service
# ============================================================================

class IndexerService:
    """Main indexing service"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.document_processor = DocumentProcessor()
        self.indexes: Dict[str, IndexMetadata] = {}
    
    def _generate_index_id(self, connection_id: str, tenant_id: str) -> str:
        """Generate unique index ID"""
        base = f"{tenant_id}_{connection_id}"
        return hashlib.md5(base.encode()).hexdigest()[:12]
    
    async def index_database(self, request: IndexRequest) -> IndexResponse:
        """Index database metadata"""
        start_time = datetime.now()
        
        try:
            # Generate index ID
            index_id = self._generate_index_id(request.connection_id, request.tenant_id)
            collection_name = f"index_{index_id}"
            
            # Check if index exists and force_reindex is False
            if not request.force_reindex:
                existing_info = await self.vector_store.get_collection_info(collection_name)
                if existing_info:
                    return IndexResponse(
                        success=True,
                        index_id=index_id,
                        total_vectors=existing_info["count"],
                        index_size_mb=0.0,  # Would need to calculate
                        processing_time_ms=0.0
                    )
            
            # Create collection
            collection_metadata = {
                "connection_id": request.connection_id,
                "tenant_id": request.tenant_id,
                "user_id": request.user_id,
                "schema_revision": request.schema_revision,
                "embedding_model": request.embedding_model,
                "created_at": datetime.now().isoformat()
            }
            
            if not await self.vector_store.create_collection(collection_name, collection_metadata):
                raise Exception("Failed to create collection")
            
            # Process metadata into documents
            documents = []
            metadatas = []
            ids = []
            
            schemas = request.metadata.get('schemas', {})
            total_vectors = 0
            
            for schema_name, tables in schemas.items():
                for table in tables:
                    # Create table document
                    table_doc, table_meta = self.document_processor.create_table_document(table)
                    table_id = f"table_{index_id}_{schema_name}_{table['name']}"
                    
                    documents.append(table_doc)
                    metadatas.append(table_meta)
                    ids.append(table_id)
                    total_vectors += 1
                    
                    # Create column documents
                    column_docs = self.document_processor.create_column_documents(table)
                    for i, (col_doc, col_meta) in enumerate(column_docs):
                        col_id = f"column_{index_id}_{schema_name}_{table['name']}_{i}"
                        
                        documents.append(col_doc)
                        metadatas.append(col_meta)
                        ids.append(col_id)
                        total_vectors += 1
            
            # Add documents to vector store
            if documents:
                if not await self.vector_store.add_documents(collection_name, documents, metadatas, ids):
                    raise Exception("Failed to add documents to vector store")
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Store index metadata
            self.indexes[index_id] = IndexMetadata(
                index_id=index_id,
                connection_id=request.connection_id,
                tenant_id=request.tenant_id,
                schema_revision=request.schema_revision,
                embedding_model=request.embedding_model,
                total_vectors=total_vectors,
                index_size_mb=0.0,  # Would need to calculate
                created_at=datetime.now(),
                last_updated=datetime.now(),
                status="active"
            )
            
            return IndexResponse(
                success=True,
                index_id=index_id,
                total_vectors=total_vectors,
                index_size_mb=0.0,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            return IndexResponse(
                success=False,
                error=str(e)
            )
    
    async def search_index(self, request: SearchRequest) -> SearchResponse:
        """Search the index for relevant tables/columns"""
        start_time = datetime.now()
        
        try:
            # Generate index ID
            index_id = self._generate_index_id(request.connection_id, request.tenant_id)
            collection_name = f"index_{index_id}"
            
            # Search vector store
            search_results = await self.vector_store.search(
                collection_name=collection_name,
                query=request.query,
                top_k=request.top_k,
                filters=request.filters
            )
            
            # Convert to search results
            results = []
            for result in search_results:
                metadata = result.get('metadata', {})
                similarity_score = 1.0 - result.get('distance', 0.0)  # Convert distance to similarity
                
                if similarity_score >= request.similarity_threshold:
                    search_result = SearchResult(
                        table_name=metadata.get('table_name', ''),
                        schema_name=metadata.get('schema_name', ''),
                        full_name=metadata.get('full_name', ''),
                        similarity_score=similarity_score,
                        business_domain=metadata.get('business_domain'),
                        table_summary=result.get('document', '')[:200] + "..." if len(result.get('document', '')) > 200 else result.get('document', '')
                    )
                    
                    # Add column matches for table results
                    if metadata.get('type') == 'table':
                        # Find related column results
                        for col_result in search_results:
                            col_meta = col_result.get('metadata', {})
                            if (col_meta.get('table_name') == metadata.get('table_name') and 
                                col_meta.get('schema_name') == metadata.get('schema_name')):
                                search_result.column_matches.append({
                                    'name': col_meta.get('column_name', ''),
                                    'data_type': col_meta.get('data_type', ''),
                                    'semantic_type': col_meta.get('semantic_type'),
                                    'similarity_score': 1.0 - col_result.get('distance', 0.0)
                                })
                    
                    results.append(search_result)
            
            # Calculate search time
            search_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return SearchResponse(
                success=True,
                results=results,
                total_results=len(results),
                search_time_ms=search_time
            )
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return SearchResponse(
                success=False,
                error=str(e)
            )
    
    async def delete_index(self, connection_id: str, tenant_id: str) -> bool:
        """Delete an index"""
        try:
            index_id = self._generate_index_id(connection_id, tenant_id)
            collection_name = f"index_{index_id}"
            
            if await self.vector_store.delete_collection(collection_name):
                if index_id in self.indexes:
                    del self.indexes[index_id]
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete index: {e}")
            return False

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="AskData Indexer Service",
    description="Vector indexing and semantic search for database metadata",
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

# Initialize services
vector_store = ChromaDBStore(host="chromadb", port=8000)
indexer_service = IndexerService(vector_store)

@app.post("/index", response_model=IndexResponse)
async def index_database(request: IndexRequest):
    """Index database metadata"""
    return await indexer_service.index_database(request)

@app.post("/search", response_model=SearchResponse)
async def search_index(request: SearchRequest):
    """Search the index"""
    return await indexer_service.search_index(request)

@app.delete("/index/{connection_id}")
async def delete_index(connection_id: str, tenant_id: str = None):
    """Delete an index"""
    if not tenant_id:
        raise HTTPException(status_code=400, detail="tenant_id is required")
    success = await indexer_service.delete_index(connection_id, tenant_id)
    if success:
        return {"success": True, "message": f"Index for {connection_id} deleted"}
    else:
        raise HTTPException(status_code=500, detail="Failed to delete index")

@app.get("/indexes/{connection_id}")
async def get_index_info(connection_id: str, tenant_id: str = None):
    """Get index information"""
    if not tenant_id:
        raise HTTPException(status_code=400, detail="tenant_id is required")
    index_id = indexer_service._generate_index_id(connection_id, tenant_id)
    collection_name = f"index_{index_id}"
    
    info = await vector_store.get_collection_info(collection_name)
    if info:
        return info
    else:
        raise HTTPException(status_code=404, detail="Index not found")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "indexer-service"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AskData Indexer Service",
        "version": "1.0.0",
        "endpoints": {
            "index": "/index",
            "search": "/search",
            "delete_index": "/index/{connection_id}",
            "get_index": "/indexes/{connection_id}",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
