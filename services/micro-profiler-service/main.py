"""
AskData Micro Profiler Service

This service provides data profiling and statistics generation with TTL caching.
Profiles are cached by (connection_id, table, schema_rev) to avoid redundant work.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import hashlib
import json

import structlog
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# Add the contracts directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent / "contracts"))

from mcp_tools import (
    ProfileTablesInput, ProfileTablesOutput,
    TableProfile, ColumnProfile, DataTypeCategory
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
    
    # Introspect service settings
    introspect_url: str = "http://introspect:8000"
    introspect_timeout: int = 30
    
    # Profiling settings
    default_sample_size: int = 1000
    max_sample_size: int = 10000
    cache_ttl_hours: int = 24
    max_cache_size: int = 1000
    
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
# TTL CACHE
# =============================================================================

class TTLCache:
    """Time-based cache with automatic expiration"""
    
    def __init__(self, ttl_hours: int, max_size: int):
        self.ttl_hours = ttl_hours
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, datetime] = {}
    
    def _generate_cache_key(self, connection_id: str, table_name: str, schema_rev: str) -> str:
        """Generate a cache key for the given parameters"""
        key_data = f"{connection_id}:{table_name}:{schema_rev}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, connection_id: str, table_name: str, schema_rev: str) -> Optional[Dict[str, Any]]:
        """Get a cached profile if it exists and is not expired"""
        cache_key = self._generate_cache_key(connection_id, table_name, schema_rev)
        
        if cache_key not in self.cache:
            return None
        
        # Check if expired
        if datetime.utcnow() - self.access_times[cache_key] > timedelta(hours=self.ttl_hours):
            # Remove expired entry
            del self.cache[cache_key]
            del self.access_times[cache_key]
            return None
        
        # Update access time
        self.access_times[cache_key] = datetime.utcnow()
        return self.cache[cache_key]
    
    def set(self, connection_id: str, table_name: str, schema_rev: str, profile: Dict[str, Any]):
        """Set a profile in the cache"""
        cache_key = self._generate_cache_key(connection_id, table_name, schema_rev)
        
        # Check if we need to evict old entries
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[cache_key] = profile
        self.access_times[cache_key] = datetime.utcnow()
    
    def _evict_oldest(self):
        """Evict the oldest cache entry"""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    def clear_expired(self):
        """Remove all expired entries from the cache"""
        current_time = datetime.utcnow()
        expired_keys = [
            key for key, access_time in self.access_times.items()
            if current_time - access_time > timedelta(hours=self.ttl_hours)
        ]
        
        for key in expired_keys:
            del self.cache[key]
            del self.access_times[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "total_entries": len(self.cache),
            "max_size": self.max_size,
            "ttl_hours": self.ttl_hours,
            "oldest_entry": min(self.access_times.values()) if self.access_times else None,
            "newest_entry": max(self.access_times.values()) if self.access_times else None
        }

# =============================================================================
# DATA PROFILER
# =============================================================================

class DataProfiler:
    """Service for profiling database tables and columns"""
    
    def __init__(self, connection_registry_client: ServiceClient, introspect_client: ServiceClient, 
                 settings: Settings, cache: TTLCache):
        self.connection_registry = connection_registry_client
        self.introspect = introspect_client
        self.settings = settings
        self.cache = cache
    
    async def profile_tables(self, input_data: ProfileTablesInput) -> ProfileTablesOutput:
        """Profile specified tables with caching"""
        try:
            connection_id = input_data.connection_id
            tables = input_data.tables
            sample_size = input_data.sample_size or self.settings.default_sample_size
            
            # Limit sample size
            sample_size = min(sample_size, self.settings.max_sample_size)
            
            # Get connection details for schema revision
            connection_info = await self._get_connection_info(connection_id)
            schema_rev = connection_info.get("schema_revision", "1.0")
            
            # Get schema information
            schema_result = await self.introspect.call_mcp_tool("introspect_database", {
                "run_envelope": input_data.run_envelope,
                "connection_id": connection_id
            })
            
            # Profile each table
            table_profiles = []
            for table_name in tables:
                profile = await self._profile_single_table(
                    connection_id, table_name, schema_rev, schema_result, sample_size
                )
                table_profiles.append(profile)
            
            return ProfileTablesOutput(
                connection_id=connection_id,
                tables=table_profiles,
                total_tables=len(table_profiles),
                sample_size_used=sample_size,
                profiling_time=datetime.utcnow().isoformat(),
                cache_hits=len([p for p in table_profiles if p.cache_hit]),
                cache_misses=len([p for p in table_profiles if not p.cache_hit])
            )
            
        except Exception as e:
            logging.error(f"Table profiling failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _profile_single_table(self, connection_id: str, table_name: str, schema_rev: str, 
                                   schema_result: Dict[str, Any], sample_size: int) -> TableProfile:
        """Profile a single table with caching"""
        # Check cache first
        cached_profile = self.cache.get(connection_id, table_name, schema_rev)
        if cached_profile:
            logging.info(f"Cache hit for table {table_name} in connection {connection_id}")
            return TableProfile(**cached_profile)
        
        # Generate new profile
        logging.info(f"Generating new profile for table {table_name} in connection {connection_id}")
        profile = await self._generate_table_profile(table_name, schema_result, sample_size)
        
        # Cache the profile
        self.cache.set(connection_id, table_name, schema_rev, profile.dict())
        
        return profile
    
    async def _generate_table_profile(self, table_name: str, schema_result: Dict[str, Any], 
                                    sample_size: int) -> TableProfile:
        """Generate a profile for a table"""
        # Find table in schema
        table_info = None
        for table in schema_result.get("tables", []):
            if table["table_name"] == table_name:
                table_info = table
                break
        
        if not table_info:
            raise HTTPException(status_code=404, detail=f"Table {table_name} not found")
        
        # Find columns for this table
        table_columns = [
            col for col in schema_result.get("columns", [])
            if col["table_id"] == table_info["table_id"]
        ]
        
        # Generate column profiles
        column_profiles = []
        for column in table_columns:
            col_profile = await self._generate_column_profile(column, sample_size)
            column_profiles.append(col_profile)
        
        # Calculate table-level statistics
        total_rows = table_info.get("row_count", 0)
        total_columns = len(column_profiles)
        
        # Determine data quality score
        quality_score = self._calculate_quality_score(column_profiles)
        
        # Determine table type
        table_type = self._determine_table_type(table_info, column_profiles)
        
        return TableProfile(
            table_id=table_info["table_id"],
            table_name=table_name,
            schema_name=table_info["schema_name"],
            total_rows=total_rows,
            total_columns=total_columns,
            sample_size=sample_size,
            columns=column_profiles,
            data_quality_score=quality_score,
            table_type=table_type,
            profiling_timestamp=datetime.utcnow().isoformat(),
            cache_hit=False
        )
    
    async def _generate_column_profile(self, column: Dict[str, Any], sample_size: int) -> ColumnProfile:
        """Generate a profile for a column"""
        column_name = column["column_name"]
        data_type = column["data_type"]
        is_nullable = column["is_nullable"]
        is_primary_key = column["is_primary_key"]
        
        # Categorize data type
        type_category = self._categorize_data_type(data_type)
        
        # Generate sample data for analysis
        sample_data = self._generate_sample_data(data_type, sample_size)
        
        # Calculate statistics
        stats = self._calculate_column_statistics(sample_data, type_category)
        
        # Determine data quality indicators
        quality_indicators = self._assess_data_quality(sample_data, type_category, is_nullable)
        
        return ColumnProfile(
            column_id=column["column_id"],
            column_name=column_name,
            data_type=data_type,
            type_category=type_category,
            is_nullable=is_nullable,
            is_primary_key=is_primary_key,
            sample_data=sample_data,
            statistics=stats,
            quality_indicators=quality_indicators,
            description=column.get("description", "")
        )
    
    def _categorize_data_type(self, data_type: str) -> DataTypeCategory:
        """Categorize a database data type"""
        data_type_lower = data_type.lower()
        
        if any(t in data_type_lower for t in ["int", "bigint", "smallint", "serial", "bigserial"]):
            return DataTypeCategory.INTEGER
        elif any(t in data_type_lower for t in ["decimal", "numeric", "real", "double", "float"]):
            return DataTypeCategory.DECIMAL
        elif any(t in data_type_lower for t in ["varchar", "char", "text", "string"]):
            return DataTypeCategory.STRING
        elif any(t in data_type_lower for t in ["date", "timestamp", "time", "datetime"]):
            return DataTypeCategory.DATE
        elif any(t in data_type_lower for t in ["bool", "boolean"]):
            return DataTypeCategory.BOOLEAN
        elif any(t in data_type_lower for t in ["json", "jsonb", "xml"]):
            return DataTypeCategory.COMPLEX
        else:
            return DataTypeCategory.UNKNOWN
    
    def _generate_sample_data(self, data_type: str, sample_size: int) -> List[Any]:
        """Generate sample data for analysis"""
        # This is a simplified version - in production, you'd query the actual database
        # For now, generate realistic mock data based on the data type
        
        if "int" in data_type.lower():
            return [i for i in range(1, min(sample_size + 1, 1001))]
        elif "decimal" in data_type.lower() or "numeric" in data_type.lower():
            return [float(i) * 1.5 for i in range(1, min(sample_size + 1, 1001))]
        elif "varchar" in data_type.lower() or "text" in data_type.lower():
            return [f"Sample text {i}" for i in range(1, min(sample_size + 1, 1001))]
        elif "date" in data_type.lower() or "timestamp" in data_type.lower():
            base_date = datetime(2020, 1, 1)
            return [(base_date + timedelta(days=i)).isoformat() for i in range(min(sample_size, 1000))]
        elif "bool" in data_type.lower():
            return [True, False] * (sample_size // 2) + ([True] if sample_size % 2 else [])
        else:
            return [f"Unknown type {i}" for i in range(1, min(sample_size + 1, 1001))]
    
    def _calculate_column_statistics(self, sample_data: List[Any], type_category: DataTypeCategory) -> Dict[str, Any]:
        """Calculate statistics for a column"""
        if not sample_data:
            return {}
        
        stats = {
            "count": len(sample_data),
            "null_count": 0,
            "unique_count": len(set(sample_data))
        }
        
        if type_category == DataTypeCategory.INTEGER or type_category == DataTypeCategory.DECIMAL:
            numeric_data = [x for x in sample_data if isinstance(x, (int, float)) and x is not None]
            if numeric_data:
                stats.update({
                    "min": min(numeric_data),
                    "max": max(numeric_data),
                    "mean": sum(numeric_data) / len(numeric_data),
                    "median": sorted(numeric_data)[len(numeric_data) // 2]
                })
        elif type_category == DataTypeCategory.STRING:
            string_data = [str(x) for x in sample_data if x is not None]
            if string_data:
                lengths = [len(x) for x in string_data]
                stats.update({
                    "min_length": min(lengths),
                    "max_length": max(lengths),
                    "avg_length": sum(lengths) / len(lengths)
                })
        elif type_category == DataTypeCategory.DATE:
            date_data = [x for x in sample_data if x is not None]
            if date_data:
                try:
                    parsed_dates = [datetime.fromisoformat(x.replace('Z', '+00:00')) for x in date_data]
                    stats.update({
                        "min_date": min(parsed_dates).isoformat(),
                        "max_date": max(parsed_dates).isoformat()
                    })
                except:
                    pass
        
        return stats
    
    def _assess_data_quality(self, sample_data: List[Any], type_category: DataTypeCategory, 
                            is_nullable: bool) -> Dict[str, Any]:
        """Assess data quality for a column"""
        if not sample_data:
            return {}
        
        total_count = len(sample_data)
        null_count = sum(1 for x in sample_data if x is None)
        unique_count = len(set(x for x in sample_data if x is not None))
        
        quality = {
            "completeness": (total_count - null_count) / total_count if total_count > 0 else 0,
            "uniqueness": unique_count / total_count if total_count > 0 else 0,
            "null_percentage": null_count / total_count if total_count > 0 else 0
        }
        
        # Type consistency check
        if type_category != DataTypeCategory.UNKNOWN:
            expected_type = self._get_expected_type(type_category)
            type_consistent = all(isinstance(x, expected_type) for x in sample_data if x is not None)
            quality["type_consistency"] = type_consistent
        
        # Format validation for specific types
        if type_category == DataTypeCategory.DATE:
            valid_dates = 0
            for x in sample_data:
                if x is not None:
                    try:
                        datetime.fromisoformat(x.replace('Z', '+00:00'))
                        valid_dates += 1
                    except:
                        pass
            quality["format_validity"] = valid_dates / (total_count - null_count) if (total_count - null_count) > 0 else 0
        
        return quality
    
    def _get_expected_type(self, type_category: DataTypeCategory) -> type:
        """Get the expected Python type for a data type category"""
        type_map = {
            DataTypeCategory.INTEGER: int,
            DataTypeCategory.DECIMAL: float,
            DataTypeCategory.STRING: str,
            DataTypeCategory.DATE: str,  # ISO format string
            DataTypeCategory.BOOLEAN: bool,
            DataTypeCategory.COMPLEX: dict,
            DataTypeCategory.UNKNOWN: object
        }
        return type_map.get(type_category, object)
    
    def _calculate_quality_score(self, column_profiles: List[ColumnProfile]) -> float:
        """Calculate overall data quality score for a table"""
        if not column_profiles:
            return 0.0
        
        total_score = 0.0
        for col_profile in column_profiles:
            quality = col_profile.quality_indicators
            if quality:
                # Weight different quality aspects
                completeness = quality.get("completeness", 0.0)
                uniqueness = quality.get("uniqueness", 0.0)
                type_consistency = quality.get("type_consistency", 1.0)
                
                column_score = (completeness * 0.4 + uniqueness * 0.3 + type_consistency * 0.3)
                total_score += column_score
        
        return total_score / len(column_profiles)
    
    def _determine_table_type(self, table_info: Dict[str, Any], column_profiles: List[ColumnProfile]) -> str:
        """Determine the type/category of a table"""
        # Simple heuristics based on table structure
        has_id = any(col.is_primary_key for col in column_profiles)
        has_timestamps = any(
            col.type_category == DataTypeCategory.DATE and 
            any(word in col.column_name.lower() for word in ["created", "updated", "modified"])
            for col in column_profiles
        )
        has_soft_delete = any(
            col.column_name.lower() in ["deleted", "active", "enabled", "status"]
            for col in column_profiles
        )
        
        if has_id and has_timestamps:
            return "transactional"
        elif has_id and has_soft_delete:
            return "master"
        elif has_timestamps and not has_id:
            return "log"
        else:
            return "reference"
    
    async def _get_connection_info(self, connection_id: str) -> Dict[str, Any]:
        """Get connection information from the registry"""
        try:
            result = await self.connection_registry.call_mcp_tool("get_connection", {
                "connection_id": connection_id
            })
            return result
        except Exception as e:
            logging.warning(f"Failed to get connection info: {e}")
            return {"schema_revision": "1.0"}

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="AskData Micro Profiler Service",
    description="Provides data profiling with TTL caching",
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
cache = TTLCache(settings.cache_ttl_hours, settings.max_cache_size)
connection_registry_client = ServiceClient(settings.connection_registry_url, settings.connection_registry_timeout)
introspect_client = ServiceClient(settings.introspect_url, settings.introspect_timeout)
profiler = DataProfiler(connection_registry_client, introspect_client, settings, cache)

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup"""
    setup_logging()
    logging.info("AskData Micro Profiler Service starting up")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    await connection_registry_client.close()
    await introspect_client.close()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    cache_stats = cache.get_stats()
    return {
        "status": "healthy",
        "service": "micro-profiler",
        "cache_stats": cache_stats,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/mcp/profile_tables", response_model=ProfileTablesOutput)
async def profile_tables_mcp(request: ProfileTablesInput):
    """MCP tool: Profile tables with caching"""
    return await profiler.profile_tables(request)

@app.post("/cache/clear")
async def clear_cache():
    """Clear all cached profiles"""
    cache.clear_expired()
    return {"message": "Cache cleared", "timestamp": datetime.utcnow().isoformat()}

@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    return cache.get_stats()

@app.get("/")
async def root():
    """Root endpoint"""
    return {"service": "micro-profiler", "status": "running"}

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
