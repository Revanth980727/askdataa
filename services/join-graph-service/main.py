"""
Join Graph Service - Table Relationship Analysis

This service is responsible for:
1. Analyzing table relationships and foreign key constraints
2. Building join graphs for query optimization
3. Discovering optimal join paths between tables
4. Providing join recommendations for queries
5. Supporting query planning and performance optimization
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque

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

class RelationshipType(str, Enum):
    """Types of table relationships"""
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"
    UNKNOWN = "unknown"

class JoinType(str, Enum):
    """Types of SQL joins"""
    INNER = "inner"
    LEFT = "left"
    RIGHT = "right"
    FULL = "full"
    CROSS = "cross"

class TableNode(BaseModel):
    """Node in the join graph representing a table"""
    table_name: str
    schema_name: str
    full_name: str
    
    # Table metadata
    estimated_rows: int = 0
    size_mb: float = 0.0
    business_domain: Optional[str] = None
    
    # Relationship information
    primary_keys: List[str] = []
    foreign_keys: List[Dict[str, Any]] = []
    referenced_by: List[Dict[str, Any]] = []
    
    # Graph properties
    degree: int = 0  # Number of connections
    centrality: float = 0.0  # Graph centrality score

class RelationshipEdge(BaseModel):
    """Edge in the join graph representing a relationship"""
    source_table: str
    target_table: str
    source_column: str
    target_column: str
    
    # Relationship details
    relationship_type: RelationshipType
    join_type: JoinType = JoinType.INNER
    confidence: float = 0.0
    
    # Performance characteristics
    estimated_join_cost: float = 0.0
    selectivity: float = 0.0  # Join selectivity factor
    
    # Metadata
    constraint_name: Optional[str] = None
    is_enforced: bool = True

class JoinPath(BaseModel):
    """Path between tables for joining"""
    source_table: str
    target_table: str
    path: List[RelationshipEdge] = []
    
    # Path characteristics
    total_cost: float = 0.0
    total_selectivity: float = 1.0
    path_length: int = 0
    
    # Alternative paths
    alternatives: List[List[RelationshipEdge]] = []
    
    # Recommendations
    recommended_join_type: JoinType = JoinType.INNER
    optimization_hints: List[str] = []

class JoinGraph(BaseModel):
    """Complete join graph for a database"""
    connection_id: str
    tenant_id: str
    
    # Graph structure
    nodes: List[TableNode] = []
    edges: List[RelationshipEdge] = []
    
    # Graph properties
    total_tables: int = 0
    total_relationships: int = 0
    connected_components: int = 0
    
    # Analysis results
    hub_tables: List[str] = []  # Tables with many connections
    isolated_tables: List[str] = []  # Tables with no connections
    circular_references: List[List[str]] = []
    
    # Metadata
    generated_at: datetime
    version: str = "1.0"

class GraphAnalysisRequest(BaseModel):
    """Request to analyze join graph"""
    connection_id: str
    tenant_id: str
    user_id: str
    
    # Analysis options
    include_performance_metrics: bool = True
    include_business_analysis: bool = True
    max_path_length: int = 5
    include_alternatives: bool = True

class GraphAnalysisResponse(BaseModel):
    """Response from join graph analysis"""
    success: bool
    graph: Optional[JoinGraph] = None
    analysis_time_ms: float = 0.0
    error: Optional[str] = None
    warnings: List[str] = []

class JoinPathRequest(BaseModel):
    """Request to find join paths between tables"""
    connection_id: str
    tenant_id: str
    user_id: str
    source_table: str
    target_table: str
    
    # Path finding options
    max_path_length: int = 5
    include_alternatives: bool = True
    optimize_for: str = "performance"  # performance, simplicity, selectivity

class JoinPathResponse(BaseModel):
    """Response with join paths between tables"""
    success: bool
    paths: List[JoinPath] = []
    optimal_path: Optional[JoinPath] = None
    total_paths_found: int = 0
    search_time_ms: float = 0.0
    error: Optional[str] = None

# ============================================================================
# Join Graph Builder
# ============================================================================

class JoinGraphBuilder:
    """Builds join graph from table metadata"""
    
    def __init__(self):
        self.graph = {}
        self.reverse_index = {}
    
    def build_graph(self, tables: List[Dict[str, Any]]) -> Tuple[List[TableNode], List[RelationshipEdge]]:
        """Build join graph from table metadata"""
        nodes = []
        edges = []
        
        # Create table nodes
        for table_info in tables:
            node = self._create_table_node(table_info)
            nodes.append(node)
            
            # Index by table name for quick lookup
            self.graph[table_info['name']] = node
        
        # Create relationship edges
        for table_info in tables:
            table_edges = self._create_relationship_edges(table_info, tables)
            edges.extend(table_edges)
        
        # Update node properties
        self._update_node_properties(nodes, edges)
        
        return nodes, edges
    
    def _create_table_node(self, table_info: Dict[str, Any]) -> TableNode:
        """Create a table node from metadata"""
        return TableNode(
            table_name=table_info.get('name', ''),
            schema_name=table_info.get('schema_name', ''),
            full_name=table_info.get('full_name', ''),
            estimated_rows=table_info.get('estimated_rows', 0),
            size_mb=table_info.get('size_mb', 0.0),
            business_domain=table_info.get('business_domain'),
            primary_keys=table_info.get('primary_keys', []),
            foreign_keys=table_info.get('foreign_keys', []),
            referenced_by=[]
        )
    
    def _create_relationship_edges(self, table_info: Dict[str, Any], 
                                 all_tables: List[Dict[str, Any]]) -> List[RelationshipEdge]:
        """Create relationship edges for a table"""
        edges = []
        
        # Process foreign keys
        for fk_info in table_info.get('foreign_keys', []):
            source_column = fk_info.get('column', '')
            target_info = fk_info.get('references', '')
            
            if target_info and '.' in target_info:
                target_table, target_column = target_info.split('.')
                
                # Find target table
                target_table_info = next(
                    (t for t in all_tables if t['name'] == target_table), None
                )
                
                if target_table_info:
                    edge = RelationshipEdge(
                        source_table=table_info['name'],
                        target_table=target_table,
                        source_column=source_column,
                        target_column=target_column,
                        relationship_type=self._determine_relationship_type(
                            table_info, target_table_info, source_column, target_column
                        ),
                        confidence=0.9,  # High confidence for explicit FKs
                        constraint_name=fk_info.get('constraint_name'),
                        is_enforced=True
                    )
                    
                    edges.append(edge)
                    
                    # Update reverse index
                    if target_table not in self.reverse_index:
                        self.reverse_index[target_table] = []
                    self.reverse_index[target_table].append({
                        'table': table_info['name'],
                        'column': source_column,
                        'edge': edge
                    })
        
        return edges
    
    def _determine_relationship_type(self, source_table: Dict[str, Any], 
                                   target_table: Dict[str, Any],
                                   source_column: str, target_column: str) -> RelationshipType:
        """Determine the type of relationship between tables"""
        # Check if target column is primary key
        target_pks = target_table.get('primary_keys', [])
        is_target_pk = target_column in target_pks
        
        # Check if source column is unique
        source_columns = source_table.get('columns', [])
        source_col_info = next(
            (col for col in source_columns if col['name'] == source_column), None
        )
        
        if source_col_info:
            # This is a simplified check - in practice, you'd analyze actual data
            if source_col_info.get('is_unique', False):
                if is_target_pk:
                    return RelationshipType.ONE_TO_ONE
                else:
                    return RelationshipType.ONE_TO_MANY
            else:
                if is_target_pk:
                    return RelationshipType.MANY_TO_ONE
                else:
                    return RelationshipType.MANY_TO_MANY
        
        # Default based on primary key
        if is_target_pk:
            return RelationshipType.MANY_TO_ONE
        else:
            return RelationshipType.UNKNOWN
    
    def _update_node_properties(self, nodes: List[TableNode], edges: List[RelationshipEdge]):
        """Update node properties based on edges"""
        # Count connections for each node
        connection_counts = defaultdict(int)
        for edge in edges:
            connection_counts[edge.source_table] += 1
            connection_counts[edge.target_table] += 1
        
        # Update node properties
        for node in nodes:
            node.degree = connection_counts.get(node.table_name, 0)
            
            # Find edges where this table is referenced
            node.referenced_by = [
                {'table': edge.source_table, 'column': edge.source_column}
                for edge in edges if edge.target_table == node.table_name
            ]
            
            # Calculate centrality (simplified)
            total_nodes = len(nodes)
            if total_nodes > 1:
                node.centrality = node.degree / (total_nodes - 1)
            else:
                node.centrality = 0.0

# ============================================================================
# Join Path Finder
# ============================================================================

class JoinPathFinder:
    """Finds optimal join paths between tables"""
    
    def __init__(self, nodes: List[TableNode], edges: List[RelationshipEdge]):
        self.nodes = {node.table_name: node for node in nodes}
        self.edges = edges
        self.adjacency_list = self._build_adjacency_list()
    
    def _build_adjacency_list(self) -> Dict[str, List[RelationshipEdge]]:
        """Build adjacency list for graph traversal"""
        adjacency = defaultdict(list)
        for edge in self.edges:
            adjacency[edge.source_table].append(edge)
            adjacency[edge.target_table].append(edge)
        return adjacency
    
    def find_join_paths(self, source_table: str, target_table: str, 
                        max_length: int = 5) -> List[JoinPath]:
        """Find all join paths between two tables"""
        if source_table not in self.nodes or target_table not in self.nodes:
            return []
        
        paths = []
        visited = set()
        
        # Use BFS to find all paths
        queue = deque([(source_table, [], 0)])  # (current_table, path, length)
        
        while queue:
            current_table, current_path, path_length = queue.popleft()
            
            if path_length > max_length:
                continue
            
            if current_table == target_table and current_path:
                # Found a path
                join_path = JoinPath(
                    source_table=source_table,
                    target_table=target_table,
                    path=current_path,
                    path_length=len(current_path),
                    total_cost=self._calculate_path_cost(current_path),
                    total_selectivity=self._calculate_path_selectivity(current_path)
                )
                paths.append(join_path)
                continue
            
            # Explore neighbors
            for edge in self.adjacency_list[current_table]:
                next_table = edge.target_table if edge.source_table == current_table else edge.source_table
                
                # Create path key to avoid cycles
                path_key = (current_table, next_table)
                if path_key not in visited:
                    visited.add(path_key)
                    
                    # Determine if this edge is in the right direction
                    if edge.source_table == current_table:
                        # Forward direction
                        new_path = current_path + [edge]
                        queue.append((next_table, new_path, path_length + 1))
                    elif edge.target_table == current_table:
                        # Reverse direction - create reverse edge
                        reverse_edge = RelationshipEdge(
                            source_table=edge.target_table,
                            target_table=edge.source_table,
                            source_column=edge.target_column,
                            target_column=edge.source_column,
                            relationship_type=edge.relationship_type,
                            join_type=edge.join_type,
                            confidence=edge.confidence,
                            estimated_join_cost=edge.estimated_join_cost,
                            selectivity=edge.selectivity
                        )
                        new_path = current_path + [reverse_edge]
                        queue.append((next_table, new_path, path_length + 1))
        
        # Sort paths by cost
        paths.sort(key=lambda x: x.total_cost)
        
        return paths
    
    def _calculate_path_cost(self, path: List[RelationshipEdge]) -> float:
        """Calculate total cost of a join path"""
        total_cost = 0.0
        for edge in path:
            total_cost += edge.estimated_join_cost
        return total_cost
    
    def _calculate_path_selectivity(self, path: List[RelationshipEdge]) -> float:
        """Calculate total selectivity of a join path"""
        total_selectivity = 1.0
        for edge in path:
            total_selectivity *= edge.selectivity
        return total_selectivity
    
    def find_optimal_path(self, source_table: str, target_table: str, 
                         optimize_for: str = "performance") -> Optional[JoinPath]:
        """Find the optimal join path based on criteria"""
        paths = self.find_join_paths(source_table, target_table)
        
        if not paths:
            return None
        
        if optimize_for == "performance":
            # Choose path with lowest cost
            return min(paths, key=lambda x: x.total_cost)
        elif optimize_for == "simplicity":
            # Choose path with fewest joins
            return min(paths, key=lambda x: x.path_length)
        elif optimize_for == "selectivity":
            # Choose path with highest selectivity
            return max(paths, key=lambda x: x.total_selectivity)
        else:
            # Default to performance
            return min(paths, key=lambda x: x.total_cost)

# ============================================================================
# Graph Analyzer
# ============================================================================

class GraphAnalyzer:
    """Analyzes join graph properties and patterns"""
    
    def __init__(self, nodes: List[TableNode], edges: List[RelationshipEdge]):
        self.nodes = nodes
        self.edges = edges
    
    def analyze_graph(self) -> Dict[str, Any]:
        """Analyze the join graph and return insights"""
        analysis = {
            'total_tables': len(self.nodes),
            'total_relationships': len(self.edges),
            'connected_components': self._count_connected_components(),
            'hub_tables': self._find_hub_tables(),
            'isolated_tables': self._find_isolated_tables(),
            'circular_references': self._find_circular_references(),
            'average_degree': self._calculate_average_degree(),
            'density': self._calculate_graph_density()
        }
        
        return analysis
    
    def _count_connected_components(self) -> int:
        """Count number of connected components in the graph"""
        visited = set()
        components = 0
        
        for node in self.nodes:
            if node.table_name not in visited:
                self._dfs_component(node.table_name, visited)
                components += 1
        
        return components
    
    def _dfs_component(self, table_name: str, visited: Set[str]):
        """Depth-first search to find connected component"""
        visited.add(table_name)
        
        # Find all edges connected to this table
        for edge in self.edges:
            if edge.source_table == table_name and edge.target_table not in visited:
                self._dfs_component(edge.target_table, visited)
            elif edge.target_table == table_name and edge.source_table not in visited:
                self._dfs_component(edge.source_table, visited)
    
    def _find_hub_tables(self) -> List[str]:
        """Find tables with many connections (hub tables)"""
        # Sort by degree and return top 5
        sorted_nodes = sorted(self.nodes, key=lambda x: x.degree, reverse=True)
        return [node.table_name for node in sorted_nodes[:5]]
    
    def _find_isolated_tables(self) -> List[str]:
        """Find tables with no connections"""
        return [node.table_name for node in self.nodes if node.degree == 0]
    
    def _find_circular_references(self) -> List[List[str]]:
        """Find circular references in the graph"""
        # This is a simplified implementation
        # In practice, you'd use a more sophisticated cycle detection algorithm
        circular_refs = []
        
        # Check for simple 2-table cycles
        for edge1 in self.edges:
            for edge2 in self.edges:
                if (edge1.source_table == edge2.target_table and 
                    edge1.target_table == edge2.source_table):
                    cycle = [edge1.source_table, edge1.target_table]
                    if cycle not in circular_refs:
                        circular_refs.append(cycle)
        
        return circular_refs
    
    def _calculate_average_degree(self) -> float:
        """Calculate average degree of all nodes"""
        if not self.nodes:
            return 0.0
        
        total_degree = sum(node.degree for node in self.nodes)
        return total_degree / len(self.nodes)
    
    def _calculate_graph_density(self) -> float:
        """Calculate graph density (edges / max possible edges)"""
        n = len(self.nodes)
        if n < 2:
            return 0.0
        
        max_edges = n * (n - 1) / 2
        return len(self.edges) / max_edges

# ============================================================================
# Main Service
# ============================================================================

class JoinGraphService:
    """Main join graph service"""
    
    def __init__(self):
        self.graph_builder = None
        self.path_finder = None
        self.graph_analyzer = None
    
    async def analyze_join_graph(self, request: GraphAnalysisRequest) -> GraphAnalysisResponse:
        """Analyze the join graph for a database connection"""
        start_time = datetime.now()
        
        try:
            # Get table metadata from introspect service
            tables = await self._get_table_metadata(request.connection_id)
            
            # Build join graph
            self.graph_builder = JoinGraphBuilder()
            nodes, edges = self.graph_builder.build_graph(tables)
            
            # Analyze graph
            self.graph_analyzer = GraphAnalyzer(nodes, edges)
            analysis = self.graph_analyzer.analyze_graph()
            
            # Create join graph
            graph = JoinGraph(
                connection_id=request.connection_id,
                tenant_id=request.tenant_id,
                nodes=nodes,
                edges=edges,
                total_tables=analysis['total_tables'],
                total_relationships=analysis['total_relationships'],
                connected_components=analysis['connected_components'],
                hub_tables=analysis['hub_tables'],
                isolated_tables=analysis['isolated_tables'],
                circular_references=analysis['circular_references'],
                generated_at=datetime.now()
            )
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return GraphAnalysisResponse(
                success=True,
                graph=graph,
                analysis_time_ms=analysis_time
            )
            
        except Exception as e:
            logger.error(f"Join graph analysis failed: {e}")
            return GraphAnalysisResponse(
                success=False,
                error=str(e)
            )
    
    async def find_join_paths(self, request: JoinPathRequest) -> JoinPathResponse:
        """Find join paths between two tables"""
        start_time = datetime.now()
        
        try:
            # Get table metadata if not already available
            if not self.graph_builder:
                tables = await self._get_table_metadata(request.connection_id)
                self.graph_builder = JoinGraphBuilder()
                nodes, edges = self.graph_builder.build_graph(tables)
            else:
                # Use existing graph
                nodes = self.graph_builder.nodes
                edges = self.graph_builder.edges
            
            # Find paths
            self.path_finder = JoinPathFinder(nodes, edges)
            paths = self.path_finder.find_join_paths(
                request.source_table, 
                request.target_table, 
                request.max_path_length
            )
            
            # Find optimal path
            optimal_path = None
            if paths:
                optimal_path = self.path_finder.find_optimal_path(
                    request.source_table, 
                    request.target_table, 
                    request.optimize_for
                )
            
            # Calculate search time
            search_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return JoinPathResponse(
                success=True,
                paths=paths,
                optimal_path=optimal_path,
                total_paths_found=len(paths),
                search_time_ms=search_time
            )
            
        except Exception as e:
            logger.error(f"Join path finding failed: {e}")
            return JoinPathResponse(
                success=False,
                error=str(e)
            )
    
    async def _get_table_metadata(self, connection_id: str) -> List[Dict[str, Any]]:
        """Get table metadata from introspect service"""
        # This would call the introspect service
        # For now, return placeholder data
        return [
            {
                "name": "users",
                "schema_name": "public",
                "full_name": "public.users",
                "estimated_rows": 1000,
                "size_mb": 5.0,
                "business_domain": "users",
                "primary_keys": ["id"],
                "foreign_keys": [],
                "columns": []
            },
            {
                "name": "orders",
                "schema_name": "public",
                "full_name": "public.orders",
                "estimated_rows": 5000,
                "size_mb": 15.0,
                "business_domain": "orders",
                "primary_keys": ["id"],
                "foreign_keys": [{"column": "user_id", "references": "users.id"}],
                "columns": []
            },
            {
                "name": "products",
                "schema_name": "public",
                "full_name": "public.products",
                "estimated_rows": 200,
                "size_mb": 2.0,
                "business_domain": "products",
                "primary_keys": ["id"],
                "foreign_keys": [],
                "columns": []
            }
        ]

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="AskData Join Graph Service",
    description="Table relationship analysis and join path discovery",
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
join_graph_service = JoinGraphService()

@app.post("/analyze", response_model=GraphAnalysisResponse)
async def analyze_join_graph(request: GraphAnalysisRequest):
    """Analyze join graph for a database connection"""
    return await join_graph_service.analyze_join_graph(request)

@app.post("/paths", response_model=JoinPathResponse)
async def find_join_paths(request: JoinPathRequest):
    """Find join paths between two tables"""
    return await join_graph_service.find_join_paths(request)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "join-graph-service"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AskData Join Graph Service",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/analyze",
            "paths": "/paths",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
