"""
SQL Validator Service

This service validates generated SQL statements for safety, correctness, and performance.
It implements strict security gates and provides detailed validation feedback.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import re

import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SQL Validator Service", version="1.0.0")

# =============================================================================
# Data Models
# =============================================================================

class ValidationLevel(str, Enum):
    """Levels of SQL validation"""
    BASIC = "basic"
    STRICT = "strict"
    PARANOID = "paranoid"

class SecurityRisk(str, Enum):
    """Types of security risks"""
    SQL_INJECTION = "sql_injection"
    DATA_EXPOSURE = "data_exposure"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    INFORMATION_DISCLOSURE = "information_disclosure"

class PerformanceRisk(str, Enum):
    """Types of performance risks"""
    CARTESIAN_PRODUCT = "cartesian_product"
    MISSING_INDEX = "missing_index"
    LARGE_RESULT_SET = "large_result_set"
    COMPLEX_JOIN = "complex_join"
    SUBQUERY_NESTING = "subquery_nesting"

class ValidationError(BaseModel):
    """SQL validation error"""
    error_type: str
    severity: str  # ERROR, WARNING, INFO
    message: str
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    suggestion: Optional[str] = None
    security_risk: Optional[SecurityRisk] = None
    performance_risk: Optional[PerformanceRisk] = None

class ValidationWarning(BaseModel):
    """SQL validation warning"""
    warning_type: str
    message: str
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    suggestion: Optional[str] = None
    impact: str = "low"

class ValidationResult(BaseModel):
    """Result of SQL validation"""
    is_valid: bool
    sql: str
    validation_level: ValidationLevel
    errors: List[ValidationError] = []
    warnings: List[ValidationWarning] = []
    security_score: float = 1.0  # 0.0 = unsafe, 1.0 = safe
    performance_score: float = 1.0  # 0.0 = poor, 1.0 = excellent
    estimated_cost: Optional[float] = None
    estimated_rows: Optional[int] = None
    execution_plan: Optional[Dict[str, Any]] = None
    recommendations: List[str] = []
    metadata: Dict[str, Any] = {}

class SQLValidationRequest(BaseModel):
    """Request to validate SQL"""
    sql: str
    connection_id: str
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    validation_level: ValidationLevel = ValidationLevel.STRICT
    dialect: str = "postgresql"
    table_metadata: List[Dict[str, Any]] = []
    options: Dict[str, Any] = {}

class SQLValidationResponse(BaseModel):
    """Response containing validation results"""
    success: bool
    validation_result: Optional[ValidationResult] = None
    error: Optional[str] = None

class SecurityAuditRequest(BaseModel):
    """Request for security audit"""
    sql: str
    connection_id: str
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    audit_level: str = "comprehensive"

class SecurityAuditResponse(BaseModel):
    """Response containing security audit results"""
    success: bool
    security_score: float = 0.0
    risks: List[Dict[str, Any]] = []
    recommendations: List[str] = []
    error: Optional[str] = None

# =============================================================================
# Core Components
# =============================================================================

class SQLParser:
    """Basic SQL parser for validation"""
    
    def __init__(self):
        self.dangerous_keywords = [
            "DROP", "DELETE", "UPDATE", "INSERT", "CREATE", "ALTER", "TRUNCATE",
            "GRANT", "REVOKE", "EXECUTE", "EXEC", "UNION", "UNION ALL"
        ]
        
        self.dangerous_functions = [
            "SLEEP", "BENCHMARK", "LOAD_FILE", "UPDATEXML", "EXTRACTVALUE",
            "GET_LOCK", "RELEASE_LOCK", "USER", "VERSION", "DATABASE"
        ]
        
        self.performance_indicators = [
            "CROSS JOIN", "FULL OUTER JOIN", "SELECT *", "ORDER BY RAND()",
            "LIKE '%pattern%'", "NOT IN", "OR conditions"
        ]
    
    def parse_sql(self, sql: str) -> Dict[str, Any]:
        """Parse SQL and extract basic information"""
        sql_upper = sql.upper().strip()
        
        # Check if it's a SELECT statement
        is_select = sql_upper.startswith("SELECT")
        
        # Count statements
        statement_count = len([s for s in sql.split(';') if s.strip()])
        
        # Extract table names (basic regex)
        table_pattern = r"FROM\s+([a-zA-Z_][a-zA-Z0-9_]*\.?[a-zA-Z_][a-zA-Z0-9_]*)"
        tables = re.findall(table_pattern, sql_upper)
        
        # Extract JOIN clauses
        join_pattern = r"JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*\.?[a-zA-Z_][a-zA-Z0-9_]*)"
        joins = re.findall(join_pattern, sql_upper)
        
        # Check for dangerous patterns
        dangerous_patterns = []
        for keyword in self.dangerous_keywords:
            if keyword in sql_upper:
                dangerous_patterns.append(keyword)
        
        for func in self.dangerous_functions:
            if func in sql_upper:
                dangerous_patterns.append(func)
        
        # Check for performance issues
        performance_issues = []
        for indicator in self.performance_indicators:
            if indicator.upper() in sql_upper:
                performance_issues.append(indicator)
        
        return {
            "is_select": is_select,
            "statement_count": statement_count,
            "tables": tables,
            "joins": joins,
            "dangerous_patterns": dangerous_patterns,
            "performance_issues": performance_issues,
            "sql_length": len(sql),
            "line_count": sql.count('\n') + 1
        }

class SecurityValidator:
    """Validates SQL for security risks"""
    
    def __init__(self):
        self.injection_patterns = [
            r"'.*--",  # SQL comment injection
            r"'.*;",   # Statement termination
            r"'.*UNION", # UNION injection
            r"'.*OR\s+1\s*=\s*1", # OR 1=1 injection
            r"'.*AND\s+1\s*=\s*1", # AND 1=1 injection
            r"'.*EXEC", # EXEC injection
            r"'.*XP_",  # Extended stored procedures
        ]
        
        self.sensitive_data_patterns = [
            r"password", r"credit_card", r"ssn", r"social_security",
            r"passport", r"license", r"account_number", r"pin"
        ]
    
    def validate_security(self, sql: str, validation_level: ValidationLevel) -> List[ValidationError]:
        """Validate SQL for security risks"""
        errors = []
        sql_lower = sql.lower()
        
        # Check for dangerous keywords
        dangerous_keywords = [
            "DROP", "DELETE", "UPDATE", "INSERT", "CREATE", "ALTER", "TRUNCATE",
            "GRANT", "REVOKE", "EXECUTE", "EXEC"
        ]
        
        for keyword in dangerous_keywords:
            if keyword.lower() in sql_lower:
                errors.append(ValidationError(
                    error_type="dangerous_keyword",
                    severity="ERROR",
                    message=f"Use of dangerous keyword: {keyword}",
                    suggestion=f"Remove or replace {keyword} with safer alternative",
                    security_risk=SecurityRisk.PRIVILEGE_ESCALATION
                ))
        
        # Check for SQL injection patterns
        for pattern in self.injection_patterns:
            if re.search(pattern, sql_lower, re.IGNORECASE):
                errors.append(ValidationError(
                    error_type="sql_injection_pattern",
                    severity="ERROR",
                    message="Potential SQL injection pattern detected",
                    suggestion="Use parameterized queries instead of string concatenation",
                    security_risk=SecurityRisk.SQL_INJECTION
                ))
        
        # Check for sensitive data exposure
        for pattern in self.sensitive_data_patterns:
            if re.search(pattern, sql_lower):
                errors.append(ValidationError(
                    error_type="sensitive_data_exposure",
                    severity="WARNING",
                    message=f"Query may expose sensitive data: {pattern}",
                    suggestion="Review if this data access is necessary and authorized",
                    security_risk=SecurityRisk.DATA_EXPOSURE
                ))
        
        # Check for resource exhaustion
        if "LIMIT" not in sql_upper and validation_level == ValidationLevel.PARANOID:
            errors.append(ValidationError(
                error_type="missing_limit",
                severity="WARNING",
                message="Query missing LIMIT clause",
                suggestion="Add LIMIT clause to prevent large result sets",
                security_risk=SecurityRisk.RESOURCE_EXHAUSTION
            ))
        
        return errors
    
    def calculate_security_score(self, errors: List[ValidationError]) -> float:
        """Calculate security score based on validation errors"""
        if not errors:
            return 1.0
        
        # Weight errors by severity
        error_weights = {
            "ERROR": 0.5,
            "WARNING": 0.2,
            "INFO": 0.1
        }
        
        total_score = 1.0
        for error in errors:
            weight = error_weights.get(error.severity, 0.1)
            total_score -= weight
        
        return max(0.0, total_score)

class PerformanceValidator:
    """Validates SQL for performance issues"""
    
    def __init__(self):
        self.performance_patterns = {
            "cartesian_product": {
                "pattern": r"FROM\s+([^,]+),\s*([^,]+)(?!\s+WHERE)",
                "severity": "WARNING",
                "message": "Potential cartesian product detected",
                "suggestion": "Use explicit JOIN clauses instead of comma-separated tables"
            },
            "missing_where": {
                "pattern": r"FROM\s+[^;]+(?!\s+WHERE)",
                "severity": "WARNING",
                "message": "Query missing WHERE clause",
                "suggestion": "Add WHERE clause to limit result set"
            },
            "select_star": {
                "pattern": r"SELECT\s+\*",
                "severity": "WARNING",
                "message": "SELECT * may return unnecessary columns",
                "suggestion": "Specify only required columns"
            },
            "complex_subquery": {
                "pattern": r"SELECT.*SELECT.*SELECT",
                "severity": "INFO",
                "message": "Complex nested subqueries detected",
                "suggestion": "Consider using CTEs or temporary tables for better performance"
            }
        }
    
    def validate_performance(self, sql: str, table_metadata: List[Dict[str, Any]]) -> List[ValidationWarning]:
        """Validate SQL for performance issues"""
        warnings = []
        sql_upper = sql.upper()
        
        # Check for performance patterns
        for issue_type, pattern_info in self.performance_patterns.items():
            if re.search(pattern_info["pattern"], sql_upper, re.IGNORECASE):
                warnings.append(ValidationWarning(
                    warning_type=issue_type,
                    message=pattern_info["message"],
                    suggestion=pattern_info["suggestion"],
                    impact="medium"
                ))
        
        # Check for specific performance issues
        if "CROSS JOIN" in sql_upper:
            warnings.append(ValidationWarning(
                warning_type="cross_join",
                message="CROSS JOIN detected - may cause performance issues",
                suggestion="Use INNER JOIN with explicit conditions",
                impact="high"
            ))
        
        if "ORDER BY RAND()" in sql_upper:
            warnings.append(ValidationWarning(
                warning_type="random_ordering",
                message="ORDER BY RAND() detected - very expensive operation",
                suggestion="Use alternative random selection methods",
                impact="high"
            ))
        
        if "LIKE '%pattern%'" in sql_upper or "LIKE '%pattern'" in sql_upper:
            warnings.append(ValidationWarning(
                warning_type="wildcard_like",
                message="Leading wildcard in LIKE clause - prevents index usage",
                suggestion="Consider full-text search or restructure query",
                impact="medium"
            ))
        
        # Check for missing indexes (basic heuristic)
        if "WHERE" in sql_upper and table_metadata:
            warnings.extend(self._check_missing_indexes(sql, table_metadata))
        
        return warnings
    
    def _check_missing_indexes(self, sql: str, table_metadata: List[Dict[str, Any]]) -> List[ValidationWarning]:
        """Check for potential missing indexes"""
        warnings = []
        
        # Extract WHERE conditions (simplified)
        where_match = re.search(r"WHERE\s+(.+?)(?:\s+ORDER\s+BY|\s+GROUP\s+BY|\s+LIMIT|$)", sql, re.IGNORECASE)
        if not where_match:
            return warnings
        
        where_clause = where_match.group(1)
        
        # Check for common indexable patterns
        indexable_patterns = [
            r"(\w+)\s*=\s*[^']",  # column = value
            r"(\w+)\s+IN\s*\([^)]+\)",  # column IN (...)
            r"(\w+)\s+LIKE\s+'[^%]%'",  # column LIKE 'pattern%'
            r"(\w+)\s+>\s*[^']",  # column > value
            r"(\w+)\s+<\s*[^']",  # column < value
        ]
        
        for pattern in indexable_patterns:
            matches = re.findall(pattern, where_clause, re.IGNORECASE)
            for match in matches:
                warnings.append(ValidationWarning(
                    warning_type="potential_missing_index",
                    message=f"Column '{match}' in WHERE clause may benefit from an index",
                    suggestion=f"Consider adding index on column '{match}'",
                    impact="low"
                ))
        
        return warnings
    
    def calculate_performance_score(self, warnings: List[ValidationWarning]) -> float:
        """Calculate performance score based on validation warnings"""
        if not warnings:
            return 1.0
        
        # Weight warnings by impact
        impact_weights = {
            "high": 0.3,
            "medium": 0.2,
            "low": 0.1
        }
        
        total_score = 1.0
        for warning in warnings:
            weight = impact_weights.get(warning.impact, 0.1)
            total_score -= weight
        
        return max(0.0, total_score)

class SQLValidator:
    """Main SQL validation service"""
    
    def __init__(self):
        self.parser = SQLParser()
        self.security_validator = SecurityValidator()
        self.performance_validator = PerformanceValidator()
    
    async def validate_sql(self, request: SQLValidationRequest) -> ValidationResult:
        """Validate SQL statement"""
        try:
            # Parse SQL
            parsed_info = self.parser.parse_sql(request.sql)
            
            # Validate security
            security_errors = self.security_validator.validate_security(
                request.sql, request.validation_level
            )
            
            # Validate performance
            performance_warnings = self.performance_validator.validate_performance(
                request.sql, request.table_metadata
            )
            
            # Check basic validity
            basic_errors = self._validate_basic_syntax(request.sql, parsed_info)
            
            # Combine all errors
            all_errors = security_errors + basic_errors
            
            # Calculate scores
            security_score = self.security_validator.calculate_security_score(all_errors)
            performance_score = self.performance_validator.calculate_performance_score(performance_warnings)
            
            # Determine overall validity
            is_valid = len([e for e in all_errors if e.severity == "ERROR"]) == 0
            
            # Generate recommendations
            recommendations = self._generate_recommendations(all_errors, performance_warnings)
            
            # Estimate cost and rows
            estimated_cost = self._estimate_query_cost(request.sql, parsed_info)
            estimated_rows = self._estimate_result_rows(request.sql, parsed_info, request.table_metadata)
            
            return ValidationResult(
                is_valid=is_valid,
                sql=request.sql,
                validation_level=request.validation_level,
                errors=all_errors,
                warnings=performance_warnings,
                security_score=security_score,
                performance_score=performance_score,
                estimated_cost=estimated_cost,
                estimated_rows=estimated_rows,
                recommendations=recommendations,
                metadata={
                    "parsed_info": parsed_info,
                    "dialect": request.dialect,
                    "validation_timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Error validating SQL: {str(e)}")
            # Return basic validation result with error
            return ValidationResult(
                is_valid=False,
                sql=request.sql,
                validation_level=request.validation_level,
                errors=[
                    ValidationError(
                        error_type="validation_error",
                        severity="ERROR",
                        message=f"Validation failed: {str(e)}"
                    )
                ]
            )
    
    def _validate_basic_syntax(self, sql: str, parsed_info: Dict[str, Any]) -> List[ValidationError]:
        """Validate basic SQL syntax"""
        errors = []
        
        # Check if it's a SELECT statement
        if not parsed_info["is_select"]:
            errors.append(ValidationError(
                error_type="non_select_statement",
                severity="ERROR",
                message="Only SELECT statements are allowed",
                suggestion="Convert to SELECT statement or use appropriate service"
            ))
        
        # Check for multiple statements
        if parsed_info["statement_count"] > 1:
            errors.append(ValidationError(
                error_type="multiple_statements",
                severity="ERROR",
                message="Multiple SQL statements detected",
                suggestion="Submit only one statement at a time"
            ))
        
        # Check SQL length
        if parsed_info["sql_length"] > 10000:
            errors.append(ValidationError(
                error_type="sql_too_long",
                severity="WARNING",
                message="SQL statement is very long",
                suggestion="Consider breaking into smaller, focused queries"
            ))
        
        return errors
    
    def _generate_recommendations(self, errors: List[ValidationError], 
                                warnings: List[ValidationWarning]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Security recommendations
        security_errors = [e for e in errors if e.security_risk]
        if security_errors:
            recommendations.append("Review and address all security issues before execution")
        
        # Performance recommendations
        if warnings:
            recommendations.append("Consider performance optimizations for better query execution")
        
        # General recommendations
        if not errors and not warnings:
            recommendations.append("SQL validation passed - ready for execution")
        
        return recommendations
    
    def _estimate_query_cost(self, sql: str, parsed_info: Dict[str, Any]) -> Optional[float]:
        """Estimate query execution cost"""
        # Simplified cost estimation
        base_cost = 1.0
        
        # Add cost for joins
        if parsed_info["joins"]:
            base_cost += len(parsed_info["joins"]) * 0.5
        
        # Add cost for complex queries
        if parsed_info["sql_length"] > 1000:
            base_cost += 0.5
        
        # Add cost for performance issues
        base_cost += len(parsed_info["performance_issues"]) * 0.3
        
        return round(base_cost, 2)
    
    def _estimate_result_rows(self, sql: str, parsed_info: Dict[str, Any], 
                            table_metadata: List[Dict[str, Any]]) -> Optional[int]:
        """Estimate result row count"""
        # Simplified row estimation
        if not table_metadata:
            return None
        
        # Use first table's estimated rows as baseline
        first_table = table_metadata[0]
        estimated_rows = first_table.get("estimated_rows", 1000)
        
        # Adjust based on WHERE conditions (simplified)
        if "WHERE" in sql.upper():
            estimated_rows = estimated_rows // 10  # Assume 10% selectivity
        
        # Adjust based on LIMIT
        limit_match = re.search(r"LIMIT\s+(\d+)", sql.upper())
        if limit_match:
            limit_value = int(limit_match.group(1))
            estimated_rows = min(estimated_rows, limit_value)
        
        return estimated_rows

class SecurityAuditor:
    """Performs comprehensive security audits"""
    
    def __init__(self):
        self.audit_patterns = self._build_audit_patterns()
    
    def _build_audit_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Build comprehensive security audit patterns"""
        return {
            "sql_injection": {
                "patterns": [
                    r"'.*--", r"'.*;", r"'.*UNION", r"'.*OR\s+1\s*=\s*1",
                    r"'.*AND\s+1\s*=\s*1", r"'.*EXEC", r"'.*XP_"
                ],
                "risk_level": "HIGH",
                "description": "Potential SQL injection vulnerability"
            },
            "privilege_escalation": {
                "patterns": [
                    r"GRANT\s+", r"REVOKE\s+", r"CREATE\s+USER", r"ALTER\s+USER"
                ],
                "risk_level": "HIGH",
                "description": "Attempted privilege escalation"
            },
            "data_exfiltration": {
                "patterns": [
                    r"SELECT\s+\*", r"SELECT.*FROM.*WHERE\s+1\s*=\s*1"
                ],
                "risk_level": "MEDIUM",
                "description": "Potential data exfiltration"
            },
            "resource_abuse": {
                "patterns": [
                    r"SELECT.*FROM.*CROSS\s+JOIN", r"SELECT.*FROM.*FULL\s+OUTER\s+JOIN"
                ],
                "risk_level": "MEDIUM",
                "description": "Potential resource abuse"
            }
        }
    
    async def audit_security(self, request: SecurityAuditRequest) -> SecurityAuditResponse:
        """Perform comprehensive security audit"""
        try:
            risks = []
            sql_upper = request.sql.upper()
            
            # Check each audit pattern
            for risk_type, pattern_info in self.audit_patterns.items():
                for pattern in pattern_info["patterns"]:
                    if re.search(pattern, sql_upper, re.IGNORECASE):
                        risks.append({
                            "risk_type": risk_type,
                            "risk_level": pattern_info["risk_level"],
                            "description": pattern_info["description"],
                            "pattern": pattern,
                            "recommendation": f"Review and validate {risk_type} risk"
                        })
            
            # Calculate security score
            high_risks = len([r for r in risks if r["risk_level"] == "HIGH"])
            medium_risks = len([r for r in risks if r["risk_level"] == "MEDIUM"])
            
            security_score = max(0.0, 1.0 - (high_risks * 0.4) - (medium_risks * 0.2))
            
            # Generate recommendations
            recommendations = []
            if high_risks > 0:
                recommendations.append("Address HIGH risk issues immediately before execution")
            if medium_risks > 0:
                recommendations.append("Review MEDIUM risk issues for potential impact")
            if not risks:
                recommendations.append("No security risks detected")
            
            return SecurityAuditResponse(
                success=True,
                security_score=security_score,
                risks=risks,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error during security audit: {str(e)}")
            return SecurityAuditResponse(
                success=False,
                error=str(e)
            )

# =============================================================================
# Service Instance
# =============================================================================

sql_validator = SQLValidator()
security_auditor = SecurityAuditor()

# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "SQL Validator Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/validate - Validate SQL for safety and performance",
            "/audit - Perform comprehensive security audit",
            "/health - Health check"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "sql-validator-service"
    }

@app.post("/validate", response_model=SQLValidationResponse)
async def validate_sql(request: SQLValidationRequest):
    """Validate SQL statement for safety and performance"""
    validation_result = await sql_validator.validate_sql(request)
    return SQLValidationResponse(
        success=True,
        validation_result=validation_result
    )

@app.post("/audit", response_model=SecurityAuditResponse)
async def audit_sql_security(request: SecurityAuditRequest):
    """Perform comprehensive security audit of SQL"""
    return await security_auditor.audit_security(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
