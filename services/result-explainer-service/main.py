"""
AskData Result Explainer Service

This service provides natural language explanations of query results, generates insights,
and creates human-readable summaries of data analysis results.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
import re

import structlog
import httpx
import pandas as pd
import numpy as np
from scipy import stats
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# Add the contracts directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent / "contracts"))

from mcp_tools import (
    ExplainResultsInput, ExplainResultsOutput,
    ResultExplanation, Insight, Summary
)

# =============================================================================
# CONFIGURATION
# =============================================================================

class Settings(BaseSettings):
    """Application settings"""
    environment: str = "local"
    log_level: str = "INFO"
    debug: bool = False
    
    # Analysis settings
    max_insights: int = 10
    min_confidence: float = 0.7
    max_summary_length: int = 500
    
    # Statistical settings
    outlier_threshold: float = 2.0  # Standard deviations for outlier detection
    correlation_threshold: float = 0.5  # Minimum correlation for insights
    
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
            structlog.stdlib.PositionalArgumentsFormatter(),
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
# DATA MODELS
# =============================================================================

class AnalysisRequest(BaseModel):
    """Request to analyze query results"""
    query_results: Dict[str, Any]
    original_query: str
    connection_id: str
    run_envelope: Dict[str, Any]
    analysis_type: str = "comprehensive"  # basic, comprehensive, insights_only

class AnalysisResponse(BaseModel):
    """Response containing analysis results"""
    success: bool
    explanation: Optional[ResultExplanation] = None
    insights: List[Insight] = []
    summary: Optional[Summary] = None
    error: Optional[str] = None

# =============================================================================
# CORE ANALYSIS COMPONENTS
# =============================================================================

class DataAnalyzer:
    """Analyzes data and generates insights"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
    
    def analyze_data(self, data: List[List[Any]], columns: List[str]) -> Dict[str, Any]:
        """Perform comprehensive data analysis"""
        try:
            # Convert to pandas DataFrame for analysis
            df = pd.DataFrame(data, columns=columns)
            
            analysis = {
                "basic_stats": self._calculate_basic_stats(df),
                "data_quality": self._assess_data_quality(df),
                "patterns": self._identify_patterns(df),
                "outliers": self._detect_outliers(df),
                "correlations": self._find_correlations(df),
                "trends": self._identify_trends(df)
            }
            
            return analysis
            
        except Exception as e:
            logging.error(f"Data analysis failed: {e}")
            return {"error": str(e)}
    
    def _calculate_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistical measures"""
        stats = {}
        
        for col in df.columns:
            col_data = df[col]
            
            # Skip non-numeric columns for statistical calculations
            if not pd.api.types.is_numeric_dtype(col_data):
                stats[col] = {
                    "type": "categorical",
                    "unique_count": col_data.nunique(),
                    "most_common": col_data.value_counts().head(3).to_dict(),
                    "missing_count": col_data.isnull().sum()
                }
                continue
            
            # Numeric column statistics
            stats[col] = {
                "type": "numeric",
                "count": col_data.count(),
                "mean": float(col_data.mean()),
                "median": float(col_data.median()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "missing_count": col_data.isnull().sum()
            }
        
        return stats
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality"""
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        duplicate_rows = df.duplicated().sum()
        
        return {
            "total_cells": total_cells,
            "missing_cells": missing_cells,
            "missing_percentage": (missing_cells / total_cells) * 100 if total_cells > 0 else 0,
            "duplicate_rows": duplicate_rows,
            "duplicate_percentage": (duplicate_rows / len(df)) * 100 if len(df) > 0 else 0,
            "quality_score": max(0, 100 - ((missing_cells / total_cells) * 50) - ((duplicate_rows / len(df)) * 50)) if total_cells > 0 and len(df) > 0 else 100
        }
    
    def _identify_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify patterns in the data"""
        patterns = {}
        
        for col in df.columns:
            col_data = df[col]
            
            if pd.api.types.is_numeric_dtype(col_data):
                # Check for even/odd patterns
                if col_data.dtype in ['int64', 'int32']:
                    even_count = (col_data % 2 == 0).sum()
                    odd_count = (col_data % 2 == 1).sum()
                    if even_count > 0 and odd_count > 0:
                        patterns[col] = {
                            "type": "numeric_pattern",
                            "even_odd_ratio": even_count / odd_count if odd_count > 0 else float('inf')
                        }
                
                # Check for sequential patterns
                if len(col_data) > 2:
                    diffs = col_data.diff().dropna()
                    if len(diffs) > 0 and (diffs == diffs.iloc[0]).all():
                        patterns[col] = {
                            "type": "sequential_pattern",
                            "step_size": float(diffs.iloc[0])
                        }
            
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                # Check for time patterns
                if len(col_data) > 2:
                    time_diffs = col_data.diff().dropna()
                    if len(time_diffs) > 0:
                        patterns[col] = {
                            "type": "time_pattern",
                            "min_interval": str(time_diffs.min()),
                            "max_interval": str(time_diffs.max()),
                            "avg_interval": str(time_diffs.mean())
                        }
        
        return patterns
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers in numeric columns"""
        outliers = {}
        
        for col in df.columns:
            col_data = df[col]
            
            if pd.api.types.is_numeric_dtype(col_data):
                # Remove missing values
                clean_data = col_data.dropna()
                
                if len(clean_data) > 0:
                    # Calculate z-scores
                    z_scores = np.abs(stats.zscore(clean_data))
                    outlier_indices = np.where(z_scores > self.settings.outlier_threshold)[0]
                    
                    if len(outlier_indices) > 0:
                        outliers[col] = {
                            "count": len(outlier_indices),
                            "percentage": (len(outlier_indices) / len(clean_data)) * 100,
                            "values": clean_data.iloc[outlier_indices].tolist()
                        }
        
        return outliers
    
    def _find_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Find correlations between numeric columns"""
        correlations = {}
        
        # Get only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            # Find significant correlations
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols):
                    if i < j:  # Avoid duplicate pairs and self-correlation
                        corr_value = corr_matrix.loc[col1, col2]
                        
                        if abs(corr_value) >= self.settings.correlation_threshold:
                            correlations[f"{col1}_vs_{col2}"] = {
                                "correlation": float(corr_value),
                                "strength": "strong" if abs(corr_value) >= 0.8 else "moderate" if abs(corr_value) >= 0.5 else "weak",
                                "direction": "positive" if corr_value > 0 else "negative"
                            }
        
        return correlations
    
    def _identify_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify trends in the data"""
        trends = {}
        
        for col in df.columns:
            col_data = df[col]
            
            if pd.api.types.is_numeric_dtype(col_data) and len(col_data) > 2:
                # Calculate linear trend
                x = np.arange(len(col_data))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, col_data.dropna())
                
                if abs(r_value) >= 0.3:  # Only report if there's a meaningful trend
                    trends[col] = {
                        "slope": float(slope),
                        "r_squared": float(r_value ** 2),
                        "trend_direction": "increasing" if slope > 0 else "decreasing",
                        "trend_strength": "strong" if abs(r_value) >= 0.8 else "moderate" if abs(r_value) >= 0.5 else "weak"
                    }
        
        return trends

class InsightGenerator:
    """Generates insights from data analysis"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
    
    def generate_insights(self, analysis: Dict[str, Any], original_query: str) -> List[Insight]:
        """Generate insights based on data analysis"""
        insights = []
        
        try:
            # Generate insights from basic stats
            insights.extend(self._generate_statistical_insights(analysis.get("basic_stats", {})))
            
            # Generate insights from data quality
            insights.extend(self._generate_quality_insights(analysis.get("data_quality", {})))
            
            # Generate insights from patterns
            insights.extend(self._generate_pattern_insights(analysis.get("patterns", {})))
            
            # Generate insights from outliers
            insights.extend(self._generate_outlier_insights(analysis.get("outliers", {})))
            
            # Generate insights from correlations
            insights.extend(self._generate_correlation_insights(analysis.get("correlations", {})))
            
            # Generate insights from trends
            insights.extend(self._generate_trend_insights(analysis.get("trends", {})))
            
            # Sort by confidence and limit
            insights.sort(key=lambda x: x.confidence, reverse=True)
            insights = insights[:self.settings.max_insights]
            
            return insights
            
        except Exception as e:
            logging.error(f"Insight generation failed: {e}")
            return []
    
    def _generate_statistical_insights(self, stats: Dict[str, Any]) -> List[Insight]:
        """Generate insights from statistical analysis"""
        insights = []
        
        for col, col_stats in stats.items():
            if col_stats.get("type") == "numeric":
                # Check for unusual distributions
                mean = col_stats.get("mean", 0)
                median = col_stats.get("median", 0)
                
                if abs(mean - median) / max(abs(mean), 1) > 0.2:
                    insights.append(Insight(
                        insight_type="distribution_skew",
                        description=f"Column '{col}' shows a skewed distribution (mean: {mean:.2f}, median: {median:.2f})",
                        confidence=0.8,
                        impact="medium",
                        recommendation="Consider using median instead of mean for this column"
                    ))
                
                # Check for wide ranges
                min_val = col_stats.get("min", 0)
                max_val = col_stats.get("max", 0)
                std = col_stats.get("std", 0)
                
                if std > 0 and (max_val - min_val) / std > 10:
                    insights.append(Insight(
                        insight_type="wide_range",
                        description=f"Column '{col}' has a very wide range ({min_val:.2f} to {max_val:.2f})",
                        confidence=0.7,
                        impact="low",
                        recommendation="Data spans a wide range - consider normalization if needed"
                    ))
            
            elif col_stats.get("type") == "categorical":
                # Check for imbalanced categories
                unique_count = col_stats.get("unique_count", 0)
                total_count = col_stats.get("count", 0)
                
                if unique_count > 0 and total_count / unique_count > 100:
                    insights.append(Insight(
                        insight_type="high_cardinality",
                        description=f"Column '{col}' has high cardinality ({unique_count} unique values)",
                        confidence=0.8,
                        impact="medium",
                        recommendation="High cardinality may affect query performance"
                    ))
        
        return insights
    
    def _generate_quality_insights(self, quality: Dict[str, Any]) -> List[Insight]:
        """Generate insights from data quality analysis"""
        insights = []
        
        missing_percentage = quality.get("missing_percentage", 0)
        duplicate_percentage = quality.get("duplicate_percentage", 0)
        quality_score = quality.get("quality_score", 100)
        
        if missing_percentage > 20:
            insights.append(Insight(
                insight_type="high_missing_data",
                description=f"Data has {missing_percentage:.1f}% missing values",
                confidence=0.9,
                impact="high",
                recommendation="Investigate missing data patterns and consider data cleaning"
            ))
        
        if duplicate_percentage > 10:
            insights.append(Insight(
                insight_type="duplicate_data",
                description=f"Data has {duplicate_percentage:.1f}% duplicate rows",
                confidence=0.9,
                impact="medium",
                recommendation="Remove duplicates to improve data quality"
            ))
        
        if quality_score < 70:
            insights.append(Insight(
                insight_type="low_quality_data",
                description=f"Overall data quality score is {quality_score:.1f}/100",
                confidence=0.8,
                impact="high",
                recommendation="Address data quality issues before analysis"
            ))
        
        return insights
    
    def _generate_pattern_insights(self, patterns: Dict[str, Any]) -> List[Insight]:
        """Generate insights from pattern analysis"""
        insights = []
        
        for col, pattern in patterns.items():
            if pattern.get("type") == "sequential_pattern":
                insights.append(Insight(
                    insight_type="sequential_data",
                    description=f"Column '{col}' shows a sequential pattern with step size {pattern.get('step_size', 0)}",
                    confidence=0.8,
                    impact="low",
                    recommendation="Sequential data may be suitable for time series analysis"
                ))
            
            elif pattern.get("type") == "time_pattern":
                insights.append(Insight(
                    insight_type="time_pattern",
                    description=f"Column '{col}' shows regular time intervals",
                    confidence=0.7,
                    impact="low",
                    recommendation="Time-based data may benefit from temporal analysis"
                ))
        
        return insights
    
    def _generate_outlier_insights(self, outliers: Dict[str, Any]) -> List[Insight]:
        """Generate insights from outlier analysis"""
        insights = []
        
        for col, outlier_info in outliers.items():
            percentage = outlier_info.get("percentage", 0)
            
            if percentage > 5:
                insights.append(Insight(
                    insight_type="high_outlier_rate",
                    description=f"Column '{col}' has {percentage:.1f}% outliers",
                    confidence=0.8,
                    impact="medium",
                    recommendation="Investigate outliers for data quality issues or business insights"
                ))
        
        return insights
    
    def _generate_correlation_insights(self, correlations: Dict[str, Any]) -> List[Insight]:
        """Generate insights from correlation analysis"""
        insights = []
        
        for pair, corr_info in correlations.items():
            strength = corr_info.get("strength", "weak")
            direction = corr_info.get("direction", "positive")
            correlation = corr_info.get("correlation", 0)
            
            if strength in ["strong", "moderate"]:
                insights.append(Insight(
                    insight_type="correlation",
                    description=f"Strong {direction} correlation ({correlation:.2f}) between {pair}",
                    confidence=0.8,
                    impact="medium",
                    recommendation=f"Consider the relationship between these variables in your analysis"
                ))
        
        return insights
    
    def _generate_trend_insights(self, trends: Dict[str, Any]) -> List[Insight]:
        """Generate insights from trend analysis"""
        insights = []
        
        for col, trend_info in trends.items():
            direction = trend_info.get("trend_direction", "stable")
            strength = trend_info.get("trend_strength", "weak")
            r_squared = trend_info.get("r_squared", 0)
            
            if strength in ["strong", "moderate"]:
                insights.append(Insight(
                    insight_type="trend",
                    description=f"Column '{col}' shows a {strength} {direction} trend (R² = {r_squared:.2f})",
                    confidence=0.8,
                    impact="medium",
                    recommendation=f"Consider the {direction} trend in your analysis and forecasting"
                ))
        
        return insights

class SummaryGenerator:
    """Generates human-readable summaries of results"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
    
    def generate_summary(self, analysis: Dict[str, Any], insights: List[Insight], 
                        original_query: str, row_count: int) -> Summary:
        """Generate a comprehensive summary of the results"""
        try:
            # Generate executive summary
            executive_summary = self._generate_executive_summary(analysis, row_count)
            
            # Generate key findings
            key_findings = self._generate_key_findings(insights)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(insights, analysis)
            
            # Generate data quality summary
            quality_summary = self._generate_quality_summary(analysis.get("data_quality", {}))
            
            return Summary(
                executive_summary=executive_summary,
                key_findings=key_findings,
                recommendations=recommendations,
                quality_summary=quality_summary,
                total_insights=len(insights),
                analysis_timestamp=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            logging.error(f"Summary generation failed: {e}")
            return Summary(
                executive_summary="Summary generation failed",
                key_findings=[],
                recommendations=[],
                quality_summary="Unable to assess data quality",
                total_insights=0,
                analysis_timestamp=datetime.utcnow().isoformat()
            )
    
    def _generate_executive_summary(self, analysis: Dict[str, Any], row_count: int) -> str:
        """Generate executive summary"""
        quality_score = analysis.get("data_quality", {}).get("quality_score", 100)
        
        summary = f"Analysis of {row_count} rows of data"
        
        if quality_score >= 90:
            summary += " shows high-quality data with minimal issues."
        elif quality_score >= 70:
            summary += " shows generally good data quality with some areas for improvement."
        else:
            summary += " reveals significant data quality concerns that should be addressed."
        
        # Add pattern information
        patterns = analysis.get("patterns", {})
        if patterns:
            summary += f" The data contains {len(patterns)} notable patterns."
        
        # Add outlier information
        outliers = analysis.get("outliers", {})
        if outliers:
            summary += f" {len(outliers)} columns contain outliers that may warrant investigation."
        
        return summary
    
    def _generate_key_findings(self, insights: List[Insight]) -> List[str]:
        """Generate key findings from insights"""
        findings = []
        
        # Group insights by type
        insight_types = {}
        for insight in insights:
            insight_type = insight.insight_type
            if insight_type not in insight_types:
                insight_types[insight_type] = []
            insight_types[insight_type].append(insight)
        
        # Generate findings for each type
        for insight_type, type_insights in insight_types.items():
            if insight_type == "correlation":
                findings.append(f"Found {len(type_insights)} significant correlations between variables")
            elif insight_type == "trend":
                findings.append(f"Identified {len(type_insights)} clear trends in the data")
            elif insight_type == "outlier":
                findings.append(f"Detected outliers in {len(type_insights)} columns")
            elif insight_type == "data_quality":
                findings.append(f"Data quality assessment revealed {len(type_insights)} areas of concern")
        
        return findings[:5]  # Limit to top 5 findings
    
    def _generate_recommendations(self, insights: List[Insight], analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # High-impact recommendations first
        high_impact = [i for i in insights if i.impact == "high"]
        for insight in high_impact[:3]:
            if insight.recommendation:
                recommendations.append(insight.recommendation)
        
        # Add data quality recommendations
        quality = analysis.get("data_quality", {})
        if quality.get("missing_percentage", 0) > 10:
            recommendations.append("Implement data validation to reduce missing values")
        
        if quality.get("duplicate_percentage", 0) > 5:
            recommendations.append("Set up duplicate detection and removal processes")
        
        # Add analysis recommendations
        if analysis.get("correlations"):
            recommendations.append("Consider multivariate analysis to explore relationships")
        
        if analysis.get("trends"):
            recommendations.append("Apply time series analysis techniques for trend forecasting")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _generate_quality_summary(self, quality: Dict[str, Any]) -> str:
        """Generate data quality summary"""
        if not quality:
            return "Data quality assessment not available"
        
        score = quality.get("quality_score", 100)
        missing = quality.get("missing_percentage", 0)
        duplicates = quality.get("duplicate_percentage", 0)
        
        summary = f"Data quality score: {score:.1f}/100"
        
        if missing > 0:
            summary += f". Missing data: {missing:.1f}%"
        
        if duplicates > 0:
            summary += f". Duplicate rows: {duplicates:.1f}%"
        
        if score >= 90:
            summary += ". Overall quality is excellent."
        elif score >= 70:
            summary += ". Overall quality is good with room for improvement."
        else:
            summary += ". Overall quality needs attention."
        
        return summary

# =============================================================================
# RESULT EXPLAINER SERVICE
# =============================================================================

class ResultExplainerService:
    """Main service for explaining query results"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.data_analyzer = DataAnalyzer(settings)
        self.insight_generator = InsightGenerator(settings)
        self.summary_generator = SummaryGenerator(settings)
    
    async def explain_results(self, request: ExplainResultsInput) -> ExplainResultsOutput:
        """Explain query results with comprehensive analysis"""
        try:
            # Extract data from request
            query_results = request.query_results
            original_query = request.original_query
            connection_id = request.connection_id
            
            # Extract data and columns
            data = query_results.get("data", [])
            columns = query_results.get("columns", [])
            
            if not data or not columns:
                return ExplainResultsOutput(
                    connection_id=connection_id,
                    explanation=ResultExplanation(
                        natural_language_summary="No data available for analysis",
                        technical_details="Empty result set",
                        data_overview="No rows returned"
                    ),
                    insights=[],
                    summary=Summary(
                        executive_summary="No data to analyze",
                        key_findings=[],
                        recommendations=["Verify query returns data"],
                        quality_summary="No data available",
                        total_insights=0,
                        analysis_timestamp=datetime.utcnow().isoformat()
                    ),
                    analysis_time=datetime.utcnow().isoformat()
                )
            
            # Perform data analysis
            analysis = self.data_analyzer.analyze_data(data, columns)
            
            # Generate insights
            insights = self.insight_generator.generate_insights(analysis, original_query)
            
            # Generate summary
            summary = self.summary_generator.generate_summary(analysis, insights, original_query, len(data))
            
            # Create natural language explanation
            explanation = self._create_explanation(analysis, insights, original_query, len(data))
            
            return ExplainResultsOutput(
                connection_id=connection_id,
                explanation=explanation,
                insights=insights,
                summary=summary,
                analysis_time=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            logging.error(f"Result explanation failed: {e}")
            return ExplainResultsOutput(
                connection_id=connection_id,
                explanation=ResultExplanation(
                    natural_language_summary=f"Analysis failed: {str(e)}",
                    technical_details="Error during analysis",
                    data_overview="Unable to process results"
                ),
                insights=[],
                summary=Summary(
                    executive_summary="Analysis failed",
                    key_findings=[],
                    recommendations=["Check data format and try again"],
                    quality_summary="Unable to assess",
                    total_insights=0,
                    analysis_timestamp=datetime.utcnow().isoformat()
                ),
                analysis_time=datetime.utcnow().isoformat()
            )
    
    def _create_explanation(self, analysis: Dict[str, Any], insights: List[Insight], 
                           original_query: str, row_count: int) -> ResultExplanation:
        """Create natural language explanation of results"""
        try:
            # Generate natural language summary
            summary = self._generate_natural_language_summary(analysis, insights, row_count)
            
            # Generate technical details
            technical_details = self._generate_technical_details(analysis)
            
            # Generate data overview
            data_overview = self._generate_data_overview(analysis, row_count)
            
            return ResultExplanation(
                natural_language_summary=summary,
                technical_details=technical_details,
                data_overview=data_overview
            )
            
        except Exception as e:
            logging.error(f"Explanation creation failed: {e}")
            return ResultExplanation(
                natural_language_summary="Unable to generate explanation",
                technical_details="Error in explanation generation",
                data_overview="Explanation unavailable"
            )
    
    def _generate_natural_language_summary(self, analysis: Dict[str, Any], 
                                         insights: List[Insight], row_count: int) -> str:
        """Generate natural language summary"""
        summary = f"Your query returned {row_count} rows of data"
        
        # Add quality information
        quality = analysis.get("data_quality", {})
        quality_score = quality.get("quality_score", 100)
        
        if quality_score >= 90:
            summary += " with excellent data quality."
        elif quality_score >= 70:
            summary += " with generally good data quality."
        else:
            summary += " with some data quality concerns."
        
        # Add insight information
        if insights:
            high_impact = [i for i in insights if i.impact == "high"]
            if high_impact:
                summary += f" Key findings include {len(high_impact)} high-impact insights."
            else:
                summary += f" Analysis revealed {len(insights)} insights."
        
        # Add pattern information
        patterns = analysis.get("patterns", {})
        if patterns:
            summary += f" The data shows {len(patterns)} notable patterns."
        
        return summary
    
    def _generate_technical_details(self, analysis: Dict[str, Any]) -> str:
        """Generate technical details"""
        details = []
        
        # Data structure
        basic_stats = analysis.get("basic_stats", {})
        numeric_cols = len([s for s in basic_stats.values() if s.get("type") == "numeric"])
        categorical_cols = len([s for s in basic_stats.values() if s.get("type") == "categorical"])
        
        details.append(f"Data structure: {numeric_cols} numeric columns, {categorical_cols} categorical columns")
        
        # Quality metrics
        quality = analysis.get("data_quality", {})
        if quality:
            details.append(f"Quality score: {quality.get('quality_score', 0):.1f}/100")
            details.append(f"Missing data: {quality.get('missing_percentage', 0):.1f}%")
        
        # Statistical measures
        if basic_stats:
            details.append(f"Statistical analysis performed on {len(basic_stats)} columns")
        
        return "; ".join(details)
    
    def _generate_data_overview(self, analysis: Dict[str, Any], row_count: int) -> str:
        """Generate data overview"""
        overview = f"Dataset contains {row_count} rows"
        
        # Column information
        basic_stats = analysis.get("basic_stats", {})
        if basic_stats:
            overview += f" across {len(basic_stats)} columns"
        
        # Data types
        numeric_cols = len([s for s in basic_stats.values() if s.get("type") == "numeric"])
        categorical_cols = len([s for s in basic_stats.values() if s.get("type") == "categorical"])
        
        if numeric_cols > 0:
            overview += f" ({numeric_cols} numeric, {categorical_cols} categorical)"
        
        overview += "."
        
        # Add quality note
        quality = analysis.get("data_quality", {})
        if quality.get("quality_score", 100) < 80:
            overview += " Data quality issues detected."
        
        return overview

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="AskData Result Explainer Service",
    description="Provides natural language explanations and insights for query results",
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
result_explainer_service = ResultExplainerService(settings)

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup"""
    setup_logging()
    logging.info("AskData Result Explainer Service starting up")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "result-explainer",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/mcp/explain_results", response_model=ExplainResultsOutput)
async def explain_results_mcp(request: ExplainResultsInput):
    """MCP tool: Explain query results"""
    return await result_explainer_service.explain_results(request)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"service": "result-explainer", "status": "running"}

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
