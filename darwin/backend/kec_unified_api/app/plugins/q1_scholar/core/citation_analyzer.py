"""
Citation Analyzer for Q1 Scholar Plugin
Analyzes citation quality, impact factors, and academic influence networks.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

from .models import CitationAnalysis, BibTeXEntry, Bibliography

logger = logging.getLogger(__name__)


class CitationAnalyzer:
    """
    Advanced citation analysis engine for Q1 academic writing.

    Features:
    - Impact factor calculation and tracking
    - Citation network analysis
    - Quality assessment metrics
    - Temporal citation patterns
    - Cross-disciplinary influence mapping
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.impact_calculator = ImpactCalculator()
        self.network_analyzer = CitationNetworkAnalyzer()
        self.temporal_analyzer = TemporalCitationAnalyzer()
        self.quality_scorer = CitationQualityScorer()

    async def analyze_citation_quality(
        self,
        bibliography: Bibliography,
        document_context: Optional[str] = None
    ) -> CitationAnalysis:
        """
        Comprehensive citation quality analysis.

        Args:
            bibliography: Bibliography to analyze
            document_context: Document context for relevance analysis

        Returns:
            Complete citation analysis report
        """
        try:
            # Calculate impact metrics
            impact_metrics = await self.impact_calculator.calculate_impact(bibliography)

            # Analyze citation network
            network_metrics = await self.network_analyzer.analyze_network(bibliography)

            # Analyze temporal patterns
            temporal_metrics = await self.temporal_analyzer.analyze_temporal_patterns(bibliography)

            # Score overall quality
            quality_score = await self.quality_scorer.calculate_quality_score(
                bibliography, impact_metrics, network_metrics, temporal_metrics
            )

            # Generate recommendations
            recommendations = await self._generate_recommendations(
                impact_metrics, network_metrics, temporal_metrics, quality_score
            )

            return CitationAnalysis(
                total_citations=len(bibliography.entries),
                unique_sources=len(bibliography.entries),  # Simplified
                average_impact_factor=impact_metrics.get('average_if', 0.0),
                recent_citations_percentage=temporal_metrics.get('recent_percentage', 0.0),
                self_citation_rate=impact_metrics.get('self_citation_rate', 0.0),
                citation_network_centrality=network_metrics.get('centrality', 0.0),
                recommendations=recommendations
            )

        except Exception as e:
            self.logger.error(f"Citation analysis failed: {e}")
            raise

    async def assess_citation_relevance(
        self,
        citations: List[BibTeXEntry],
        document_context: str,
        domain: str
    ) -> Dict[str, float]:
        """
        Assess relevance of citations to document context.

        Args:
            citations: Citations to assess
            document_context: Document content
            domain: Research domain

        Returns:
            Relevance scores for each citation
        """
        try:
            relevance_scores = {}

            for citation in citations:
                score = await self._calculate_relevance_score(
                    citation, document_context, domain
                )
                relevance_scores[citation.key] = score

            return relevance_scores

        except Exception as e:
            self.logger.error(f"Relevance assessment failed: {e}")
            raise

    async def detect_citation_biases(
        self,
        bibliography: Bibliography
    ) -> Dict[str, Any]:
        """
        Detect potential citation biases and imbalances.

        Args:
            bibliography: Bibliography to analyze

        Returns:
            Bias analysis results
        """
        try:
            biases = {
                "geographic_bias": await self._analyze_geographic_distribution(bibliography),
                "institutional_bias": await self._analyze_institutional_distribution(bibliography),
                "temporal_bias": await self._analyze_temporal_distribution(bibliography),
                "methodological_bias": await self._analyze_methodological_distribution(bibliography)
            }

            return biases

        except Exception as e:
            self.logger.error(f"Bias detection failed: {e}")
            raise


class ImpactCalculator:
    """Calculates impact factors and citation metrics"""

    async def calculate_impact(self, bibliography: Bibliography) -> Dict[str, Any]:
        """Calculate impact metrics for bibliography"""
        try:
            impact_metrics = {
                "average_if": 0.0,
                "median_if": 0.0,
                "max_if": 0.0,
                "h_index": 0,
                "total_citations": 0,
                "self_citation_rate": 0.0
            }

            # Placeholder calculations - will be implemented with real data
            entries_with_impact = []

            for entry in bibliography.entries:
                # Simulate impact factor lookup
                impact_factor = await self._lookup_impact_factor(entry)
                if impact_factor:
                    entries_with_impact.append(impact_factor)

            if entries_with_impact:
                impact_metrics["average_if"] = sum(entries_with_impact) / len(entries_with_impact)
                impact_metrics["median_if"] = sorted(entries_with_impact)[len(entries_with_impact) // 2]
                impact_metrics["max_if"] = max(entries_with_impact)

            # Calculate H-index (simplified)
            impact_metrics["h_index"] = await self._calculate_h_index(bibliography)

            return impact_metrics

        except Exception as e:
            logger.error(f"Impact calculation failed: {e}")
            raise

    async def _lookup_impact_factor(self, entry: BibTeXEntry) -> Optional[float]:
        """Lookup impact factor for a journal"""
        # Placeholder - will be implemented with journal databases
        journal_name = entry.fields.get('journal', '').lower()

        # Mock impact factors for common journals
        impact_factors = {
            'nature': 49.962,
            'science': 47.728,
            'cell': 41.582,
            'lancet': 202.731,
            'new england journal of medicine': 176.079,
            'nature materials': 39.737,
            'advanced materials': 32.086,
            'biomaterials': 15.304,
            'materials science & engineering': 7.654
        }

        for journal, if_score in impact_factors.items():
            if journal in journal_name:
                return if_score

        # Default impact factor for unknown journals
        return 3.5

    async def _calculate_h_index(self, bibliography: Bibliography) -> int:
        """Calculate H-index for bibliography"""
        # Simplified H-index calculation
        # In reality, this would require citation counts for each paper
        return min(len(bibliography.entries), 15)  # Mock value


class CitationNetworkAnalyzer:
    """Analyzes citation networks and academic influence"""

    async def analyze_network(self, bibliography: Bibliography) -> Dict[str, Any]:
        """Analyze citation network properties"""
        try:
            network_metrics = {
                "centrality": 0.0,
                "clustering_coefficient": 0.0,
                "network_density": 0.0,
                "influential_nodes": [],
                "communities": []
            }

            # Placeholder network analysis
            # In a real implementation, this would build a citation graph

            # Calculate centrality (simplified)
            network_metrics["centrality"] = min(0.8, len(bibliography.entries) / 50.0)

            # Mock clustering coefficient
            network_metrics["clustering_coefficient"] = 0.65

            # Calculate network density
            total_possible_connections = len(bibliography.entries) * (len(bibliography.entries) - 1) / 2
            if total_possible_connections > 0:
                network_metrics["network_density"] = min(1.0, len(bibliography.entries) / total_possible_connections)

            return network_metrics

        except Exception as e:
            logger.error(f"Network analysis failed: {e}")
            raise


class TemporalCitationAnalyzer:
    """Analyzes temporal patterns in citations"""

    async def analyze_temporal_patterns(self, bibliography: Bibliography) -> Dict[str, Any]:
        """Analyze temporal distribution of citations"""
        try:
            temporal_metrics = {
                "recent_percentage": 0.0,
                "average_age": 0.0,
                "temporal_distribution": {},
                "trending_topics": []
            }

            current_year = datetime.now().year
            years = []

            for entry in bibliography.entries:
                year_str = entry.fields.get('year')
                if year_str:
                    try:
                        year = int(year_str)
                        years.append(year)
                    except ValueError:
                        continue

            if years:
                # Calculate recent citations (last 5 years)
                recent_years = [y for y in years if current_year - y <= 5]
                temporal_metrics["recent_percentage"] = len(recent_years) / len(years)

                # Calculate average age
                ages = [current_year - y for y in years]
                temporal_metrics["average_age"] = sum(ages) / len(ages)

                # Create temporal distribution
                distribution = {}
                for year in sorted(set(years)):
                    distribution[str(year)] = years.count(year)
                temporal_metrics["temporal_distribution"] = distribution

            return temporal_metrics

        except Exception as e:
            logger.error(f"Temporal analysis failed: {e}")
            raise


class CitationQualityScorer:
    """Scores overall citation quality"""

    async def calculate_quality_score(
        self,
        bibliography: Bibliography,
        impact_metrics: Dict[str, Any],
        network_metrics: Dict[str, Any],
        temporal_metrics: Dict[str, Any]
    ) -> float:
        """Calculate overall citation quality score"""
        try:
            score = 0.0

            # Impact factor component (40%)
            avg_if = impact_metrics.get('average_if', 0.0)
            if avg_if >= 10:
                score += 0.4
            elif avg_if >= 5:
                score += 0.3
            elif avg_if >= 2:
                score += 0.2
            else:
                score += 0.1

            # Network centrality component (20%)
            centrality = network_metrics.get('centrality', 0.0)
            score += centrality * 0.2

            # Recency component (20%)
            recent_pct = temporal_metrics.get('recent_percentage', 0.0)
            score += recent_pct * 0.2

            # Diversity component (10%) - placeholder
            score += 0.1

            # Completeness component (10%) - placeholder
            score += 0.1

            return min(score, 1.0)

        except Exception as e:
            logger.error(f"Quality scoring failed: {e}")
            return 0.0

    # Private helper methods for CitationAnalyzer

    async def _calculate_relevance_score(
        self,
        citation: BibTeXEntry,
        document_context: str,
        domain: str
    ) -> float:
        """Calculate relevance score for a citation"""
        score = 0.0

        # Title relevance
        title = citation.fields.get('title', '').lower()
        context_lower = document_context.lower()

        title_words = set(title.split())
        context_words = set(context_lower.split())

        overlap = len(title_words.intersection(context_words))
        if overlap > 0:
            score += min(0.4, overlap / len(title_words))

        # Abstract relevance (if available)
        abstract = citation.fields.get('abstract', '').lower()
        if abstract:
            abstract_words = set(abstract.split())
            abstract_overlap = len(abstract_words.intersection(context_words))
            if abstract_overlap > 0:
                score += min(0.3, abstract_overlap / len(abstract_words))

        # Domain relevance
        journal = citation.fields.get('journal', '').lower()
        if domain.lower() in journal:
            score += 0.2

        # Recency bonus
        year_str = citation.fields.get('year')
        if year_str:
            try:
                year = int(year_str)
                current_year = datetime.now().year
                if current_year - year <= 5:
                    score += 0.1
            except ValueError:
                pass

        return min(score, 1.0)

    async def _analyze_geographic_distribution(self, bibliography: Bibliography) -> Dict[str, Any]:
        """Analyze geographic distribution of citations"""
        # Placeholder - would analyze author affiliations
        return {"bias_detected": False, "recommendations": []}

    async def _analyze_institutional_distribution(self, bibliography: Bibliography) -> Dict[str, Any]:
        """Analyze institutional distribution of citations"""
        # Placeholder - would analyze author institutions
        return {"bias_detected": False, "recommendations": []}

    async def _analyze_temporal_distribution(self, bibliography: Bibliography) -> Dict[str, Any]:
        """Analyze temporal distribution for bias detection"""
        # Placeholder - would detect temporal biases
        return {"bias_detected": False, "recommendations": []}

    async def _analyze_methodological_distribution(self, bibliography: Bibliography) -> Dict[str, Any]:
        """Analyze methodological distribution of citations"""
        # Placeholder - would analyze research methods
        return {"bias_detected": False, "recommendations": []}

    async def _generate_recommendations(
        self,
        impact_metrics: Dict[str, Any],
        network_metrics: Dict[str, Any],
        temporal_metrics: Dict[str, Any],
        quality_score: float
    ) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []

        # Impact factor recommendations
        avg_if = impact_metrics.get('average_if', 0.0)
        if avg_if < 5.0:
            recommendations.append("Consider including more citations from high-impact journals (IF > 5.0)")

        # Recency recommendations
        recent_pct = temporal_metrics.get('recent_percentage', 0.0)
        if recent_pct < 0.6:
            recommendations.append("Include more recent citations (last 5 years) to show current research trends")

        # Network recommendations
        centrality = network_metrics.get('centrality', 0.0)
        if centrality < 0.5:
            recommendations.append("Diversify citation sources to improve network centrality")

        # Quality recommendations
        if quality_score < 0.7:
            recommendations.append("Overall citation quality needs improvement - review selection criteria")

        return recommendations