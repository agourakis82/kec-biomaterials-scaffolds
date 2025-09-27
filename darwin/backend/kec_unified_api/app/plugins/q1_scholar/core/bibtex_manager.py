"""
BibTeX Manager for Q1 Scholar Plugin
Handles BibTeX processing, citation management, and academic database integration.
"""

import re
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode

from .models import BibTeXEntry, Bibliography, Citation

logger = logging.getLogger(__name__)


class BibTeXManager:
    """
    Advanced BibTeX management system for Q1 academic writing.

    Features:
    - BibTeX parsing and validation
    - Citation quality analysis
    - Impact factor tracking
    - Academic database integration
    - Citation style adaptation
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.database_connector = None  # Will be implemented in Sprint 3
        self.citation_analyzer = None   # Will be implemented in Sprint 3
        self.style_adapter = None       # Will be implemented in Sprint 3

    async def process_bibliography(
        self,
        bibtex_content: str,
        target_journal: Optional[str] = None
    ) -> Bibliography:
        """
        Process and optimize bibliography for Q1 standards.

        Args:
            bibtex_content: Raw BibTeX content
            target_journal: Target journal for optimization

        Returns:
            Processed and optimized bibliography
        """
        try:
            # Parse BibTeX content
            entries = await self._parse_bibtex(bibtex_content)

            # Validate and clean entries
            validated_entries = await self._validate_entries(entries)

            # Enrich with metadata from academic databases
            enriched_entries = await self._enrich_metadata(validated_entries)

            # Optimize for target journal if specified
            if target_journal:
                enriched_entries = await self._optimize_for_journal(enriched_entries, target_journal)

            # Create bibliography object
            bibliography = Bibliography(
                entries=enriched_entries,
                style=self._detect_citation_style(bibtex_content),
                raw_content=bibtex_content
            )

            return bibliography

        except Exception as e:
            self.logger.error(f"Bibliography processing failed: {e}")
            raise

    async def suggest_citations(
        self,
        context: str,
        domain: str,
        current_bibliography: Optional[Bibliography] = None,
        max_suggestions: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Suggest relevant citations based on context and domain.

        Args:
            context: Text context for citation suggestions
            domain: Research domain (e.g., 'biomaterials', 'neuroscience')
            current_bibliography: Current bibliography to avoid duplicates
            max_suggestions: Maximum number of suggestions

        Returns:
            List of citation suggestions with relevance scores
        """
        try:
            # Extract keywords and concepts from context
            keywords = await self._extract_keywords(context)
            concepts = await self._extract_concepts(context, domain)

            # Search academic databases
            candidates = await self._search_databases(keywords, concepts, domain)

            # Filter out existing citations
            if current_bibliography:
                candidates = await self._filter_existing(candidates, current_bibliography)

            # Rank and score candidates
            ranked_candidates = await self._rank_candidates(candidates, context, domain)

            # Return top suggestions
            return ranked_candidates[:max_suggestions]

        except Exception as e:
            self.logger.error(f"Citation suggestion failed: {e}")
            raise

    async def validate_citations(
        self,
        bibliography: Bibliography,
        document_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate citation quality and consistency.

        Args:
            bibliography: Bibliography to validate
            document_context: Document context for relevance checking

        Returns:
            Validation results with issues and recommendations
        """
        try:
            validation_results = {
                "syntax_errors": [],
                "missing_fields": [],
                "quality_issues": [],
                "consistency_issues": [],
                "relevance_score": 0.0,
                "recommendations": []
            }

            # Check BibTeX syntax
            validation_results["syntax_errors"] = await self._check_syntax(bibliography)

            # Validate required fields
            validation_results["missing_fields"] = await self._check_required_fields(bibliography)

            # Assess citation quality
            validation_results["quality_issues"] = await self._assess_quality(bibliography)

            # Check consistency
            validation_results["consistency_issues"] = await self._check_consistency(bibliography)

            # Calculate relevance score
            if document_context:
                validation_results["relevance_score"] = await self._calculate_relevance(
                    bibliography, document_context
                )

            # Generate recommendations
            validation_results["recommendations"] = await self._generate_recommendations(validation_results)

            return validation_results

        except Exception as e:
            self.logger.error(f"Citation validation failed: {e}")
            raise

    async def adapt_citation_style(
        self,
        bibliography: Bibliography,
        target_style: str
    ) -> Bibliography:
        """
        Adapt bibliography to different citation styles.

        Args:
            bibliography: Original bibliography
            target_style: Target citation style (APA, IEEE, Nature, etc.)

        Returns:
            Bibliography adapted to target style
        """
        try:
            # This will be implemented with citation style adapter
            # For now, return original bibliography
            self.logger.info(f"Adapting bibliography to {target_style} style")
            return bibliography

        except Exception as e:
            self.logger.error(f"Style adaptation failed: {e}")
            raise

    # Private helper methods

    async def _parse_bibtex(self, bibtex_content: str) -> List[BibTeXEntry]:
        """Parse BibTeX content into structured entries"""
        try:
            # Configure parser
            parser = BibTexParser()
            parser.customization = convert_to_unicode

            # Parse content
            bib_database = bibtexparser.loads(bibtex_content, parser=parser)

            # Convert to our model
            entries = []
            for entry_dict in bib_database.entries:
                entry = BibTeXEntry(
                    key=entry_dict.get('ID', ''),
                    entry_type=entry_dict.get('ENTRYTYPE', ''),
                    fields=entry_dict,
                    raw_entry=self._dict_to_bibtex(entry_dict)
                )

                # Extract additional metadata
                entry.doi = entry_dict.get('doi')
                entry.pmid = entry_dict.get('pmid')
                entry.arxiv_id = entry_dict.get('arxiv_id') or entry_dict.get('eprint')

                entries.append(entry)

            return entries

        except Exception as e:
            self.logger.error(f"BibTeX parsing failed: {e}")
            raise

    async def _validate_entries(self, entries: List[BibTeXEntry]) -> List[BibTeXEntry]:
        """Validate and clean BibTeX entries"""
        validated = []

        for entry in entries:
            # Basic validation
            if not entry.key or not entry.entry_type:
                self.logger.warning(f"Skipping invalid entry: {entry}")
                continue

            # Clean and normalize fields
            cleaned_entry = await self._clean_entry(entry)
            validated.append(cleaned_entry)

        return validated

    async def _enrich_metadata(self, entries: List[BibTeXEntry]) -> List[BibTeXEntry]:
        """Enrich entries with metadata from academic databases"""
        # Placeholder - will be implemented with database connector
        # For now, return entries as-is
        return entries

    async def _optimize_for_journal(
        self,
        entries: List[BibTeXEntry],
        journal: str
    ) -> List[BibTeXEntry]:
        """Optimize bibliography for specific journal requirements"""
        # Placeholder - will be implemented with journal-specific logic
        return entries

    def _detect_citation_style(self, bibtex_content: str) -> str:
        """Detect citation style from BibTeX content"""
        # Simple detection based on common patterns
        if '@article' in bibtex_content.lower():
            return 'APA'  # Default assumption
        return 'APA'

    async def _extract_keywords(self, context: str) -> List[str]:
        """Extract keywords from context"""
        # Simple keyword extraction - will be enhanced with NLP
        words = re.findall(r'\b\w+\b', context.lower())
        keywords = [word for word in words if len(word) > 3]
        return list(set(keywords))[:20]  # Top 20 unique keywords

    async def _extract_concepts(self, context: str, domain: str) -> List[str]:
        """Extract research concepts from context"""
        # Placeholder - will be implemented with domain-specific concept extraction
        return [domain]

    async def _search_databases(
        self,
        keywords: List[str],
        concepts: List[str],
        domain: str
    ) -> List[Dict[str, Any]]:
        """Search academic databases for relevant papers"""
        # Placeholder - will be implemented with actual database APIs
        return []

    async def _filter_existing(
        self,
        candidates: List[Dict[str, Any]],
        bibliography: Bibliography
    ) -> List[Dict[str, Any]]:
        """Filter out citations already in bibliography"""
        existing_dois = {entry.doi for entry in bibliography.entries if entry.doi}

        filtered = []
        for candidate in candidates:
            candidate_doi = candidate.get('doi')
            if candidate_doi not in existing_dois:
                filtered.append(candidate)

        return filtered

    async def _rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        context: str,
        domain: str
    ) -> List[Dict[str, Any]]:
        """Rank candidates by relevance"""
        # Simple ranking - will be enhanced with ML models
        for candidate in candidates:
            # Calculate relevance score based on various factors
            relevance_score = await self._calculate_relevance_score(candidate, context, domain)
            candidate['relevance_score'] = relevance_score

        # Sort by relevance score
        candidates.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return candidates

    async def _calculate_relevance_score(
        self,
        candidate: Dict[str, Any],
        context: str,
        domain: str
    ) -> float:
        """Calculate relevance score for a candidate citation"""
        score = 0.0

        # Title relevance
        title = candidate.get('title', '').lower()
        context_lower = context.lower()
        if any(word in title for word in context_lower.split()):
            score += 0.3

        # Abstract relevance
        abstract = candidate.get('abstract', '').lower()
        if any(word in abstract for word in context_lower.split()):
            score += 0.4

        # Recency bonus
        year = candidate.get('year')
        if year and isinstance(year, (int, str)):
            try:
                year_int = int(year)
                current_year = datetime.now().year
                years_old = current_year - year_int
                if years_old <= 5:
                    score += 0.3
                elif years_old <= 10:
                    score += 0.1
            except ValueError:
                pass

        return min(score, 1.0)  # Cap at 1.0

    async def _check_syntax(self, bibliography: Bibliography) -> List[str]:
        """Check BibTeX syntax errors"""
        errors = []

        for entry in bibliography.entries:
            # Check for missing required fields based on entry type
            required_fields = self._get_required_fields(entry.entry_type)
            missing = [field for field in required_fields if field not in entry.fields]
            if missing:
                errors.append(f"Entry {entry.key}: missing required fields {missing}")

        return errors

    async def _check_required_fields(self, bibliography: Bibliography) -> List[str]:
        """Check for missing required fields"""
        issues = []

        for entry in bibliography.entries:
            required = self._get_required_fields(entry.entry_type)
            missing = [field for field in required if field not in entry.fields]
            if missing:
                issues.append(f"{entry.key}: missing {', '.join(missing)}")

        return issues

    async def _assess_quality(self, bibliography: Bibliography) -> List[str]:
        """Assess citation quality issues"""
        issues = []

        for entry in bibliography.entries:
            # Check for DOI
            if not entry.doi:
                issues.append(f"{entry.key}: missing DOI")

            # Check publication year
            year = entry.fields.get('year')
            if not year:
                issues.append(f"{entry.key}: missing publication year")
            else:
                try:
                    year_int = int(year)
                    if year_int < 1950 or year_int > datetime.now().year + 1:
                        issues.append(f"{entry.key}: suspicious publication year {year}")
                except ValueError:
                    issues.append(f"{entry.key}: invalid publication year format")

        return issues

    async def _check_consistency(self, bibliography: Bibliography) -> List[str]:
        """Check bibliography consistency"""
        issues = []

        # Check for duplicate keys
        keys = [entry.key for entry in bibliography.entries]
        duplicates = set([key for key in keys if keys.count(key) > 1])
        for duplicate in duplicates:
            issues.append(f"Duplicate entry key: {duplicate}")

        # Check for duplicate DOIs
        dois = [entry.doi for entry in bibliography.entries if entry.doi]
        duplicate_dois = set([doi for doi in dois if dois.count(doi) > 1])
        for duplicate in duplicate_dois:
            issues.append(f"Duplicate DOI: {duplicate}")

        return issues

    async def _calculate_relevance(
        self,
        bibliography: Bibliography,
        document_context: str
    ) -> float:
        """Calculate overall bibliography relevance"""
        # Placeholder - simple relevance calculation
        return 0.75

    async def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []

        if validation_results["syntax_errors"]:
            recommendations.append("Fix BibTeX syntax errors")

        if validation_results["missing_fields"]:
            recommendations.append("Add missing required fields to entries")

        if validation_results["quality_issues"]:
            recommendations.append("Improve citation quality (add DOIs, verify years)")

        if validation_results["consistency_issues"]:
            recommendations.append("Resolve duplicate entries and inconsistencies")

        if validation_results["relevance_score"] < 0.7:
            recommendations.append("Consider adding more recent and relevant citations")

        return recommendations

    def _get_required_fields(self, entry_type: str) -> List[str]:
        """Get required fields for entry type"""
        required_fields = {
            'article': ['author', 'title', 'journal', 'year'],
            'book': ['author', 'title', 'publisher', 'year'],
            'inproceedings': ['author', 'title', 'booktitle', 'year'],
            'phdthesis': ['author', 'title', 'school', 'year'],
            'mastersthesis': ['author', 'title', 'school', 'year']
        }
        return required_fields.get(entry_type.lower(), ['author', 'title', 'year'])

    async def _clean_entry(self, entry: BibTeXEntry) -> BibTeXEntry:
        """Clean and normalize entry fields"""
        # Basic cleaning - will be enhanced
        return entry

    def _dict_to_bibtex(self, entry_dict: Dict[str, Any]) -> str:
        """Convert entry dict back to BibTeX format"""
        # Simple conversion - will be enhanced
        bibtex = f"@{entry_dict.get('ENTRYTYPE', 'article')}{{{entry_dict.get('ID', 'key')},\n"
        for key, value in entry_dict.items():
            if key not in ['ENTRYTYPE', 'ID']:
                bibtex += f"  {key} = {{{value}}},\n"
        bibtex += "}\n"
        return bibtex