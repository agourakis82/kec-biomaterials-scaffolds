"""
Tests for Q1 Scholar Core functionality.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.kec_unified_api.plugins.q1_scholar.core.q1_scholar_core import Q1ScholarCore
from src.kec_unified_api.plugins.q1_scholar.core.models import LaTeXDocument, AcademicProject


class TestQ1ScholarCore:
    """Test cases for Q1 Scholar Core"""

    @pytest.fixture
    def q1_scholar_core(self):
        """Create Q1 Scholar Core instance for testing"""
        return Q1ScholarCore()

    @pytest.fixture
    def sample_latex_document(self):
        """Create sample LaTeX document for testing"""
        return LaTeXDocument(
            content=r"""
            \documentclass{article}
            \title{Sample Research Paper}
            \author{Test Author}

            \begin{document}
            \maketitle

            \section{Introduction}
            This is a sample research paper for testing.

            \section{Methods}
            We used standard methods for this research.

            \section{Results}
            Our results show significant findings.

            \section{Discussion}
            These results have important implications.

            \bibliography{references}
            \end{document}
            """,
            metadata={"title": "Sample Research Paper", "author": "Test Author"}
        )

    @pytest.fixture
    def sample_academic_project(self):
        """Create sample academic project for testing"""
        return AcademicProject(
            id="test_project_001",
            title="Test Research Project",
            authors=["Test Author"],
            target_journal="Nature",
            domain="biomaterials",
            document=None
        )

    @pytest.mark.asyncio
    async def test_initialization(self, q1_scholar_core):
        """Test Q1 Scholar Core initialization"""
        assert q1_scholar_core is not None
        assert q1_scholar_core.darwin_core is None  # No DARWIN core provided
        assert q1_scholar_core.latex_processor is None  # Not implemented yet
        assert q1_scholar_core.bibtex_manager is None  # Not implemented yet

    @pytest.mark.asyncio
    async def test_analyze_document_basic(self, q1_scholar_core, sample_latex_document):
        """Test basic document analysis functionality"""
        result = await q1_scholar_core.analyze_document(sample_latex_document)

        assert isinstance(result, dict)
        assert "document_info" in result
        assert "analysis_results" in result
        assert "generated_at" in result

        # Check document info
        doc_info = result["document_info"]
        assert "title" in doc_info
        assert "word_count" in doc_info
        assert "sections" in doc_info

    @pytest.mark.asyncio
    async def test_analyze_document_with_journal(self, q1_scholar_core, sample_latex_document):
        """Test document analysis with specific journal target"""
        result = await q1_scholar_core.analyze_document(
            sample_latex_document,
            journal="Nature"
        )

        assert result["target_journal"] == "Nature"
        assert "journal_fit" in result["analysis_results"]

    @pytest.mark.asyncio
    async def test_validate_document(self, q1_scholar_core, sample_latex_document):
        """Test document validation functionality"""
        validation_result = await q1_scholar_core.validate_document(sample_latex_document)

        assert hasattr(validation_result, 'is_q1_ready')
        assert hasattr(validation_result, 'confidence_score')
        assert hasattr(validation_result, 'required_improvements')
        assert hasattr(validation_result, 'estimated_acceptance_probability')

        # Validation result should be a boolean
        assert isinstance(validation_result.is_q1_ready, bool)

        # Confidence score should be between 0 and 100
        assert 0 <= validation_result.confidence_score <= 100

    @pytest.mark.asyncio
    async def test_get_journal_templates(self, q1_scholar_core):
        """Test retrieval of journal templates"""
        templates = await q1_scholar_core.get_journal_templates()

        assert isinstance(templates, list)
        assert len(templates) > 0

        # Check template structure
        for template in templates:
            assert hasattr(template, 'journal_name')
            assert hasattr(template, 'tier')
            assert hasattr(template, 'citation_style')
            assert hasattr(template, 'word_limits')
            assert hasattr(template, 'template_content')

    @pytest.mark.asyncio
    async def test_create_collaboration_session(self, q1_scholar_core, sample_academic_project):
        """Test collaboration session creation"""
        participants = ["user1@example.com", "user2@example.com"]

        session = await q1_scholar_core.create_collaboration_session(
            sample_academic_project,
            participants
        )

        assert hasattr(session, 'id')
        assert hasattr(session, 'project')
        assert hasattr(session, 'participants')
        assert hasattr(session, 'is_active')

        assert session.project.id == sample_academic_project.id
        assert session.participants == participants
        assert session.is_active is True

    @pytest.mark.asyncio
    async def test_optimize_for_journal_basic(self, q1_scholar_core, sample_latex_document):
        """Test basic journal optimization"""
        optimized = await q1_scholar_core.optimize_for_journal(
            sample_latex_document,
            "Nature",
            "basic"
        )

        assert isinstance(optimized, LaTeXDocument)
        assert optimized.content is not None
        assert "optimization_validation" in optimized.metadata

    @pytest.mark.asyncio
    async def test_optimize_for_journal_comprehensive(self, q1_scholar_core, sample_latex_document):
        """Test comprehensive journal optimization"""
        optimized = await q1_scholar_core.optimize_for_journal(
            sample_latex_document,
            "Science",
            "comprehensive"
        )

        assert isinstance(optimized, LaTeXDocument)
        assert "optimization_validation" in optimized.metadata

    @pytest.mark.asyncio
    async def test_document_parsing(self, q1_scholar_core):
        """Test LaTeX document parsing"""
        latex_content = r"""
        \documentclass{article}
        \begin{document}
        \section{Test Section}
        This is test content.
        \end{document}
        """

        parsed = await q1_scholar_core._parse_latex_document(latex_content)

        assert isinstance(parsed, LaTeXDocument)
        assert parsed.content == latex_content

    @pytest.mark.asyncio
    async def test_quality_analysis(self, q1_scholar_core, sample_latex_document):
        """Test quality analysis functionality"""
        quality_report = await q1_scholar_core._analyze_quality(sample_latex_document)

        assert hasattr(quality_report, 'overall_score')
        assert hasattr(quality_report, 'methodology_score')
        assert hasattr(quality_report, 'citations')
        assert 0 <= quality_report.overall_score <= 100

    @pytest.mark.asyncio
    async def test_citation_analysis(self, q1_scholar_core, sample_latex_document):
        """Test citation analysis functionality"""
        citation_analysis = await q1_scholar_core._analyze_citations(sample_latex_document)

        assert hasattr(citation_analysis, 'total_citations')
        assert hasattr(citation_analysis, 'average_impact_factor')
        assert hasattr(citation_analysis, 'recent_citations_percentage')
        assert citation_analysis.total_citations >= 0

    @pytest.mark.asyncio
    async def test_error_handling(self, q1_scholar_core):
        """Test error handling in various scenarios"""
        # Test with invalid journal
        with pytest.raises(ValueError):
            await q1_scholar_core.optimize_for_journal(
                "invalid content",
                "NonExistentJournal"
            )

        # Test with empty document
        with pytest.raises(ValueError):
            await q1_scholar_core._parse_latex_document("")

    @pytest.mark.asyncio
    async def test_darwin_integration_placeholder(self, q1_scholar_core, sample_latex_document):
        """Test DARWIN integration placeholder functionality"""
        # Test without DARWIN core
        ai_analysis = await q1_scholar_core._ai_enhanced_analysis(sample_latex_document)

        assert "ai_analysis" in ai_analysis or "ai_insights" in ai_analysis

    def test_core_attributes(self, q1_scholar_core):
        """Test core attributes and configuration"""
        assert hasattr(q1_scholar_core, 'logger')
        assert hasattr(q1_scholar_core, 'darwin_core')
        assert hasattr(q1_scholar_core, 'latex_processor')
        assert hasattr(q1_scholar_core, 'bibtex_manager')
        assert hasattr(q1_scholar_core, 'quality_validator')
        assert hasattr(q1_scholar_core, 'collaboration_engine')
        assert hasattr(q1_scholar_core, 'template_engine')