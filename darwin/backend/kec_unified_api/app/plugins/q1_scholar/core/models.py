"""
Pydantic models for Q1 Scholar Plugin
Data structures for LaTeX documents, citations, quality reports, and validation results.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator


class DocumentSection(str, Enum):
    """LaTeX document sections"""
    TITLE = "title"
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    METHODS = "methods"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    APPENDIX = "appendix"


class JournalTier(str, Enum):
    """Journal quality tiers"""
    Q1 = "q1"
    Q2 = "q2"
    Q3 = "q3"
    Q4 = "q4"


class CitationStyle(str, Enum):
    """Citation styles supported"""
    APA = "apa"
    IEEE = "ieee"
    NATURE = "nature"
    SCIENCE = "science"
    CELL = "cell"
    AMA = "ama"
    CHICAGO = "chicago"


class LaTeXNode(BaseModel):
    """AST node for LaTeX document structure"""
    type: str = Field(..., description="Node type (command, environment, text)")
    content: Union[str, List['LaTeXNode']] = Field(..., description="Node content")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Node attributes")
    position: Optional[Dict[str, int]] = Field(None, description="Position in source")

    class Config:
        allow_population_by_field_name = True


class LaTeXDocument(BaseModel):
    """Complete LaTeX document representation"""
    content: str = Field(..., description="Raw LaTeX content")
    ast: Optional[List[LaTeXNode]] = Field(None, description="Parsed AST")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    sections: Dict[DocumentSection, str] = Field(default_factory=dict, description="Extracted sections")
    bibliography: Optional['Bibliography'] = Field(None, description="Document bibliography")
    citations: List['Citation'] = Field(default_factory=list, description="Document citations")
    figures: List['Figure'] = Field(default_factory=list, description="Document figures")
    tables: List['Table'] = Field(default_factory=list, description="Document tables")
    equations: List['Equation'] = Field(default_factory=list, description="Document equations")

    @validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError("Document content cannot be empty")
        return v


class BibTeXEntry(BaseModel):
    """BibTeX entry representation"""
    key: str = Field(..., description="BibTeX citation key")
    entry_type: str = Field(..., description="Entry type (article, book, etc.)")
    fields: Dict[str, str] = Field(..., description="BibTeX fields")
    raw_entry: str = Field(..., description="Raw BibTeX entry")
    doi: Optional[str] = Field(None, description="DOI if available")
    pmid: Optional[str] = Field(None, description="PubMed ID if available")
    arxiv_id: Optional[str] = Field(None, description="ArXiv ID if available")


class Bibliography(BaseModel):
    """Document bibliography"""
    entries: List[BibTeXEntry] = Field(default_factory=list, description="BibTeX entries")
    style: CitationStyle = Field(CitationStyle.APA, description="Citation style")
    raw_content: Optional[str] = Field(None, description="Raw bibliography content")


class Citation(BaseModel):
    """Citation reference in document"""
    key: str = Field(..., description="Citation key")
    context: str = Field(..., description="Surrounding context")
    position: Dict[str, int] = Field(..., description="Position in document")
    bibtex_entry: Optional[BibTeXEntry] = Field(None, description="Associated BibTeX entry")


class Figure(BaseModel):
    """Figure in document"""
    label: str = Field(..., description="Figure label")
    caption: str = Field(..., description="Figure caption")
    path: Optional[str] = Field(None, description="Figure file path")
    position: Dict[str, int] = Field(..., description="Position in document")


class Table(BaseModel):
    """Table in document"""
    label: str = Field(..., description="Table label")
    caption: str = Field(..., description="Table caption")
    content: str = Field(..., description="Table content")
    position: Dict[str, int] = Field(..., description="Position in document")


class Equation(BaseModel):
    """Equation in document"""
    label: str = Field(..., description="Equation label")
    content: str = Field(..., description="Equation LaTeX content")
    position: Dict[str, int] = Field(..., description="Position in document")


class QualityScore(BaseModel):
    """Quality score with detailed breakdown"""
    score: float = Field(..., ge=0, le=100, description="Quality score (0-100)")
    max_score: float = Field(100, description="Maximum possible score")
    criteria: Dict[str, float] = Field(default_factory=dict, description="Individual criteria scores")
    feedback: List[str] = Field(default_factory=list, description="Detailed feedback")


class QualityReport(BaseModel):
    """Complete quality assessment report"""
    overall_score: float = Field(..., ge=0, le=100, description="Overall quality score")
    methodology_score: QualityScore = Field(..., description="Methodology quality")
    originality_score: QualityScore = Field(..., description="Originality assessment")
    citation_score: QualityScore = Field(..., description="Citation quality")
    statistical_score: QualityScore = Field(..., description="Statistical rigor")
    reproducibility_score: QualityScore = Field(..., description="Reproducibility")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
    generated_at: datetime = Field(default_factory=datetime.now, description="Report generation time")


class CitationAnalysis(BaseModel):
    """Citation quality analysis"""
    total_citations: int = Field(..., description="Total number of citations")
    unique_sources: int = Field(..., description="Number of unique sources")
    average_impact_factor: float = Field(..., description="Average impact factor")
    recent_citations_percentage: float = Field(..., description="Percentage of citations from last 5 years")
    self_citation_rate: float = Field(..., description="Self-citation rate")
    citation_network_centrality: float = Field(..., description="Citation network centrality")
    recommendations: List[str] = Field(default_factory=list, description="Citation improvement suggestions")


class JournalTemplate(BaseModel):
    """Journal-specific template configuration"""
    journal_name: str = Field(..., description="Journal name")
    tier: JournalTier = Field(..., description="Journal quality tier")
    citation_style: CitationStyle = Field(..., description="Required citation style")
    word_limits: Dict[str, int] = Field(default_factory=dict, description="Section word limits")
    formatting_rules: Dict[str, Any] = Field(default_factory=dict, description="Formatting requirements")
    template_content: str = Field(..., description="LaTeX template content")
    validation_rules: Dict[str, Any] = Field(default_factory=dict, description="Validation rules")


class Q1ValidationResult(BaseModel):
    """Q1-specific validation result"""
    is_q1_ready: bool = Field(..., description="Document ready for Q1 submission")
    confidence_score: float = Field(..., ge=0, le=100, description="Confidence in assessment")
    required_improvements: List[str] = Field(default_factory=list, description="Required improvements")
    optional_improvements: List[str] = Field(default_factory=list, description="Optional improvements")
    estimated_acceptance_probability: float = Field(..., ge=0, le=100, description="Estimated acceptance probability")
    alternative_journals: List[str] = Field(default_factory=list, description="Alternative journal suggestions")
    validated_at: datetime = Field(default_factory=datetime.now, description="Validation timestamp")


class AcademicProject(BaseModel):
    """Academic writing project"""
    id: str = Field(..., description="Project unique identifier")
    title: str = Field(..., description="Project title")
    authors: List[str] = Field(default_factory=list, description="Project authors")
    target_journal: str = Field(..., description="Target journal")
    domain: str = Field(..., description="Research domain")
    document: Optional[LaTeXDocument] = Field(None, description="Current document")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")


class CollaborationSession(BaseModel):
    """Real-time collaboration session"""
    id: str = Field(..., description="Session unique identifier")
    project: AcademicProject = Field(..., description="Associated project")
    participants: List[str] = Field(default_factory=list, description="Active participants")
    document_version: str = Field(..., description="Current document version")
    is_active: bool = Field(True, description="Session active status")
    created_at: datetime = Field(default_factory=datetime.now, description="Session creation time")
    last_activity: datetime = Field(default_factory=datetime.now, description="Last activity timestamp")


class DocumentEdit(BaseModel):
    """Document edit operation"""
    operation: str = Field(..., description="Edit operation type")
    position: Dict[str, int] = Field(..., description="Edit position")
    content: str = Field(..., description="Edit content")
    user_id: str = Field(..., description="User who made the edit")
    timestamp: datetime = Field(default_factory=datetime.now, description="Edit timestamp")


class Conflict(BaseModel):
    """Edit conflict in collaborative editing"""
    id: str = Field(..., description="Conflict unique identifier")
    conflicting_edits: List[DocumentEdit] = Field(..., description="Conflicting edits")
    resolution_options: List[str] = Field(default_factory=list, description="Possible resolutions")
    resolved: bool = Field(False, description="Conflict resolution status")
    resolved_by: Optional[str] = Field(None, description="User who resolved conflict")


# Update forward references
LaTeXNode.update_forward_refs()
LaTeXDocument.update_forward_refs()