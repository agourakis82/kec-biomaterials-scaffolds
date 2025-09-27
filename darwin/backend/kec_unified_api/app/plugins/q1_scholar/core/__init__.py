"""
Q1 Scholar Core Module
Contains the main Q1 Scholar processing engines and utilities.
"""

from .models import (
    LaTeXDocument,
    BibTeXEntry,
    QualityReport,
    CitationAnalysis,
    JournalTemplate,
    Q1ValidationResult
)
from .q1_scholar_core import Q1ScholarCore

__all__ = [
    "LaTeXDocument",
    "BibTeXEntry",
    "QualityReport",
    "CitationAnalysis",
    "JournalTemplate",
    "Q1ValidationResult",
    "Q1ScholarCore"
]