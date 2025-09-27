"""
DARWIN Q1 Scholar Plugin
Academic writing plugin for Q1 journal publications with native LaTeX/BibTeX support.

This plugin provides:
- Advanced LaTeX document processing and optimization
- BibTeX management with impact factor tracking
- Q1 quality gates validation
- Real-time collaborative writing
- Journal-specific template optimization
- AI-powered academic content enhancement
"""

from .core import Q1ScholarCore
from .api.routes import router as q1_scholar_router

__version__ = "1.0.0"
__author__ = "DARWIN Platform"
__description__ = "Q1 Academic Writing Plugin for DARWIN"

# Plugin metadata for DARWIN integration
PLUGIN_METADATA = {
    "name": "q1_scholar",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "capabilities": [
        "latex_processing",
        "bibtex_management",
        "quality_validation",
        "collaborative_writing",
        "journal_optimization",
        "academic_ai_enhancement"
    ],
    "dependencies": [
        "pylatexenc>=2.10",
        "bibtexparser>=1.4",
        "scholarly>=1.7",
        "crossrefapi>=1.1",
        "habanero>=1.2",
        "arxiv>=1.4"
    ],
    "api_endpoints": [
        "/q1-scholar/analyze",
        "/q1-scholar/optimize",
        "/q1-scholar/validate",
        "/q1-scholar/collaborate",
        "/q1-scholar/templates"
    ]
}

# Global plugin instance
_q1_scholar_instance = None

def get_q1_scholar() -> Q1ScholarCore:
    """Get or create Q1 Scholar plugin instance"""
    global _q1_scholar_instance
    if _q1_scholar_instance is None:
        _q1_scholar_instance = Q1ScholarCore()
    return _q1_scholar_instance

def initialize_plugin(darwin_core):
    """Initialize Q1 Scholar plugin with DARWIN core"""
    global _q1_scholar_instance
    _q1_scholar_instance = Q1ScholarCore(darwin_core=darwin_core)
    return _q1_scholar_instance

# Export main components
__all__ = [
    "Q1ScholarCore",
    "q1_scholar_router",
    "get_q1_scholar",
    "initialize_plugin",
    "PLUGIN_METADATA"
]