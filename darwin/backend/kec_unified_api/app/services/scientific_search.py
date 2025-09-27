"""Scientific Search Service

Serviço especializado para busca científica avançada com validação,
análise de citações e enriquecimento de metadados.
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
from urllib.parse import urlparse, quote
import hashlib

# Importações opcionais
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    HTTP_AVAILABLE = True
except ImportError:
    requests = None
    HTTP_AVAILABLE = False

try:
    import feedparser
    RSS_AVAILABLE = True
except ImportError:
    feedparser = None
    RSS_AVAILABLE = False

try:
    import xml.etree.ElementTree as ET
    XML_AVAILABLE = True
except ImportError:
    ET = None
    XML_AVAILABLE = False

from .vertex_ai_client import VertexAIClient, get_vertex_client
from .rag_engine import RAGEngine, get_rag_engine, QueryResult
from ..models.rag_models import (
    QueryDomain, SourceType, ScientificSearchRequest,
    BiomaterialsQueryRequest, CrossDomainRequest,
    DiscoveryRequest, DiscoveryResponse
)

logger = logging.getLogger(__name__)


class DOIResolver:
    """Resolvedor de DOI para enriquecimento de metadados"""
    
    def __init__(self):
        self.session = None
        if HTTP_AVAILABLE:
            self.session = requests.Session()
            
            # Configuração de retry
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
            
        self.cache = {}
    
    async def resolve_doi(self, doi: str) -> Dict[str, Any]:
        """Resolve DOI e retorna metadados enriquecidos"""
        if not self.session:
            return {}
            
        # Limpa DOI
        clean_doi = doi.strip().replace("doi:", "").replace("https://doi.org/", "")
        
        if clean_doi in self.cache:
            return self.cache[clean_doi]
        
        try:
            # CrossRef API
            url = f"https://api.crossref.org/works/{clean_doi}"
            headers = {
                "Accept": "application/json",
                "User-Agent": "DARWIN-RAG++ (mailto:research@example.com)"
            }
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.session.get(url, headers=headers, timeout=10)
            )
            
            if response.status_code == 200:
                data = response.json()
                work = data.get("message", {})
                
                metadata = {
                    "doi": clean_doi,
                    "title": " ".join(work.get("title", [])),
                    "authors": [
                        f"{author.get('given', '')} {author.get('family', '')}"
                        for author in work.get("author", [])
                    ],
                    "journal": work.get("container-title", [""])[0] if work.get("container-title") else "",
                    "published_date": self._parse_date(work.get("published-print") or work.get("published-online")),
                    "abstract": work.get("abstract", ""),
                    "citation_count": work.get("is-referenced-by-count", 0),
                    "references": len(work.get("reference", [])),
                    "url": f"https://doi.org/{clean_doi}",
                    "publisher": work.get("publisher", ""),
                    "type": work.get("type", ""),
                    "subject": work.get("subject", [])
                }
                
                self.cache[clean_doi] = metadata
                return metadata
                
        except Exception as e:
            logger.error(f"Error resolving DOI {clean_doi}: {e}")
            
        return {"doi": clean_doi, "error": "Could not resolve"}
    
    def _parse_date(self, date_parts: Optional[Dict[str, Any]]) -> Optional[str]:
        """Parse date parts from CrossRef"""
        if not date_parts or not isinstance(date_parts, dict):
            return None
            
        date_array = date_parts.get("date-parts", [[]])
        if date_array and date_array[0]:
            parts = date_array[0]
            year = parts[0] if len(parts) > 0 else None
            month = parts[1] if len(parts) > 1 else 1
            day = parts[2] if len(parts) > 2 else 1
            
            if year:
                return f"{year:04d}-{month:02d}-{day:02d}"
        
        return None


class CitationNetworkAnalyzer:
    """Analisador de redes de citação"""
    
    def __init__(self):
        self.citation_graph = defaultdict(set)  # doi -> set of citing dois
        self.impact_scores = {}
    
    def add_citation(self, cited_doi: str, citing_doi: str):
        """Adiciona citação ao grafo"""
        self.citation_graph[cited_doi].add(citing_doi)
    
    def calculate_impact_score(self, doi: str, decay_factor: float = 0.1) -> float:
        """Calcula score de impacto com decaimento temporal"""
        if doi in self.impact_scores:
            return self.impact_scores[doi]
        
        # Score baseado em citações diretas
        direct_citations = len(self.citation_graph.get(doi, set()))
        
        # Score baseado em citações indiretas (simplified PageRank-like)
        indirect_score = 0.0
        for citing_doi in self.citation_graph.get(doi, set()):
            citing_citations = len(self.citation_graph.get(citing_doi, set()))
            indirect_score += citing_citations * decay_factor
        
        total_score = direct_citations + indirect_score
        self.impact_scores[doi] = total_score
        
        return total_score
    
    def find_related_papers(self, doi: str, max_depth: int = 2) -> Set[str]:
        """Encontra papers relacionados via rede de citação"""
        related = set()
        visited = set()
        queue = [(doi, 0)]
        
        while queue and len(related) < 50:  # Limit results
            current_doi, depth = queue.pop(0)
            
            if current_doi in visited or depth > max_depth:
                continue
                
            visited.add(current_doi)
            
            # Add papers that cite this one
            for citing_doi in self.citation_graph.get(current_doi, set()):
                if citing_doi not in visited:
                    related.add(citing_doi)
                    if depth < max_depth:
                        queue.append((citing_doi, depth + 1))
        
        return related


class ScientificSourceValidator:
    """Validador de fontes científicas"""
    
    def __init__(self):
        # Lista de journals confiáveis (simplificada)
        self.trusted_journals = {
            "nature", "science", "cell", "lancet", "nejm", "pnas",
            "jacs", "angew", "biomaterials", "acta biomaterialia",
            "ieee", "acm", "springer", "elsevier", "wiley"
        }
        
        # Padrões para identificar preprints
        self.preprint_patterns = {
            "arxiv.org", "biorxiv.org", "medrxiv.org", "chemrxiv.org"
        }
    
    def validate_source(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Valida e classifica fonte científica"""
        validation = {
            "is_peer_reviewed": False,
            "is_preprint": False,
            "journal_tier": "unknown",
            "trust_score": 0.0,
            "validation_flags": []
        }
        
        # Verifica se é preprint
        url = metadata.get("url", "").lower()
        for pattern in self.preprint_patterns:
            if pattern in url:
                validation["is_preprint"] = True
                validation["validation_flags"].append("preprint")
                break
        
        # Verifica journal
        journal = metadata.get("journal", "").lower()
        source = metadata.get("source", "").lower()
        
        for trusted in self.trusted_journals:
            if trusted in journal or trusted in source:
                validation["is_peer_reviewed"] = True
                validation["journal_tier"] = "tier1" if trusted in ["nature", "science", "cell"] else "tier2"
                validation["trust_score"] += 0.8
                validation["validation_flags"].append("trusted_journal")
                break
        
        # Verifica DOI
        if metadata.get("doi"):
            validation["trust_score"] += 0.2
            validation["validation_flags"].append("has_doi")
        
        # Verifica citações
        citations = metadata.get("citation_count", 0)
        if citations > 100:
            validation["trust_score"] += 0.3
            validation["validation_flags"].append("highly_cited")
        elif citations > 10:
            validation["trust_score"] += 0.1
            validation["validation_flags"].append("cited")
        
        # Verifica recência
        pub_date = metadata.get("published_date")
        if pub_date:
            try:
                pub_datetime = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                age_days = (datetime.now() - pub_datetime).days
                
                if age_days < 365:
                    validation["trust_score"] += 0.1
                    validation["validation_flags"].append("recent")
                elif age_days > 3650:  # 10 years
                    validation["trust_score"] -= 0.1
                    validation["validation_flags"].append("old")
            except:
                pass
        
        # Normaliza score
        validation["trust_score"] = min(validation["trust_score"], 1.0)
        
        return validation


class FeedMonitor:
    """Monitor de feeds RSS científicos"""
    
    def __init__(self):
        self.feeds_config = {
            "biomaterials": [
                "https://www.sciencedirect.com/journal/biomaterials/rss",
                "https://onlinelibrary.wiley.com/feed/15527004/most-recent"
            ],
            "neuroscience": [
                "https://www.nature.com/subjects/neuroscience/rss",
                "https://www.frontiersin.org/journals/neuroscience/rss"
            ],
            "arxiv_ai": [
                "https://export.arxiv.org/rss/cs.AI",
                "https://export.arxiv.org/rss/cs.LG"
            ],
            "pubmed_biomaterials": [
                "https://pubmed.ncbi.nlm.nih.gov/rss/search/1YQWGAcLmGAXwZgzQF4YPAHcvJ8LUdacfZl_T4L9gVG7XHYB-x/?limit=15&utm_campaign=pubmed-2&fc=20210915185900"
            ]
        }
        
        self.session = None
        if HTTP_AVAILABLE:
            self.session = requests.Session()
    
    async def fetch_feed_entries(self, feed_url: str, max_entries: int = 15) -> List[Dict[str, Any]]:
        """Busca entradas de um feed RSS"""
        if not self.session:
            return []
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.session.get(feed_url, timeout=30)
            )
            
            if response.status_code == 200:
                return self._parse_feed(response.text, max_entries)
                
        except Exception as e:
            logger.error(f"Error fetching feed {feed_url}: {e}")
            
        return []
    
    def _parse_feed(self, feed_content: str, max_entries: int) -> List[Dict[str, Any]]:
        """Parse feed RSS/Atom"""
        entries = []
        
        if RSS_AVAILABLE and feedparser:
            # Use feedparser se disponível
            try:
                parsed = feedparser.parse(feed_content)
                for entry in parsed.entries[:max_entries]:
                    entries.append({
                        "title": getattr(entry, "title", ""),
                        "link": getattr(entry, "link", ""),
                        "summary": getattr(entry, "summary", ""),
                        "content": getattr(entry, "summary", ""),
                        "published": getattr(entry, "published", ""),
                        "authors": [author.get("name", "") for author in getattr(entry, "authors", [])],
                        "source": "rss_feed"
                    })
                return entries
            except Exception as e:
                logger.error(f"Feedparser error: {e}")
        
        # Fallback XML parsing
        if XML_AVAILABLE and ET:
            try:
                root = ET.fromstring(feed_content)
                for item in root.findall(".//item")[:max_entries]:
                    entries.append({
                        "title": item.findtext("title", ""),
                        "link": item.findtext("link", ""),
                        "summary": item.findtext("description", ""),
                        "content": item.findtext("description", ""),
                        "published": item.findtext("pubDate", ""),
                        "source": "rss_feed"
                    })
            except Exception as e:
                logger.error(f"XML parsing error: {e}")
        
        return entries


class ScientificSearchService:
    """Serviço principal de busca científica"""
    
    def __init__(self):
        self.vertex_client: Optional[VertexAIClient] = None
        self.rag_engine: Optional[RAGEngine] = None
        self.doi_resolver = DOIResolver()
        self.citation_analyzer = CitationNetworkAnalyzer()
        self.source_validator = ScientificSourceValidator()
        self.feed_monitor = FeedMonitor()
        
        # Caches
        self.search_cache = {}
        self.domain_concept_map = self._build_domain_concept_map()
    
    async def initialize(self):
        """Inicializa o serviço"""
        try:
            self.vertex_client = await get_vertex_client()
            self.rag_engine = await get_rag_engine()
            logger.info("Scientific Search Service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Scientific Search Service: {e}")
            raise
    
    def _build_domain_concept_map(self) -> Dict[str, List[str]]:
        """Constrói mapa de conceitos por domínio"""
        return {
            QueryDomain.BIOMATERIALS.value: [
                "scaffold", "porosity", "biocompatibility", "tissue engineering",
                "hydrogel", "nanofibers", "mechanical properties", "cell adhesion",
                "biodegradation", "implant", "regenerative medicine"
            ],
            QueryDomain.NEUROSCIENCE.value: [
                "neural network", "plasticity", "EEG", "fMRI", "neuron",
                "synapse", "cognition", "memory", "learning", "brain",
                "neuroplasticity", "neural oscillations"
            ],
            QueryDomain.PHILOSOPHY.value: [
                "epistemology", "ontology", "logic", "ethics", "metaphysics",
                "consciousness", "mind", "knowledge", "reality", "truth",
                "reasoning", "argument", "existence"
            ],
            QueryDomain.QUANTUM.value: [
                "quantum mechanics", "superposition", "entanglement", "decoherence",
                "quantum computing", "quantum field theory", "wave function",
                "measurement", "uncertainty principle", "quantum state"
            ],
            QueryDomain.PSYCHIATRY.value: [
                "depression", "anxiety", "therapy", "psychopharmacology",
                "cognitive behavioral", "mental health", "disorder",
                "diagnosis", "treatment", "psychiatric", "neurotransmitter"
            ]
        }
    
    async def scientific_search(self, request: ScientificSearchRequest) -> Dict[str, Any]:
        """Busca científica especializada"""
        start_time = time.time()
        
        # Expande query com conceitos do domínio
        expanded_query = await self._expand_query_with_concepts(request.query, request.domains)
        
        # Busca documentos
        results = await self._search_multiple_sources(expanded_query, request)
        
        # Enriquece metadados
        enriched_results = await self._enrich_results_metadata(results, request)
        
        # Filtra e ranqueia
        filtered_results = await self._filter_and_rank_results(enriched_results, request)
        
        # Analisa rede de citações se solicitado
        citation_network = {}
        if request.citation_network:
            citation_network = await self._analyze_citation_network(filtered_results)
        
        elapsed_time = (time.time() - start_time) * 1000
        
        return {
            "query": request.query,
            "expanded_query": expanded_query,
            "results": filtered_results,
            "citation_network": citation_network,
            "search_metadata": {
                "domains_searched": request.domains,
                "sources_searched": request.sources,
                "total_results": len(results),
                "filtered_results": len(filtered_results),
                "search_time_ms": elapsed_time
            }
        }
    
    async def biomaterials_search(self, request: BiomaterialsQueryRequest) -> Dict[str, Any]:
        """Busca especializada em biomateriais"""
        # Constrói query expandida
        query_parts = [request.query]
        
        if request.scaffold_type:
            query_parts.append(f"scaffold type: {request.scaffold_type}")
        
        if request.material_properties:
            query_parts.append("properties: " + ", ".join(request.material_properties))
        
        if request.tissue_type:
            query_parts.append(f"tissue: {request.tissue_type}")
        
        expanded_query = " AND ".join(query_parts)
        
        # Converte para ScientificSearchRequest
        sci_request = ScientificSearchRequest(
            query=expanded_query,
            domains=[QueryDomain.BIOMATERIALS],
            sources=[SourceType.PUBMED, SourceType.SPRINGER],
            require_doi=True,
            min_impact_factor=None,
            temporal_weight=1.0,
            citation_network=True
        )
        
        return await self.scientific_search(sci_request)
    
    async def cross_domain_search(self, request: CrossDomainRequest) -> Dict[str, Any]:
        """Busca interdisciplinar entre domínios"""
        results_by_domain = {}
        
        # Busca em cada domínio
        all_domains = [request.primary_domain] + request.secondary_domains
        
        for domain in all_domains:
            domain_request = ScientificSearchRequest(
                query=request.primary_query,
                domains=[domain],
                sources=[],
                require_doi=False,
                min_impact_factor=None,
                temporal_weight=1.0,
                citation_network=request.concept_mapping
            )
            
            domain_results = await self.scientific_search(domain_request)
            results_by_domain[domain.value] = domain_results["results"][:5]  # Top 5 per domain
        
        # Analisa conexões entre domínios
        connections = await self._find_cross_domain_connections(results_by_domain, request)
        
        # Combina e ranqueia resultados
        combined_results = []
        for domain_results in results_by_domain.values():
            combined_results.extend(domain_results)
        
        # Ranqueia por relevância cross-domain
        ranked_results = await self._rank_cross_domain_results(combined_results, request)
        
        return {
            "query": request.primary_query,
            "primary_domain": request.primary_domain.value,
            "secondary_domains": [d.value for d in request.secondary_domains],
            "results": ranked_results,
            "cross_domain_connections": connections,
            "results_by_domain": {k: len(v) for k, v in results_by_domain.items()}
        }
    
    async def run_discovery(self, request: DiscoveryRequest) -> DiscoveryResponse:
        """Executa descoberta científica automática"""
        start_time = time.time()
        
        fetched_total = 0
        novel_total = 0
        added_total = 0
        errors_total = 0
        
        # Monitora feeds por domínio
        if not request.domains:
            domains_to_search = list(QueryDomain)
        else:
            domains_to_search = request.domains
        
        for domain in domains_to_search:
            try:
                domain_results = await self._discover_domain_content(domain, request)
                
                fetched_total += domain_results["fetched"]
                novel_total += domain_results["novel"]
                added_total += domain_results["added"]
                errors_total += domain_results["errors"]
                
                if request.run_once:
                    break
                    
            except Exception as e:
                logger.error(f"Discovery error for domain {domain}: {e}")
                errors_total += 1
        
        elapsed_time = (time.time() - start_time) * 1000
        
        return DiscoveryResponse(
            status="completed",
            fetched=fetched_total,
            novel=novel_total,
            added=added_total,
            errors=errors_total,
            timestamp=datetime.now()
        )
    
    async def _expand_query_with_concepts(self, query: str, domains: List[QueryDomain]) -> str:
        """Expande query com conceitos específicos do domínio"""
        if not domains or not self.vertex_client:
            return query
        
        # Coleta conceitos relevantes
        relevant_concepts = []
        for domain in domains:
            domain_concepts = self.domain_concept_map.get(domain.value, [])
            relevant_concepts.extend(domain_concepts[:5])  # Top 5 concepts per domain
        
        # Usa IA para expandir query inteligentemente
        expansion_prompt = f"""
        Original query: {query}
        Domain concepts: {', '.join(relevant_concepts)}
        
        Expand the query with 2-3 most relevant domain-specific terms that would improve scientific search results.
        Return only the expanded query without explanation:
        """
        
        try:
            expanded = await self.vertex_client.generate_text(expansion_prompt)
            return expanded.strip()
        except:
            return query
    
    async def _search_multiple_sources(self, query: str, request: ScientificSearchRequest) -> List[QueryResult]:
        """Busca em múltiplas fontes"""
        if not self.rag_engine:
            return []
        
        # Busca no RAG engine principal
        rag_results = await self.rag_engine.search_documents(
            query, 
            top_k=20,  # Busca mais para ter opções para filtrar
            domain=request.domains[0] if request.domains else None
        )
        
        # Aqui poderia integrar outras APIs (PubMed, arXiv, etc.)
        # Por simplicidade, usando apenas o RAG engine interno
        
        return rag_results
    
    async def _enrich_results_metadata(self, results: List[QueryResult], request: ScientificSearchRequest) -> List[QueryResult]:
        """Enriquece resultados com metadados adicionais"""
        enriched_results = []
        
        for result in results:
            try:
                # Enriquece com DOI se disponível
                doi = result.metadata.get("doi")
                if doi:
                    doi_metadata = await self.doi_resolver.resolve_doi(doi)
                    result.metadata.update(doi_metadata)
                
                # Adiciona validação científica
                validation = self.source_validator.validate_source(result.metadata)
                result.metadata["validation"] = validation
                
                # Calcula score temporal se relevante
                if request.temporal_weight > 0:
                    temporal_score = self._calculate_temporal_score(result.metadata)
                    result.metadata["temporal_score"] = temporal_score
                
                enriched_results.append(result)
                
            except Exception as e:
                logger.error(f"Error enriching result {result.doc_id}: {e}")
                enriched_results.append(result)  # Include anyway
        
        return enriched_results
    
    async def _filter_and_rank_results(self, results: List[QueryResult], request: ScientificSearchRequest) -> List[Dict[str, Any]]:
        """Filtra e ranqueia resultados baseado nos critérios"""
        filtered = []
        
        for result in results:
            # Filtro por DOI se exigido
            if request.require_doi and not result.metadata.get("doi"):
                continue
            
            # Filtro por impacto mínimo
            if request.min_impact_factor:
                impact_factor = result.metadata.get("impact_factor", 0)
                if impact_factor < request.min_impact_factor:
                    continue
            
            # Calcula score final
            final_score = result.score
            
            # Boost por validação científica
            validation = result.metadata.get("validation", {})
            trust_score = validation.get("trust_score", 0)
            final_score += trust_score * 0.2
            
            # Boost temporal
            temporal_score = result.metadata.get("temporal_score", 0)
            final_score += temporal_score * request.temporal_weight * 0.1
            
            result.score = min(final_score, 1.0)  # Normalize
            filtered.append(self._result_to_dict(result))
        
        # Ordena por score final
        filtered.sort(key=lambda x: x["score"], reverse=True)
        
        return filtered
    
    def _calculate_temporal_score(self, metadata: Dict[str, Any]) -> float:
        """Calcula score baseado em recência"""
        pub_date = metadata.get("published_date")
        if not pub_date:
            return 0.0
        
        try:
            pub_datetime = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
            age_days = (datetime.now() - pub_datetime).days
            
            # Score decai exponencialmente com idade
            if age_days < 365:
                return 1.0
            elif age_days < 365 * 3:
                return 0.8
            elif age_days < 365 * 5:
                return 0.6
            elif age_days < 365 * 10:
                return 0.4
            else:
                return 0.2
                
        except:
            return 0.0
    
    async def _analyze_citation_network(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analisa rede de citações dos resultados"""
        network_stats = {
            "total_papers": len(results),
            "papers_with_citations": 0,
            "avg_citations": 0,
            "citation_clusters": []
        }
        
        citation_counts = []
        for result in results:
            citations = result.get("citation_count", 0)
            if citations > 0:
                network_stats["papers_with_citations"] += 1
                citation_counts.append(citations)
        
        if citation_counts:
            network_stats["avg_citations"] = sum(citation_counts) / len(citation_counts)
        
        return network_stats
    
    async def _find_cross_domain_connections(self, results_by_domain: Dict[str, List], request: CrossDomainRequest) -> List[Dict[str, Any]]:
        """Encontra conexões entre domínios"""
        connections = []
        
        # Análise simplificada de sobreposição conceitual
        primary_concepts = set()
        secondary_concepts = defaultdict(set)
        
        for domain, domain_results in results_by_domain.items():
            for result in domain_results:
                title = result.get("title", "").lower()
                abstract = result.get("abstract", "").lower()
                content = f"{title} {abstract}"
                
                # Extrai conceitos (palavras-chave simples)
                words = re.findall(r'\b\w{4,}\b', content)
                concepts = set(words)
                
                if domain == request.primary_domain.value:
                    primary_concepts.update(concepts)
                else:
                    secondary_concepts[domain].update(concepts)
        
        # Encontra sobreposições
        for domain, domain_concepts in secondary_concepts.items():
            overlap = primary_concepts.intersection(domain_concepts)
            if overlap:
                connections.append({
                    "primary_domain": request.primary_domain.value,
                    "secondary_domain": domain,
                    "shared_concepts": list(overlap)[:10],  # Top 10
                    "connection_strength": len(overlap) / max(len(primary_concepts), 1)
                })
        
        return connections
    
    async def _rank_cross_domain_results(self, results: List[Dict[str, Any]], request: CrossDomainRequest) -> List[Dict[str, Any]]:
        """Ranqueia resultados para busca cross-domain"""
        # Score adicional para papers que mencionam múltiplos domínios
        for result in results:
            cross_domain_score = 0
            content = f"{result.get('title', '')} {result.get('abstract', '')}".lower()
            
            domains_mentioned = 0
            all_domains = [request.primary_domain] + request.secondary_domains
            
            for domain in all_domains:
                domain_concepts = self.domain_concept_map.get(domain.value, [])
                for concept in domain_concepts:
                    if concept.lower() in content:
                        domains_mentioned += 1
                        break
            
            if domains_mentioned > 1:
                cross_domain_score = domains_mentioned / len(all_domains)
                result["score"] = result.get("score", 0) + cross_domain_score * 0.3
        
        # Ordena por score final
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results
    
    async def _discover_domain_content(self, domain: QueryDomain, request: DiscoveryRequest) -> Dict[str, int]:
        """Descobre conteúdo novo para um domínio específico"""
        stats = {"fetched": 0, "novel": 0, "added": 0, "errors": 0}
        
        try:
            # Busca em feeds RSS configurados para o domínio
            feed_urls = self.feed_monitor.feeds_config.get(domain.value, [])
            
            for feed_url in feed_urls:
                try:
                    entries = await self.feed_monitor.fetch_feed_entries(feed_url)
                    stats["fetched"] += len(entries)
                    
                    for entry in entries:
                        # Verifica se é novel (não existe na base)
                        if await self._is_novel_content(entry):
                            stats["novel"] += 1
                            
                            # Adiciona à base de conhecimento
                            if await self._add_discovered_content(entry, domain):
                                stats["added"] += 1
                            
                except Exception as e:
                    logger.error(f"Error processing feed {feed_url}: {e}")
                    stats["errors"] += 1
                    
        except Exception as e:
            logger.error(f"Error in domain discovery for {domain}: {e}")
            stats["errors"] += 1
        
        return stats
    
    async def _is_novel_content(self, entry: Dict[str, Any]) -> bool:
        """Verifica se conteúdo é novo"""
        # Simplificado - verifica por hash do título
        title = entry.get("title", "")
        if not title:
            return False
        
        content_hash = hashlib.md5(title.encode()).hexdigest()
        
        # Busca por título similar no RAG engine
        if self.rag_engine:
            similar = await self.rag_engine.search_documents(title, top_k=1)
            if similar and similar[0].score > 0.9:  # Very similar
                return False
        
        return True
    
    async def _add_discovered_content(self, entry: Dict[str, Any], domain: QueryDomain) -> bool:
        """Adiciona conteúdo descoberto à base"""
        if not self.rag_engine:
            return False
        
        try:
            content = entry.get("content", "") or entry.get("summary", "")
            if not content:
                return False
            
            metadata = {
                "title": entry.get("title", ""),
                "url": entry.get("link", ""),
                "source": "discovery",
                "domain": domain.value,
                "published_date": entry.get("published", ""),
                "authors": entry.get("authors", []),
                "discovery_timestamp": datetime.now().isoformat()
            }
            
            doc_id = await self.rag_engine.index_document(content, metadata)
            logger.info(f"Added discovered content: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding discovered content: {e}")
            return False
    
    def _result_to_dict(self, result: QueryResult) -> Dict[str, Any]:
        """Converte QueryResult para dict"""
        return {
            "doc_id": result.doc_id,
            "score": result.score,
            "title": result.metadata.get("title", ""),
            "abstract": result.metadata.get("abstract", ""),
            "url": result.metadata.get("url", ""),
            "doi": result.metadata.get("doi", ""),
            "authors": result.metadata.get("authors", []),
            "journal": result.metadata.get("journal", ""),
            "published_date": result.metadata.get("published_date", ""),
            "citation_count": result.metadata.get("citation_count", 0),
            "domain": result.metadata.get("domain", ""),
            "source": result.metadata.get("source", ""),
            "validation": result.metadata.get("validation", {}),
            "temporal_score": result.metadata.get("temporal_score", 0.0)
        }
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Verifica saúde do serviço"""
        try:
            components = {
                "vertex_ai": self.vertex_client is not None,
                "rag_engine": self.rag_engine is not None,
                "doi_resolver": self.doi_resolver.session is not None if HTTP_AVAILABLE else False,
                "feed_monitor": True,
                "citation_analyzer": True,
                "source_validator": True
            }
            
            healthy = all(components.values())
            
            return {
                "healthy": healthy,
                "components": components,
                "cache_stats": {
                    "search_cache_size": len(self.search_cache),
                    "doi_cache_size": len(self.doi_resolver.cache)
                }
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "components": {"error": True}
            }


# Instância singleton global
_scientific_search_service: Optional[ScientificSearchService] = None


async def get_scientific_search_service() -> ScientificSearchService:
    """Obtém instância global do serviço"""
    global _scientific_search_service
    
    if _scientific_search_service is None:
        _scientific_search_service = ScientificSearchService()
        await _scientific_search_service.initialize()
    
    return _scientific_search_service


async def cleanup_scientific_search_service():
    """Limpa instância global"""
    global _scientific_search_service
    _scientific_search_service = None