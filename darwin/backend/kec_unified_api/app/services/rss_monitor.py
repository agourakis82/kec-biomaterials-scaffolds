"""
DARWIN SCIENTIFIC DISCOVERY - RSS Monitor
Sistema robusto de monitoramento de feeds RSS científicos por domínio
"""

import asyncio
import hashlib
import logging
import os
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from xml.etree import ElementTree as ET

import requests
from pydantic import ValidationError

from ..models.discovery_models import (
    FeedConfig, 
    FeedStatus, 
    PaperMetadata, 
    ScientificDomain,
    DiscoveryError
)

# Optional dependencies with fallbacks
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    feedparser = None
    FEEDPARSER_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BeautifulSoup = None
    BS4_AVAILABLE = False

try:
    import dateutil.parser as date_parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    date_parser = None
    DATEUTIL_AVAILABLE = False


logger = logging.getLogger(__name__)


# =============================================================================
# SCIENTIFIC FEEDS REGISTRY
# =============================================================================

SCIENTIFIC_FEEDS: Dict[ScientificDomain, List[Dict[str, Any]]] = {
    ScientificDomain.BIOMATERIALS: [
        {
            "name": "Biomaterials Journal",
            "url": "https://journals.elsevier.com/biomaterials/rss",
            "priority": 10,
            "max_entries": 20
        },
        {
            "name": "Advanced Functional Materials",
            "url": "https://onlinelibrary.wiley.com/journal/15526563/rss",
            "priority": 9,
            "max_entries": 15
        },
        {
            "name": "ACS Biomaterials Science & Engineering",
            "url": "https://pubs.acs.org/journal/abseba/rss",
            "priority": 9,
            "max_entries": 15
        },
        {
            "name": "Journal of Biomedical Materials Research",
            "url": "https://onlinelibrary.wiley.com/journal/10974636/rss",
            "priority": 8,
            "max_entries": 12
        },
        {
            "name": "Materials Science & Engineering C",
            "url": "https://journals.elsevier.com/materials-science-and-engineering-c/rss",
            "priority": 7,
            "max_entries": 10
        },
        {
            "name": "Tissue Engineering",
            "url": "https://home.liebertpub.com/rss/ten/54/10",
            "priority": 8,
            "max_entries": 12
        }
    ],
    
    ScientificDomain.NEUROSCIENCE: [
        {
            "name": "Nature Neuroscience",
            "url": "https://www.nature.com/neuro/rss",
            "priority": 10,
            "max_entries": 15
        },
        {
            "name": "eLife Neuroscience",
            "url": "https://elifesciences.org/rss/recent/neuroscience",
            "priority": 9,
            "max_entries": 15
        },
        {
            "name": "PLoS ONE Neuroscience",
            "url": "https://journals.plos.org/plosone/neuroscience/rss",
            "priority": 7,
            "max_entries": 20
        },
        {
            "name": "Journal of Neuroscience",
            "url": "https://www.jneurosci.org/rss/current.xml",
            "priority": 9,
            "max_entries": 15
        },
        {
            "name": "Neuron",
            "url": "https://journals.elsevier.com/neuron/rss",
            "priority": 10,
            "max_entries": 12
        },
        {
            "name": "Brain Research",
            "url": "https://journals.elsevier.com/brain-research/rss",
            "priority": 7,
            "max_entries": 12
        }
    ],
    
    ScientificDomain.PHILOSOPHY: [
        {
            "name": "Mind & Language",
            "url": "https://onlinelibrary.wiley.com/journal/12746001/rss",
            "priority": 9,
            "max_entries": 8
        },
        {
            "name": "Mind Journal",
            "url": "https://academic.oup.com/mind/rss",
            "priority": 10,
            "max_entries": 8
        },
        {
            "name": "Philosophy of Science",
            "url": "https://www.journals.uchicago.edu/journal/phos/rss",
            "priority": 9,
            "max_entries": 6
        },
        {
            "name": "Stanford Encyclopedia of Philosophy",
            "url": "https://plato.stanford.edu/rss/sep.xml",
            "priority": 8,
            "max_entries": 5
        },
        {
            "name": "Journal of Consciousness Studies",
            "url": "https://www.ingentaconnect.com/content/imp/jcs/rss",
            "priority": 8,
            "max_entries": 6
        }
    ],
    
    ScientificDomain.QUANTUM_MECHANICS: [
        {
            "name": "Physical Review A",
            "url": "https://journals.aps.org/pra/rss",
            "priority": 10,
            "max_entries": 15
        },
        {
            "name": "Nature Physics",
            "url": "https://www.nature.com/nphys/rss",
            "priority": 10,
            "max_entries": 12
        },
        {
            "name": "Quantum Science and Technology",
            "url": "https://iopscience.iop.org/journal/2058-9565/rss",
            "priority": 9,
            "max_entries": 10
        },
        {
            "name": "Physical Review Quantum",
            "url": "https://journals.aps.org/prquantum/rss",
            "priority": 9,
            "max_entries": 8
        },
        {
            "name": "arXiv Quantum Physics",
            "url": "https://export.arxiv.org/rss/quant-ph",
            "priority": 7,
            "max_entries": 25
        }
    ],
    
    ScientificDomain.MATHEMATICS: [
        {
            "name": "arXiv Mathematics",
            "url": "https://export.arxiv.org/rss/math",
            "priority": 8,
            "max_entries": 20
        },
        {
            "name": "Journal of Mathematical Analysis",
            "url": "https://journals.elsevier.com/journal-of-mathematical-analysis-and-applications/rss",
            "priority": 7,
            "max_entries": 10
        },
        {
            "name": "Applied Mathematics Letters",
            "url": "https://journals.elsevier.com/applied-mathematics-letters/rss",
            "priority": 6,
            "max_entries": 8
        }
    ]
}

# Fallback feeds para testes e desenvolvimento
FALLBACK_FEEDS = [
    {
        "name": "arXiv AI",
        "url": "https://export.arxiv.org/rss/cs.AI",
        "domain": ScientificDomain.COMPUTER_SCIENCE,
        "priority": 5,
        "max_entries": 15
    },
    {
        "name": "arXiv Machine Learning",
        "url": "https://export.arxiv.org/rss/cs.LG",
        "domain": ScientificDomain.COMPUTER_SCIENCE,
        "priority": 5,
        "max_entries": 15
    }
]


# =============================================================================
# RSS MONITOR CLASS
# =============================================================================

class RSSMonitor:
    """
    Monitor robusto de feeds RSS científicos.
    """
    
    def __init__(self, rate_limit_seconds: int = 60):
        """
        Inicializa o RSS Monitor.
        
        Args:
            rate_limit_seconds: Tempo mínimo entre requests para o mesmo feed
        """
        self.rate_limit = rate_limit_seconds
        self.last_fetch_times: Dict[str, float] = {}
        self.feed_configs: List[FeedConfig] = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DARWIN-Scientific-Discovery/1.0 (Academic Research)'
        })
        
        # Cache de artigos processados (evita duplicatas)
        self.processed_articles: Set[str] = set()
        
        # Estatísticas
        self.stats = {
            'total_feeds_processed': 0,
            'total_articles_found': 0,
            'total_articles_new': 0,
            'total_errors': 0,
            'last_sync': None
        }
        
        logger.info(f"RSS Monitor initialized - feedparser: {FEEDPARSER_AVAILABLE}, bs4: {BS4_AVAILABLE}")
    
    def _generate_article_hash(self, title: str, url: str, published: Optional[str] = None) -> str:
        """Gera hash único para identificar artigos duplicados."""
        content = f"{title}|{url}|{published or ''}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _should_fetch_feed(self, feed_url: str) -> bool:
        """Verifica se deve fazer fetch baseado no rate limiting."""
        last_fetch = self.last_fetch_times.get(feed_url, 0)
        return (time.time() - last_fetch) >= self.rate_limit
    
    def _update_fetch_time(self, feed_url: str):
        """Atualiza timestamp do último fetch."""
        self.last_fetch_times[feed_url] = time.time()
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse robusto de datas de feeds RSS."""
        if not date_str:
            return None
            
        try:
            # Usar dateutil se disponível (mais robusto)
            if DATEUTIL_AVAILABLE:
                return date_parser.parse(date_str)
            
            # Fallback para formatos comuns
            common_formats = [
                "%a, %d %b %Y %H:%M:%S %z",
                "%a, %d %b %Y %H:%M:%S GMT",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d"
            ]
            
            for fmt in common_formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
                    
        except Exception as e:
            logger.warning(f"Failed to parse date '{date_str}': {e}")
            
        return None
    
    def _extract_doi(self, content: str, url: str) -> Optional[str]:
        """Extrai DOI do conteúdo ou URL."""
        if not content:
            return None
        
        # Padrões comuns de DOI
        doi_patterns = [
            r'doi:?\s*([10]\.\d+/[^\s]+)',
            r'https?://dx\.doi\.org/([10]\.\d+/[^\s]+)',
            r'https?://doi\.org/([10]\.\d+/[^\s]+)'
        ]
        
        search_text = f"{content} {url}"
        
        for pattern in doi_patterns:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _classify_domain(self, title: str, content: str, journal: str = "") -> ScientificDomain:
        """Classificação automática de domínio baseada em conteúdo."""
        text = f"{title} {content} {journal}".lower()
        
        # Palavras-chave por domínio
        domain_keywords = {
            ScientificDomain.BIOMATERIALS: [
                'biomaterial', 'scaffold', 'tissue engineering', 'hydrogel',
                'nanofiber', 'biocompatible', 'implant', 'drug delivery',
                'bioactive', 'biodegradable', 'cytotoxicity', 'cell culture'
            ],
            ScientificDomain.NEUROSCIENCE: [
                'neuron', 'brain', 'neural', 'synapse', 'cortex', 'hippocampus',
                'fmri', 'eeg', 'cognitive', 'memory', 'learning', 'plasticity'
            ],
            ScientificDomain.PHILOSOPHY: [
                'consciousness', 'epistemology', 'ontology', 'ethics', 'metaphysics',
                'phenomenology', 'logic', 'philosophy', 'moral', 'existence'
            ],
            ScientificDomain.QUANTUM_MECHANICS: [
                'quantum', 'qubit', 'entanglement', 'superposition', 'decoherence',
                'quantum computing', 'quantum field', 'wave function', 'spin'
            ],
            ScientificDomain.MATHEMATICS: [
                'theorem', 'proof', 'algebra', 'geometry', 'topology',
                'differential', 'integral', 'mathematical', 'algorithm'
            ]
        }
        
        # Pontuação por domínio
        scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                scores[domain] = score
        
        # Retorna domínio com maior pontuação ou default
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return ScientificDomain.COMPUTER_SCIENCE  # Default fallback
    
    def _parse_feed_advanced(self, url: str, content: str) -> List[Dict[str, Any]]:
        """Parse avançado de feed RSS/Atom usando feedparser."""
        if not FEEDPARSER_AVAILABLE:
            return self._parse_feed_fallback(content)
        
        try:
            parsed = feedparser.parse(content)
            entries = []
            
            for entry in parsed.entries:
                # Extrair dados básicos
                title = getattr(entry, 'title', '').strip()
                link = getattr(entry, 'link', '').strip()
                summary = getattr(entry, 'summary', '').strip()
                
                # Data de publicação
                published_str = getattr(entry, 'published', '') or getattr(entry, 'updated', '')
                published = self._parse_date(published_str)
                
                # Autores
                authors = []
                if hasattr(entry, 'authors'):
                    authors = [author.get('name', '') for author in entry.authors if author.get('name')]
                elif hasattr(entry, 'author'):
                    authors = [entry.author]
                
                # Tags/keywords
                tags = []
                if hasattr(entry, 'tags'):
                    tags = [tag.get('term', '') for tag in entry.tags if tag.get('term')]
                
                # Journal/source
                journal = ""
                if hasattr(parsed.feed, 'title'):
                    journal = parsed.feed.title
                
                if title and link:
                    entries.append({
                        'title': title,
                        'link': link,
                        'summary': summary,
                        'content': summary,  # Para compatibilidade
                        'published': published,
                        'authors': authors,
                        'tags': tags,
                        'journal': journal,
                        'doi': self._extract_doi(summary, link)
                    })
            
            logger.debug(f"Parsed {len(entries)} entries from {url} using feedparser")
            return entries
            
        except Exception as e:
            logger.error(f"Feedparser failed for {url}: {e}")
            return self._parse_feed_fallback(content)
    
    def _parse_feed_fallback(self, content: str) -> List[Dict[str, Any]]:
        """Parse fallback usando XML nativo."""
        entries = []
        try:
            root = ET.fromstring(content)
            
            # Tentar RSS 2.0
            items = root.findall(".//item")
            if not items:
                # Tentar Atom
                items = root.findall(".//{http://www.w3.org/2005/Atom}entry")
            
            for item in items:
                title = ""
                link = ""
                summary = ""
                published = None
                
                if item.tag == "item":  # RSS
                    title = item.findtext("title") or ""
                    link = item.findtext("link") or ""
                    summary = item.findtext("description") or ""
                    pub_date = item.findtext("pubDate") or ""
                    published = self._parse_date(pub_date)
                    
                else:  # Atom
                    title = item.findtext("{http://www.w3.org/2005/Atom}title") or ""
                    link_elem = item.find("{http://www.w3.org/2005/Atom}link")
                    if link_elem is not None:
                        link = link_elem.get("href") or ""
                    summary = item.findtext("{http://www.w3.org/2005/Atom}summary") or ""
                    updated = item.findtext("{http://www.w3.org/2005/Atom}updated") or ""
                    published = self._parse_date(updated)
                
                if title and link:
                    entries.append({
                        'title': title.strip(),
                        'link': link.strip(),
                        'summary': summary.strip(),
                        'content': summary.strip(),
                        'published': published,
                        'authors': [],
                        'tags': [],
                        'journal': "",
                        'doi': self._extract_doi(summary, link)
                    })
            
            logger.debug(f"Parsed {len(entries)} entries using fallback parser")
            
        except ET.ParseError as e:
            logger.error(f"XML parsing failed: {e}")
        except Exception as e:
            logger.error(f"Fallback parser failed: {e}")
        
        return entries
    
    async def fetch_feed_async(self, feed_config: FeedConfig) -> List[PaperMetadata]:
        """Fetch assíncrono de um feed RSS."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.fetch_feed, feed_config
        )
    
    def fetch_feed(self, feed_config: FeedConfig) -> List[PaperMetadata]:
        """
        Fetch e parse de um feed RSS específico.
        
        Args:
            feed_config: Configuração do feed
            
        Returns:
            Lista de metadados de papers processados
        """
        if not self._should_fetch_feed(feed_config.url):
            logger.debug(f"Rate limited: {feed_config.url}")
            return []
        
        try:
            logger.info(f"Fetching feed: {feed_config.name} ({feed_config.url})")
            
            # HTTP request com timeout e headers
            headers = {'User-Agent': 'DARWIN-Scientific-Discovery/1.0'}
            headers.update(feed_config.custom_headers)
            
            response = self.session.get(
                feed_config.url,
                headers=headers,
                timeout=30,
                allow_redirects=True
            )
            response.raise_for_status()
            
            self._update_fetch_time(feed_config.url)
            
            # Parse do conteúdo
            entries = self._parse_feed_advanced(feed_config.url, response.text)
            
            # Limitar número de entradas
            entries = entries[:feed_config.max_entries]
            
            # Converter para PaperMetadata
            papers = []
            for entry in entries:
                try:
                    # Gerar hash para deduplicação
                    article_hash = self._generate_article_hash(
                        entry['title'], 
                        entry['link'], 
                        str(entry.get('published'))
                    )
                    
                    if article_hash in self.processed_articles:
                        continue
                    
                    # Classificar domínio automaticamente
                    domain = self._classify_domain(
                        entry['title'],
                        entry.get('summary', ''),
                        entry.get('journal', '')
                    )
                    
                    # Criar PaperMetadata
                    paper = PaperMetadata(
                        doc_id=article_hash,
                        title=entry['title'],
                        abstract=entry.get('summary', ''),
                        authors=entry.get('authors', []),
                        url=entry.get('link'),
                        doi=entry.get('doi'),
                        publication_date=entry.get('published'),
                        journal=entry.get('journal') or feed_config.name,
                        keywords=entry.get('tags', []),
                        domain=domain,
                        source_feed=feed_config.name
                    )
                    
                    papers.append(paper)
                    self.processed_articles.add(article_hash)
                    
                except ValidationError as e:
                    logger.warning(f"Invalid paper data from {feed_config.name}: {e}")
                except Exception as e:
                    logger.error(f"Error processing entry from {feed_config.name}: {e}")
            
            # Atualizar estatísticas
            self.stats['total_articles_found'] += len(entries)
            self.stats['total_articles_new'] += len(papers)
            
            logger.info(f"Processed {len(papers)} new papers from {feed_config.name}")
            return papers
            
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error fetching {feed_config.url}: {e}")
            self.stats['total_errors'] += 1
        except Exception as e:
            logger.error(f"Unexpected error fetching {feed_config.url}: {e}")
            self.stats['total_errors'] += 1
        
        return []
    
    def load_default_feeds(self) -> List[FeedConfig]:
        """Carrega feeds RSS padrão por domínio."""
        configs = []
        
        for domain, feeds in SCIENTIFIC_FEEDS.items():
            for feed_data in feeds:
                try:
                    config = FeedConfig(
                        name=feed_data['name'],
                        url=feed_data['url'],
                        domain=domain,
                        max_entries=feed_data.get('max_entries', 15),
                        priority=feed_data.get('priority', 5),
                        status=FeedStatus.ACTIVE
                    )
                    configs.append(config)
                except Exception as e:
                    logger.error(f"Invalid feed config: {feed_data}, error: {e}")
        
        # Adicionar fallback feeds se necessário
        if not configs:
            for feed_data in FALLBACK_FEEDS:
                try:
                    config = FeedConfig(
                        name=feed_data['name'],
                        url=feed_data['url'],
                        domain=feed_data['domain'],
                        max_entries=feed_data.get('max_entries', 15),
                        priority=feed_data.get('priority', 3)
                    )
                    configs.append(config)
                except Exception as e:
                    logger.error(f"Invalid fallback feed config: {feed_data}, error: {e}")
        
        logger.info(f"Loaded {len(configs)} default feed configurations")
        return configs
    
    def add_custom_feeds(self, domain: ScientificDomain, feed_urls: List[str]) -> List[FeedConfig]:
        """Adiciona feeds personalizados para um domínio."""
        configs = []
        
        for i, url in enumerate(feed_urls):
            try:
                config = FeedConfig(
                    name=f"Custom {domain.value} Feed {i+1}",
                    url=url,
                    domain=domain,
                    max_entries=10,
                    priority=5,
                    status=FeedStatus.ACTIVE
                )
                configs.append(config)
                
            except Exception as e:
                logger.error(f"Invalid custom feed URL {url}: {e}")
        
        return configs
    
    async def sync_all_feeds_async(self, feed_configs: List[FeedConfig]) -> List[PaperMetadata]:
        """Sincronização assíncrona de todos os feeds."""
        tasks = [self.fetch_feed_async(config) for config in feed_configs if config.status == FeedStatus.ACTIVE]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_papers = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Feed sync failed for {feed_configs[i].name}: {result}")
                self.stats['total_errors'] += 1
            else:
                all_papers.extend(result)
        
        self.stats['total_feeds_processed'] += len([r for r in results if not isinstance(r, Exception)])
        self.stats['last_sync'] = datetime.now(timezone.utc)
        
        return all_papers
    
    def sync_all_feeds(self, feed_configs: List[FeedConfig]) -> List[PaperMetadata]:
        """Sincronização síncrona de todos os feeds."""
        all_papers = []
        
        for config in feed_configs:
            if config.status != FeedStatus.ACTIVE:
                continue
                
            papers = self.fetch_feed(config)
            all_papers.extend(papers)
        
        self.stats['total_feeds_processed'] += len([c for c in feed_configs if c.status == FeedStatus.ACTIVE])
        self.stats['last_sync'] = datetime.now(timezone.utc)
        
        return all_papers
    
    def get_feed_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do monitor."""
        return self.stats.copy()
    
    def clear_processed_cache(self):
        """Limpa cache de artigos processados."""
        self.processed_articles.clear()
        logger.info("Cleared processed articles cache")


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

# Instância global do RSS Monitor
_rss_monitor = None

def get_rss_monitor() -> RSSMonitor:
    """Retorna instância singleton do RSS Monitor."""
    global _rss_monitor
    if _rss_monitor is None:
        _rss_monitor = RSSMonitor()
    return _rss_monitor