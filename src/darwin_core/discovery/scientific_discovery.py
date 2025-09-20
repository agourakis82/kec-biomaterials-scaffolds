"""
Scientific Discovery System - Sistema de Descoberta Científica Automatizada
=========================================================================

Sistema automatizado que executa buscas horárias por:
- Inovações tecnológicas emergentes
- Fronteiras da ciência relevantes
- Papers e pesquisas recentes
- Tendências em biotecnologia e materiais
- Atualizações metodológicas
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import hashlib
import sqlite3

logger = logging.getLogger(__name__)


@dataclass
class DiscoverySource:
    """Fonte de descoberta científica."""
    name: str
    url: str
    type: str  # "arxiv", "pubmed", "google_scholar", "patent", "news"
    keywords: List[str]
    enabled: bool = True
    last_check: Optional[datetime] = None
    check_interval_hours: int = 1
    priority: int = 1  # 1-5, 5 = highest


@dataclass
class ScientificFinding:
    """Descoberta científica encontrada."""
    id: str
    title: str
    abstract: str
    authors: List[str]
    source: str
    url: str
    published_date: datetime
    discovery_timestamp: datetime
    relevance_score: float
    keywords_matched: List[str]
    category: str  # "biomaterials", "AI", "materials_science", "biotech", etc.
    novelty_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveryConfig:
    """Configuração do sistema de descoberta."""
    enabled: bool = True
    check_interval_hours: int = 1
    max_findings_per_source: int = 10
    relevance_threshold: float = 0.6
    novelty_threshold: float = 0.5
    storage_retention_days: int = 90
    notification_threshold: float = 0.8
    parallel_sources: bool = True
    max_concurrent_sources: int = 5


class ScientificDiscoverySystem:
    """
    Sistema de descoberta científica automatizada que:
    - Monitora múltiplas fontes científicas 24/7
    - Filtra por relevância para o projeto KEC
    - Detecta breakthrough technologies
    - Armazena discoveries com contexto
    - Gera alertas para findings importantes
    """
    
    def __init__(self, config: Optional[DiscoveryConfig] = None):
        self.config = config or DiscoveryConfig()
        self.db_path = Path("data/memory/scientific_discoveries.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Fontes configuradas
        self.sources = self._setup_discovery_sources()
        self._discovery_task: Optional[asyncio.Task] = None
        self._running = False
        
    def _setup_discovery_sources(self) -> List[DiscoverySource]:
        """Configura fontes de descoberta."""
        
        return [
            # ArXiv - Papers de AI/ML/Materials
            DiscoverySource(
                name="ArXiv Materials Science",
                url="http://export.arxiv.org/api/query?search_query=cat:cond-mat.mtrl-sci",
                type="arxiv",
                keywords=["biomaterials", "scaffold", "porous", "materials", "entropy", "topology"],
                priority=5,
                check_interval_hours=2
            ),
            
            DiscoverySource(
                name="ArXiv AI/ML",
                url="http://export.arxiv.org/api/query?search_query=cat:cs.AI+OR+cat:cs.LG",
                type="arxiv", 
                keywords=["graph neural networks", "tree search", "MCTS", "RAG", "retrieval"],
                priority=4,
                check_interval_hours=3
            ),
            
            # PubMed - Biomaterials e bioengenharia
            DiscoverySource(
                name="PubMed Biomaterials",
                url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=biomaterials",
                type="pubmed",
                keywords=["biomaterial", "scaffold", "porosity", "tissue engineering", "regenerative medicine"],
                priority=5,
                check_interval_hours=4
            ),
            
            # Nature - Breakthrough research
            DiscoverySource(
                name="Nature Research",
                url="https://www.nature.com/search?q=biomaterials&format=rss",
                type="nature",
                keywords=["breakthrough", "innovation", "materials science", "bioengineering"],
                priority=5,
                check_interval_hours=6
            ),
            
            # IEEE - Engineering advances
            DiscoverySource(
                name="IEEE Biomedical",
                url="https://ieeexplore.ieee.org/search/rss",
                type="ieee",
                keywords=["biomedical engineering", "computational methods", "AI healthcare"],
                priority=3,
                check_interval_hours=8
            ),
            
            # Science Direct - Comprehensive
            DiscoverySource(
                name="ScienceDirect Materials",
                url="https://www.sciencedirect.com/search/api", 
                type="sciencedirect",
                keywords=["advanced materials", "computational modeling", "network analysis"],
                priority=4,
                check_interval_hours=6
            )
        ]
    
    async def initialize(self) -> None:
        """Inicializa sistema de descoberta."""
        
        await self._setup_database()
        logger.info("Sistema de descoberta científica inicializado")
        
        if self.config.enabled:
            await self.start_continuous_discovery()
    
    async def _setup_database(self) -> None:
        """Configura banco de dados para discoveries."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabela de discoveries
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scientific_findings (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                abstract TEXT,
                authors TEXT,  -- JSON array
                source TEXT NOT NULL,
                url TEXT,
                published_date TEXT,
                discovery_timestamp TEXT NOT NULL,
                relevance_score REAL NOT NULL,
                keywords_matched TEXT,  -- JSON array
                category TEXT NOT NULL,
                novelty_score REAL NOT NULL,
                metadata TEXT  -- JSON object
            )
        """)
        
        # Tabela de status das fontes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS discovery_sources_status (
                source_name TEXT PRIMARY KEY,
                last_check TEXT,
                last_success TEXT,
                total_findings INTEGER DEFAULT 0,
                errors_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active'
            )
        """)
        
        # Índices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_findings_timestamp ON scientific_findings(discovery_timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_findings_relevance ON scientific_findings(relevance_score)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_findings_category ON scientific_findings(category)")
        
        conn.commit()
        conn.close()
    
    async def start_continuous_discovery(self) -> None:
        """Inicia descoberta contínua em background."""
        
        if self._running:
            logger.warning("Descoberta contínua já está rodando")
            return
        
        self._running = True
        self._discovery_task = asyncio.create_task(self._discovery_loop())
        logger.info("Descoberta científica contínua iniciada")
    
    async def stop_continuous_discovery(self) -> None:
        """Para descoberta contínua."""
        
        self._running = False
        if self._discovery_task:
            self._discovery_task.cancel()
            try:
                await self._discovery_task
            except asyncio.CancelledError:
                pass
        logger.info("Descoberta científica contínua parada")
    
    async def _discovery_loop(self) -> None:
        """Loop principal de descoberta."""
        
        while self._running:
            try:
                logger.info("Executando ciclo de descoberta científica...")
                
                # Executa descoberta em todas as fontes ativas
                if self.config.parallel_sources:
                    await self._parallel_discovery()
                else:
                    await self._sequential_discovery()
                
                # Aguarda próximo ciclo
                await asyncio.sleep(self.config.check_interval_hours * 3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erro no loop de descoberta: {e}")
                await asyncio.sleep(300)  # Aguarda 5 min em caso de erro
    
    async def _parallel_discovery(self) -> None:
        """Executa descoberta paralela em múltiplas fontes."""
        
        # Filtra fontes que precisam ser verificadas
        sources_to_check = []
        for source in self.sources:
            if source.enabled and self._should_check_source(source):
                sources_to_check.append(source)
        
        if not sources_to_check:
            logger.debug("Nenhuma fonte precisa ser verificada no momento")
            return
        
        # Executa em batches para limitar concorrência
        batch_size = self.config.max_concurrent_sources
        
        for i in range(0, len(sources_to_check), batch_size):
            batch = sources_to_check[i:i + batch_size]
            tasks = [self._discover_from_source(source) for source in batch]
            
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for source, result in zip(batch, results):
                    if isinstance(result, Exception):
                        logger.error(f"Erro na fonte {source.name}: {result}")
                    else:
                        logger.info(f"Descoberta {source.name}: {len(result)} findings")
                        
            except Exception as e:
                logger.error(f"Erro no batch de descoberta: {e}")
    
    async def _sequential_discovery(self) -> None:
        """Executa descoberta sequencial."""
        
        for source in self.sources:
            if source.enabled and self._should_check_source(source):
                try:
                    findings = await self._discover_from_source(source)
                    logger.info(f"Fonte {source.name}: {len(findings)} findings")
                except Exception as e:
                    logger.error(f"Erro na fonte {source.name}: {e}")
    
    def _should_check_source(self, source: DiscoverySource) -> bool:
        """Verifica se fonte deve ser checada agora."""
        
        if not source.last_check:
            return True
        
        time_since_check = datetime.now() - source.last_check
        return time_since_check.total_seconds() >= (source.check_interval_hours * 3600)
    
    async def _discover_from_source(self, source: DiscoverySource) -> List[ScientificFinding]:
        """Executa descoberta de uma fonte específica."""
        
        logger.debug(f"Descobrindo de {source.name}...")
        source.last_check = datetime.now()
        
        try:
            if source.type == "arxiv":
                findings = await self._discover_arxiv(source)
            elif source.type == "pubmed":
                findings = await self._discover_pubmed(source)
            else:
                # Fallback: discovery mock para desenvolvimento
                findings = await self._mock_discovery(source)
            
            # Filtra por relevância e novelty
            filtered_findings = [
                f for f in findings 
                if f.relevance_score >= self.config.relevance_threshold 
                and f.novelty_score >= self.config.novelty_threshold
            ]
            
            # Armazena findings
            for finding in filtered_findings:
                await self._store_finding(finding)
            
            await self._update_source_status(source.name, success=True, findings_count=len(filtered_findings))
            
            return filtered_findings
            
        except Exception as e:
            logger.error(f"Erro ao descobrir de {source.name}: {e}")
            await self._update_source_status(source.name, success=False, error=str(e))
            return []
    
    async def _discover_arxiv(self, source: DiscoverySource) -> List[ScientificFinding]:
        """Descoberta específica do ArXiv."""
        
        # TODO: Implementar integração real com ArXiv API
        # Por enquanto, simulação para demonstração
        
        findings = []
        
        # Mock findings baseados nas keywords
        for i, keyword in enumerate(source.keywords[:3]):  # Limit para performance
            finding = ScientificFinding(
                id=f"arxiv_{hashlib.md5(f'{keyword}{datetime.now()}'.encode()).hexdigest()[:8]}",
                title=f"Recent Advances in {keyword.title()} Research",
                abstract=f"Latest developments in {keyword} showing promising results for biomedical applications...",
                authors=[f"Author {i+1}", f"Author {i+2}"],
                source=source.name,
                url=f"https://arxiv.org/abs/2024.{i+1:04d}.{i*100+1:05d}",
                published_date=datetime.now() - timedelta(days=i+1),
                discovery_timestamp=datetime.now(),
                relevance_score=0.8 - (i * 0.1),
                keywords_matched=[keyword],
                category="biomaterials" if "material" in keyword else "AI",
                novelty_score=0.7 + (i * 0.05),
                metadata={"source_type": "arxiv", "category": source.type}
            )
            findings.append(finding)
        
        return findings
    
    async def _discover_pubmed(self, source: DiscoverySource) -> List[ScientificFinding]:
        """Descoberta específica do PubMed."""
        
        # TODO: Implementar integração real com PubMed API
        # Simulação para desenvolvimento
        
        findings = []
        
        for i, keyword in enumerate(source.keywords[:2]):
            finding = ScientificFinding(
                id=f"pubmed_{hashlib.md5(f'{keyword}{datetime.now()}'.encode()).hexdigest()[:8]}",
                title=f"Clinical Applications of {keyword.title()} in Regenerative Medicine",
                abstract=f"This study investigates {keyword} applications in clinical settings...",
                authors=[f"Dr. {chr(65+i)}", f"Dr. {chr(66+i)}"],
                source=source.name,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{30000000 + i}",
                published_date=datetime.now() - timedelta(days=i*2),
                discovery_timestamp=datetime.now(),
                relevance_score=0.9 - (i * 0.1),
                keywords_matched=[keyword],
                category="biomedicine",
                novelty_score=0.8,
                metadata={"source_type": "pubmed", "medical_relevance": True}
            )
            findings.append(finding)
        
        return findings
    
    async def _mock_discovery(self, source: DiscoverySource) -> List[ScientificFinding]:
        """Discovery mock para desenvolvimento."""
        
        findings = []
        
        # Simula 1-3 discoveries por fonte
        num_findings = min(3, len(source.keywords))
        
        for i in range(num_findings):
            keyword = source.keywords[i % len(source.keywords)]
            
            finding = ScientificFinding(
                id=f"mock_{source.name.lower().replace(' ', '_')}_{i}_{int(datetime.now().timestamp())}",
                title=f"Emerging Trends in {keyword.title()} Technology",
                abstract=f"Novel approaches to {keyword} are revolutionizing the field with applications in biotechnology and materials science...",
                authors=[f"Researcher {i+1}", f"Scientist {i+2}"],
                source=source.name,
                url=f"https://example.com/paper/{i}",
                published_date=datetime.now() - timedelta(days=i),
                discovery_timestamp=datetime.now(),
                relevance_score=0.7 + (0.1 * (3-i)),  # Decreasing relevance
                keywords_matched=[keyword],
                category=self._categorize_by_keyword(keyword),
                novelty_score=0.6 + (0.1 * i),
                metadata={"source_type": "mock", "development_mode": True}
            )
            
            findings.append(finding)
        
        return findings
    
    def _categorize_by_keyword(self, keyword: str) -> str:
        """Categoriza discovery baseado em keyword."""
        
        keyword_lower = keyword.lower()
        
        if any(term in keyword_lower for term in ["biomat", "scaffold", "tissue", "regenerative"]):
            return "biomaterials"
        elif any(term in keyword_lower for term in ["ai", "ml", "neural", "algorithm"]):
            return "artificial_intelligence"
        elif any(term in keyword_lower for term in ["material", "polymer", "composite"]):
            return "materials_science"
        elif any(term in keyword_lower for term in ["biotech", "bioeng", "medical"]):
            return "biotechnology"
        else:
            return "general_science"
    
    async def _store_finding(self, finding: ScientificFinding) -> None:
        """Armazena discovery no banco."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO scientific_findings
            (id, title, abstract, authors, source, url, published_date,
             discovery_timestamp, relevance_score, keywords_matched, 
             category, novelty_score, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            finding.id,
            finding.title,
            finding.abstract,
            json.dumps(finding.authors),
            finding.source,
            finding.url,
            finding.published_date.isoformat(),
            finding.discovery_timestamp.isoformat(),
            finding.relevance_score,
            json.dumps(finding.keywords_matched),
            finding.category,
            finding.novelty_score,
            json.dumps(finding.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    async def _update_source_status(self, source_name: str, success: bool, 
                                  findings_count: int = 0, error: Optional[str] = None) -> None:
        """Atualiza status da fonte."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT OR REPLACE INTO discovery_sources_status
            (source_name, last_check, last_success, total_findings, errors_count, status)
            VALUES (?, ?, ?, 
                    COALESCE((SELECT total_findings FROM discovery_sources_status WHERE source_name = ?), 0) + ?,
                    COALESCE((SELECT errors_count FROM discovery_sources_status WHERE source_name = ?), 0) + ?,
                    ?)
        """, (
            source_name, 
            now,
            now if success else None,
            source_name, findings_count,
            source_name, 0 if success else 1,
            "active" if success else "error"
        ))
        
        conn.commit()
        conn.close()
    
    async def get_recent_discoveries(self, 
                                   hours: int = 24,
                                   category: Optional[str] = None,
                                   min_relevance: float = 0.5) -> List[ScientificFinding]:
        """Recupera discoveries recentes."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        where_clauses = ["discovery_timestamp > ?", "relevance_score >= ?"]
        params = [
            (datetime.now() - timedelta(hours=hours)).isoformat(),
            min_relevance
        ]
        
        if category:
            where_clauses.append("category = ?")
            params.append(category)
        
        sql = f"""
            SELECT * FROM scientific_findings
            WHERE {' AND '.join(where_clauses)}
            ORDER BY relevance_score DESC, discovery_timestamp DESC
            LIMIT 50
        """
        
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()
        
        findings = []
        for row in rows:
            findings.append(ScientificFinding(
                id=row[0],
                title=row[1],
                abstract=row[2] or "",
                authors=json.loads(row[3] or "[]"),
                source=row[4],
                url=row[5] or "",
                published_date=datetime.fromisoformat(row[6]),
                discovery_timestamp=datetime.fromisoformat(row[7]),
                relevance_score=row[8],
                keywords_matched=json.loads(row[9] or "[]"),
                category=row[10],
                novelty_score=row[11],
                metadata=json.loads(row[12] or "{}")
            ))
        
        return findings
    
    async def get_breakthrough_discoveries(self, min_novelty: float = 0.9) -> List[ScientificFinding]:
        """Recupera discoveries breakthrough (alta novelty)."""
        
        return await self.get_recent_discoveries(
            hours=168,  # 1 semana
            min_relevance=0.8,
        )
    
    async def generate_discovery_report(self, hours: int = 24) -> Dict[str, Any]:
        """Gera relatório de discoveries."""
        
        discoveries = await self.get_recent_discoveries(hours=hours)
        
        if not discoveries:
            return {
                "period_hours": hours,
                "total_discoveries": 0,
                "message": "Nenhuma discovery encontrada no período"
            }
        
        # Estatísticas por categoria
        categories = {}
        sources = {}
        
        for discovery in discoveries:
            categories[discovery.category] = categories.get(discovery.category, 0) + 1
            sources[discovery.source] = sources.get(discovery.source, 0) + 1
        
        # Top discoveries
        top_discoveries = sorted(discoveries, key=lambda x: x.relevance_score, reverse=True)[:5]
        
        return {
            "period_hours": hours,
            "total_discoveries": len(discoveries),
            "categories": categories,
            "sources": sources,
            "avg_relevance": sum(d.relevance_score for d in discoveries) / len(discoveries),
            "avg_novelty": sum(d.novelty_score for d in discoveries) / len(discoveries),
            "top_discoveries": [
                {
                    "title": d.title,
                    "source": d.source,
                    "relevance": d.relevance_score,
                    "novelty": d.novelty_score,
                    "url": d.url
                }
                for d in top_discoveries
            ],
            "breakthrough_count": len([d for d in discoveries if d.novelty_score > 0.8])
        }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Status completo do sistema de descoberta."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Estatísticas gerais
        cursor.execute("SELECT COUNT(*) FROM scientific_findings")
        total_findings = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM scientific_findings WHERE discovery_timestamp > ?", 
                      [(datetime.now() - timedelta(days=1)).isoformat()])
        recent_findings = cursor.fetchone()[0]
        
        # Status das fontes
        cursor.execute("SELECT * FROM discovery_sources_status")
        sources_status = cursor.fetchall()
        
        conn.close()
        
        return {
            "system": "scientific_discovery",
            "running": self._running,
            "total_findings": total_findings,
            "recent_findings_24h": recent_findings,
            "active_sources": len([s for s in self.sources if s.enabled]),
            "sources_status": [
                {
                    "name": row[0],
                    "last_check": row[1],
                    "last_success": row[2], 
                    "total_findings": row[3],
                    "errors": row[4],
                    "status": row[5]
                }
                for row in sources_status
            ],
            "config": {
                "check_interval_hours": self.config.check_interval_hours,
                "relevance_threshold": self.config.relevance_threshold,
                "novelty_threshold": self.config.novelty_threshold
            }
        }


# Instância global
_discovery_system = ScientificDiscoverySystem()


async def get_discovery_system() -> ScientificDiscoverySystem:
    """Factory function para sistema de descoberta."""
    if not _discovery_system.db_path.exists():
        await _discovery_system.initialize()
    return _discovery_system