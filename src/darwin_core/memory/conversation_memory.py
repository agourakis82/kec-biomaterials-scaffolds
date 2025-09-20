"""
Conversation Memory System - Sistema de Memória de Conversação
============================================================

Sistema avançado para armazenar e recuperar histórico completo de conversas
com diferentes LLMs, mantendo contexto de projeto e continuidade.
"""

import json
import sqlite3
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ConversationEntry:
    """Entrada individual de conversação."""
    id: str
    timestamp: datetime
    llm_provider: str  # "gpt-4", "gemini-pro", "claude-3", etc.
    user_message: str
    assistant_response: str
    context_type: str  # "project_planning", "code_implementation", "analysis", etc.
    project_phase: str  # "architecture", "development", "testing", "deployment"
    tags: List[str]
    metadata: Dict[str, Any]
    relevance_score: float = 1.0


@dataclass
class ProjectContext:
    """Contexto completo do projeto."""
    project_id: str
    current_phase: str
    active_tasks: List[str]
    completed_tasks: List[str]
    pending_decisions: List[str]
    architecture_decisions: Dict[str, Any]
    file_modifications: List[Dict[str, Any]]
    next_steps: List[str]
    preferences: Dict[str, Any]
    last_updated: datetime


class ConversationMemorySystem:
    """
    Sistema de memória de conversação que:
    - Armazena histórico completo de todas as conversas
    - Mantém contexto de projeto automaticamente
    - Recupera contexto relevante no início das sessões
    - Identifica continuidade e próximos passos
    """
    
    def __init__(self, db_path: str = "data/memory/conversation_memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_project_context: Optional[ProjectContext] = None
        
    async def initialize(self) -> None:
        """Inicializa sistema de memória."""
        
        await self._setup_database()
        await self._load_current_project_context()
        logger.info("Sistema de memória de conversação inicializado")
    
    async def _setup_database(self) -> None:
        """Configura banco de dados SQLite."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabela de conversas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                llm_provider TEXT NOT NULL,
                user_message TEXT NOT NULL,
                assistant_response TEXT NOT NULL,
                context_type TEXT NOT NULL,
                project_phase TEXT NOT NULL,
                tags TEXT,  -- JSON array
                metadata TEXT,  -- JSON object
                relevance_score REAL DEFAULT 1.0
            )
        """)
        
        # Tabela de contexto de projeto
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS project_context (
                project_id TEXT PRIMARY KEY,
                current_phase TEXT NOT NULL,
                active_tasks TEXT,  -- JSON array
                completed_tasks TEXT,  -- JSON array
                pending_decisions TEXT,  -- JSON array
                architecture_decisions TEXT,  -- JSON object
                file_modifications TEXT,  -- JSON array
                next_steps TEXT,  -- JSON array
                preferences TEXT,  -- JSON object
                last_updated TEXT NOT NULL
            )
        """)
        
        # Índices para performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_context ON conversations(context_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_llm ON conversations(llm_provider)")
        
        conn.commit()
        conn.close()
    
    async def store_conversation(self, 
                               user_message: str,
                               assistant_response: str,
                               llm_provider: str,
                               context_type: str = "general",
                               project_phase: str = "development",
                               tags: Optional[List[str]] = None,
                               metadata: Optional[Dict[str, Any]] = None) -> str:
        """Armazena uma conversação completa."""
        
        # Gera ID único baseado no conteúdo
        content_hash = hashlib.md5(
            f"{user_message}{assistant_response}{datetime.now().isoformat()}".encode()
        ).hexdigest()
        
        entry = ConversationEntry(
            id=content_hash,
            timestamp=datetime.now(),
            llm_provider=llm_provider,
            user_message=user_message,
            assistant_response=assistant_response,
            context_type=context_type,
            project_phase=project_phase,
            tags=tags or [],
            metadata=metadata or {},
            relevance_score=await self._calculate_relevance_score(user_message, assistant_response)
        )
        
        await self._store_entry(entry)
        await self._update_project_context_from_conversation(entry)
        
        logger.info(f"Conversação armazenada: {entry.id} ({llm_provider})")
        return entry.id
    
    async def _store_entry(self, entry: ConversationEntry) -> None:
        """Armazena entrada no banco."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO conversations 
            (id, timestamp, llm_provider, user_message, assistant_response, 
             context_type, project_phase, tags, metadata, relevance_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.id,
            entry.timestamp.isoformat(),
            entry.llm_provider,
            entry.user_message,
            entry.assistant_response,
            entry.context_type,
            entry.project_phase,
            json.dumps(entry.tags),
            json.dumps(entry.metadata),
            entry.relevance_score
        ))
        
        conn.commit()
        conn.close()
    
    async def _calculate_relevance_score(self, user_msg: str, assistant_msg: str) -> float:
        """Calcula score de relevância baseado no conteúdo."""
        
        # Keywords importantes para o projeto
        important_keywords = [
            "kec", "biomat", "metrics", "rag", "puct", "tree_search", 
            "architecture", "modular", "backend", "api", "cloud",
            "entropy", "curvature", "small_world", "analytics"
        ]
        
        combined_text = (user_msg + " " + assistant_msg).lower()
        
        # Score baseado em keywords
        keyword_score = sum(1 for kw in important_keywords if kw in combined_text)
        keyword_score = min(keyword_score / len(important_keywords), 1.0)
        
        # Score baseado no tamanho (conversas mais longas são mais valiosas)
        length_score = min(len(combined_text) / 2000, 1.0)
        
        # Score final combinado
        final_score = (keyword_score * 0.7) + (length_score * 0.3)
        
        return max(0.1, min(1.0, final_score))
    
    async def retrieve_relevant_context(self, 
                                      query: str,
                                      llm_provider: Optional[str] = None,
                                      max_results: int = 10,
                                      time_window_days: int = 30) -> List[ConversationEntry]:
        """Recupera contexto relevante para uma query."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Query com filtros opcionais
        where_clauses = ["timestamp > ?"]
        params = [(datetime.now() - timedelta(days=time_window_days)).isoformat()]
        
        if llm_provider:
            where_clauses.append("llm_provider = ?")
            params.append(llm_provider)
        
        # Busca por palavra-chave (simplificada)
        if query:
            where_clauses.append("(user_message LIKE ? OR assistant_response LIKE ?)")
            query_param = f"%{query}%"
            params.extend([query_param, query_param])
        
        sql = f"""
            SELECT * FROM conversations 
            WHERE {' AND '.join(where_clauses)}
            ORDER BY relevance_score DESC, timestamp DESC
            LIMIT ?
        """
        params.append(max_results)
        
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()
        
        # Converte para objetos ConversationEntry
        entries = []
        for row in rows:
            entries.append(ConversationEntry(
                id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                llm_provider=row[2],
                user_message=row[3],
                assistant_response=row[4],
                context_type=row[5],
                project_phase=row[6],
                tags=json.loads(row[7] or "[]"),
                metadata=json.loads(row[8] or "{}"),
                relevance_score=row[9]
            ))
        
        logger.info(f"Recuperadas {len(entries)} conversações relevantes para: '{query}'")
        return entries
    
    async def get_session_startup_context(self, llm_provider: str = "claude") -> Dict[str, Any]:
        """
        Gera contexto completo para início de sessão.
        
        Returns:
            Dict com contexto atual do projeto, conversas recentes, próximos passos
        """
        
        # Conversas recentes relevantes
        recent_conversations = await self.retrieve_relevant_context(
            query="",
            llm_provider=llm_provider,
            max_results=5,
            time_window_days=7
        )
        
        # Contexto atual do projeto
        project_context = self.current_project_context
        
        # Análise de continuidade
        continuity_analysis = await self._analyze_conversation_continuity(recent_conversations)
        
        startup_context = {
            "session_info": {
                "timestamp": datetime.now().isoformat(),
                "llm_provider": llm_provider,
                "project_id": project_context.project_id if project_context else "kec-biomaterials"
            },
            "project_status": {
                "current_phase": project_context.current_phase if project_context else "development",
                "active_tasks": project_context.active_tasks if project_context else [],
                "completed_tasks": project_context.completed_tasks if project_context else [],
                "next_steps": project_context.next_steps if project_context else []
            },
            "recent_conversations": [
                {
                    "timestamp": conv.timestamp.isoformat(),
                    "context_type": conv.context_type,
                    "summary": conv.user_message[:200] + "..." if len(conv.user_message) > 200 else conv.user_message,
                    "relevance": conv.relevance_score
                }
                for conv in recent_conversations
            ],
            "continuity": continuity_analysis,
            "architecture_status": "modular_backend_implemented",
            "last_session": recent_conversations[0].timestamp.isoformat() if recent_conversations else None
        }
        
        logger.info(f"Contexto de startup gerado para {llm_provider}")
        return startup_context
    
    async def _analyze_conversation_continuity(self, conversations: List[ConversationEntry]) -> Dict[str, Any]:
        """Analisa continuidade das conversas para identificar onde paramos."""
        
        if not conversations:
            return {"status": "new_session", "recommendations": ["Começar nova sessão"]}
        
        latest = conversations[0]
        
        # Identifica tipo de última conversa
        if "implementation" in latest.context_type.lower() or "code" in latest.context_type.lower():
            continuity_type = "implementation_in_progress"
            recommendations = [
                "Continuar implementação de código",
                "Verificar testes de integração",
                "Revisar arquitetura modular"
            ]
        elif "planning" in latest.context_type.lower() or "architecture" in latest.context_type.lower():
            continuity_type = "planning_phase"
            recommendations = [
                "Finalizar planejamento arquitetural", 
                "Implementar módulos pendentes",
                "Criar testes de validação"
            ]
        else:
            continuity_type = "general_session"
            recommendations = ["Continuar trabalho geral do projeto"]
        
        # Analisa padrões nas últimas conversas
        context_types = [conv.context_type for conv in conversations]
        most_common_context = max(set(context_types), key=context_types.count)
        
        return {
            "status": continuity_type,
            "last_conversation_age_hours": (datetime.now() - latest.timestamp).total_seconds() / 3600,
            "most_common_context": most_common_context,
            "recommendations": recommendations,
            "session_continuity": "high" if len(conversations) > 2 else "medium"
        }
    
    async def _update_project_context_from_conversation(self, entry: ConversationEntry) -> None:
        """Atualiza contexto do projeto baseado na conversação."""
        
        if not self.current_project_context:
            self.current_project_context = ProjectContext(
                project_id="kec-biomaterials-scaffolds",
                current_phase="development",
                active_tasks=[],
                completed_tasks=[],
                pending_decisions=[],
                architecture_decisions={},
                file_modifications=[],
                next_steps=[],
                preferences={},
                last_updated=datetime.now()
            )
        
        # Extrai informações da conversação
        context = self.current_project_context
        
        # Analisa conteúdo para extrair tarefas e decisões
        content = entry.user_message + " " + entry.assistant_response
        
        # Identifica tarefas concluídas
        if any(keyword in content.lower() for keyword in ["completed", "finished", "done", "implemented"]):
            # Extração simples de tarefas concluídas
            if "modular" in content.lower() and "architecture" in content.lower():
                task = "Arquitetura modular backend implementada"
                if task not in context.completed_tasks:
                    context.completed_tasks.append(task)
        
        # Identifica próximos passos
        if any(keyword in content.lower() for keyword in ["next", "próximo", "todo", "pendente"]):
            if "memory" in content.lower() and "system" in content.lower():
                next_step = "Implementar sistema de memória avançado"
                if next_step not in context.next_steps:
                    context.next_steps.append(next_step)
        
        # Atualiza fase do projeto
        if entry.context_type == "architecture" and entry.project_phase == "planning":
            context.current_phase = "architecture_design"
        elif "implementation" in entry.context_type:
            context.current_phase = "implementation"
        
        context.last_updated = datetime.now()
        
        await self._store_project_context(context)
    
    async def _load_current_project_context(self) -> None:
        """Carrega contexto atual do projeto."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM project_context 
            WHERE project_id = ? 
            ORDER BY last_updated DESC LIMIT 1
        """, ("kec-biomaterials-scaffolds",))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            self.current_project_context = ProjectContext(
                project_id=row[0],
                current_phase=row[1],
                active_tasks=json.loads(row[2] or "[]"),
                completed_tasks=json.loads(row[3] or "[]"),
                pending_decisions=json.loads(row[4] or "[]"),
                architecture_decisions=json.loads(row[5] or "{}"),
                file_modifications=json.loads(row[6] or "[]"),
                next_steps=json.loads(row[7] or "[]"),
                preferences=json.loads(row[8] or "{}"),
                last_updated=datetime.fromisoformat(row[9])
            )
            logger.info("Contexto do projeto carregado")
        else:
            logger.info("Contexto do projeto não encontrado - criando novo")
    
    async def _store_project_context(self, context: ProjectContext) -> None:
        """Armazena contexto do projeto."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO project_context 
            (project_id, current_phase, active_tasks, completed_tasks, 
             pending_decisions, architecture_decisions, file_modifications,
             next_steps, preferences, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            context.project_id,
            context.current_phase,
            json.dumps(context.active_tasks),
            json.dumps(context.completed_tasks),
            json.dumps(context.pending_decisions),
            json.dumps(context.architecture_decisions),
            json.dumps(context.file_modifications),
            json.dumps(context.next_steps),
            json.dumps(context.preferences),
            context.last_updated.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    async def get_conversation_summary(self, days: int = 7) -> Dict[str, Any]:
        """Gera resumo das conversações recentes."""
        
        recent_conversations = await self.retrieve_relevant_context(
            query="",
            max_results=50,
            time_window_days=days
        )
        
        if not recent_conversations:
            return {"period": f"last_{days}_days", "total_conversations": 0}
        
        # Estatísticas
        llm_usage = {}
        context_types = {}
        
        for conv in recent_conversations:
            llm_usage[conv.llm_provider] = llm_usage.get(conv.llm_provider, 0) + 1
            context_types[conv.context_type] = context_types.get(conv.context_type, 0) + 1
        
        return {
            "period": f"last_{days}_days",
            "total_conversations": len(recent_conversations),
            "llm_usage": llm_usage,
            "context_distribution": context_types,
            "avg_relevance": sum(conv.relevance_score for conv in recent_conversations) / len(recent_conversations),
            "most_active_llm": max(llm_usage.items(), key=lambda x: x[1])[0] if llm_usage else None,
            "most_common_context": max(context_types.items(), key=lambda x: x[1])[0] if context_types else None
        }
    
    async def export_conversation_history(self, 
                                        output_file: str,
                                        format: str = "json",
                                        llm_provider: Optional[str] = None) -> bool:
        """Exporta histórico de conversações."""
        
        conversations = await self.retrieve_relevant_context(
            query="",
            llm_provider=llm_provider,
            max_results=1000,
            time_window_days=365
        )
        
        try:
            if format == "json":
                export_data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_conversations": len(conversations),
                    "conversations": [asdict(conv) for conv in conversations]
                }
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"Histórico exportado para {output_file}: {len(conversations)} conversações")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao exportar histórico: {e}")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """Status do sistema de memória."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM conversations")
        total_conversations = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT llm_provider) FROM conversations")
        unique_llms = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "system": "conversation_memory",
            "database_path": str(self.db_path),
            "total_conversations": total_conversations,
            "unique_llm_providers": unique_llms,
            "current_project": self.current_project_context.project_id if self.current_project_context else None,
            "project_phase": self.current_project_context.current_phase if self.current_project_context else None,
            "status": "ready"
        }


# Instância global
_conversation_memory = ConversationMemorySystem()


async def get_conversation_memory() -> ConversationMemorySystem:
    """Factory function para sistema de memória."""
    if not _conversation_memory._db_path.exists():
        await _conversation_memory.initialize()
    return _conversation_memory