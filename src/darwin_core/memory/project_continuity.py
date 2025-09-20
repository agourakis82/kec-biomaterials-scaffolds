"""
Project Continuity System - Sistema de Continuidade de Projeto
=============================================================

Sistema que assegura continuidade completa do projeto, lembrando:
- Arquivos modificados recentemente
- Tarefas em andamento
- Próximos passos planejados
- Configurações e preferências
- Decisões de design e arquitetura
"""

import json
import sqlite3
import os
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import logging
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class FileModification:
    """Modificação de arquivo rastreada."""
    file_path: str
    modification_type: str  # "created", "modified", "deleted"
    timestamp: datetime
    size_bytes: int
    content_hash: str
    user_action: str  # "manual_edit", "generated", "refactored"
    related_task: Optional[str] = None
    importance: int = 1  # 1-5, 5 = critical


@dataclass
class TaskProgress:
    """Progresso de uma tarefa."""
    task_id: str
    description: str
    status: str  # "pending", "in_progress", "completed", "blocked"
    created_at: datetime
    updated_at: datetime
    estimated_completion: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    completion_percentage: int = 0
    blockers: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ArchitecturalDecision:
    """Decisão arquitetural documentada."""
    decision_id: str
    title: str
    description: str
    rationale: str
    alternatives_considered: List[str]
    decision_date: datetime
    status: str  # "proposed", "accepted", "implemented", "deprecated"
    impact_assessment: str
    related_files: List[str] = field(default_factory=list)


@dataclass
class ProjectPreferences:
    """Preferências estabelecidas do projeto."""
    coding_style: Dict[str, str]
    architectural_patterns: List[str]
    testing_approach: str
    deployment_preferences: Dict[str, str]
    llm_preferences: Dict[str, Any]
    workflow_preferences: Dict[str, Any]


class ProjectContinuitySystem:
    """
    Sistema de continuidade que:
    - Monitora mudanças de arquivos automaticamente
    - Rastreia progresso de tarefas
    - Mantém contexto de decisões arquiteturais
    - Preserva preferências e configurações
    - Gera contexto de "onde paramos"
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.db_path = Path("data/memory/project_continuity.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Cache de estados
        self._file_cache: Dict[str, FileModification] = {}
        self._active_tasks: Dict[str, TaskProgress] = {}
        self._decisions: Dict[str, ArchitecturalDecision] = {}
        self._preferences: Optional[ProjectPreferences] = None
        
        # Monitoramento
        self._monitoring = False
        self._last_scan = datetime.now()
    
    async def initialize(self) -> None:
        """Inicializa sistema de continuidade."""
        
        await self._setup_database()
        await self._load_current_state()
        await self._scan_recent_changes()
        
        logger.info("Sistema de continuidade de projeto inicializado")
    
    async def _setup_database(self) -> None:
        """Configura banco de dados."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabela de modificações de arquivos
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_modifications (
                id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                modification_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                size_bytes INTEGER,
                content_hash TEXT,
                user_action TEXT,
                related_task TEXT,
                importance INTEGER DEFAULT 1
            )
        """)
        
        # Tabela de tarefas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_progress (
                task_id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                estimated_completion TEXT,
                dependencies TEXT,  -- JSON array
                completion_percentage INTEGER DEFAULT 0,
                blockers TEXT,  -- JSON array
                notes TEXT
            )
        """)
        
        # Tabela de decisões arquiteturais
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS architectural_decisions (
                decision_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                rationale TEXT NOT NULL,
                alternatives_considered TEXT,  -- JSON array
                decision_date TEXT NOT NULL,
                status TEXT NOT NULL,
                impact_assessment TEXT,
                related_files TEXT  -- JSON array
            )
        """)
        
        # Tabela de preferências
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS project_preferences (
                preference_type TEXT PRIMARY KEY,
                preferences_data TEXT NOT NULL,  -- JSON object
                last_updated TEXT NOT NULL
            )
        """)
        
        # Índices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_timestamp ON file_modifications(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON task_progress(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_date ON architectural_decisions(decision_date)")
        
        conn.commit()
        conn.close()
    
    async def track_file_modification(self, 
                                    file_path: str,
                                    modification_type: str,
                                    user_action: str = "manual_edit",
                                    related_task: Optional[str] = None,
                                    importance: int = 1) -> None:
        """Rastreia modificação de arquivo."""
        
        try:
            full_path = self.project_root / file_path
            
            # Calcula hash do conteúdo se arquivo existe
            content_hash = ""
            size_bytes = 0
            
            if full_path.exists() and modification_type != "deleted":
                with open(full_path, 'rb') as f:
                    content = f.read()
                    content_hash = hashlib.md5(content).hexdigest()
                    size_bytes = len(content)
            
            modification = FileModification(
                file_path=file_path,
                modification_type=modification_type,
                timestamp=datetime.now(),
                size_bytes=size_bytes,
                content_hash=content_hash,
                user_action=user_action,
                related_task=related_task,
                importance=importance
            )
            
            mod_id = f"{file_path}_{modification.timestamp.isoformat()}"
            
            await self._store_file_modification(mod_id, modification)
            self._file_cache[file_path] = modification
            
            logger.debug(f"Arquivo rastreado: {file_path} ({modification_type})")
            
        except Exception as e:
            logger.error(f"Erro ao rastrear arquivo {file_path}: {e}")
    
    async def _store_file_modification(self, mod_id: str, modification: FileModification) -> None:
        """Armazena modificação no banco."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO file_modifications
            (id, file_path, modification_type, timestamp, size_bytes,
             content_hash, user_action, related_task, importance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            mod_id,
            modification.file_path,
            modification.modification_type,
            modification.timestamp.isoformat(),
            modification.size_bytes,
            modification.content_hash,
            modification.user_action,
            modification.related_task,
            modification.importance
        ))
        
        conn.commit()
        conn.close()
    
    async def update_task_progress(self, 
                                 task_id: str,
                                 description: str,
                                 status: str,
                                 completion_percentage: int = 0,
                                 notes: str = "",
                                 blockers: Optional[List[str]] = None) -> None:
        """Atualiza progresso de tarefa."""
        
        now = datetime.now()
        
        if task_id in self._active_tasks:
            task = self._active_tasks[task_id]
            task.status = status
            task.updated_at = now
            task.completion_percentage = completion_percentage
            task.notes = notes
            task.blockers = blockers or []
        else:
            task = TaskProgress(
                task_id=task_id,
                description=description,
                status=status,
                created_at=now,
                updated_at=now,
                completion_percentage=completion_percentage,
                notes=notes,
                blockers=blockers or []
            )
            self._active_tasks[task_id] = task
        
        await self._store_task_progress(task)
        logger.info(f"Tarefa atualizada: {task_id} ({status}, {completion_percentage}%)")
    
    async def _store_task_progress(self, task: TaskProgress) -> None:
        """Armazena progresso da tarefa."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO task_progress
            (task_id, description, status, created_at, updated_at,
             estimated_completion, dependencies, completion_percentage, blockers, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task.task_id,
            task.description,
            task.status,
            task.created_at.isoformat(),
            task.updated_at.isoformat(),
            task.estimated_completion.isoformat() if task.estimated_completion else None,
            json.dumps(task.dependencies),
            task.completion_percentage,
            json.dumps(task.blockers),
            task.notes
        ))
        
        conn.commit()
        conn.close()
    
    async def record_architectural_decision(self, 
                                          title: str,
                                          description: str,
                                          rationale: str,
                                          alternatives: List[str],
                                          impact_assessment: str,
                                          related_files: Optional[List[str]] = None) -> str:
        """Registra decisão arquitetural."""
        
        decision_id = hashlib.md5(f"{title}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        
        decision = ArchitecturalDecision(
            decision_id=decision_id,
            title=title,
            description=description,
            rationale=rationale,
            alternatives_considered=alternatives,
            decision_date=datetime.now(),
            status="accepted",
            impact_assessment=impact_assessment,
            related_files=related_files or []
        )
        
        self._decisions[decision_id] = decision
        await self._store_architectural_decision(decision)
        
        logger.info(f"Decisão arquitetural registrada: {title}")
        return decision_id
    
    async def _store_architectural_decision(self, decision: ArchitecturalDecision) -> None:
        """Armazena decisão arquitetural."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO architectural_decisions
            (decision_id, title, description, rationale, alternatives_considered,
             decision_date, status, impact_assessment, related_files)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            decision.decision_id,
            decision.title,
            decision.description,
            decision.rationale,
            json.dumps(decision.alternatives_considered),
            decision.decision_date.isoformat(),
            decision.status,
            decision.impact_assessment,
            json.dumps(decision.related_files)
        ))
        
        conn.commit()
        conn.close()
    
    async def _scan_recent_changes(self) -> None:
        """Escaneia mudanças recentes no projeto."""
        
        # Escaneia diretório src/ para mudanças recentes (último dia)
        src_dir = self.project_root / "src"
        if not src_dir.exists():
            return
        
        cutoff_time = time.time() - (24 * 3600)  # 24 horas atrás
        
        for file_path in src_dir.rglob("*.py"):
            try:
                stat = file_path.stat()
                
                if stat.st_mtime > cutoff_time:
                    # Arquivo modificado recentemente
                    rel_path = str(file_path.relative_to(self.project_root))
                    
                    # Verifica se já está no cache
                    if rel_path not in self._file_cache:
                        await self.track_file_modification(
                            rel_path,
                            "modified",
                            "auto_detected",
                            importance=3
                        )
            except Exception as e:
                logger.debug(f"Erro ao escanear {file_path}: {e}")
    
    async def get_project_status_snapshot(self) -> Dict[str, Any]:
        """Gera snapshot completo do status atual do projeto."""
        
        # Arquivos modificados recentemente (72h)
        recent_files = await self._get_recent_file_modifications(hours=72)
        
        # Tarefas ativas
        active_tasks = [task for task in self._active_tasks.values() 
                       if task.status in ["pending", "in_progress"]]
        
        # Decisões recentes
        recent_decisions = [decision for decision in self._decisions.values()
                          if (datetime.now() - decision.decision_date).days <= 7]
        
        # Próximos passos inferidos
        next_steps = await self._infer_next_steps()
        
        return {
            "snapshot_timestamp": datetime.now().isoformat(),
            "project_health": {
                "active_files": len(recent_files),
                "active_tasks": len(active_tasks),
                "completed_tasks": len([t for t in self._active_tasks.values() if t.status == "completed"]),
                "blocked_tasks": len([t for t in active_tasks if t.blockers]),
                "recent_decisions": len(recent_decisions)
            },
            "current_focus": {
                "most_active_directory": self._get_most_active_directory(recent_files),
                "primary_task": active_tasks[0].description if active_tasks else None,
                "latest_decision": recent_decisions[0].title if recent_decisions else None
            },
            "recent_activity": {
                "files_modified_24h": len([f for f in recent_files if (datetime.now() - f.timestamp).hours <= 24]),
                "files_modified_72h": len(recent_files),
                "critical_files_modified": len([f for f in recent_files if f.importance >= 4])
            },
            "continuity_indicators": {
                "session_gap_hours": (datetime.now() - self._last_scan).total_seconds() / 3600,
                "project_momentum": self._calculate_project_momentum(),
                "context_richness": len(self._active_tasks) + len(recent_files) + len(recent_decisions)
            },
            "recommended_actions": next_steps
        }
    
    async def _get_recent_file_modifications(self, hours: int = 24) -> List[FileModification]:
        """Recupera modificações recentes de arquivos."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        cursor.execute("""
            SELECT * FROM file_modifications
            WHERE timestamp > ?
            ORDER BY importance DESC, timestamp DESC
        """, (cutoff,))
        
        rows = cursor.fetchall()
        conn.close()
        
        modifications = []
        for row in rows:
            modifications.append(FileModification(
                file_path=row[1],
                modification_type=row[2],
                timestamp=datetime.fromisoformat(row[3]),
                size_bytes=row[4],
                content_hash=row[5],
                user_action=row[6],
                related_task=row[7],
                importance=row[8]
            ))
        
        return modifications
    
    def _get_most_active_directory(self, recent_files: List[FileModification]) -> Optional[str]:
        """Identifica diretório mais ativo."""
        
        if not recent_files:
            return None
        
        dir_counts: Dict[str, int] = {}
        
        for file_mod in recent_files:
            directory = str(Path(file_mod.file_path).parent)
            dir_counts[directory] = dir_counts.get(directory, 0) + 1
        
        return max(dir_counts.items(), key=lambda x: x[1])[0] if dir_counts else None
    
    def _calculate_project_momentum(self) -> float:
        """Calcula momentum do projeto (0.0 a 1.0)."""
        
        # Fatores que influenciam momentum
        recent_tasks = len([t for t in self._active_tasks.values() 
                          if (datetime.now() - t.updated_at).days <= 1])
        
        completion_rate = sum(t.completion_percentage for t in self._active_tasks.values()) / max(1, len(self._active_tasks) * 100)
        
        recent_files = len([f for f in self._file_cache.values()
                          if (datetime.now() - f.timestamp).hours <= 24])
        
        # Normaliza e combina fatores
        task_momentum = min(recent_tasks / 5.0, 1.0)
        file_momentum = min(recent_files / 10.0, 1.0)
        
        return (task_momentum * 0.4) + (completion_rate * 0.3) + (file_momentum * 0.3)
    
    async def _infer_next_steps(self) -> List[str]:
        """Infere próximos passos baseado no estado atual."""
        
        next_steps = []
        
        # Baseado em tarefas pendentes
        pending_tasks = [t for t in self._active_tasks.values() if t.status == "pending"]
        if pending_tasks:
            next_steps.append(f"Continuar tarefa: {pending_tasks[0].description}")
        
        # Baseado em tarefas bloqueadas
        blocked_tasks = [t for t in self._active_tasks.values() if t.blockers]
        if blocked_tasks:
            next_steps.append(f"Resolver bloqueio: {blocked_tasks[0].blockers[0]}")
        
        # Baseado em arquivos recentes
        recent_files = await self._get_recent_file_modifications(hours=24)
        if recent_files:
            most_recent = recent_files[0]
            if "test" in most_recent.file_path:
                next_steps.append("Executar testes de integração")
            elif most_recent.modification_type == "created":
                next_steps.append(f"Validar implementação: {most_recent.file_path}")
        
        # Baseado na fase do projeto (inferida)
        if len([f for f in recent_files if "src/" in f.file_path]) > 5:
            next_steps.append("Revisar qualidade de código e arquitetura")
        
        # Defaults se nada específico for encontrado
        if not next_steps:
            next_steps = [
                "Continuar desenvolvimento dos módulos",
                "Executar testes de integração",
                "Atualizar documentação"
            ]
        
        return next_steps[:5]  # Limita a 5 próximos passos
    
    async def get_session_resumption_context(self) -> Dict[str, Any]:
        """Gera contexto completo para retomar sessão."""
        
        # Status atual do projeto
        project_snapshot = await self.get_project_status_snapshot()
        
        # Tarefas pendentes prioritárias
        active_tasks = [
            {
                "id": task.task_id,
                "description": task.description,
                "status": task.status,
                "completion": task.completion_percentage,
                "blockers": task.blockers,
                "priority": "high" if task.blockers or task.status == "in_progress" else "normal"
            }
            for task in self._active_tasks.values()
            if task.status in ["pending", "in_progress"]
        ]
        
        # Arquivos que precisam de atenção
        recent_files = await self._get_recent_file_modifications(hours=48)
        files_needing_attention = [
            {
                "path": f.file_path,
                "type": f.modification_type,
                "importance": f.importance,
                "age_hours": (datetime.now() - f.timestamp).total_seconds() / 3600
            }
            for f in recent_files
            if f.importance >= 3 or f.modification_type == "created"
        ]
        
        return {
            "resumption_context": {
                "timestamp": datetime.now().isoformat(),
                "project_momentum": project_snapshot["continuity_indicators"]["project_momentum"],
                "session_gap_hours": project_snapshot["continuity_indicators"]["session_gap_hours"],
                "context_richness": project_snapshot["continuity_indicators"]["context_richness"]
            },
            "immediate_priorities": {
                "active_tasks": active_tasks[:3],  # Top 3 tarefas
                "files_needing_attention": files_needing_attention[:5],  # Top 5 arquivos
                "next_steps": project_snapshot["recommended_actions"][:3]  # Top 3 next steps
            },
            "project_status": {
                "current_phase": self._infer_current_phase(),
                "health_indicators": project_snapshot["project_health"],
                "recent_focus": project_snapshot["current_focus"]
            },
            "continuity_recommendations": self._generate_continuity_recommendations(project_snapshot)
        }
    
    def _infer_current_phase(self) -> str:
        """Infere fase atual do projeto baseado na atividade."""
        
        recent_files = list(self._file_cache.values())
        
        # Analisa tipos de arquivos modificados recentemente
        file_types = {}
        for file_mod in recent_files:
            if "/tests/" in file_mod.file_path:
                file_types["testing"] = file_types.get("testing", 0) + 1
            elif "/docs/" in file_mod.file_path:
                file_types["documentation"] = file_types.get("documentation", 0) + 1
            elif file_mod.modification_type == "created":
                file_types["implementation"] = file_types.get("implementation", 0) + 1
            else:
                file_types["maintenance"] = file_types.get("maintenance", 0) + 1
        
        if not file_types:
            return "planning"
        
        # Retorna fase mais ativa
        most_active = max(file_types.items(), key=lambda x: x[1])[0]
        
        phase_mapping = {
            "testing": "validation",
            "documentation": "documentation", 
            "implementation": "active_development",
            "maintenance": "maintenance"
        }
        
        return phase_mapping.get(most_active, "development")
    
    def _generate_continuity_recommendations(self, snapshot: Dict[str, Any]) -> List[str]:
        """Gera recomendações para continuidade."""
        
        recommendations = []
        
        momentum = snapshot["continuity_indicators"]["project_momentum"]
        
        if momentum > 0.7:
            recommendations.append("Alto momentum detectado - continuar implementação atual")
        elif momentum < 0.3:
            recommendations.append("Baixo momentum - considerar revisão de prioridades")
        
        gap_hours = snapshot["continuity_indicators"]["session_gap_hours"]
        
        if gap_hours > 24:
            recommendations.append("Gap longo desde última sessão - revisar contexto recente")
        elif gap_hours < 1:
            recommendations.append("Sessão contínua - manter foco atual")
        
        active_tasks = snapshot["project_health"]["active_tasks"]
        blocked_tasks = snapshot["project_health"]["blocked_tasks"]
        
        if blocked_tasks > 0:
            recommendations.append(f"Resolver {blocked_tasks} tarefa(s) bloqueada(s) prioritariamente")
        
        if active_tasks > 5:
            recommendations.append("Muitas tarefas ativas - considerar consolidação")
        
        return recommendations
    
    async def _load_current_state(self) -> None:
        """Carrega estado atual do banco."""
        
        # Carrega tarefas ativas
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM task_progress WHERE status IN ('pending', 'in_progress')")
        task_rows = cursor.fetchall()
        
        for row in task_rows:
            task = TaskProgress(
                task_id=row[0],
                description=row[1],
                status=row[2],
                created_at=datetime.fromisoformat(row[3]),
                updated_at=datetime.fromisoformat(row[4]),
                estimated_completion=datetime.fromisoformat(row[5]) if row[5] else None,
                dependencies=json.loads(row[6] or "[]"),
                completion_percentage=row[7],
                blockers=json.loads(row[8] or "[]"),
                notes=row[9] or ""
            )
            self._active_tasks[task.task_id] = task
        
        conn.close()
        logger.info(f"Estado carregado: {len(self._active_tasks)} tarefas ativas")
    
    async def get_status(self) -> Dict[str, Any]:
        """Status do sistema de continuidade."""
        
        return {
            "system": "project_continuity",
            "database_path": str(self.db_path),
            "monitoring": self._monitoring,
            "cached_files": len(self._file_cache),
            "active_tasks": len(self._active_tasks),
            "architectural_decisions": len(self._decisions),
            "last_scan": self._last_scan.isoformat(),
            "status": "ready"
        }


# Instância global
_continuity_system = ProjectContinuitySystem()


async def get_continuity_system() -> ProjectContinuitySystem:
    """Factory function para sistema de continuidade."""
    if not _continuity_system.db_path.exists():
        await _continuity_system.initialize()
    return _continuity_system