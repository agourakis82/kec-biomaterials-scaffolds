"""
DARWIN SCIENTIFIC DISCOVERY - Scheduler Service
Serviço de scheduling para discovery científico contínuo e tarefas automatizadas
"""

import asyncio
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field

from ..models.discovery_models import (
    DiscoveryRequest,
    ScientificDomain,
    DiscoveryStatus
)

# APScheduler imports with fallback
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.jobstores.memory import MemoryJobStore
    from apscheduler.executors.asyncio import AsyncIOExecutor
    from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, JobExecutionEvent
    APSCHEDULER_AVAILABLE = True
except ImportError:
    AsyncIOScheduler = None
    IntervalTrigger = None
    CronTrigger = None
    MemoryJobStore = None
    AsyncIOExecutor = None
    EVENT_JOB_EXECUTED = None
    EVENT_JOB_ERROR = None
    JobExecutionEvent = None
    APSCHEDULER_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# SCHEDULER CONFIGURATION
# =============================================================================

@dataclass
class SchedulerConfig:
    """Configuração do scheduler."""
    max_workers: int = 3
    coalesce: bool = True  # Combinar jobs duplicados
    misfire_grace_time: int = 300  # 5 minutos
    timezone: str = "UTC"
    job_defaults: Dict[str, Any] = field(default_factory=lambda: {
        'coalesce': True,
        'max_instances': 1,
        'misfire_grace_time': 300
    })


@dataclass
class ScheduledJob:
    """Informações de um job agendado."""
    job_id: str
    name: str
    function: str
    trigger_type: str
    trigger_config: Dict[str, Any]
    status: str = "scheduled"
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    success_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# SCIENTIFIC DISCOVERY SCHEDULER
# =============================================================================

class ScientificDiscoveryScheduler:
    """
    Scheduler dedicado para discovery científico contínuo.
    """
    
    def __init__(self, config: Optional[SchedulerConfig] = None):
        """
        Inicializa o scheduler.
        
        Args:
            config: Configuração personalizada do scheduler
        """
        self.config = config or SchedulerConfig()
        
        # Scheduler principal
        self.scheduler: Optional[AsyncIOScheduler] = None
        self._is_running = False
        
        # Jobs registry
        self.jobs_registry: Dict[str, ScheduledJob] = {}
        
        # Discovery engine reference (lazy loading)
        self._discovery_engine = None
        
        # Estatísticas
        self.stats = {
            'jobs_scheduled': 0,
            'jobs_executed': 0,
            'jobs_failed': 0,
            'scheduler_start_time': None,
            'last_job_execution': None,
            'uptime_seconds': 0
        }
        
        if APSCHEDULER_AVAILABLE:
            self._initialize_scheduler()
        else:
            logger.error("APScheduler not available - scheduler functionality disabled")
    
    def _initialize_scheduler(self):
        """Inicializa o APScheduler."""
        if not APSCHEDULER_AVAILABLE:
            return
        
        # Configurar job stores e executors
        jobstores = {
            'default': MemoryJobStore()
        }
        
        executors = {
            'default': AsyncIOExecutor(max_workers=self.config.max_workers)
        }
        
        # Criar scheduler
        self.scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=self.config.job_defaults,
            timezone=self.config.timezone
        )
        
        # Adicionar event listeners
        self.scheduler.add_listener(
            self._job_executed_listener,
            EVENT_JOB_EXECUTED | EVENT_JOB_ERROR
        )
        
        logger.info("Scientific Discovery Scheduler initialized")
    
    def _job_executed_listener(self, event: JobExecutionEvent):
        """Listener para eventos de execução de jobs."""
        job_id = event.job_id
        
        if job_id in self.jobs_registry:
            job_info = self.jobs_registry[job_id]
            job_info.last_run = datetime.now(timezone.utc)
            
            if event.exception:
                # Job failed
                job_info.error_count += 1
                job_info.last_error = str(event.exception)
                job_info.status = "error"
                self.stats['jobs_failed'] += 1
                
                logger.error(f"Scheduled job '{job_info.name}' failed: {event.exception}")
            else:
                # Job succeeded
                job_info.success_count += 1
                job_info.status = "completed"
                job_info.last_error = None
                self.stats['jobs_executed'] += 1
                
                logger.info(f"Scheduled job '{job_info.name}' completed successfully")
            
            # Update next run time
            job = self.scheduler.get_job(job_id)
            if job:
                job_info.next_run = job.next_run_time
        
        self.stats['last_job_execution'] = datetime.now(timezone.utc)
    
    @property
    def discovery_engine(self):
        """Lazy loading do Discovery Engine."""
        if self._discovery_engine is None:
            from .discovery_engine import get_discovery_engine
            self._discovery_engine = get_discovery_engine()
        return self._discovery_engine
    
    async def start(self) -> bool:
        """Inicia o scheduler."""
        if not APSCHEDULER_AVAILABLE:
            logger.error("Cannot start scheduler - APScheduler not available")
            return False
        
        if self._is_running:
            logger.warning("Scheduler already running")
            return True
        
        try:
            self.scheduler.start()
            self._is_running = True
            self.stats['scheduler_start_time'] = datetime.now(timezone.utc)
            
            logger.info("🕒 Scientific Discovery Scheduler started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            return False
    
    async def stop(self, wait: bool = True) -> bool:
        """Para o scheduler."""
        if not self._is_running or not self.scheduler:
            logger.warning("Scheduler not running")
            return True
        
        try:
            self.scheduler.shutdown(wait=wait)
            self._is_running = False
            
            logger.info("🛑 Scientific Discovery Scheduler stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop scheduler: {e}")
            return False
    
    def is_running(self) -> bool:
        """Verifica se o scheduler está rodando."""
        return self._is_running and self.scheduler is not None
    
    # =============================================================================
    # DISCOVERY SCHEDULING METHODS
    # =============================================================================
    
    async def schedule_continuous_discovery(
        self,
        interval_minutes: int = 120,
        domains: Optional[List[ScientificDomain]] = None,
        max_papers: int = 100,
        job_id: str = "continuous_discovery"
    ) -> bool:
        """
        Agenda discovery científico contínuo.
        
        Args:
            interval_minutes: Intervalo em minutos entre execuções
            domains: Domínios científicos (None = todos)
            max_papers: Número máximo de papers por execução
            job_id: ID único do job
        """
        if not APSCHEDULER_AVAILABLE or not self.scheduler:
            logger.error("Cannot schedule job - scheduler not available")
            return False
        
        try:
            # Remover job existente se houver
            if self.scheduler.get_job(job_id):
                self.scheduler.remove_job(job_id)
                if job_id in self.jobs_registry:
                    del self.jobs_registry[job_id]
            
            # Configurar trigger
            trigger = IntervalTrigger(minutes=interval_minutes)
            
            # Agendar job
            self.scheduler.add_job(
                self._execute_discovery,
                trigger=trigger,
                id=job_id,
                name=f"Continuous Discovery ({interval_minutes}min)",
                args=[domains, max_papers, job_id],
                replace_existing=True
            )
            
            # Registrar job
            job_info = ScheduledJob(
                job_id=job_id,
                name=f"Continuous Discovery ({interval_minutes}min)",
                function="execute_discovery",
                trigger_type="interval",
                trigger_config={"minutes": interval_minutes},
                next_run=datetime.now(timezone.utc) + timedelta(minutes=interval_minutes)
            )
            
            self.jobs_registry[job_id] = job_info
            self.stats['jobs_scheduled'] += 1
            
            logger.info(f"Scheduled continuous discovery: every {interval_minutes} minutes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to schedule continuous discovery: {e}")
            return False
    
    async def schedule_daily_discovery(
        self,
        hour: int = 2,
        minute: int = 0,
        domains: Optional[List[ScientificDomain]] = None,
        max_papers: int = 200,
        job_id: str = "daily_discovery"
    ) -> bool:
        """
        Agenda discovery diário.
        
        Args:
            hour: Hora do dia (0-23)
            minute: Minuto da hora (0-59)
            domains: Domínios científicos
            max_papers: Número máximo de papers
            job_id: ID do job
        """
        if not APSCHEDULER_AVAILABLE or not self.scheduler:
            return False
        
        try:
            # Configurar trigger para execução diária
            trigger = CronTrigger(hour=hour, minute=minute)
            
            self.scheduler.add_job(
                self._execute_discovery,
                trigger=trigger,
                id=job_id,
                name=f"Daily Discovery ({hour:02d}:{minute:02d})",
                args=[domains, max_papers, job_id],
                replace_existing=True
            )
            
            # Registrar job
            next_run = datetime.now(timezone.utc).replace(
                hour=hour, minute=minute, second=0, microsecond=0
            )
            if next_run <= datetime.now(timezone.utc):
                next_run += timedelta(days=1)
            
            job_info = ScheduledJob(
                job_id=job_id,
                name=f"Daily Discovery ({hour:02d}:{minute:02d})",
                function="execute_discovery",
                trigger_type="cron",
                trigger_config={"hour": hour, "minute": minute},
                next_run=next_run
            )
            
            self.jobs_registry[job_id] = job_info
            self.stats['jobs_scheduled'] += 1
            
            logger.info(f"Scheduled daily discovery at {hour:02d}:{minute:02d}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to schedule daily discovery: {e}")
            return False
    
    async def schedule_domain_specific_discovery(
        self,
        domain: ScientificDomain,
        interval_hours: int = 6,
        max_papers: int = 50
    ) -> bool:
        """
        Agenda discovery específico para um domínio.
        
        Args:
            domain: Domínio científico específico
            interval_hours: Intervalo em horas
            max_papers: Número máximo de papers
        """
        job_id = f"domain_{domain.value}_discovery"
        
        try:
            trigger = IntervalTrigger(hours=interval_hours)
            
            self.scheduler.add_job(
                self._execute_discovery,
                trigger=trigger,
                id=job_id,
                name=f"{domain.value.title()} Discovery ({interval_hours}h)",
                args=[[domain], max_papers, job_id],
                replace_existing=True
            )
            
            job_info = ScheduledJob(
                job_id=job_id,
                name=f"{domain.value.title()} Discovery",
                function="execute_discovery",
                trigger_type="interval",
                trigger_config={"hours": interval_hours},
                next_run=datetime.now(timezone.utc) + timedelta(hours=interval_hours)
            )
            
            self.jobs_registry[job_id] = job_info
            self.stats['jobs_scheduled'] += 1
            
            logger.info(f"Scheduled {domain.value} discovery: every {interval_hours} hours")
            return True
            
        except Exception as e:
            logger.error(f"Failed to schedule {domain.value} discovery: {e}")
            return False
    
    async def _execute_discovery(
        self, 
        domains: Optional[List[ScientificDomain]],
        max_papers: int,
        job_id: str
    ):
        """Executa discovery científico (método interno do scheduler)."""
        try:
            logger.info(f"🔬 Executing scheduled discovery job: {job_id}")
            
            # Criar request
            request = DiscoveryRequest(
                domains=domains or list(ScientificDomain),
                max_papers=max_papers,
                run_once=True
            )
            
            # Executar discovery
            result = await self.discovery_engine.run_discovery_async(request)
            
            if result.status == DiscoveryStatus.COMPLETED:
                logger.info(
                    f"✅ Scheduled discovery completed: "
                    f"{result.papers_novel} novel papers, "
                    f"{result.insights_generated} insights, "
                    f"{result.processing_time_seconds:.1f}s"
                )
            else:
                logger.warning(f"⚠️ Scheduled discovery had issues: {result.error_message}")
                
        except Exception as e:
            logger.error(f"❌ Scheduled discovery failed: {e}")
            raise  # Re-raise para que o scheduler registre o erro
    
    # =============================================================================
    # JOB MANAGEMENT
    # =============================================================================
    
    def remove_job(self, job_id: str) -> bool:
        """Remove um job agendado."""
        if not self.scheduler:
            return False
        
        try:
            self.scheduler.remove_job(job_id)
            if job_id in self.jobs_registry:
                del self.jobs_registry[job_id]
            
            logger.info(f"Removed scheduled job: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove job {job_id}: {e}")
            return False
    
    def pause_job(self, job_id: str) -> bool:
        """Pausa um job agendado."""
        if not self.scheduler:
            return False
        
        try:
            self.scheduler.pause_job(job_id)
            if job_id in self.jobs_registry:
                self.jobs_registry[job_id].status = "paused"
            
            logger.info(f"Paused job: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pause job {job_id}: {e}")
            return False
    
    def resume_job(self, job_id: str) -> bool:
        """Resume um job pausado."""
        if not self.scheduler:
            return False
        
        try:
            self.scheduler.resume_job(job_id)
            if job_id in self.jobs_registry:
                self.jobs_registry[job_id].status = "scheduled"
            
            logger.info(f"Resumed job: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume job {job_id}: {e}")
            return False
    
    def get_job_info(self, job_id: str) -> Optional[ScheduledJob]:
        """Retorna informações de um job específico."""
        return self.jobs_registry.get(job_id)
    
    def list_jobs(self) -> List[ScheduledJob]:
        """Lista todos os jobs agendados."""
        # Atualizar next_run times dos jobs
        if self.scheduler:
            for job_id, job_info in self.jobs_registry.items():
                job = self.scheduler.get_job(job_id)
                if job:
                    job_info.next_run = job.next_run_time
        
        return list(self.jobs_registry.values())
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do scheduler."""
        stats = self.stats.copy()
        
        if self.stats['scheduler_start_time']:
            uptime = (datetime.now(timezone.utc) - self.stats['scheduler_start_time']).total_seconds()
            stats['uptime_seconds'] = uptime
        
        stats.update({
            'is_running': self._is_running,
            'apscheduler_available': APSCHEDULER_AVAILABLE,
            'active_jobs': len(self.jobs_registry),
            'success_rate': (
                self.stats['jobs_executed'] / max(1, self.stats['jobs_executed'] + self.stats['jobs_failed'])
                if self.stats['jobs_executed'] or self.stats['jobs_failed'] else 0
            )
        })
        
        return stats


# =============================================================================
# GLOBAL SCHEDULER INSTANCE
# =============================================================================

# Instância global do scheduler
_scheduler_instance = None

def get_scheduler(config: Optional[SchedulerConfig] = None) -> ScientificDiscoveryScheduler:
    """Retorna instância singleton do Scheduler."""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = ScientificDiscoveryScheduler(config)
    return _scheduler_instance


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def start_continuous_discovery(interval_minutes: int = 120) -> bool:
    """Função conveniente para iniciar discovery contínuo."""
    scheduler = get_scheduler()
    
    if not scheduler.is_running():
        await scheduler.start()
    
    return await scheduler.schedule_continuous_discovery(interval_minutes)


async def stop_continuous_discovery() -> bool:
    """Função conveniente para parar discovery contínuo."""
    scheduler = get_scheduler()
    
    # Remove job de discovery contínuo
    scheduler.remove_job("continuous_discovery")
    
    # Para scheduler se não tiver mais jobs
    if not scheduler.list_jobs():
        return await scheduler.stop()
    
    return True


async def schedule_daily_biomaterials_discovery() -> bool:
    """Agenda discovery diário focado em biomateriais às 2:00 AM."""
    scheduler = get_scheduler()
    
    if not scheduler.is_running():
        await scheduler.start()
    
    return await scheduler.schedule_daily_discovery(
        hour=2,
        minute=0,
        domains=[ScientificDomain.BIOMATERIALS],
        max_papers=150,
        job_id="daily_biomaterials"
    )


async def setup_default_discovery_schedule() -> Dict[str, bool]:
    """Configura agenda padrão de discovery científico."""
    scheduler = get_scheduler()
    
    if not scheduler.is_running():
        await scheduler.start()
    
    results = {}
    
    # Discovery geral a cada 2 horas
    results['continuous_general'] = await scheduler.schedule_continuous_discovery(
        interval_minutes=120,
        domains=None,  # Todos os domínios
        max_papers=100
    )
    
    # Discovery diário de biomateriais
    results['daily_biomaterials'] = await scheduler.schedule_daily_discovery(
        hour=2, minute=0,
        domains=[ScientificDomain.BIOMATERIALS],
        max_papers=150,
        job_id="daily_biomaterials"
    )
    
    # Discovery de neurociência a cada 6 horas
    results['neuroscience_6h'] = await scheduler.schedule_domain_specific_discovery(
        domain=ScientificDomain.NEUROSCIENCE,
        interval_hours=6,
        max_papers=75
    )
    
    # Discovery de filosofia a cada 12 horas
    results['philosophy_12h'] = await scheduler.schedule_domain_specific_discovery(
        domain=ScientificDomain.PHILOSOPHY,
        interval_hours=12,
        max_papers=30
    )
    
    return results