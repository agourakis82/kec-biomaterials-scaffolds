"""Application lifespan management for DARWIN META-RESEARCH BRAIN."""

import asyncio
import signal
from typing import TYPE_CHECKING

from .logging import get_logger

if TYPE_CHECKING:
    from main import DarwinMetaResearchBrain

logger = get_logger("lifespan")


async def initialize_brain(brain: "DarwinMetaResearchBrain") -> None:
    """Initialize all DARWIN brain systems."""
    logger.info("🧠 Initializing DARWIN META-RESEARCH BRAIN systems...")
    
    try:
        # Initialize core systems
        await brain.initialize_core_systems()
        
        # Initialize domain engines  
        await brain.initialize_domain_engines()
        
        logger.info("✅ DARWIN brain initialization completed successfully")
        
    except Exception as e:
        logger.error(f"❌ DARWIN brain initialization failed: {e}")
        raise


async def shutdown_brain(brain: "DarwinMetaResearchBrain") -> None:
    """Shutdown all DARWIN brain systems gracefully."""
    logger.info("🛑 Shutting down DARWIN META-RESEARCH BRAIN systems...")
    
    try:
        # Shutdown domain engines
        if brain.domain_engines:
            for domain_name, engine in brain.domain_engines.items():
                try:
                    if hasattr(engine, 'shutdown'):
                        await engine.shutdown()
                    logger.info(f"✅ {domain_name} domain engine shutdown")
                except Exception as e:
                    logger.error(f"❌ Error shutting down {domain_name} engine: {e}")
        
        # Shutdown core systems
        if brain.knowledge_graph:
            try:
                await brain.knowledge_graph.shutdown()
                logger.info("✅ Knowledge graph shutdown")
            except Exception as e:
                logger.error(f"❌ Error shutting down knowledge graph: {e}")
        
        if brain.multi_ai_hub:
            try:
                await brain.multi_ai_hub.shutdown()
                logger.info("✅ Multi-AI hub shutdown")
            except Exception as e:
                logger.error(f"❌ Error shutting down multi-AI hub: {e}")
        
        if brain.cache_manager:
            try:
                await brain.cache_manager.shutdown()
                logger.info("✅ Cache manager shutdown")
            except Exception as e:
                logger.error(f"❌ Error shutting down cache manager: {e}")
                
        if brain.auth_manager:
            try:
                await brain.auth_manager.shutdown()
                logger.info("✅ Auth manager shutdown") 
            except Exception as e:
                logger.error(f"❌ Error shutting down auth manager: {e}")
        
        logger.info("✅ DARWIN brain shutdown completed successfully")
        
    except Exception as e:
        logger.error(f"❌ DARWIN brain shutdown failed: {e}")
        raise


def setup_signal_handlers(brain: "DarwinMetaResearchBrain") -> None:
    """Setup signal handlers for graceful shutdown."""
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(shutdown_brain(brain))
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


__all__ = [
    "initialize_brain",
    "shutdown_brain", 
    "setup_signal_handlers"
]