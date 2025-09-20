"""
Helio Service - Serviço Principal do PCS Helio
=============================================

Serviço integrado que combina analytics, integração PCS Meta e funcionalidades avançadas.
"""

from typing import Dict, Any, Optional, List
import logging
import asyncio

from ..analytics import AnalyticsEngine, AnalyticsConfig
from ..integration import PCSMetaBridge

logger = logging.getLogger(__name__)


class HelioService:
    """
    Serviço principal do PCS Helio que integra:
    - Analytics avançados
    - Integração com pcs-meta-repo
    - Processamento de dados especializado
    """
    
    def __init__(self, 
                 analytics_config: Optional[AnalyticsConfig] = None,
                 pcs_meta_path: Optional[str] = None):
        self.analytics = AnalyticsEngine(analytics_config)
        self.pcs_bridge = PCSMetaBridge(pcs_meta_path)
        self._initialized = False
        
    async def initialize(self) -> bool:
        """Inicializa todos os componentes do serviço."""
        
        try:
            # Inicializa ponte PCS Meta
            pcs_ready = await self.pcs_bridge.initialize()
            
            self._initialized = True
            logger.info(f"Helio Service inicializado (PCS Meta: {'OK' if pcs_ready else 'Limited'})")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao inicializar Helio Service: {e}")
            return False
    
    async def analyze_with_pcs_integration(self, 
                                         kec_metrics: Dict[str, float],
                                         metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Análise KEC com integração PCS Meta."""
        
        # Análise local primeiro
        local_results = await self.analytics.analyze_kec_metrics(kec_metrics, metadata)
        
        # Tenta análise avançada via PCS Meta
        try:
            pcs_results = await self.pcs_bridge.execute_pcs_analysis(
                {"kec_metrics": kec_metrics, "metadata": metadata},
                analysis_type="kec_biomaterials"
            )
            
            # Combina resultados
            combined_results = {
                "local_analysis": {
                    "summary_stats": local_results.summary_stats,
                    "insights": local_results.insights
                },
                "pcs_analysis": pcs_results,
                "integration_status": "success"
            }
            
        except Exception as e:
            logger.warning(f"PCS Meta análise falhou: {e}")
            combined_results = {
                "local_analysis": {
                    "summary_stats": local_results.summary_stats,
                    "insights": local_results.insights
                },
                "pcs_analysis": None,
                "integration_status": "failed",
                "error": str(e)
            }
        
        return combined_results
    
    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """Status abrangente do Helio Service."""
        
        analytics_health = await self.analytics.get_health_metrics()
        pcs_status = await self.pcs_bridge.get_status()
        available_tools = await self.pcs_bridge.get_available_tools()
        
        return {
            "service": "helio_service",
            "initialized": self._initialized,
            "components": {
                "analytics": analytics_health,
                "pcs_bridge": pcs_status
            },
            "capabilities": {
                "advanced_analytics": analytics_health["status"] == "ready",
                "pcs_integration": pcs_status["initialized"],
                "available_pcs_tools": len(available_tools)
            },
            "status": "ready" if self._initialized else "initializing"
        }