"""
PCS Meta Repository Bridge
=========================

Ponte de integração com pcs-meta-repo para funcionalidades avançadas.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import importlib.util

logger = logging.getLogger(__name__)


class PCSMetaBridge:
    """
    Ponte entre kec-biomaterials-scaffolds e pcs-meta-repo.
    
    Funcionalidades:
    - Import dinâmico de módulos do pcs-meta-repo
    - Sincronização de sistemas de memória
    - Acesso a analytics avançados
    - Integração com pipeline de dados
    """
    
    def __init__(self, pcs_meta_path: Optional[str] = None):
        self.pcs_meta_path = pcs_meta_path or self._find_pcs_meta_repo()
        self._modules_cache: Dict[str, Any] = {}
        self._initialized = False
        
    def _find_pcs_meta_repo(self) -> Optional[str]:
        """Encontra o caminho do pcs-meta-repo."""
        
        # Verifica submódulo Git
        possible_paths = [
            "external/pcs-meta-repo",
            "../pcs-meta-repo",
            "/app/external/pcs-meta-repo"
        ]
        
        for path in possible_paths:
            full_path = Path(path).resolve()
            if full_path.exists() and (full_path / "pcs").exists():
                logger.info(f"pcs-meta-repo encontrado em: {full_path}")
                return str(full_path)
        
        # Verifica PYTHONPATH
        for path in sys.path:
            pcs_path = Path(path) / "pcs"
            if pcs_path.exists():
                logger.info(f"pcs-meta-repo via PYTHONPATH: {path}")
                return path
        
        logger.warning("pcs-meta-repo não encontrado")
        return None
    
    async def initialize(self) -> bool:
        """Inicializa conexão com pcs-meta-repo."""
        
        if self._initialized:
            return True
            
        if not self.pcs_meta_path:
            logger.warning("pcs-meta-repo não disponível")
            return False
        
        try:
            # Adiciona ao Python path se necessário
            if self.pcs_meta_path not in sys.path:
                sys.path.insert(0, self.pcs_meta_path)
            
            # Testa import básico
            test_import = await self._safe_import("pcs")
            if test_import:
                self._initialized = True
                logger.info("PCS Meta Bridge inicializado com sucesso")
                return True
                
        except Exception as e:
            logger.error(f"Erro ao inicializar PCS Meta Bridge: {e}")
        
        return False
    
    async def _safe_import(self, module_name: str) -> Optional[Any]:
        """Import seguro de módulo do pcs-meta-repo."""
        
        if module_name in self._modules_cache:
            return self._modules_cache[module_name]
        
        try:
            module = importlib.import_module(module_name)
            self._modules_cache[module_name] = module
            return module
        except ImportError as e:
            logger.debug(f"Módulo {module_name} não disponível: {e}")
            return None
    
    async def get_advanced_analytics(self) -> Optional[Any]:
        """Acessa funcionalidades de analytics avançados."""
        
        if not self._initialized:
            await self.initialize()
        
        try:
            # Tenta importar módulos de analytics do pcs-meta-repo
            analytics_modules = [
                "pcs.analytics",
                "pcs.ml_pipeline", 
                "pcs.statistical_analysis"
            ]
            
            for module_name in analytics_modules:
                module = await self._safe_import(module_name)
                if module:
                    logger.info(f"Analytics module carregado: {module_name}")
                    return module
            
        except Exception as e:
            logger.error(f"Erro ao carregar analytics: {e}")
        
        return None
    
    async def sync_memory_systems(self, local_memory: Dict[str, Any]) -> Dict[str, Any]:
        """Sincroniza sistemas de memória entre repositórios."""
        
        if not self._initialized:
            await self.initialize()
        
        try:
            memory_module = await self._safe_import("pcs.memory_system")
            if memory_module and hasattr(memory_module, "sync_memories"):
                return memory_module.sync_memories(local_memory)
            
        except Exception as e:
            logger.error(f"Erro na sincronização de memória: {e}")
        
        # Fallback: retorna memória local
        return local_memory
    
    async def get_ml_pipeline(self) -> Optional[Any]:
        """Acessa pipeline de ML do pcs-meta-repo."""
        
        if not self._initialized:
            await self.initialize()
        
        try:
            ml_module = await self._safe_import("pcs.ml_pipeline")
            return ml_module
            
        except Exception as e:
            logger.error(f"Erro ao acessar ML pipeline: {e}")
        
        return None
    
    async def execute_pcs_analysis(self, 
                                 data: Dict[str, Any], 
                                 analysis_type: str = "default") -> Dict[str, Any]:
        """Executa análise usando ferramentas do pcs-meta-repo."""
        
        if not self._initialized:
            await self.initialize()
        
        try:
            analysis_module = await self._safe_import("pcs.analysis_engine")
            if analysis_module and hasattr(analysis_module, "run_analysis"):
                return analysis_module.run_analysis(data, analysis_type)
            
        except Exception as e:
            logger.error(f"Erro na análise PCS: {e}")
        
        # Fallback: análise básica local
        return {
            "analysis_type": analysis_type,
            "data_summary": {
                "keys": list(data.keys()),
                "size": len(data)
            },
            "status": "fallback_analysis",
            "message": "pcs-meta-repo não disponível, usando análise local"
        }
    
    async def get_available_tools(self) -> List[str]:
        """Lista ferramentas disponíveis no pcs-meta-repo."""
        
        if not self._initialized:
            await self.initialize()
        
        available_tools = []
        
        # Lista de módulos/tools para verificar
        potential_tools = [
            "pcs.analytics",
            "pcs.ml_pipeline",
            "pcs.statistical_analysis", 
            "pcs.memory_system",
            "pcs.data_processing",
            "pcs.visualization",
            "pcs.optimization",
            "pcs.causal_inference"
        ]
        
        for tool in potential_tools:
            module = await self._safe_import(tool)
            if module:
                available_tools.append(tool)
        
        return available_tools
    
    async def get_status(self) -> Dict[str, Any]:
        """Retorna status da ponte PCS Meta."""
        
        available_tools = await self.get_available_tools()
        
        return {
            "bridge": "pcs_meta_bridge",
            "initialized": self._initialized,
            "pcs_meta_path": self.pcs_meta_path,
            "available_tools": available_tools,
            "cache_size": len(self._modules_cache),
            "status": "ready" if self._initialized else "limited"
        }


# Factory function
async def create_pcs_meta_bridge(pcs_meta_path: Optional[str] = None) -> PCSMetaBridge:
    """Cria e inicializa bridge com pcs-meta-repo."""
    
    bridge = PCSMetaBridge(pcs_meta_path)
    await bridge.initialize()
    return bridge