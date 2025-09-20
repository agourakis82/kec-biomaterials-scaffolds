"""
KEC Biomat Configuration System
===============================

Sistema de configuração para métricas KEC 2.0 e processamento de biomateriais.
"""

import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class KECConfig:
    """Configuração KEC 2.0 com valores padrão."""
    
    # Geral
    seed: int = 42
    
    # Segmentação
    segmentation_method: str = "otsu_local"
    bilateral_filter: bool = True
    bilateral_sigma_color: int = 75
    bilateral_sigma_space: int = 75
    cnn_proxy: bool = False
    
    # Extração de grafo
    use_porespy: bool = True
    throat_weights: list = None
    
    # Entropia
    entropy_method: str = "spectral"
    k_eigs: int = 64
    entropy_tol: float = 1e-8
    stochastic_trace: bool = False
    
    # Curvatura
    forman_enabled: bool = True
    include_triangles: bool = True
    ollivier_enabled: bool = True
    ollivier_method: str = "sinkhorn"
    sample_frac: float = 0.2
    sinkhorn_eps: float = 0.01
    sinkhorn_n_iter: int = 100
    
    # Coerência
    sigma_small_world: bool = True
    swp: bool = True
    n_random: int = 20
    sigma_Q: bool = False
    
    # Modelagem
    baseline_model: str = "random_forest"
    gnn_enabled: bool = True
    embedding_mode: bool = True
    gnn_layers: int = 3
    hidden_dim: int = 64
    dropout: float = 0.3
    
    # Reporting
    null_models: bool = True
    preserve_degree_nulls: int = 100
    ci_bootstrap: int = 1000
    
    def __post_init__(self):
        if self.throat_weights is None:
            self.throat_weights = ["diameter", "length", "conductance_proxy"]


def load_config(config_path: Optional[str] = None) -> KECConfig:
    """
    Carrega configuração de arquivo YAML ou usa padrão.
    
    Args:
        config_path: Caminho para arquivo YAML. Se None, usa configuração padrão.
        
    Returns:
        Instância KECConfig
    """
    
    # Configuração padrão
    config_data = {}
    
    # Tenta carregar de arquivo se especificado
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
                if yaml_data:
                    config_data = _flatten_yaml_config(yaml_data)
                    logger.info(f"Configuração carregada de {config_path}")
        except Exception as e:
            logger.warning(f"Erro ao carregar configuração {config_path}: {e}")
            logger.info("Usando configuração padrão")
    
    # Tenta carregar de variável de ambiente
    env_config_path = os.getenv('KEC_CONFIG_PATH')
    if not config_path and env_config_path and os.path.exists(env_config_path):
        return load_config(env_config_path)
    
    # Cria configuração com overrides do YAML
    return KECConfig(**config_data)


def _flatten_yaml_config(yaml_data: Dict[str, Any]) -> Dict[str, Any]:
    """Converte estrutura YAML aninhada para argumentos KECConfig."""
    
    flat_config = {}
    
    # Mapeamento direto
    if 'seed' in yaml_data:
        flat_config['seed'] = yaml_data['seed']
    
    # Segmentação
    if 'segmentation' in yaml_data:
        seg = yaml_data['segmentation']
        flat_config['segmentation_method'] = seg.get('method', 'otsu_local')
        flat_config['bilateral_filter'] = seg.get('bilateral_filter', True)
        flat_config['bilateral_sigma_color'] = seg.get('bilateral_sigma_color', 75)
        flat_config['bilateral_sigma_space'] = seg.get('bilateral_sigma_space', 75)
        flat_config['cnn_proxy'] = seg.get('cnn_proxy', False)
    
    # Extração de grafo
    if 'graph_extraction' in yaml_data:
        graph = yaml_data['graph_extraction']
        flat_config['use_porespy'] = graph.get('use_porespy', True)
        flat_config['throat_weights'] = graph.get('throat_weights', 
                                                 ["diameter", "length", "conductance_proxy"])
    
    # Entropia
    if 'entropy' in yaml_data:
        entropy = yaml_data['entropy']
        flat_config['entropy_method'] = entropy.get('method', 'spectral')
        flat_config['k_eigs'] = entropy.get('k_eigs', 64)
        flat_config['entropy_tol'] = entropy.get('tol', 1e-8)
        flat_config['stochastic_trace'] = entropy.get('stochastic_trace', False)
    
    # Curvatura
    if 'curvature' in yaml_data:
        curv = yaml_data['curvature']
        
        if 'forman' in curv:
            forman = curv['forman']
            flat_config['forman_enabled'] = forman.get('enabled', True)
            flat_config['include_triangles'] = forman.get('include_triangles', True)
        
        if 'ollivier' in curv:
            ollivier = curv['ollivier']
            flat_config['ollivier_enabled'] = ollivier.get('enabled', True)
            flat_config['ollivier_method'] = ollivier.get('method', 'sinkhorn')
            flat_config['sample_frac'] = ollivier.get('sample_frac', 0.2)
            flat_config['sinkhorn_eps'] = ollivier.get('sinkhorn_eps', 0.01)
            flat_config['sinkhorn_n_iter'] = ollivier.get('n_iter', 100)
    
    # Coerência
    if 'coherence' in yaml_data:
        coh = yaml_data['coherence']
        flat_config['sigma_small_world'] = coh.get('sigma_small_world', True)
        flat_config['swp'] = coh.get('swp', True)
        flat_config['n_random'] = coh.get('n_random', 20)
        flat_config['sigma_Q'] = coh.get('sigma_Q', False)
    
    # Modelagem
    if 'modeling' in yaml_data:
        model = yaml_data['modeling']
        flat_config['baseline_model'] = model.get('baseline', 'random_forest')
        
        if 'gnn' in model:
            gnn = model['gnn']
            flat_config['gnn_enabled'] = gnn.get('enabled', True)
            flat_config['embedding_mode'] = gnn.get('embedding_mode', True)
            flat_config['gnn_layers'] = gnn.get('layers', 3)
            flat_config['hidden_dim'] = gnn.get('hidden_dim', 64)
            flat_config['dropout'] = gnn.get('dropout', 0.3)
    
    # Reporting
    if 'reporting' in yaml_data:
        rep = yaml_data['reporting']
        flat_config['null_models'] = rep.get('null_models', True)
        flat_config['preserve_degree_nulls'] = rep.get('preserve_degree_nulls', 100)
        flat_config['ci_bootstrap'] = rep.get('ci_bootstrap', 1000)
    
    return flat_config


def get_default_config() -> KECConfig:
    """Retorna configuração padrão."""
    return KECConfig()


def save_config(config: KECConfig, output_path: str) -> bool:
    """
    Salva configuração em arquivo YAML.
    
    Args:
        config: Configuração a salvar
        output_path: Caminho de saída
        
    Returns:
        True se salvou com sucesso
    """
    
    try:
        # Converte para estrutura YAML aninhada
        yaml_data = {
            'seed': config.seed,
            'segmentation': {
                'method': config.segmentation_method,
                'bilateral_filter': config.bilateral_filter,
                'bilateral_sigma_color': config.bilateral_sigma_color,
                'bilateral_sigma_space': config.bilateral_sigma_space,
                'cnn_proxy': config.cnn_proxy
            },
            'graph_extraction': {
                'use_porespy': config.use_porespy,
                'throat_weights': config.throat_weights
            },
            'entropy': {
                'method': config.entropy_method,
                'k_eigs': config.k_eigs,
                'tol': config.entropy_tol,
                'stochastic_trace': config.stochastic_trace
            },
            'curvature': {
                'forman': {
                    'enabled': config.forman_enabled,
                    'include_triangles': config.include_triangles
                },
                'ollivier': {
                    'enabled': config.ollivier_enabled,
                    'method': config.ollivier_method,
                    'sample_frac': config.sample_frac,
                    'sinkhorn_eps': config.sinkhorn_eps,
                    'n_iter': config.sinkhorn_n_iter
                }
            },
            'coherence': {
                'sigma_small_world': config.sigma_small_world,
                'swp': config.swp,
                'n_random': config.n_random,
                'sigma_Q': config.sigma_Q
            },
            'modeling': {
                'baseline': config.baseline_model,
                'gnn': {
                    'enabled': config.gnn_enabled,
                    'embedding_mode': config.embedding_mode,
                    'layers': config.gnn_layers,
                    'hidden_dim': config.hidden_dim,
                    'dropout': config.dropout
                }
            },
            'reporting': {
                'null_models': config.null_models,
                'preserve_degree_nulls': config.preserve_degree_nulls,
                'ci_bootstrap': config.ci_bootstrap
            }
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, indent=2, 
                     allow_unicode=True)
        
        logger.info(f"Configuração salva em {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Erro ao salvar configuração: {e}")
        return False