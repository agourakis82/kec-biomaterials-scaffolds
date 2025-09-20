"""
Testes para sistema de configuração KEC
======================================
"""

import tempfile
import os
from pathlib import Path

from ..configs import KECConfig, load_config, save_config, get_default_config


def test_config_loading():
    """Teste carregamento de configuração padrão."""
    config = get_default_config()
    
    # Verifica valores padrão
    assert config.seed == 42
    assert config.segmentation_method == "otsu_local"
    assert config.k_eigs == 64
    assert config.n_random == 20
    assert config.sigma_Q == False


def test_config_from_yaml():
    """Teste carregamento de configuração de YAML."""
    yaml_content = """
seed: 123
entropy:
  k_eigs: 32
  method: "spectral"
coherence:
  sigma_Q: true
  n_random: 10
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name
    
    try:
        config = load_config(temp_path)
        
        assert config.seed == 123
        assert config.k_eigs == 32
        assert config.sigma_Q == True
        assert config.n_random == 10
        
    finally:
        os.unlink(temp_path)


def test_config_save_load():
    """Teste salvar e carregar configuração."""
    
    # Configura valores customizados
    config = KECConfig()
    config.seed = 999
    config.k_eigs = 128
    config.sigma_Q = True
    
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
        temp_path = f.name
    
    try:
        # Salva
        success = save_config(config, temp_path)
        assert success
        
        # Carrega
        loaded_config = load_config(temp_path)
        
        assert loaded_config.seed == 999
        assert loaded_config.k_eigs == 128
        assert loaded_config.sigma_Q == True
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    test_config_loading()
    test_config_from_yaml()
    print("✅ Testes de configuração passaram!")