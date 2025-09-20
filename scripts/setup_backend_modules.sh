#!/bin/bash
"""
Setup Backend Modules - Configuração dos Módulos Backend
========================================================

Script para configurar PYTHONPATH e integração com pcs-meta-repo.
"""

set -e

echo "🚀 Configurando módulos backend KEC Biomaterials..."

# Diretório base do projeto
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "📁 Projeto: $PROJECT_ROOT"

# Configurações de path
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/external/pcs-meta-repo:$PYTHONPATH"
export KEC_CONFIG_PATH="$PROJECT_ROOT/src/kec_biomat/configs"
export DARWIN_MEMORY_PATH="$PROJECT_ROOT/src/darwin_core/memory"

echo "✅ PYTHONPATH configurado:"
echo "   - $PROJECT_ROOT/src"
echo "   - $PROJECT_ROOT/external/pcs-meta-repo"

# Verifica se submódulo pcs-meta-repo existe
if [ -d "$PROJECT_ROOT/external/pcs-meta-repo" ]; then
    echo "✅ pcs-meta-repo encontrado"
    
    # Atualiza submódulo se necessário
    echo "🔄 Atualizando submódulo pcs-meta-repo..."
    cd "$PROJECT_ROOT"
    git submodule update --init --recursive
else
    echo "⚠️  pcs-meta-repo não encontrado em external/"
    echo "   Execute: git submodule update --init --recursive"
fi

# Cria arquivo de ambiente
ENV_FILE="$PROJECT_ROOT/.env.modules"
cat > "$ENV_FILE" << EOF
# Backend Modules Configuration
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/external/pcs-meta-repo:\$PYTHONPATH"
export KEC_CONFIG_PATH="$PROJECT_ROOT/src/kec_biomat/configs"
export DARWIN_MEMORY_PATH="$PROJECT_ROOT/src/darwin_core/memory"

# Aliases úteis
alias kec-test="cd $PROJECT_ROOT && python -m pytest src/kec_biomat/tests/ -v"
alias kec-metrics="cd $PROJECT_ROOT && python -c 'from kec_biomat.metrics import compute_kec_metrics; import networkx as nx; print(compute_kec_metrics(nx.erdos_renyi_graph(50, 0.1)))'"
EOF

echo "✅ Arquivo de ambiente criado: $ENV_FILE"
echo ""
echo "Para usar os módulos, execute:"
echo "   source $ENV_FILE"
echo ""

# Testa imports básicos
echo "🧪 Testando imports dos módulos..."

cd "$PROJECT_ROOT"

python3 -c "
import sys
sys.path.insert(0, 'src')

try:
    from darwin_core.rag import RAGPlusEngine
    print('✅ darwin_core.rag importado')
except Exception as e:
    print(f'❌ darwin_core.rag: {e}')

try:
    from kec_biomat.metrics import compute_kec_metrics
    print('✅ kec_biomat.metrics importado')
except Exception as e:
    print(f'❌ kec_biomat.metrics: {e}')

try:
    from pcs_helio.analytics import AnalyticsEngine
    print('✅ pcs_helio.analytics importado')
except Exception as e:
    print(f'❌ pcs_helio.analytics: {e}')

try:
    from philosophy.reasoning import LogicEngine
    print('✅ philosophy.reasoning importado')
except Exception as e:
    print(f'❌ philosophy.reasoning: {e}')
"

echo ""
echo "🎉 Setup completo! Backend modular configurado."
echo ""
echo "Próximos passos:"
echo "1. source $ENV_FILE"
echo "2. Testar com: kec-test"
echo "3. Verificar métricas: kec-metrics"