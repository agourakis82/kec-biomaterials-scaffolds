#!/bin/bash
# Script para configurar e iniciar o Darwin KEC localmente

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== DARWIN KEC - Configuração Local ===${NC}"

# Verificar se o .env existe
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Arquivo .env não encontrado. Criando a partir do exemplo...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}⚠️ Por favor, edite o arquivo .env antes de continuar!${NC}"
    exit 1
fi

# Verificar diretório de dados
DATA_DIR="/mnt/darwin_data"
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${YELLOW}Criando diretório de dados em $DATA_DIR...${NC}"
    sudo mkdir -p $DATA_DIR
    sudo chown -R $USER:$USER $DATA_DIR
    mkdir -p $DATA_DIR/{redis,chroma,ollama,mcp,models,rag,discovery,logs,notebooks}
fi

# Iniciar os serviços
echo -e "${BLUE}Iniciando serviços com Docker Compose...${NC}"
docker-compose -f docker-compose.local.yml down
docker-compose -f docker-compose.local.yml up -d

# Aguardar inicialização
echo -e "${BLUE}Aguardando inicialização dos serviços...${NC}"
sleep 15

# Verificar status
echo -e "${BLUE}Verificando status dos serviços...${NC}"
BACKEND_PORT="${HOST_BACKEND_PORT:-8090}"
FRONTEND_PORT="${HOST_FRONTEND_PORT:-3000}"

echo -e "${YELLOW}Testando backend: http://localhost:$BACKEND_PORT/health${NC}"
if curl -s "http://localhost:$BACKEND_PORT/health" > /dev/null; then
    echo -e "${GREEN}✓ Backend está funcionando!${NC}"
else
    echo -e "${YELLOW}⚠️ Backend pode não estar pronto ainda. Verifique os logs com 'docker-compose -f docker-compose.local.yml logs api'${NC}"
fi

echo -e "${YELLOW}Testando frontend: http://localhost:$FRONTEND_PORT/api/health${NC}"
if curl -s "http://localhost:$FRONTEND_PORT/api/health" > /dev/null; then
    echo -e "${GREEN}✓ Frontend está funcionando!${NC}"
else
    echo -e "${YELLOW}⚠️ Frontend pode não estar pronto ainda. Verifique os logs com 'docker-compose -f docker-compose.local.yml logs web'${NC}"
fi

# Instruções para modelos Ollama
echo -e "\n${BLUE}=== Configuração de Modelos de IA ===${NC}"
echo -e "${YELLOW}Para carregar modelos no Ollama, execute:${NC}"
echo -e "docker exec -it \$(docker ps | grep ollama | awk '{print \$1}') ollama pull llama3"
echo -e "docker exec -it \$(docker ps | grep ollama | awk '{print \$1}') ollama pull mistral"

echo -e "\n${GREEN}=== Configuração concluída! ===${NC}"
echo -e "Backend: http://localhost:$BACKEND_PORT"
echo -e "Frontend: http://localhost:$FRONTEND_PORT"
echo -e "VectorDB: http://localhost:${CHROMA_PORT:-8000}"
echo -e "Ollama: http://localhost:${OLLAMA_PORT:-11434}"
echo -e "Jupyter: http://localhost:${JUPYTER_PORT:-8888}"
echo -e "\nPara ver os logs: docker-compose -f docker-compose.local.yml logs -f"
echo -e "Para parar os serviços: docker-compose -f docker-compose.local.yml down"
