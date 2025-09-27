#!/bin/bash
# Script para carregar modelos no Ollama

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Configuração de Modelos Ollama ===${NC}"

# Aguardar Ollama iniciar
echo -e "${YELLOW}Aguardando Ollama iniciar...${NC}"
until curl --output /dev/null --silent --fail http://localhost:11434/api/tags; do
  printf '.'
  sleep 5
done
echo -e "${GREEN}Ollama está online!${NC}"

# Encontrar o container Ollama
OLLAMA_CONTAINER=$(docker ps | grep ollama | awk '{print $1}')

if [ -z "$OLLAMA_CONTAINER" ]; then
    echo -e "${YELLOW}Container Ollama não encontrado. Verifique se o serviço está rodando.${NC}"
    exit 1
fi

echo -e "${GREEN}Encontrado container Ollama: $OLLAMA_CONTAINER${NC}"

# Lista de modelos para baixar
# Você pode ajustar esta lista conforme suas necessidades
MODELS=(
    "llama3"
    "mistral"
    "codellama"
    "phi3"
)

# Baixar cada modelo
for model in "${MODELS[@]}"; do
    echo -e "\n${YELLOW}Baixando modelo $model...${NC}"
    docker exec -it $OLLAMA_CONTAINER ollama pull $model
    echo -e "${GREEN}Modelo $model baixado com sucesso!${NC}"
done

echo -e "\n${GREEN}=== Todos os modelos foram baixados com sucesso! ===${NC}"
echo -e "Os seguintes modelos estão disponíveis:"

# Listar modelos disponíveis
docker exec -it $OLLAMA_CONTAINER ollama list

echo -e "\n${BLUE}Para usar estes modelos no Darwin, ajuste as configurações no arquivo .env${NC}"