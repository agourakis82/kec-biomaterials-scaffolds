#!/bin/bash
# Script de backup para o Darwin KEC

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configurações
BACKUP_DIR="/mnt/darwin_data/backups"
DATE_FORMAT=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="$BACKUP_DIR/darwin_backup_$DATE_FORMAT.tar.gz"

echo -e "${BLUE}=== Darwin KEC - Backup ===${NC}"

# Criar diretório de backup se não existir
mkdir -p $BACKUP_DIR

# Fazer backup dos dados essenciais
echo -e "${YELLOW}Iniciando backup do Darwin KEC...${NC}"

# 1. Redis (fazer dump do Redis primeiro)
echo -e "${YELLOW}Criando snapshot do Redis...${NC}"
docker exec $(docker ps | grep redis | awk '{print $1}') redis-cli SAVE

# 2. Compactar dados
echo -e "${YELLOW}Compactando dados...${NC}"
tar -czf $BACKUP_FILE \
    /mnt/darwin_data/redis \
    /mnt/darwin_data/chroma \
    /mnt/darwin_data/mcp \
    /mnt/darwin_data/rag \
    .env \
    docker-compose.local.yml

# 3. Limpeza de backups antigos (manter últimos 7)
echo -e "${YELLOW}Limpando backups antigos...${NC}"
ls -t $BACKUP_DIR/darwin_backup_*.tar.gz | tail -n +8 | xargs -r rm

echo -e "${GREEN}Backup concluído: $BACKUP_FILE${NC}"
echo -e "${GREEN}Tamanho do backup: $(du -h $BACKUP_FILE | cut -f1)${NC}"