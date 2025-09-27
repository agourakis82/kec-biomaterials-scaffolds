#!/bin/bash

set -e

# Diretório base
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

# Instalar dependências backend se necessário
BACKEND_DIR="backend/kec_unified_api"
if [ -d "$BACKEND_DIR" ]; then
    cd "$BACKEND_DIR"
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    source venv/bin/activate
    if [ ! -f "requirements.txt" ] || [ ! -d "venv/lib" ]; then
        pip install --upgrade pip
        pip install -r requirements.txt
    fi
    deactivate
    cd - >/dev/null
else
    echo "Diretório backend não encontrado: $BACKEND_DIR"
    exit 1
fi

# Instalar dependências frontend se necessário
FRONTEND_DIR="frontend/ui"
if [ -d "$FRONTEND_DIR" ]; then
    cd "$FRONTEND_DIR"
    if [ ! -d "node_modules" ]; then
        npm install
    fi
    cd - >/dev/null
else
    echo "Diretório frontend não encontrado: $FRONTEND_DIR"
    exit 1
fi

# Configurar .env
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo ".env criado a partir de .env.example. Edite com suas chaves API."
    else
        echo "Arquivo .env.example não encontrado. Crie um com as variáveis necessárias."
        exit 1
    fi
fi

# Iniciar serviços com Docker Compose
if [ -f "docker-compose.yml" ]; then
    docker compose up -d
    echo "Serviços iniciados com Docker Compose."
    echo "Verifique health check: curl http://localhost:8090/healthz"
else
    echo "docker-compose.yml não encontrado. Crie-o primeiro."
    exit 1
fi

echo "Setup concluído!"
