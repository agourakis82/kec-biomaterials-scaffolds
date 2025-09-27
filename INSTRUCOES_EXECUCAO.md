# Instruções para Execução do Darwin KEC

Este documento contém as instruções passo-a-passo para configurar e executar o Darwin KEC em seu ambiente.

## Pré-requisitos

Certifique-se de que os seguintes softwares estão instalados em sua máquina:

- Docker e Docker Compose
- Python 3.11
- Node.js e npm
- Git (para clonar o repositório)

## Passos para Instalação

### 1. Preparação do Ambiente

```bash
# Criar diretório principal para os dados
sudo mkdir -p /mnt/darwin_data

# Criar subdiretórios para diferentes componentes
sudo mkdir -p /mnt/darwin_data/{redis,chroma,ollama,mcp,models,rag,discovery,logs,notebooks,backups}

# Ajustar permissões para seu usuário
sudo chown -R $USER:$USER /mnt/darwin_data
```

### 2. Clonar o Repositório (se ainda não o fez)

```bash
# Navegar para um diretório onde você deseja instalar
cd ~

# Clonar o repositório (substitua a URL pelo repositório real)
git clone https://github.com/repo/darwin.git
cd darwin
```

### 3. Configuração do Ambiente

```bash
# Copiar o arquivo .env de exemplo
cp .env.example .env

# Editar o arquivo .env com suas configurações
nano .env

# Tornar os scripts executáveis
chmod +x setup.sh setup_models.sh backup_darwin.sh
```

### 4. Iniciar o Sistema

```bash
# Executar o script de setup
./setup.sh
```

### 5. Carregar Modelos de IA

Após o sistema estar em execução:

```bash
# Executar o script para carregar modelos no Ollama
./setup_models.sh
```

### 6. Verificar o Funcionamento

Abra seu navegador e acesse:

- Frontend: http://localhost:3000
- Backend health check: http://localhost:8090/health
- VectorDB: http://localhost:8000
- Jupyter: http://localhost:8888

### 7. Comandos Úteis

```bash
# Ver logs de todos os serviços
docker-compose -f docker-compose.local.yml logs -f

# Ver logs de um serviço específico (ex: api)
docker-compose -f docker-compose.local.yml logs -f api

# Reiniciar um serviço (ex: api)
docker-compose -f docker-compose.local.yml restart api

# Parar todos os serviços
docker-compose -f docker-compose.local.yml down

# Fazer backup do sistema
./backup_darwin.sh
```

### 8. Configuração de Domínios (opcional)

Se você já tem domínios configurados com Cloudflare:

1. Configure seu roteador para encaminhar as portas 80 e 443 para seu servidor
2. Use Nginx ou Traefik como proxy reverso para rotear o tráfego para os serviços corretos

### 9. Problemas Comuns e Soluções

- Se um serviço não iniciar, verifique os logs: `docker-compose -f docker-compose.local.yml logs SERVICE_NAME`
- Se Ollama não carregar os modelos, verifique espaço em disco e memória disponível
- Para acesso remoto, configure um proxy reverso e certifique-se que os domínios apontam para o IP do seu servidor

### 10. Próximos Passos

- Explore a interface web para conhecer os diferentes módulos
- Teste a integração com ChatGPT para salvar suas conversas
- Adicione sua própria base de conhecimento ao sistema RAG
- Configure agentes especializados para suas áreas de interesse

## Conclusão

O Darwin KEC agora deve estar configurado e pronto para uso em seu ambiente local. Lembre-se que este é um sistema complexo, então leve um tempo para explorar e entender todas as funcionalidades disponíveis.

Se encontrar problemas durante a configuração ou uso, consulte a documentação ou entre em contato com o suporte.
