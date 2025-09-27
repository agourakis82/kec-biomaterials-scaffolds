# DARWIN Infrastructure Deployment Guide

**Guia Completo de Deployment da Infraestrutura Production-Ready**

---

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Pré-requisitos](#pré-requisitos)
3. [Configuração Inicial](#configuração-inicial)
4. [Deployment da Infraestrutura](#deployment-da-infraestrutura)
5. [Deployment das Aplicações](#deployment-das-aplicações)
6. [Configuração de DNS](#configuração-de-dns)
7. [Verificação e Testes](#verificação-e-testes)
8. [Monitoramento](#monitoramento)
9. [Troubleshooting](#troubleshooting)
10. [Manutenção](#manutenção)

---

## 🎯 Visão Geral

A plataforma DARWIN é uma infraestrutura production-ready no Google Cloud Platform que inclui:

### 🏗️ Componentes da Infraestrutura
- **Backend JAX-powered:** API FastAPI com suporte a GPU/TPU
- **Frontend React TypeScript:** Next.js com PWA support
- **Database:** PostgreSQL com extensões vector (pgvector)
- **Cache:** Redis Memorystore para performance
- **Storage:** Cloud Storage com CDN global
- **Monitoring:** Dashboards e alerting completos
- **Security:** Criptografia, WAF, IAM com least privilege

### 🌐 URLs de Produção
- **Frontend:** https://darwin.agourakis.med.br
- **API:** https://api.agourakis.med.br
- **Documentação API:** https://api.agourakis.med.br/docs

---

## ✅ Pré-requisitos

### 🔧 Ferramentas Necessárias

```bash
# Google Cloud SDK
curl https://sdk.cloud.google.com | bash
gcloud init

# Terraform
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
unzip terraform_1.6.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/

# Outras ferramentas
sudo apt-get install -y jq curl git
```

### 🔑 Permissões Necessárias

**Usuário deve ter as seguintes permissões no GCP:**
- `Project Owner` ou `Project Editor`
- `Billing Account User`
- `Cloud Build Editor`
- `Service Account Admin`
- `Security Admin` (para políticas de segurança)

### 💳 Configuração de Billing

1. Acesse [Google Cloud Console](https://console.cloud.google.com)
2. Configure billing account ativo
3. Anote o **Billing Account ID** (formato: `123456-789012-345678`)

---

## ⚙️ Configuração Inicial

### 1. Autenticação

```bash
# Login no gcloud
gcloud auth login

# Configurar projeto
gcloud config set project YOUR_PROJECT_ID

# Autenticação para aplicações
gcloud auth application-default login
```

### 2. Preparar Scripts

```bash
# Tornar scripts executáveis
chmod +x scripts/*.sh

# Ou usar o script de setup
./scripts/setup_scripts.sh
```

### 3. Configurar Variáveis de Ambiente

```bash
# Exportar variáveis principais
export DARWIN_PROJECT_ID="your-project-id"
export DARWIN_BILLING_ACCOUNT_ID="123456-789012-345678"
export DARWIN_ENVIRONMENT="production"
export DARWIN_REGION="us-central1"

# Opcional: Configurações de notificação
export DARWIN_EMAIL_ADDRESSES="admin@company.com,ops@company.com"
export DARWIN_SLACK_WEBHOOK="https://hooks.slack.com/services/..."
export DARWIN_BUDGET_AMOUNT="500"
```

---

## 🚀 Deployment da Infraestrutura

### Passo 1: Deploy Completo da Infraestrutura

```bash
# Deployment automático completo
./scripts/deploy_infrastructure.sh \
    --project-id YOUR_PROJECT_ID \
    --billing-account YOUR_BILLING_ACCOUNT_ID \
    --environment production \
    --region us-central1

# Ou usando variáveis de ambiente
./scripts/deploy_infrastructure.sh
```

### Passo 2: Verificar Deployment

```bash
# Verificar status da infraestrutura
gcloud run services list --platform=managed
gcloud sql instances list
gcloud redis instances list --region=us-central1
gcloud compute networks list
```

### ⏱️ Tempo Esperado
- **Infraestrutura:** 15-20 minutos
- **SSL Certificates:** 10-60 minutos (automático)

---

## 📱 Deployment das Aplicações

### Passo 1: Deploy do Backend

```bash
# Deploy apenas do backend
./scripts/deploy_applications.sh \
    --project-id YOUR_PROJECT_ID \
    --backend \
    --environment production

# Verificar health do backend
curl https://api.agourakis.med.br/health
```

### Passo 2: Deploy do Frontend

```bash
# Deploy apenas do frontend
./scripts/deploy_applications.sh \
    --project-id YOUR_PROJECT_ID \
    --frontend \
    --environment production

# Ou deploy de ambos em paralelo
./scripts/deploy_applications.sh \
    --project-id YOUR_PROJECT_ID \
    --both \
    --parallel
```

### ⏱️ Tempo Esperado
- **Backend:** 10-15 minutos
- **Frontend:** 8-12 minutos
- **Paralelo:** 12-15 minutos

---

## 🌐 Configuração de DNS

### Obter IP do Load Balancer

```bash
# Via Terraform outputs
cd infrastructure/terraform
terraform output load_balancer_ip

# Ou via gcloud
gcloud compute addresses list --global
```

### Configurar Registros DNS

**No seu provedor de DNS (ex: Cloudflare, Route53, etc.):**

```dns
# Registros A necessários
api.agourakis.med.br.     300   IN   A   LOAD_BALANCER_IP
darwin.agourakis.med.br.  300   IN   A   LOAD_BALANCER_IP

# Opcional: CNAME para www
www.darwin.agourakis.med.br.  300  IN  CNAME  darwin.agourakis.med.br.
```

### Verificar Propagação DNS

```bash
# Verificar propagação
nslookup api.agourakis.med.br
nslookup darwin.agourakis.med.br

# Teste com dig
dig +short api.agourakis.med.br
dig +short darwin.agourakis.med.br
```

---

## 🧪 Verificação e Testes

### Verificação Básica

```bash
# Verificar serviços
./scripts/setup_monitoring.sh --project-id YOUR_PROJECT_ID --verify

# Status das aplicações
curl -s https://api.agourakis.med.br/health | jq
curl -s https://darwin.agourakis.med.br/api/health
```

### Testes Abrangentes

```bash
# Executar suite completa de testes
./scripts/run_tests.sh \
    --project-id YOUR_PROJECT_ID \
    --environment production \
    --all

# Ou testes específicos
./scripts/run_tests.sh \
    --project-id YOUR_PROJECT_ID \
    --integration-tests \
    --load-tests
```

### Verificação de SSL

```bash
# Verificar certificados SSL
gcloud compute ssl-certificates list

# Testar SSL manualmente
openssl s_client -connect api.agourakis.med.br:443 -servername api.agourakis.med.br
openssl s_client -connect darwin.agourakis.med.br:443 -servername darwin.agourakis.med.br
```

---

## 📊 Monitoramento

### Dashboard Principal

Acesse: https://console.cloud.google.com/monitoring/dashboards?project=YOUR_PROJECT_ID

### Configurar Alertas

```bash
# Configurar monitoramento completo
./scripts/setup_monitoring.sh \
    --project-id YOUR_PROJECT_ID \
    --email "admin@company.com,ops@company.com" \
    --slack-webhook "https://hooks.slack.com/services/..." \
    --budget 500
```

### URLs de Monitoramento

- **Cloud Monitoring:** https://console.cloud.google.com/monitoring
- **Cloud Logging:** https://console.cloud.google.com/logs
- **Error Reporting:** https://console.cloud.google.com/errors
- **Cloud Trace:** https://console.cloud.google.com/traces
- **Uptime Checks:** https://console.cloud.google.com/monitoring/uptime

---

## 🗄️ Configuração do Banco de Dados

### Setup Automatizado

```bash
# Configurar PostgreSQL + pgvector
./scripts/setup_database.sh \
    --project-id YOUR_PROJECT_ID \
    --setup-pgvector \
    --create-read-replica \
    --run-migrations
```

### Verificação Manual

```bash
# Conectar ao banco via Cloud SQL Proxy
gcloud sql connect darwin-production-db --user=postgres

# Verificar extensões
\dx

# Verificar tabelas
\dt

# Testar vector search
SELECT vector('[1,2,3]') <-> vector('[4,5,6]');
```

---

## 🔒 Configuração de Segurança

### Setup Automatizado

```bash
# Configurar segurança completa
./scripts/setup_security.sh \
    --project-id YOUR_PROJECT_ID \
    --enable-security-center \
    --verify
```

### Verificações de Segurança

- **Service Accounts:** Least privilege configurado
- **KMS:** Chaves de criptografia rotacionadas automaticamente
- **Secrets:** Senhas seguras no Secret Manager
- **Firewall:** Regras restritivas aplicadas
- **SSL/TLS:** Certificados gerenciados e HSTS ativo
- **Cloud Armor:** WAF e proteção DDoS configurada

---

## 🚨 Troubleshooting

### Problemas Comuns

#### ❌ Erro: "SSL certificate not ready"
```bash
# Verificar status dos certificados
gcloud compute ssl-certificates list

# Aguardar provisioning (pode levar 60 minutos)
# Verificar se DNS está configurado corretamente
```

#### ❌ Erro: "Service not accessible"
```bash
# Verificar status do serviço
gcloud run services describe darwin-production-backend --region=us-central1

# Verificar logs
gcloud logs read "resource.type=cloud_run_revision" --limit=50
```

#### ❌ Erro: "Database connection failed"
```bash
# Verificar status do banco
gcloud sql instances describe darwin-production-db

# Verificar conectividade via VPC
gcloud compute networks vpc-access connectors list --region=us-central1
```

#### ❌ Erro: "Permission denied"
```bash
# Verificar permissões do service account
gcloud projects get-iam-policy YOUR_PROJECT_ID

# Verificar billing
gcloud billing accounts list
gcloud billing projects describe YOUR_PROJECT_ID
```

### Logs Importantes

```bash
# Logs da aplicação
gcloud logs read "resource.type=cloud_run_revision" --limit=100

# Logs de build
gcloud logs read "resource.type=build" --limit=50

# Logs de rede
gcloud logs read "resource.type=http_load_balancer" --limit=50
```

---

## 🔄 Manutenção

### Backups Automatizados

- **Database:** Backup diário às 03:00 UTC
- **Storage:** Versionamento ativo
- **Terraform State:** Backup automático
- **Configurações:** Versionadas no Git

### Atualizações Regulares

```bash
# Atualizar dependências (mensal)
./scripts/update_dependencies.sh

# Rotacionar secrets (trimestral)  
./scripts/rotate_secrets.sh

# Verificar segurança (semanal)
./scripts/setup_security.sh --verify
```

### Monitoramento de Custos

```bash
# Verificar custos atuais
gcloud billing budgets list --billing-account=YOUR_BILLING_ACCOUNT_ID

# Relatório de custos
gcloud billing accounts get-iam-policy YOUR_BILLING_ACCOUNT_ID
```

---

## 📈 Scaling e Performance

### Auto-scaling Configurado

- **Backend:** 2-20 instâncias baseado em CPU/memória
- **Frontend:** 1-10 instâncias baseado em tráfego
- **Database:** Read replicas automáticas
- **Redis:** Memory scaling baseado em uso

### Otimização de Performance

```bash
# Verificar métricas de performance
./scripts/run_tests.sh --load-tests --verbose

# Análise de performance do banco
./scripts/setup_database.sh --verify
```

---

## 🔗 URLs e Recursos Importantes

### 🌐 Produção
- **Frontend:** https://darwin.agourakis.med.br
- **API:** https://api.agourakis.med.br
- **API Docs:** https://api.agourakis.med.br/docs
- **Health Check:** https://api.agourakis.med.br/health

### 📊 Monitoramento  
- **Dashboard:** https://console.cloud.google.com/monitoring
- **Logs:** https://console.cloud.google.com/logs
- **Alertas:** https://console.cloud.google.com/monitoring/alerting
- **Uptime:** https://console.cloud.google.com/monitoring/uptime

### 🏗️ Infraestrutura
- **Cloud Run:** https://console.cloud.google.com/run
- **Cloud SQL:** https://console.cloud.google.com/sql
- **VPC Networks:** https://console.cloud.google.com/networking
- **Load Balancing:** https://console.cloud.google.com/net-services/loadbalancing

### 🔒 Segurança
- **IAM:** https://console.cloud.google.com/iam-admin
- **Secret Manager:** https://console.cloud.google.com/security/secret-manager
- **KMS:** https://console.cloud.google.com/security/kms
- **Security Center:** https://console.cloud.google.com/security

---

## 🚀 Quick Start

### Deployment Completo em 3 Comandos

```bash
# 1. Deploy da infraestrutura
./scripts/deploy_infrastructure.sh -p YOUR_PROJECT_ID -b YOUR_BILLING_ACCOUNT

# 2. Deploy das aplicações
./scripts/deploy_applications.sh -p YOUR_PROJECT_ID --both --parallel

# 3. Verificar deployment
./scripts/setup_monitoring.sh -p YOUR_PROJECT_ID --verify
```

### Configuração DNS (Manual)

```bash
# Obter IP do load balancer
terraform output load_balancer_ip

# Configurar no seu provedor DNS:
# api.agourakis.med.br -> LOAD_BALANCER_IP
# darwin.agourakis.med.br -> LOAD_BALANCER_IP
```

---

## 💰 Custos Estimados

### 📊 Breakdown Mensal (USD)

| Componente | Custo Estimado |
|------------|----------------|
| Cloud Run (Backend) | $50-150 |
| Cloud Run (Frontend) | $20-80 |
| Cloud SQL PostgreSQL | $80-200 |
| Redis Memorystore | $40-120 |
| Cloud Storage + CDN | $20-60 |
| Load Balancer | $20-40 |
| Monitoring + Logging | $10-30 |
| **Total Estimado** | **$240-680** |

### 💡 Otimização de Custos

- **Ambiente Dev:** ~$100/mês (recursos mínimos)
- **Ambiente Staging:** ~$200/mês (recursos reduzidos)
- **Ambiente Produção:** ~$500/mês (alta disponibilidade)

---

## 🔄 CI/CD Pipeline

### Deployment Automático

```bash
# Via Cloud Build
gcloud builds submit --config=infrastructure/cloudbuild/infrastructure-deploy.yaml
gcloud builds submit --config=infrastructure/cloudbuild/backend-deploy.yaml
gcloud builds submit --config=infrastructure/cloudbuild/frontend-deploy.yaml
```

### Pipeline Completo

1. **Infrastructure Deploy:** Terraform modules
2. **Backend Deploy:** JAX-powered API
3. **Frontend Deploy:** React TypeScript
4. **Integration Tests:** Comprehensive testing
5. **Security Scans:** Vulnerability assessment
6. **Performance Tests:** Load testing

---

## 📚 Documentação Adicional

- **[ARCHITECTURE.md](ARCHITECTURE.md):** Arquitetura detalhada
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md):** Resolução de problemas
- **[API_REFERENCE.md](API_REFERENCE.md):** Documentação da API
- **[SECURITY.md](docs/SECURITY.md):** Políticas de segurança

---

## 🆘 Suporte

### 📞 Contatos

- **Admin:** admin@agourakis.med.br
- **DevOps:** devops@agourakis.med.br
- **Security:** security@agourakis.med.br

### 🔗 Links Úteis

- **Repositório:** https://github.com/your-org/darwin-platform
- **Documentação:** https://docs.darwin.agourakis.med.br
- **Status Page:** https://status.darwin.agourakis.med.br

---

## ⚠️ Notas Importantes

### 🔴 Antes do Deployment

1. ✅ Verificar billing account ativo
2. ✅ Confirmar permissões necessárias
3. ✅ Fazer backup dos dados existentes
4. ✅ Testar em ambiente de staging primeiro

### 🟡 Durante o Deployment

1. ⏳ SSL certificates podem demorar até 60 minutos
2. ⏳ DNS propagation pode levar até 48 horas
3. ⏳ Primeiro build pode ser mais lento (cache vazio)

### 🟢 Após o Deployment

1. ✅ Configurar monitoramento e alertas
2. ✅ Testar todos os endpoints
3. ✅ Verificar backups automatizados
4. ✅ Documentar procedimentos específicos

---

## 📝 Checklist de Deployment

### Pré-deployment
- [ ] GCP Project criado e billing ativo
- [ ] Usuário com permissões adequadas
- [ ] Ferramentas instaladas (gcloud, terraform, etc.)
- [ ] DNS provider configurado
- [ ] Backup de dados existentes (se aplicável)

### Deployment
- [ ] Infraestrutura deployada com sucesso
- [ ] Backend deployado e health check OK
- [ ] Frontend deployado e acessível
- [ ] DNS configurado corretamente
- [ ] SSL certificates provisionados

### Pós-deployment
- [ ] Todos os testes passando
- [ ] Monitoramento configurado
- [ ] Alertas funcionando
- [ ] Backups configurados
- [ ] Documentação atualizada
- [ ] Team treinado nos procedimentos

---

## 🔄 Rollback Procedures

### Rollback Rápido

```bash
# Rollback para versão anterior
gcloud run services update-traffic darwin-production-backend --to-revisions=PREVIOUS_REVISION=100

# Rollback da infraestrutura
cd infrastructure/terraform
terraform plan -destroy
# (apenas se necessário)
```

### Backup e Recovery

```bash
# Restaurar backup do banco
gcloud sql backups restore BACKUP_ID --restore-instance=darwin-production-db

# Verificar integridade
./scripts/setup_database.sh --verify
```

---

**🎉 Parabéns! A infraestrutura DARWIN está production-ready!**

Para questões ou problemas, consulte o [TROUBLESHOOTING.md](TROUBLESHOOTING.md) ou entre em contato com a equipe de suporte.