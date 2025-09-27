# DARWIN Troubleshooting Guide

**Guia Completo de Resolução de Problemas da Infraestrutura DARWIN**

---

## 📋 Índice

1. [Problemas de Deployment](#problemas-de-deployment)
2. [Problemas de Conectividade](#problemas-de-conectividade)
3. [Problemas de Performance](#problemas-de-performance)
4. [Problemas de Banco de Dados](#problemas-de-banco-de-dados)
5. [Problemas de SSL/DNS](#problemas-de-ssldns)
6. [Problemas de Monitoramento](#problemas-de-monitoramento)
7. [Problemas de Segurança](#problemas-de-segurança)
8. [Logs e Debugging](#logs-e-debugging)
9. [Procedimentos de Emergência](#procedimentos-de-emergência)

---

## 🚀 Problemas de Deployment

### ❌ Erro: "terraform apply failed"

**Sintomas:**
```
Error: Error creating instance: googleapi: Error 403: 
Access Denied, accessDenied
```

**Diagnóstico:**
```bash
# Verificar permissões
gcloud auth list
gcloud config get-value project

# Verificar billing
gcloud billing accounts list
gcloud billing projects describe PROJECT_ID
```

**Solução:**
```bash
# 1. Verificar autenticação
gcloud auth login
gcloud auth application-default login

# 2. Configurar projeto
gcloud config set project YOUR_PROJECT_ID

# 3. Verificar billing
gcloud billing projects link YOUR_PROJECT_ID --billing-account=BILLING_ACCOUNT_ID

# 4. Habilitar APIs necessárias
./scripts/deploy_infrastructure.sh --project-id YOUR_PROJECT_ID --billing-account BILLING_ACCOUNT_ID
```

---

### ❌ Erro: "Cloud Build permission denied"

**Sintomas:**
```
ERROR: gcloud crashed (HttpError): 
HttpError accessing <https://cloudbuild.googleapis.com/v1/projects/PROJECT_ID/builds>: 
response: <{'status': '403'}>
```

**Solução:**
```bash
# 1. Habilitar Cloud Build API
gcloud services enable cloudbuild.googleapis.com

# 2. Configurar service account do Cloud Build
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:PROJECT_NUMBER@cloudbuild.gserviceaccount.com" \
  --role="roles/run.admin"

# 3. Verificar IAM bindings
gcloud projects get-iam-policy PROJECT_ID
```

---

### ❌ Erro: "Container image not found"

**Sintomas:**
```
Error: The user-provided container image gcr.io/PROJECT_ID/darwin-backend:latest does not exist
```

**Solução:**
```bash
# 1. Verificar imagens disponíveis
gcloud container images list --repository=gcr.io/PROJECT_ID

# 2. Build da imagem se necessário
gcloud builds submit --config=infrastructure/cloudbuild/backend-deploy.yaml .

# 3. Verificar registry permissions
gcloud container images list-tags gcr.io/PROJECT_ID/darwin-backend
```

---

## 🌐 Problemas de Conectividade

### ❌ Erro: "Service Unavailable (503)"

**Sintomas:**
- API retorna 503
- Frontend não carrega
- Timeout em requests

**Diagnóstico:**
```bash
# Verificar status dos serviços
gcloud run services list --platform=managed

# Verificar logs em tempo real
gcloud logs tail "resource.type=cloud_run_revision" --follow
```

**Solução:**
```bash
# 1. Verificar health dos serviços
curl -v https://api.agourakis.med.br/health
curl -v https://darwin.agourakis.med.br/api/health

# 2. Verificar instâncias ativas
gcloud run services describe darwin-production-backend --region=us-central1

# 3. Forçar redeploy se necessário
gcloud run services update darwin-production-backend \
  --region=us-central1 \
  --min-instances=1
```

---

### ❌ Erro: "Database connection refused"

**Sintomas:**
```
psycopg2.OperationalError: connection to server at "10.x.x.x", port 5432 failed
```

**Diagnóstico:**
```bash
# Verificar status do banco
gcloud sql instances describe darwin-production-db

# Verificar VPC connector
gcloud compute networks vpc-access connectors describe \
  darwin-production-connector --region=us-central1
```

**Solução:**
```bash
# 1. Verificar conectividade VPC
gcloud compute networks vpc-access connectors list --region=us-central1

# 2. Testar conexão via proxy
gcloud sql connect darwin-production-db --user=postgres

# 3. Verificar firewall rules
gcloud compute firewall-rules list --filter="name~darwin"
```

---

## ⚡ Problemas de Performance

### ❌ Problema: "High latency responses"

**Sintomas:**
- Responses > 5 segundos
- Timeouts frequentes
- High CPU/Memory usage

**Diagnóstico:**
```bash
# Verificar métricas de performance
gcloud monitoring metrics list --filter="metric.type:run.googleapis.com"

# Verificar uso de recursos
gcloud run services describe darwin-production-backend \
  --region=us-central1 \
  --format="table(spec.template.spec.template.spec.containers[0].resources)"
```

**Solução:**
```bash
# 1. Aumentar recursos do Cloud Run
gcloud run services update darwin-production-backend \
  --region=us-central1 \
  --cpu=4 \
  --memory=8Gi

# 2. Otimizar database
./scripts/setup_database.sh --project-id PROJECT_ID --verify

# 3. Verificar cache hit rate
gcloud redis instances describe darwin-production-redis --region=us-central1
```

---

### ❌ Problema: "Cold start delays"

**Sintomas:**
- Primeira request muito lenta
- Intermittent timeouts
- High response time variance

**Solução:**
```bash
# 1. Aumentar min instances
gcloud run services update darwin-production-backend \
  --region=us-central1 \
  --min-instances=2

# 2. Habilitar CPU boost
gcloud run services update darwin-production-backend \
  --region=us-central1 \
  --cpu-boost

# 3. Otimizar container startup
# (Optimize Dockerfile and dependencies)
```

---

## 🗄️ Problemas de Banco de Dados

### ❌ Erro: "Too many connections"

**Sintomas:**
```
FATAL: remaining connection slots are reserved for non-replication superuser connections
```

**Diagnóstico:**
```bash
# Verificar conexões ativas
gcloud sql operations list --instance=darwin-production-db

# Verificar configuração
gcloud sql instances describe darwin-production-db \
  --format="value(settings.databaseFlags[].name,settings.databaseFlags[].value)"
```

**Solução:**
```bash
# 1. Aumentar max_connections
gcloud sql instances patch darwin-production-db \
  --database-flags=max_connections=300

# 2. Implementar connection pooling
# (Update application configuration)

# 3. Verificar vazamentos de conexão
# (Review application code for unclosed connections)
```

---

### ❌ Erro: "pgvector extension not found"

**Sintomas:**
```
ERROR: extension "vector" is not available
```

**Solução:**
```bash
# 1. Verificar extensões habilitadas
gcloud sql instances describe darwin-production-db \
  --format="value(settings.databaseFlags[])"

# 2. Configurar pgvector
./scripts/setup_database.sh \
  --project-id PROJECT_ID \
  --setup-pgvector

# 3. Reiniciar instância se necessário
gcloud sql instances restart darwin-production-db
```

---

### ❌ Problema: "Slow database queries"

**Diagnóstico:**
```sql
-- Conectar ao banco
gcloud sql connect darwin-production-db --user=postgres

-- Verificar queries lentas
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- Verificar índices não utilizados
SELECT schemaname, tablename, indexname, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
WHERE idx_tup_read = 0;
```

**Solução:**
```sql
-- Criar índices otimizados
CREATE INDEX CONCURRENTLY idx_documents_embedding_cosine 
ON documents USING ivfflat (embedding vector_cosine_ops);

-- Analisar tabelas
ANALYZE;

-- Atualizar estatísticas
SELECT pg_stat_reset();
```

---

## 🔒 Problemas de SSL/DNS

### ❌ Problema: "SSL certificate not ready"

**Sintomas:**
- HTTPS não funciona
- Certificate warnings
- ERR_SSL_VERSION_OR_CIPHER_MISMATCH

**Diagnóstico:**
```bash
# Verificar status dos certificados
gcloud compute ssl-certificates list

# Verificar domínios
gcloud compute ssl-certificates describe darwin-production-ssl-cert
```

**Solução:**
```bash
# 1. Verificar DNS configuração
nslookup api.agourakis.med.br
nslookup darwin.agourakis.med.br

# 2. Aguardar provisioning (pode levar até 60 minutos)
watch -n 60 'gcloud compute ssl-certificates list'

# 3. Verificar domain verification
# Certificados managed precisam que o domínio aponte para o LB
```

---

### ❌ Problema: "Domain not resolving"

**Sintomas:**
- nslookup fails
- DNS_PROBE_FINISHED_NXDOMAIN
- Domain not accessible

**Diagnóstico:**
```bash
# Verificar propagação DNS
dig +trace api.agourakis.med.br
dig +trace darwin.agourakis.med.br

# Verificar load balancer IP
gcloud compute addresses list --global
```

**Solução:**
```bash
# 1. Obter IP correto do load balancer
terraform output load_balancer_ip

# 2. Configurar DNS records
# api.agourakis.med.br -> LOAD_BALANCER_IP
# darwin.agourakis.med.br -> LOAD_BALANCER_IP

# 3. Aguardar propagação (até 48h)
```

---

## 📊 Problemas de Monitoramento

### ❌ Problema: "Alerts not firing"

**Sintomas:**
- Serviços down mas sem alertas
- Métricas não aparecem
- Dashboard vazio

**Diagnóstico:**
```bash
# Verificar notification channels
gcloud alpha monitoring channels list

# Verificar alert policies
gcloud alpha monitoring policies list

# Verificar métricas
gcloud monitoring metrics list --filter="metric.type:run.googleapis.com"
```

**Solução:**
```bash
# 1. Configurar notification channels
./scripts/setup_monitoring.sh \
  --project-id PROJECT_ID \
  --email "admin@company.com"

# 2. Verificar permissões de monitoring
gcloud projects get-iam-policy PROJECT_ID | grep monitoring

# 3. Forçar update do monitoring
gcloud alpha monitoring policies list --uri
```

---

### ❌ Problema: "Logs not appearing"

**Sintomas:**
- Cloud Logging vazio
- Logs não aparecem em tempo real
- Missing application logs

**Solução:**
```bash
# 1. Verificar log routing
gcloud logging sinks list

# 2. Verificar service account permissions
gcloud projects get-iam-policy PROJECT_ID | grep "roles/logging.logWriter"

# 3. Verificar aplicação está logando
# (Check application code for proper logging setup)

# 4. Forçar flush de logs
# (Restart Cloud Run services if needed)
```

---

## 🔐 Problemas de Segurança

### ❌ Erro: "Secret not found"

**Sintomas:**
```
Error: Secret "darwin-production-database-url" not found
```

**Diagnóstico:**
```bash
# Verificar secrets existentes
gcloud secrets list --filter="name~darwin"

# Verificar permissões
gcloud secrets get-iam-policy SECRET_NAME
```

**Solução:**
```bash
# 1. Criar secret manualmente
gcloud secrets create darwin-production-database-url \
  --data-file=- <<< "postgresql://user:pass@host:5432/db"

# 2. Configurar permissões
gcloud secrets add-iam-policy-binding darwin-production-database-url \
  --member="serviceAccount:backend-sa@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

# 3. Verificar acesso
gcloud secrets versions access latest --secret="darwin-production-database-url"
```

---

### ❌ Problema: "WAF blocking legitimate traffic"

**Sintomas:**
- 403 Forbidden errors
- Legitimate users blocked
- API requests rejected

**Diagnóstico:**
```bash
# Verificar Cloud Armor logs
gcloud logging read "resource.type=http_load_balancer AND jsonPayload.enforcedSecurityPolicy.name~'darwin'"

# Verificar security policy
gcloud compute security-policies describe darwin-production-security-policy
```

**Solução:**
```bash
# 1. Ajustar rate limiting
gcloud compute security-policies rules update 1000 \
  --security-policy=darwin-production-security-policy \
  --rate-limit-threshold-count=2000

# 2. Adicionar whitelist IPs
gcloud compute security-policies rules create 500 \
  --security-policy=darwin-production-security-policy \
  --src-ip-ranges="TRUSTED_IP/32" \
  --action=allow

# 3. Verificar logs após mudanças
gcloud logging read "resource.type=http_load_balancer" --limit=20
```

---

## 📈 Problemas de Performance

### ❌ Problema: "High memory usage"

**Sintomas:**
- OOM kills frequentes
- Memory usage > 90%
- Service restarts

**Diagnóstico:**
```bash
# Verificar uso de memória
gcloud monitoring metrics list --filter="metric.type:run.googleapis.com/container/memory"

# Verificar logs de OOM
gcloud logs read "resource.type=cloud_run_revision AND textPayload~'out of memory'"
```

**Solução:**
```bash
# 1. Aumentar limite de memória
gcloud run services update darwin-production-backend \
  --region=us-central1 \
  --memory=8Gi

# 2. Otimizar código (memory leaks)
# (Review application code for memory optimization)

# 3. Configurar garbage collection
# (Optimize JVM/Python GC settings)
```

---

### ❌ Problema: "Database connection pool exhausted"

**Sintomas:**
```
psycopg2.pool.PoolError: connection pool exhausted
```

**Solução:**
```bash
# 1. Aumentar pool size na aplicação
# DATABASE_POOL_SIZE=50

# 2. Aumentar max_connections no banco
gcloud sql instances patch darwin-production-db \
  --database-flags=max_connections=300

# 3. Implementar connection pooling externo
# (Consider PgBouncer or similar)
```

---

## 🔍 Logs e Debugging

### 📊 Localizações de Logs Importantes

#### Application Logs
```bash
# Backend logs
gcloud logs read "resource.type=cloud_run_revision AND resource.labels.service_name=darwin-production-backend" --limit=100

# Frontend logs  
gcloud logs read "resource.type=cloud_run_revision AND resource.labels.service_name=darwin-production-frontend" --limit=100

# Error logs only
gcloud logs read "resource.type=cloud_run_revision AND severity>=ERROR" --limit=50
```

#### Infrastructure Logs
```bash
# Load balancer logs
gcloud logs read "resource.type=http_load_balancer" --limit=50

# Firewall logs
gcloud logs read "resource.type=gce_firewall_rule" --limit=50

# VPC flow logs
gcloud logs read "resource.type=gce_subnetwork" --limit=50
```

#### Security Logs
```bash
# Audit logs
gcloud logs read "logName:cloudaudit.googleapis.com" --limit=50

# IAM changes
gcloud logs read "protoPayload.methodName:SetIamPolicy" --limit=20

# Service account activity
gcloud logs read "protoPayload.authenticationInfo.principalEmail~'@PROJECT_ID.iam.gserviceaccount.com'" --limit=20
```

### 🔍 Debugging Específico por Componente

#### Backend Debugging
```bash
# JAX computation errors
gcloud logs read "resource.type=cloud_run_revision AND textPayload~'JAX'" --limit=20

# Database query errors
gcloud logs read "resource.type=cloud_run_revision AND textPayload~'psycopg2'" --limit=20

# Vector search errors
gcloud logs read "resource.type=cloud_run_revision AND textPayload~'vector'" --limit=20
```

#### Frontend Debugging
```bash
# Next.js build errors
gcloud logs read "resource.type=cloud_run_revision AND textPayload~'next'" --limit=20

# React hydration errors
gcloud logs read "resource.type=cloud_run_revision AND textPayload~'hydration'" --limit=20

# API communication errors
gcloud logs read "resource.type=cloud_run_revision AND textPayload~'fetch'" --limit=20
```

---

## 🚨 Procedimentos de Emergência

### 🔥 Incident Response Checklist

#### Severidade 1: Serviços Completamente Down
```bash
# 1. Verificar status geral
./scripts/setup_monitoring.sh --project-id PROJECT_ID --verify

# 2. Rollback imediato se necessário
gcloud run services update-traffic darwin-production-backend \
  --to-revisions=PREVIOUS_REVISION=100

# 3. Verificar infraestrutura crítica
gcloud sql instances list
gcloud redis instances list --region=us-central1

# 4. Comunicar status
# (Update status page, notify stakeholders)
```

#### Severidade 2: Degradação de Performance
```bash
# 1. Aumentar recursos temporariamente
gcloud run services update darwin-production-backend \
  --region=us-central1 \
  --min-instances=5 \
  --max-instances=50

# 2. Verificar database performance
gcloud sql instances describe darwin-production-db

# 3. Verificar cache hit rate
# (Monitor Redis metrics)
```

### 🛠️ Emergency Scaling

```bash
# Scale up backend immediately
gcloud run services update darwin-production-backend \
  --region=us-central1 \
  --min-instances=10 \
  --cpu=4 \
  --memory=8Gi

# Scale up frontend
gcloud run services update darwin-production-frontend \
  --region=us-central1 \
  --min-instances=5

# Scale up database (if needed)
gcloud sql instances patch darwin-production-db \
  --tier=db-n1-standard-4
```

### 🔄 Emergency Rollback

```bash
# 1. Listar revisões disponíveis
gcloud run revisions list --service=darwin-production-backend --region=us-central1

# 2. Rollback para revisão anterior
gcloud run services update-traffic darwin-production-backend \
  --region=us-central1 \
  --to-revisions=REVISION_NAME=100

# 3. Verificar health após rollback
curl https://api.agourakis.med.br/health
```

---

## 🔧 Comandos de Diagnóstico Úteis

### 📊 Status Geral do Sistema

```bash
# Script de diagnóstico completo
cat > diagnose_system.sh << 'EOF'
#!/bin/bash
echo "🔍 DARWIN System Diagnosis"
echo "=========================="

echo "📊 Cloud Run Services:"
gcloud run services list --platform=managed

echo "🗄️ Database Status:"
gcloud sql instances list

echo "🔴 Redis Status:"
gcloud redis instances list --region=us-central1

echo "🌐 Load Balancer Status:"
gcloud compute forwarding-rules list --global

echo "🔒 SSL Certificates:"
gcloud compute ssl-certificates list

echo "💰 Current Costs (last 7 days):"
gcloud billing projects describe PROJECT_ID

echo "🚨 Recent Errors (last 1 hour):"
gcloud logs read "severity>=ERROR AND timestamp>='-1h'" --limit=10
EOF

chmod +x diagnose_system.sh
./diagnose_system.sh
```

### 🔍 Performance Profiling

```bash
# CPU profiling
gcloud profiler profiles list --project=PROJECT_ID

# Memory analysis
gcloud logging read "resource.type=cloud_run_revision AND jsonPayload.memory" --limit=20

# Request tracing
gcloud trace traces list --limit=10
```

### 🛡️ Security Audit

```bash
# IAM audit
gcloud projects get-iam-policy PROJECT_ID --format=json | jq '.bindings[] | select(.role=="roles/owner")'

# Service account audit
gcloud iam service-accounts list --filter="displayName~DARWIN"

# Firewall audit
gcloud compute firewall-rules list --filter="allowed[].ports:('22','3389')"

# Secret access audit
gcloud logs read "protoPayload.methodName~'secret'" --limit=20
```

---

## 📞 Escalation Procedures

### 🔴 Critical Issues (Production Down)

1. **Immediate Response (0-15 minutes):**
   - Execute emergency scaling
   - Check system status dashboard
   - Attempt automatic rollback

2. **Investigation (15-30 minutes):**
   - Review recent deployments
   - Analyze error logs
   - Check external dependencies

3. **Resolution (30-60 minutes):**
   - Apply targeted fixes
   - Verify system stability
   - Update incident status

4. **Post-Incident (1-24 hours):**
   - Post-mortem analysis
   - Update runbooks
   - Implement preventive measures

### 🟡 Non-Critical Issues

1. **Initial Assessment:**
   - Categorize impact and urgency
   - Gather relevant logs and metrics
   - Document reproduction steps

2. **Investigation:**
   - Use troubleshooting guides
   - Consult architecture documentation
   - Collaborate with team members

3. **Resolution:**
   - Test fixes in staging first
   - Apply changes with minimal impact
   - Monitor for regressions

---

## 📈 Monitoring and Alerting

### 🚨 Alert Response Playbooks

#### High Error Rate Alert
```bash
# 1. Check error distribution
gcloud logs read "resource.type=cloud_run_revision AND severity>=ERROR" --limit=20

# 2. Identify error patterns
gcloud logs read "resource.type=cloud_run_revision" --format="value(textPayload)" | grep -E "(error|exception|fail)" | sort | uniq -c

# 3. Check recent deployments
gcloud run revisions list --service=darwin-production-backend --region=us-central1

# 4. Rollback if deployment-related
gcloud run services update-traffic darwin-production-backend --to-revisions=PREVIOUS_REVISION=100
```

#### High Latency Alert
```bash
# 1. Check performance metrics
gcloud monitoring metrics list --filter="metric.type:run.googleapis.com/request_latencies"

# 2. Identify slow endpoints
gcloud logs read "resource.type=cloud_run_revision" --format="value(httpRequest.latency,httpRequest.requestUrl)" | sort -nr

# 3. Check database performance
gcloud sql operations list --instance=darwin-production-db --limit=10

# 4. Scale up if needed
gcloud run services update darwin-production-backend --min-instances=5
```

---

## 🛠️ Maintenance Procedures

### 🔄 Regular Maintenance Tasks

#### Weekly Tasks
```bash
# 1. Review performance metrics
./scripts/run_tests.sh --project-id PROJECT_ID --performance-tests

# 2. Check security vulnerabilities
./scripts/setup_security.sh --project-id PROJECT_ID --verify

# 3. Verify backup integrity
gcloud sql backups list --instance=darwin-production-db

# 4. Review cost reports
gcloud billing accounts get-iam-policy BILLING_ACCOUNT_ID
```

#### Monthly Tasks
```bash
# 1. Update dependencies
# (Review and update container base images)

# 2. Rotate secrets
# (Generate new JWT secrets, API keys)

# 3. Review access permissions
gcloud projects get-iam-policy PROJECT_ID

# 4. Capacity planning
# (Review usage trends and plan scaling)
```

### 🔧 Preventive Maintenance

```bash
# Database maintenance
gcloud sql instances patch darwin-production-db --maintenance-window-day=7 --maintenance-window-hour=4

# Redis maintenance
gcloud redis instances update darwin-production-redis --region=us-central1 --maintenance-window-day=SUNDAY --maintenance-window-start-time=03:00

# Application updates
./scripts/deploy_applications.sh --project-id PROJECT_ID --both --parallel
```

---

## 📚 Documentation and Resources

### 🔗 Useful Links

- **Cloud Run Troubleshooting:** https://cloud.google.com/run/docs/troubleshooting
- **Cloud SQL Troubleshooting:** https://cloud.google.com/sql/docs/troubleshooting
- **VPC Troubleshooting:** https://cloud.google.com/vpc/docs/troubleshooting
- **Load Balancer Troubleshooting:** https://cloud.google.com/load-balancing/docs/troubleshooting

### 📋 Incident Template

```markdown
## Incident Report: [TITLE]

**Date:** [DATE]
**Severity:** [P0/P1/P2/P3]
**Duration:** [START] - [END]
**Affected Services:** [SERVICES]

### Impact
- [Describe user impact]
- [Quantify affected users/requests]

### Timeline
- [HH:MM] Issue detected
- [HH:MM] Investigation started
- [HH:MM] Root cause identified
- [HH:MM] Fix applied
- [HH:MM] Service restored

### Root Cause
[Detailed analysis of what caused the issue]

### Resolution
[Steps taken to resolve the issue]

### Prevention
[Actions to prevent recurrence]

### Action Items
- [ ] [Action 1]
- [ ] [Action 2]
```

---

## 🆘 Emergency Contacts

### 📞 Escalation Matrix

| Severity | Response Time | Contact |
|----------|---------------|---------|
| P0 (Critical) | Immediate | On-call engineer + Team lead |
| P1 (High) | 30 minutes | Team lead |
| P2 (Medium) | 2 hours | Assigned engineer |
| P3 (Low) | Next business day | Team backlog |

### 📧 Contact Information

- **Primary On-call:** oncall@agourakis.med.br
- **DevOps Team:** devops@agourakis.med.br
- **Security Team:** security@agourakis.med.br
- **Database Team:** dba@agourakis.med.br

---

## ✅ Prevention Best Practices

### 🔒 Security Best Practices
1. Regular security scans and updates
2. Least privilege access reviews
3. Secret rotation schedules
4. Vulnerability assessment quarterly

### 📊 Monitoring Best Practices
1. Comprehensive alerting on key metrics
2. Regular dashboard reviews
3. SLO monitoring and adjustment
4. Capacity planning based on trends

### 🚀 Deployment Best Practices
1. Always test in staging first
2. Use gradual rollouts for major changes
3. Maintain rollback procedures
4. Document all changes

### 💰 Cost Optimization
1. Regular cost reviews and optimization
2. Resource right-sizing based on usage
3. Automated scaling policies
4. Unused resource cleanup

---

**💡 Lembre-se:** 
- Sempre teste correções em staging primeiro
- Documente todas as soluções aplicadas
- Mantenha logs de incidentes para análise futura
- Use automação para tarefas repetitivas

Para problemas não cobertos neste guia, consulte a documentação oficial do GCP ou entre em contato com a equipe de suporte.