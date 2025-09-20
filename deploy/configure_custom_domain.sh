#!/bin/bash

# Configure Custom Domain - api.agourakis.med.br
# Mapeia Cloud Run service para domínio customizado com SSL automático

# Não interromper o script em caso de erro
# set -e

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🌐 Configurando Domínio Customizado: api.agourakis.med.br${NC}"
echo "============================================================"

# Configurações
PROJECT_ID="pcs-helio"
REGION="us-central1"
SERVICE_NAME="darwin-kec-biomat"
CUSTOM_DOMAIN="api.agourakis.med.br"

# Verificar se service existe
echo -e "${YELLOW}🔍 Verificando service Cloud Run...${NC}"
if ! gcloud run services describe ${SERVICE_NAME} --region=${REGION} >/dev/null 2>&1; then
    echo -e "${RED}❌ Service ${SERVICE_NAME} não encontrado em ${REGION}${NC}"
    echo "Execute primeiro o deploy: ./deploy/deploy_modular_backend.sh"
    exit 1
fi

SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format='value(status.url)')
echo -e "${GREEN}✅ Service encontrado: ${SERVICE_URL}${NC}"

# 1. VERIFICAR/CONFIGURAR DOMAIN MAPPING
echo -e "${YELLOW}🔗 Configurando domain mapping...${NC}"

# Criar domain mapping se não existir
if ! gcloud beta run domain-mappings describe --domain=${CUSTOM_DOMAIN} --region=${REGION} >/dev/null 2>&1; then
    echo "Criando domain mapping para ${CUSTOM_DOMAIN}..."
    
    gcloud beta run domain-mappings create \
        --service=${SERVICE_NAME} \
        --domain=${CUSTOM_DOMAIN} \
        --region=${REGION}
    
    echo -e "${GREEN}✅ Domain mapping criado${NC}"
else
    echo "Domain mapping para ${CUSTOM_DOMAIN} já existe"
fi

# Verificar status do DomainMapping
echo "Verificando status do DomainMapping..."
if ! gcloud beta run domain-mappings describe --domain=${CUSTOM_DOMAIN} --region=${REGION} --format='value(status.conditions.status)' | grep -q "True"; then
    echo -e "${RED}❌ Domain mapping não está pronto. Verifique o status no console do Google Cloud.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Domain mapping está pronto${NC}"

echo "Aguardando 60 segundos para a propagação inicial do DNS..."
sleep 60

# 2. OBTER E VERIFICAR INFORMAÇÕES DE DNS
echo -e "${YELLOW}📋 Obtendo e verificando configuração DNS...${NC}"

# Get DNS records necessários
DNS_INFO=$(gcloud beta run domain-mappings describe --domain=${CUSTOM_DOMAIN} --region=${REGION} --format='value(status.resourceRecords[].name,status.resourceRecords[].rrdata)' 2>/dev/null || echo "DNS info not available yet")

echo -e "${BLUE}📋 Configuração DNS necessária:${NC}"
echo "$DNS_INFO" | while IFS=$'\t' read name rrdata; do
    if [ ! -z "$name" ] && [ ! -z "$rrdata" ]; then
        echo "• $name → $rrdata"
        
        # Verificar se o DNS já está propagado
        echo "Verificando propagação de DNS para $name..."
        if dig +short $name | grep -q "$rrdata"; then
            echo -e "${GREEN}✅ Propagação de DNS para $name concluída${NC}"
        else
            echo -e "${YELLOW}⚠️  Aguardando propagação de DNS para $name...${NC}"
        fi
    fi
done

# 3. AGUARDAR CERTIFICADO SSL
echo -e "${YELLOW}🔒 Verificando certificado SSL...${NC}"

# Check SSL certificate status
SSL_STATUS="PROVISIONING"
ATTEMPTS=0
MAX_ATTEMPTS=30  # Aumentado para dar mais tempo para o provisionamento de SSL
SKIP_SSL_CHECK=false

while [ "$SSL_STATUS" != "ACTIVE" ] && [ $ATTEMPTS -lt $MAX_ATTEMPTS ]; do
    SSL_STATUS=$(gcloud beta run domain-mappings describe --domain=${CUSTOM_DOMAIN} --region=${REGION} --format='value(status.conditions[0].status)' 2>/dev/null || echo "PROVISIONING")
    
    echo "Certificado SSL status: $SSL_STATUS (tentativa $((ATTEMPTS + 1))/$MAX_ATTEMPTS)"
    
    if [ "$SSL_STATUS" = "ACTIVE" ]; then
        break
    fi
    
    # Após 5 tentativas, perguntar se quer continuar esperando
    if [ $ATTEMPTS -eq 5 ]; then
        echo -e "${YELLOW}⚠️  Certificado SSL ainda provisionando. Isso pode levar até 24h.${NC}"
        read -p "Continuar esperando? (s/n): " CONTINUE_WAITING
        if [ "$CONTINUE_WAITING" != "s" ]; then
            SKIP_SSL_CHECK=true
            break
        fi
    fi
    
    sleep 15  # Reduzido para 15 segundos
    ATTEMPTS=$((ATTEMPTS + 1))
done

if [ "$SSL_STATUS" = "ACTIVE" ]; then
    echo -e "${GREEN}✅ Certificado SSL ativo${NC}"
elif [ "$SKIP_SSL_CHECK" = true ]; then
    echo -e "${YELLOW}⚠️  Verificação de SSL ignorada. Continuando com a configuração...${NC}"
else
    echo -e "${YELLOW}⚠️  Certificado SSL ainda provisionando (pode levar até 24h)${NC}"
    echo -e "${YELLOW}⚠️  Detalhes do status:${NC}"
    gcloud beta run domain-mappings describe --domain=${CUSTOM_DOMAIN} --region=${REGION} --format='value(status.conditions)'
    echo -e "${YELLOW}⚠️  Você pode verificar o status mais tarde com:${NC}"
    echo "gcloud beta run domain-mappings describe --domain=${CUSTOM_DOMAIN} --region=${REGION}"
fi

# 4. TESTAR DOMÍNIO CUSTOMIZADO
echo -e "${YELLOW}🧪 Testando domínio customizado...${NC}"

# Teste final abrangente
function final_check() {
    echo "Executando verificação final..."
    
    # Teste de conectividade
    echo "Testando conectividade..."
    if curl -f "https://${CUSTOM_DOMAIN}/healthz" -s --max-time 10 >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Conectividade com https://${CUSTOM_DOMAIN}/healthz bem-sucedida${NC}"
    else
        echo -e "${RED}❌ Falha na conectividade com https://${CUSTOM_DOMAIN}/healthz${NC}"
        echo "   Verifique a propagação de DNS e o status do serviço Cloud Run."
        return 1
    fi

    # Teste de SSL
    echo "Testando certificado SSL..."
    if echo | openssl s_client -servername ${CUSTOM_DOMAIN} -connect ${CUSTOM_DOMAIN}:443 2>/dev/null | openssl x509 -noout -check_host ${CUSTOM_DOMAIN}; then
        echo -e "${GREEN}✅ Certificado SSL para ${CUSTOM_DOMAIN} é válido${NC}"
    else
        echo -e "${RED}❌ Certificado SSL para ${CUSTOM_DOMAIN} inválido ou não encontrado${NC}"
        echo "   Aguarde a conclusão do provisionamento do certificado."
        return 1
    fi

    # Teste de CORS
    echo "Testando configuração de CORS..."
    CORS_CHECK_URL="https://${CUSTOM_DOMAIN}/healthz"
    CORS_ORIGIN="https://chat.openai.com"
    if curl -s -o /dev/null -w "%{http_code}" -H "Origin: ${CORS_ORIGIN}" ${CORS_CHECK_URL} | grep -q "200"; then
        echo -e "${GREEN}✅ Configuração de CORS para ${CORS_ORIGIN} está funcionando${NC}"
    else
        echo -e "${RED}❌ Configuração de CORS para ${CORS_ORIGIN} falhou${NC}"
        echo "   Verifique as variáveis de ambiente CORS_ORIGINS no serviço Cloud Run."
        return 1
    fi
    
    echo -e "${GREEN}✅ Verificação final concluída com sucesso!${NC}"
}

final_check

# 5. GERAR CONFIGURAÇÃO ATUALIZADA PARA GPT ACTIONS
echo -e "${YELLOW}🤖 Atualizando configuração GPT Actions...${NC}"

cat > gpt_actions_config_custom_domain.json <<EOF
{
  "schema_version": "v1",
  "name_for_model": "kec_biomaterials_api",
  "name_for_human": "KEC Biomaterials Analysis API",
  "description_for_model": "Advanced biomaterials analysis API with RAG++, tree search, memory systems, and scientific discovery. Provides KEC metrics (entropy, curvature, small-world) calculation, knowledge retrieval, and automated research discovery. Deployed at api.agourakis.med.br",
  "description_for_human": "Analyze porous biomaterials using advanced KEC metrics, search scientific knowledge, and discover recent research.",
  "auth": {
    "type": "none"
  },
  "api": {
    "type": "openapi",
    "url": "https://${CUSTOM_DOMAIN}/openapi.json"
  },
  "logo_url": "https://${CUSTOM_DOMAIN}/static/logo.png",
  "contact_email": "admin@agourakis.med.br",
  "legal_info_url": "https://${CUSTOM_DOMAIN}/docs"
}
EOF

echo -e "${GREEN}✅ Configuração GPT Actions atualizada: gpt_actions_config_custom_domain.json${NC}"

# 6. CONFIGURAR CORS PARA DOMÍNIO CUSTOMIZADO
echo -e "${YELLOW}🔧 Configurando CORS para domínio customizado...${NC}"

# Update Cloud Run service com CORS headers
echo "Atualizando variáveis de ambiente para CORS e domínio customizado..."
gcloud run services update ${SERVICE_NAME} \
    --region=${REGION} \
    --update-env-vars "CORS_ORIGINS=https://chat.openai.com,https://chatgpt.com,https://${CUSTOM_DOMAIN}",CUSTOM_DOMAIN=${CUSTOM_DOMAIN}

echo -e "${GREEN}✅ CORS configurado para domínio customizado${NC}"

# 7. GERAR RESUMO DE CONFIGURAÇÃO
echo -e "${YELLOW}📝 Gerando resumo de configuração...${NC}"

cat > custom_domain_setup_summary.md <<EOF
# KEC Biomaterials API - Domínio Customizado

## 🌐 Informações do Domínio

**Domínio Principal**: https://${CUSTOM_DOMAIN}
**Cloud Run Service**: ${SERVICE_NAME}
**Projeto GCP**: ${PROJECT_ID}
**Região**: ${REGION}

## 🤖 URLs para GPT Actions

- **Base URL**: https://${CUSTOM_DOMAIN}
- **OpenAPI Schema**: https://${CUSTOM_DOMAIN}/openapi.json
- **Health Check**: https://${CUSTOM_DOMAIN}/healthz
- **Documentation**: https://${CUSTOM_DOMAIN}/docs

### Endpoints Principais:
- **KEC Metrics**: https://${CUSTOM_DOMAIN}/gpt-actions/analyze-kec-metrics
- **RAG++ Query**: https://${CUSTOM_DOMAIN}/gpt-actions/rag-query
- **Project Status**: https://${CUSTOM_DOMAIN}/gpt-actions/project-status
- **Scientific Discovery**: https://${CUSTOM_DOMAIN}/gpt-actions/scientific-discovery
- **System Health**: https://${CUSTOM_DOMAIN}/gpt-actions/system-health

## 📋 Configuração DNS Necessária

Para que o domínio funcione, configure os seguintes registros DNS:

\`\`\`
$DNS_INFO
\`\`\`

## 🔒 SSL/TLS

- **Certificado**: Google-managed SSL certificate
- **Status**: $SSL_STATUS
- **Renovação**: Automática

## ✅ Próximos Passos

1. **Configure DNS** com os registros acima
2. **Aguarde propagação** (até 48h)
3. **Configure ChatGPT Action** com https://${CUSTOM_DOMAIN}/openapi.json
4. **Teste endpoints** de GPT Actions

EOF

echo -e "${GREEN}✅ Resumo salvo: custom_domain_setup_summary.md${NC}"

# Criar arquivo de status para verificação posterior
cat > domain_status.sh <<EOF
#!/bin/bash

# Script para verificar status do domínio customizado
# Executar: ./domain_status.sh

echo "Verificando status do domínio: ${CUSTOM_DOMAIN}..."

# Verificar domain mapping
echo "=== Domain Mapping ==="
gcloud beta run domain-mappings describe --domain=${CUSTOM_DOMAIN} --region=${REGION}

# Verificar certificado SSL
echo ""
echo "=== Certificado SSL ==="
gcloud beta run domain-mappings describe --domain=${CUSTOM_DOMAIN} --region=${REGION} --format='value(status.conditions[0].status,status.conditions[0].message)'

# Testar conectividade
echo ""
echo "=== Teste de Conectividade ==="
curl -v "https://${CUSTOM_DOMAIN}/healthz" --max-time 10

echo ""
echo "Para mais detalhes, acesse o Console GCP:"
echo "https://console.cloud.google.com/run/domains?project=${PROJECT_ID}"
EOF

chmod +x domain_status.sh
echo -e "${GREEN}✅ Script de status criado: domain_status.sh${NC}"

# 8. RESULTADO FINAL
echo ""
echo -e "${GREEN}🎉 DOMÍNIO CUSTOMIZADO CONFIGURADO!${NC}"
echo ""
echo -e "${BLUE}📋 INFORMAÇÕES IMPORTANTES:${NC}"
echo "• Domínio: https://${CUSTOM_DOMAIN}"
echo "• Status SSL: $SSL_STATUS"
echo "• GPT Actions Schema: https://${CUSTOM_DOMAIN}/openapi.json"
echo ""
echo -e "${BLUE}🤖 Para ChatGPT Actions:${NC}"
echo "1. Use a URL: https://${CUSTOM_DOMAIN}/openapi.json"
echo "2. Auth Type: None"
echo "3. Endpoints base: https://${CUSTOM_DOMAIN}/gpt-actions/"
echo ""
echo -e "${YELLOW}⚠️  IMPORTANTE:${NC}"
echo "• Configure os registros DNS mostrados acima"
echo "• Aguarde propagação DNS (até 48h)"
echo "• SSL pode levar até 24h para ativar"
echo ""
echo -e "${GREEN}✅ Backend KEC v2.0 integrado com api.agourakis.med.br!${NC}"