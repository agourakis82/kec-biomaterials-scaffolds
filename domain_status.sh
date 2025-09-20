#!/bin/bash

# Script para verificar status do domínio customizado
# Executar: ./domain_status.sh

echo "Verificando status do domínio: api.agourakis.med.br..."

# Verificar domain mapping
echo "=== Domain Mapping ==="
gcloud beta run domain-mappings describe --domain=api.agourakis.med.br --region=us-central1

# Verificar certificado SSL
echo ""
echo "=== Certificado SSL ==="
gcloud beta run domain-mappings describe --domain=api.agourakis.med.br --region=us-central1 --format='value(status.conditions[0].status,status.conditions[0].message)'

# Testar conectividade
echo ""
echo "=== Teste de Conectividade ==="
curl -v "https://api.agourakis.med.br/healthz" --max-time 10

echo ""
echo "Para mais detalhes, acesse o Console GCP:"
echo "https://console.cloud.google.com/run/domains?project=pcs-helio"
