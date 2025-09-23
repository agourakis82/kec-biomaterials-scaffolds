# KEC Biomaterials API - Domínio Customizado

## 🌐 Informações do Domínio

**Domínio Principal**: https://api.agourakis.med.br
**Cloud Run Service**: darwin-kec-biomat
**Projeto GCP**: pcs-helio
**Região**: us-central1

## 🤖 URLs para GPT Actions

- **Base URL**: https://api.agourakis.med.br
- **OpenAPI Schema**: https://api.agourakis.med.br/openapi.json
- **Health Check**: https://api.agourakis.med.br/healthz
- **Documentation**: https://api.agourakis.med.br/docs

### Endpoints Principais:
- **KEC Metrics**: https://api.agourakis.med.br/gpt-actions/analyze-kec-metrics
- **RAG++ Query**: https://api.agourakis.med.br/gpt-actions/rag-query
- **Project Status**: https://api.agourakis.med.br/gpt-actions/project-status
- **Scientific Discovery**: https://api.agourakis.med.br/gpt-actions/scientific-discovery
- **System Health**: https://api.agourakis.med.br/gpt-actions/system-health

## 📋 Configuração DNS Necessária

Para que o domínio funcione, configure os seguintes registros DNS:

```
api	ghs.googlehosted.com.
```

## 🔒 SSL/TLS

- **Certificado**: Google-managed SSL certificate
- **Status**: True
- **Renovação**: Automática

## ✅ Próximos Passos

1. **Configure DNS** com os registros acima
2. **Aguarde propagação** (até 48h)
3. **Configure ChatGPT Action** com https://api.agourakis.med.br/openapi.json
4. **Teste endpoints** de GPT Actions

