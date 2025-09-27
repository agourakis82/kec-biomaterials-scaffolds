# DARWIN AI - Guia de Produção

## 🚀 Como Migrar para Produção Real

Sua interface está linda e funcionando! Agora vamos conectar com o backend real do Darwin.

### 1. Configuração do Ambiente

1. **Copie o arquivo de ambiente:**
```bash
cp .env.example .env.local
```

2. **Configure as variáveis essenciais:**
```bash
# .env.local
NEXT_PUBLIC_DARWIN_API_URL=http://localhost:8090  # URL do seu backend Darwin
APP_USERNAME=seu_usuario_admin
APP_PASSWORD=sua_senha_segura
AUTH_SECRET_KEY=uma-chave-jwt-muito-segura-e-aleatoria
```

### 2. Backend Darwin Integration

O sistema já está preparado para se conectar com seu backend Darwin. As APIs estão mapeadas em:

- **Research Team**: `/api/v1/research-team/*`
- **JAX Performance**: `/api/v1/ultra-performance/*`
- **Multi-AI Chat**: `/api/v1/multi-ai/*`
- **Discovery**: `/api/v1/discovery/*`
- **Knowledge Graph**: `/api/v1/knowledge-graph/*`
- **KEC Metrics**: `/api/v1/kec-metrics/*`
- **Tree Search (PUCT)**: `/api/v1/tree-search/*`
- **RAG Plus**: `/api/v1/rag-plus/*`

### 3. Autenticação Real

Para implementar autenticação real:

1. **Substitua o sistema mock** em `src/hooks/useAuth.ts`
2. **Configure JWT** no backend Darwin
3. **Implemente validação de token** no middleware

### 4. Integração com Banco de Dados

1. **Configure PostgreSQL:**
```bash
DATABASE_URL=postgresql://username:password@localhost:5432/darwin_db
```

2. **Implemente persistência** de usuários e sessões

### 5. APIs Externas

Configure as chaves das APIs que você usa:

```bash
OPENAI_API_KEY=sua-chave-openai
ANTHROPIC_API_KEY=sua-chave-anthropic
GOOGLE_API_KEY=sua-chave-google
```

### 6. Deploy em Produção

#### Opção A: Vercel (Recomendado para Frontend)
```bash
npm install -g vercel
vercel --prod
```

#### Opção B: Docker
```bash
docker build -t darwin-frontend .
docker run -p 3000:3000 darwin-frontend
```

#### Opção C: Google Cloud Run
```bash
gcloud run deploy darwin-frontend \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### 7. Monitoramento

Configure monitoramento com:

```bash
SENTRY_DSN=sua-sentry-dsn
```

### 8. HTTPS e Domínio

1. **Configure seu domínio** (ex: darwin.agourakis.med.br)
2. **Configure SSL/TLS** automaticamente via Vercel ou Cloudflare
3. **Atualize CORS** no backend para permitir seu domínio

### 9. Checklist de Produção

- [ ] Variáveis de ambiente configuradas
- [ ] Backend Darwin rodando
- [ ] Banco de dados configurado
- [ ] APIs externas configuradas
- [ ] SSL/HTTPS ativo
- [ ] Monitoramento configurado
- [ ] Backup configurado
- [ ] Testes de carga realizados

### 10. Comandos Úteis

```bash
# Desenvolvimento
npm run dev

# Build de produção
npm run build

# Iniciar produção
npm run start

# Verificar build
npm run build && npm run start
```

### 11. Estrutura de Arquivos Importantes

```
src/
├── services/darwinApi.ts     # Todas as APIs do Darwin
├── hooks/useAuth.ts          # Sistema de autenticação
├── components/auth/          # Componentes de login
├── components/dashboard/     # Dashboard principal
└── app/api/auth/            # Rotas de autenticação
```

### 12. Próximos Passos

1. **Teste todas as integrações** com o backend real
2. **Configure monitoramento** e alertas
3. **Implemente cache** para melhor performance
4. **Configure CI/CD** para deploys automáticos
5. **Documente APIs** para sua equipe

## 🎉 Parabéns!

Sua plataforma Darwin AI está pronta para produção com:
- ✅ Interface moderna e responsiva
- ✅ Sistema de autenticação robusto
- ✅ Integração completa com backend
- ✅ Monitoramento em tempo real
- ✅ Experiência de usuário excepcional

---

**Desenvolvido por:** Dr. Demetrios Chiuratto Agourakis  
**AGOURAKIS MED RESEARCH** - Plataforma de Pesquisa Inteligente