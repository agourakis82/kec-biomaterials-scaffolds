# DARWIN_RELEASE_CHECKLIST.md

Checklist de Release — Darwin RAG++

Este checklist garante que cada release do Darwin RAG++ seja seguro, auditável e reprodutível.

## 1. Pré-release
- [ ] CI/CD: Todos os testes de integração e unitários passaram (BigQuery, Vertex AI, Cloud Run, pipeline, endpoints).
- [ ] Linting: Código limpo, sem warnings críticos (pre-commit, markdownlint, yamllint).
- [ ] Dependências: `requirements.txt` e dependências de sistema revisadas e auditadas.
- [ ] Documentação: Atualizada (`README.md`, notebooks, docs/).
- [ ] Variáveis de ambiente: Checadas e documentadas.
- [ ] Chaves/segredos: Não versionados, rotacionados se necessário.

## 2. Segurança
- [ ] Revisão de permissões e escopos de API.
- [ ] Rate limiting validado.
- [ ] Auditoria de logs e acessos sensíveis.
- [ ] Testes de penetração (automatizados ou manuais).

## 3. Dados e Backup
- [ ] Backup completo do banco de embeddings/documentos.
- [ ] Teste de restauração de backup.
- [ ] Checagem de versionamento de dados críticos.

## 4. Deploy
- [ ] Deploy automatizado (GitHub Actions, scripts, etc).
- [ ] Rollback testado e documentado.
- [ ] Monitoramento ativo pós-deploy (logs, alertas, dashboards).

## 5. Pós-release
- [ ] Tag e changelog publicados.
- [ ] Relatório de status final gerado (`DARWIN_STATUS_REPORT.md`).
- [ ] Checklist de compliance revisado.
- [ ] Feedback de usuários coletado e registrado.

---

**Atenção:** Releases só devem ser aprovados se todos os itens acima estiverem marcados como concluídos.
