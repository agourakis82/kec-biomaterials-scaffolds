% DARWIN RAG++ – Visão Geral

DARWIN RAG++ é uma plataforma de pesquisa assistida por IA que combina:
- Backend (FastAPI/Cloud Run) com serviços RAG/Iterative/PUCT e Discovery.
- UI Next.js (App Router) com proxy server-side, evitando expor chaves.
- App Desktop (Tauri) para uso offline/empacotado e configs locais.

Componentes principais:
- RAG Plus: busca contextual, síntese com citações [n].
- Iterative (ReAct): iterações de raciocínio + consultas guiadas.
- PUCT (MCTS): exploração de árvore de decisões com métricas value/visits.
- Discovery: coleta periódica de novas fontes para ingestão/corpus.
- Perfis: domínios, include/exclude tags e modo padrão para contexto.

Fluxo (alto nível):
1) UI recebe pergunta + perfil → chama rotas `/api` (proxy).
2) API aplica RAG++/Iterative/PUCT conforme parâmetros.
3) Resposta retorna com fontes → UI renderiza markdown, [n], PDFs.
4) Discovery roda via Scheduler, grava no BQ e disponibiliza na API.

Arquitetura de segurança:
- `DARWIN_SERVER_KEY` somente no servidor/proxy (Next Route Handlers) ou no Desktop via Keychain/DPAPI.
- UI pública usa apenas `NEXT_PUBLIC_DARWIN_URL`.

## Pastas
- `infra/darwin/api`: FastAPI da plataforma (Cloud Run).
- `ui/`: Next.js App Router + Tauri (desktop).
- `scripts/`: utilitários (smoke tests, etc.).
- `docs/`: documentação auxiliar.

## Perfis
- Exemplos iniciais: Biomateriais, Filosofia. Ajuste `domain/include/exclude` e observe efeito nas buscas.

## Licenças
- Verifique licenças de dados/ativos antes da publicação.
