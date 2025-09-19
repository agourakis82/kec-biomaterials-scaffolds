# KEC_BIOMAT — Pipeline KEC 2.0 (Adendo Computacional, 2025-09-19)

Este pacote contém **arquivos prontos** para inclusão no seu repositório e para ingestão no
Darwin/ChatGPT, atualizando a implementação **sem alterar os objetivos nem os outputs** do
projeto aprovado (Entropia **H**, Curvatura **κ**, Coerência **σ/ϕ**).

## Conteúdo
- `docs/ADENDO_Metodologico_KEC2.0.md` — Adendo metodológico nivel Q1 (pronto para colar).
- `configs/kec_config.yaml` — Parâmetros padrão do pipeline KEC 2.0.
- `pipeline/kec_metrics.py` — Implementação de **H_espectral**, **κ_Forman**, **σ/ϕ** e **σ_Q** (opcional).
- `tests/test_kec_metrics.py` — Testes mínimos de sanidade (unitários rápidos).
- `darwin/memory_kec_v2.json` — Memória estruturada para ingestão no **Darwin**.
- `gpt_ingest/KEC_BIOMAT_memory.md` — Resumo curto para colar no ChatGPT (memória do projeto).
- `CITATION.cff` — Metadados de citação (ORCID/DOI placeholders).
- `CHANGELOG.md` — Mudanças principais.
- `LICENSE` — MIT.
- `environment.yml` — Ambiente sugerido (conda/mamba) para reproducibilidade.

## Como usar (resumo)
1. **Repo**: copie todo o conteúdo para a raiz do repositório (ou `tools/KEC2.0/`), faça commit.
2. **Darwin**: importe `darwin/memory_kec_v2.json` como memória do projeto / chave de orquestração.
3. **ChatGPT**: cole o conteúdo de `gpt_ingest/KEC_BIOMAT_memory.md` em uma nova thread e fixe.
4. **Execução**: ajuste `configs/kec_config.yaml` e chame as funções de `pipeline/kec_metrics.py` no seu fluxo.
5. **Testes**: rode `pytest -q` em `tests/`. (Os testes são rápidos e sem dependências pesadas).

> **Invariantes:** outputs permanecem **H/κ/σ** (e **ϕ**), com **σ_Q (opcional)** como feature extra.
