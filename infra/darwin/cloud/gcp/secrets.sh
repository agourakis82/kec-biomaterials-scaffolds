#!/usr/bin/env bash
set -euo pipefail
PROJECT_ID="${GCP_PROJECT_ID:-PREENCHER_PROJETO}"

# Crie secrets:
# DARWIN_API_KEY, OPENAI_API_KEY (se necessario como fallback), CHATGPT_ACTIONS_BEARER
# VERTEX_MODELS_JSON: {"chat":"text-bison","embed":"textembedding-gecko@003"}
# DISCOVERY_FEEDS_YML: conteudo do schedules.yml

create_secret () {
  local name="$1" ; local value="$2"
  printf "%s" "$value" | gcloud secrets create "$name" --data-file=- --project "$PROJECT_ID" || \
  printf "%s" "$value" | gcloud secrets versions add "$name" --data-file=- --project "$PROJECT_ID"
}

create_secret "DARWIN_API_KEY" "PREENCHER_RANDOM_LONGO"
create_secret "VERTEX_MODELS_JSON" '{"chat":"text-bison","embed":"textembedding-gecko@003"}'
# opcional
#create_secret "OPENAI_API_KEY" "PREENCHER_OPENAI"
#create_secret "CHATGPT_ACTIONS_BEARER" "PREENCHER_BEARER"

# schedules.yml padrao
create_secret "DISCOVERY_FEEDS_YML" "$(cat <<'YML'
feeds:
  - name: arXiv AI
    url: https://export.arxiv.org/rss/cs.AI
    max: 15
  - name: arXiv LG
    url: https://export.arxiv.org/rss/cs.LG
    max: 15
  - name: Nature ML
    url: https://www.nature.com/subjects/machine-learning/rss
    max: 10
YML
)"
echo "OK"
