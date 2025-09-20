# Validação – Build/Run

UI dev:

```
cd ui
cp .env.local.example .env.local
# edite NEXT_PUBLIC_DARWIN_URL; deixe DARWIN_SERVER_KEY no servidor (rotas /api)
npm i
npm run dev
```

Desktop:

```
npm run build
npm run tauri:build
```

Testes de fumaça:

```
bash scripts/smoke.sh
```

Checklist de Aceitação:
- `npm run dev` OK; `npm run tauri:build` gera .exe/.msi e .app/.dmg
- API responde `/rag-plus/search` com fontes; `/iterative` com 2–5 iterações; `/puct` com árvore; `/discovery/run` grava no BQ
- Prompt Lab gera prompts p/ GPT‑5, 5‑pro, Gemini
- UI aponta para `https://darwin.agourakis.med.br`
- READMEs claros; sem segredos no repo
