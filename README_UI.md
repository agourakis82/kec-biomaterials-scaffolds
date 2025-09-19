# UI – Next.js + Tauri

Dev (Web):

```
cd ui
cp .env.local.example .env.local
# edite NEXT_PUBLIC_DARWIN_URL (não exponha DARWIN_SERVER_KEY)
npm i
npm run dev
```

Build (Desktop Tauri):

```
cd ui
npm run build
npm run tauri:build
```

Desktop (Config via Settings):
- Abra Configurações → seção "Desktop (Tauri) – Config".
- Clique "Load" para ler config atual (se existir).
- Preencha DARWIN_URL e DARWIN_SERVER_KEY e clique "Save".
- No modo desktop, as chamadas usam secure_fetch com Keychain/DPAPI (não expõe chave ao browser).

Variáveis (`ui/.env.local`):
- `NEXT_PUBLIC_DARWIN_URL` (URL pública da API)
- `DARWIN_SERVER_KEY` (somente no servidor/proxy — não no client)

Perfis & Atalhos:
- Perfis em `Configurações` (sheet): domain/include/exclude/mode.
- Command Palette: ⌘/Ctrl+K; atalhos `g d`, `g p`, `g a`.

Observações:
- UI chama apenas `/api/*` (proxy server-side) para não expor chaves.
- Export estático em `ui/out` para empacotar no Tauri.
