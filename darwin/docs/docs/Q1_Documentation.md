# Q1 Documentation — DARWIN UI & Proxy (English)

## Overview
This document summarizes the recent work performed in Q1 to scaffold a Next.js-based UI for the DARWIN RAG++ project, server-side proxy routes to avoid exposing API keys, and the initial component and provider libraries (shadcn-like primitives). It also lists the environment variables, how to run the UI locally, and next steps.

## What was added
- New `ui/` application (Next.js App Router, TypeScript) with Tailwind CSS and design tokens supporting dark mode by class (`class="dark"` on `html`).
- UI primitives under `ui/src/components/ui/` (Button, Input, Textarea, Card, Badge, Tabs, Dialog, Sheet, Command, Slider, Progress, Tooltip, Separator, Skeleton, Toast, ScrollArea). These are small shadcn/ui-inspired components.
- Global providers: `QueryClientProvider` (TanStack Query) and a `Toaster` for notifications (`ui/src/components/providers.tsx`).
- Utility helpers:
  - `ui/src/lib/utils.ts` — `cn()` combining `clsx` and `tailwind-merge`.
  - `ui/src/lib/query.ts` — QueryClient singleton.
- API proxy route handlers (Next Route Handlers) to forward requests to the DARWIN backend while injecting `X-API-KEY` from server-only env:
  - `POST /api/rag/search` → `${DARWIN_URL}/rag-plus/search`
  - `POST /api/rag/iterative` → `${DARWIN_URL}/rag-plus/iterative`
  - `POST /api/puct` → `${DARWIN_URL}/tree-search/puct`
  - `POST /api/discovery/run` → `${DARWIN_URL}/discovery/run`
  - `GET /api/admin/status` → `${DARWIN_URL}/openapi.json`
- Client helper `ui/src/lib/darwin.ts` that calls the proxied `/api` endpoints (so client never touches `DARWIN_SERVER_KEY`).
- Example `.env.local.example` for local configuration.

## Key Files
- `ui/package.json` — scripts and dependencies (Next 15, React 18, Tailwind, etc.).
- `ui/src/app/layout.tsx` — root layout with Inter font and dark mode class; wraps content with `Providers`.
- `ui/src/app/(app)/layout.tsx` — app shell containing the header and main content area.
- `ui/src/app/globals.css` — Tailwind base with custom CSS variables for design tokens.
- `ui/src/lib/env.ts` — runtime validation of environment variables (server-only `DARWIN_SERVER_KEY` is not exported to client code).
- `ui/src/app/api/.../route.ts` — server-side route handlers (proxy).
- `ui/src/lib/darwin.ts` — frontend-safe client that calls `/api/*` proxied routes.

## Environment variables
- `NEXT_PUBLIC_DARWIN_URL` (public) — the public endpoint of the DARWIN backend, e.g. `https://darwin.agourakis.med.br`.
- `DARWIN_SERVER_KEY` (server-only) — the secret API key used in server-side proxy routes. Do NOT commit this key or export it to client-side code.

Use the provided `ui/.env.local.example` as a template.

## How to run locally (UI)
1. Copy environment example:

```bash
cd ui
cp .env.local.example .env.local
# Edit NEXT_PUBLIC_DARWIN_URL as needed. Do NOT set DARWIN_SERVER_KEY in the client .env; the key must be set in the server environment where Next runs.
npm install
npm run dev
```

2. The UI will run at `http://localhost:3000` by default.

## Notes on server-side secrets
- The `route.ts` handlers use the `serverEnv` object parsed in `ui/src/lib/env.ts`. Ensure the production server sets `DARWIN_SERVER_KEY` as an environment variable for the runtime (Cloud Run, Vercel, or Tauri host).
- The client helper `ui/src/lib/darwin.ts` only calls `/api/*` endpoints and uses `NEXT_PUBLIC_DARWIN_URL` only for validation or informational purposes.

## Next steps / TODOs
- Implement pages (home, discovery, puct, compare, prompt-lab, admin) and wire UI components to `darwin` client.
- Add unit and integration tests for API proxies.
- Add Tauri config and commands for desktop packaging.
- Add GitHub Actions workflow for CI/CD and desktop artifact builds.
- Add detailed READMEs: `README_UI.md`, `README_cloud.md`, `README_DARWIN.md` (planned in subsequent tasks).

## Changes committed in this round (recommended commit message)
"feat(ui): scaffold Next.js UI, Tailwind, shadcn components; add API proxy routes and docs (Q1)"

## Contact / Ownership
- Dev: Agourakis
- Repo: `kec-biomaterials-scaffolds`

---
Generated Q1 doc for handoff and tracking. Update this doc as the UI pages and behaviors are implemented.
