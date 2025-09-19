import { isTauri, tauriSecureFetch } from './desktop'

export const darwin = {
  async ragSearch(query: string, options?: any) {
    if (isTauri()) {
      return tauriSecureFetch('/rag-plus/search', 'POST', { query, ...options })
    }
    const response = await fetch('/api/rag/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, ...options }),
    })
    return response.json()
  },

  async ragIterative(query: string, options?: any) {
    if (isTauri()) {
      return tauriSecureFetch('/rag-plus/iterative', 'POST', { query, ...options })
    }
    const response = await fetch('/api/rag/iterative', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, ...options }),
    })
    return response.json()
  },

  async puct(root: string, budget: number, cPuct: number) {
    if (isTauri()) {
      return tauriSecureFetch('/tree-search/puct', 'POST', { root, budget, c_puct: cPuct })
    }
    const response = await fetch('/api/puct', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ root, budget, c_puct: cPuct }),
    })
    return response.json()
  },

  async discoveryRun(runOnce: boolean = true) {
    if (isTauri()) {
      return tauriSecureFetch('/discovery/run', 'POST', { run_once: runOnce })
    }
    const response = await fetch('/api/discovery/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ run_once: runOnce }),
    })
    return response.json()
  },

  async adminStatus() {
    if (isTauri()) {
      return tauriSecureFetch('/openapi.json', 'GET')
    }
    const response = await fetch('/api/admin/status')
    return response.json()
  },
}
