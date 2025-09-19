// Utilities for Tauri desktop integration

export function isTauri(): boolean {
  return typeof window !== 'undefined' && typeof (window as any).__TAURI_IPC__ === 'function'
}

export async function tauriSecureFetch(endpoint: string, method: string, body?: any): Promise<any> {
  const payload = body === undefined ? null : JSON.stringify(body)
  const { invoke } = await import('@tauri-apps/api/tauri')
  const result = await invoke<string>('secure_fetch', {
    endpoint,
    method,
    bodyJson: payload,
  })
  try {
    return JSON.parse(result)
  } catch {
    return result
  }
}

