export type ModelId = "gpt-5" | "gpt-5-pro" | "gemini-2-pro"

export interface PromptParams {
  TASK: string
  STYLE: string
  DOMAINS: string
  CONSTRAINTS: string
  k: number
  depth: number
  iters: number
  c_puct: number
  injectProfile: boolean
}

export interface PromptPreset {
  id: string
  name: string
  params: Partial<PromptParams>
}

