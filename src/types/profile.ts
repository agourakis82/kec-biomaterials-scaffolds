export type InferenceMode = "rag" | "iterative" | "puct"

export interface Profile {
  id: string
  name: string
  domain?: string
  includeTags?: string[]
  excludeTags?: string[]
  defaultMode?: InferenceMode
  notes?: string
}

export interface AppSettings {
  activeProfileId: string | null
  profiles: Profile[]
  offlineCache: Record<string, unknown>
}

