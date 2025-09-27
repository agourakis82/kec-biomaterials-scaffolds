"use client"

import { create } from "zustand"
import { persist } from "zustand/middleware"
import type { AppSettings, Profile, InferenceMode } from "@/types/profile"

// Simple local nanoid fallback (if nanoid dep not present, inline mini impl)
function nid() {
  return (
    Date.now().toString(36) + Math.random().toString(36).slice(2)
  )
}

const defaultProfiles: Profile[] = [
  {
    id: nid(),
    name: "Biomateriais",
    domain: "biomaterials",
    includeTags: ["biomat", "scaffold", "porosity", "pore"],
    excludeTags: ["unrelated"],
    defaultMode: "rag",
    notes: "Perfil focado em biomateriais e scaffolds.",
  },
  {
    id: nid(),
    name: "Filosofia",
    domain: "philosophy",
    includeTags: ["ethics", "epistemology"],
    excludeTags: [],
    defaultMode: "iterative",
    notes: "Perfil para estudos conceituais e comparativos.",
  },
]

export interface SettingsState extends AppSettings {
  // getters
  getActiveProfile: () => Profile | null
  // actions
  setActiveProfile: (id: string) => void
  addProfile: (p?: Partial<Profile>) => string
  duplicateProfile: (id: string) => string | null
  updateProfile: (id: string, patch: Partial<Profile>) => void
  deleteProfile: (id: string) => void
  setDefaultMode: (id: string, mode: InferenceMode) => void
}

export const useSettings = create<SettingsState>()(
  persist(
    (set, get) => ({
      activeProfileId: defaultProfiles[0]?.id ?? null,
      profiles: defaultProfiles,
      offlineCache: {},

      getActiveProfile: () => {
        const { profiles, activeProfileId } = get()
        const p = profiles.find((x) => x.id === activeProfileId)
        return (
          p ??
          (profiles[0] ?? null)
        )
      },

      setActiveProfile: (id) => set({ activeProfileId: id }),

      addProfile: (p) => {
        const id = nid()
        const base: Profile = {
          id,
          name: p?.name ?? "Novo Perfil",
          domain: p?.domain ?? "",
          includeTags: p?.includeTags ?? [],
          excludeTags: p?.excludeTags ?? [],
          defaultMode: p?.defaultMode ?? "rag",
          notes: p?.notes ?? "",
        }
        set((s) => ({ profiles: [...s.profiles, base] }))
        return id
      },

      duplicateProfile: (id) => {
        const src = get().profiles.find((p) => p.id === id)
        if (!src) return null
        const newId = nid()
        const dup: Profile = { ...src, id: newId, name: `${src.name} (cópia)` }
        set((s) => ({ profiles: [...s.profiles, dup] }))
        return newId
      },

      updateProfile: (id, patch) => {
        set((s) => ({
          profiles: s.profiles.map((p) => (p.id === id ? { ...p, ...patch } : p)),
        }))
      },

      deleteProfile: (id) => {
        set((s) => {
          const next = s.profiles.filter((p) => p.id !== id)
          const activeProfileId = s.activeProfileId === id ? next[0]?.id ?? null : s.activeProfileId
          return { profiles: next, activeProfileId }
        })
      },

      setDefaultMode: (id, mode) => {
        set((s) => ({
          profiles: s.profiles.map((p) => (p.id === id ? { ...p, defaultMode: mode } : p)),
        }))
      },
    }),
    {
      name: "darwin-settings",
      partialize: (s) => ({
        activeProfileId: s.activeProfileId,
        profiles: s.profiles,
        offlineCache: s.offlineCache,
      }),
    }
  )
)
