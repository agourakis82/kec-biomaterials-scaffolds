"use client"

import * as React from "react"
import { useSettings } from "@/store/settings"
import type { Profile } from "@/types/profile"

export function useActiveProfile(): Profile | null {
  const getActiveProfile = useSettings((s) => s.getActiveProfile)
  const profiles = useSettings((s) => s.profiles)
  const activeProfileId = useSettings((s) => s.activeProfileId)

  // derive memoized active profile
  return React.useMemo(() => {
    const p = getActiveProfile()
    if (!p) return null
    return {
      id: p.id,
      name: p.name,
      domain: p.domain ?? "",
      includeTags: p.includeTags ?? [],
      excludeTags: p.excludeTags ?? [],
      defaultMode: p.defaultMode ?? "rag",
      notes: p.notes ?? "",
    }
  }, [getActiveProfile, profiles, activeProfileId])
}

