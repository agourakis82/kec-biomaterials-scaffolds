"use client"

import { create } from "zustand"
import type { ModelId, PromptParams } from "./types"

interface PGState extends PromptParams {
  model: ModelId
  set: (p: Partial<PGState>) => void
}

export const usePromptGen = create<PGState>((set) => ({
  model: "gpt-5",
  TASK: "",
  STYLE: "",
  DOMAINS: "",
  CONSTRAINTS: "",
  k: 6,
  depth: 2,
  iters: 0,
  c_puct: 1.2,
  injectProfile: true,
  set: (p) => set((s) => ({ ...s, ...p })),
}))

