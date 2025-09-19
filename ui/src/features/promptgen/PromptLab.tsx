"use client"

import * as React from "react"
import { usePromptGen } from "./store"
import { PRESETS } from "./presets"
import { MODEL_OPTIONS, openInChat } from "./modelMap"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { useActiveProfile } from "@/hooks/useActiveProfile"
import { motion } from "framer-motion"

function buildPrompts(state: ReturnType<typeof usePromptGen.getState>, profile: ReturnType<typeof useActiveProfile>) {
  const lines: string[] = []
  if (state.injectProfile && profile) {
    lines.push(`Domain: ${profile.domain}`)
    if (profile.includeTags?.length) lines.push(`Include: ${profile.includeTags.join(", ")}`)
    if (profile.excludeTags?.length) lines.push(`Exclude: ${profile.excludeTags.join(", ")}`)
  }
  if (state.DOMAINS) lines.push(`DOMAINS: ${state.DOMAINS}`)
  if (state.CONSTRAINTS) lines.push(`CONSTRAINTS: ${state.CONSTRAINTS}`)
  lines.push(`k=${state.k} depth=${state.depth} iters=${state.iters} c_puct=${state.c_puct}`)
  const system = [state.STYLE, ...lines].filter(Boolean).join("\n")
  const user = state.TASK
  return { system, user }
}

export function PromptLab() {
  const s = usePromptGen()
  const set = usePromptGen((x) => x.set)
  const profile = useActiveProfile()
  const prompts = buildPrompts(usePromptGen.getState(), profile)

  const applyPreset = (id: string) => {
    const p = PRESETS.find((x) => x.id === id)
    if (!p) return
    set(p.params as any)
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Prompt-Lab</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
            <div>
              <label className="text-sm text-muted-foreground">Modelo</label>
              <select
                className="mt-2 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                value={s.model}
                onChange={(e) => set({ model: e.target.value as any })}
              >
                {MODEL_OPTIONS.map((m) => (
                  <option key={m.id} value={m.id}>{m.name}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-sm text-muted-foreground">Preset</label>
              <select
                className="mt-2 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                onChange={(e) => applyPreset(e.target.value)}
                defaultValue=""
              >
                <option value="" disabled>Escolha…</option>
                {PRESETS.map((p) => (
                  <option key={p.id} value={p.id}>{p.name}</option>
                ))}
              </select>
            </div>
            <label className="mt-6 inline-flex items-center gap-2 text-sm">
              <input type="checkbox" checked={s.injectProfile} onChange={(e) => set({ injectProfile: e.target.checked })} />
              Injetar Perfil no Prompt
            </label>
          </div>

          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <Textarea placeholder="TASK" value={s.TASK} onChange={(e) => set({ TASK: e.target.value })} />
            <Textarea placeholder="STYLE" value={s.STYLE} onChange={(e) => set({ STYLE: e.target.value })} />
            <Textarea placeholder="DOMAINS" value={s.DOMAINS} onChange={(e) => set({ DOMAINS: e.target.value })} />
            <Textarea placeholder="CONSTRAINTS" value={s.CONSTRAINTS} onChange={(e) => set({ CONSTRAINTS: e.target.value })} />
          </div>

          <div className="grid grid-cols-1 gap-6 md:grid-cols-4 items-center">
            <div>
              <div className="mb-2 text-sm">k: {s.k}</div>
              <Slider min={1} max={20} value={[s.k]} onValueChange={([v]) => set({ k: v })} />
            </div>
            <div>
              <div className="mb-2 text-sm">depth: {s.depth}</div>
              <Slider min={1} max={10} value={[s.depth]} onValueChange={([v]) => set({ depth: v })} />
            </div>
            <div>
              <div className="mb-2 text-sm">iters: {s.iters}</div>
              <Slider min={0} max={10} value={[s.iters]} onValueChange={([v]) => set({ iters: v })} />
            </div>
            <div>
              <div className="mb-2 text-sm">c_puct: {s.c_puct.toFixed(1)}</div>
              <Slider min={0} max={4} step={0.1} value={[s.c_puct]} onValueChange={([v]) => set({ c_puct: v })} />
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            <motion.div whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.98 }}>
              <Button onClick={() => openInChat(s.model as any, prompts.system, prompts.user)}>Abrir em Chat</Button>
            </motion.div>
            <Button variant="outline" onClick={() => { void navigator.clipboard.writeText(prompts.system) }}>Copiar System</Button>
            <Button variant="outline" onClick={() => { void navigator.clipboard.writeText(prompts.user) }}>Copiar User</Button>
          </div>

          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <Textarea className="min-h-[160px]" value={prompts.system} readOnly />
            <Textarea className="min-h-[160px]" value={prompts.user} readOnly />
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

