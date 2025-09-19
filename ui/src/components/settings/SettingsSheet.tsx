"use client"

import * as React from "react"
import { useSettings } from "@/store/settings"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import {
  Sheet,
  SheetTrigger,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
  SheetFooter,
} from "@/components/ui/sheet"
import { Separator } from "@/components/ui/separator"

import type { InferenceMode } from "@/types/profile"
import { isTauri } from "@/lib/desktop"

const modes: { label: string; value: InferenceMode }[] = [
  { label: "RAG", value: "rag" },
  { label: "Iterative (ReAct)", value: "iterative" },
  { label: "PUCT", value: "puct" },
]

export function SettingsSheet() {
  const {
    profiles,
    activeProfileId,
    setActiveProfile,
    addProfile,
    duplicateProfile,
    deleteProfile,
    updateProfile,
  } = useSettings()

  const [open, setOpen] = React.useState(false)
  const active = profiles.find((p) => p.id === activeProfileId) ?? profiles[0]

  const onAdd = () => {
    const id = addProfile({ name: "Novo Perfil" })
    setActiveProfile(id)
  }

  const onDuplicate = () => {
    if (!active) return
    const id = duplicateProfile(active.id)
    if (id) setActiveProfile(id)
  }

  const onDelete = () => {
    if (!active) return
    deleteProfile(active.id)
  }

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button variant="secondary">Configurações</Button>
      </SheetTrigger>
      <SheetContent className="sm:max-w-xl">
        <SheetHeader>
          <SheetTitle>Configurações</SheetTitle>
          <SheetDescription>Gerencie perfis e contexto padrão</SheetDescription>
        </SheetHeader>

        <div className="mt-4 space-y-6">
          <div>
            <label className="text-sm text-muted-foreground">Perfil ativo</label>
            <select
              className="mt-2 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              value={active?.id ?? ""}
              onChange={(e) => setActiveProfile(e.target.value)}
            >
              {profiles.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
            <div className="mt-3 flex gap-2">
              <Button size="sm" onClick={onAdd}>
                Adicionar
              </Button>
              <Button size="sm" variant="outline" onClick={onDuplicate}>
                Duplicar
              </Button>
              <Button size="sm" variant="destructive" onClick={onDelete}>
                Excluir
              </Button>
            </div>
          </div>

          {active && (
            <div className="space-y-3">
              <Input
                value={active.name}
                onChange={(e) => updateProfile(active.id, { name: e.target.value })}
                placeholder="Nome do perfil"
              />
              <Input
                value={active.domain ?? ""}
                onChange={(e) => updateProfile(active.id, { domain: e.target.value })}
                placeholder="Domínio (ex: biomaterials)"
              />
              <Textarea
                value={(active.includeTags ?? []).join(", ")}
                onChange={(e) =>
                  updateProfile(active.id, {
                    includeTags: e.target.value
                      .split(",")
                      .map((s) => s.trim())
                      .filter(Boolean),
                  })
                }
                placeholder="Include tags, separado por vírgula"
              />
              <Textarea
                value={(active.excludeTags ?? []).join(", ")}
                onChange={(e) =>
                  updateProfile(active.id, {
                    excludeTags: e.target.value
                      .split(",")
                      .map((s) => s.trim())
                      .filter(Boolean),
                  })
                }
                placeholder="Exclude tags, separado por vírgula"
              />
              <div>
                <label className="text-sm text-muted-foreground">Modo padrão</label>
                <div className="mt-2 grid grid-cols-3 gap-2">
                  {modes.map((m) => (
                    <button
                      key={m.value}
                      onClick={() => updateProfile(active.id, { defaultMode: m.value })}
                      className={`rounded-md border px-3 py-2 text-sm ${
                        active.defaultMode === m.value
                          ? "bg-primary text-primary-foreground"
                          : "bg-background hover:bg-accent"
                      }`}
                    >
                      {m.label}
                    </button>
                  ))}
                </div>
              </div>

              <Textarea
                value={active.notes ?? ""}
                onChange={(e) => updateProfile(active.id, { notes: e.target.value })}
                placeholder="Notas"
              />
            </div>
          )}

          <Separator />

          {isTauri() && (
            <div className="space-y-3">
              <div className="text-sm font-medium">Desktop (Tauri) – Config</div>
              <TauriConfigPanel />
              <Separator />
            </div>
          )}
        </div>

        <SheetFooter className="mt-4">
          <Button onClick={() => setOpen(false)}>Fechar</Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  )
}

function TauriConfigPanel() {
  const [url, setUrl] = React.useState("")
  const [key, setKey] = React.useState("")
  const [loading, setLoading] = React.useState(false)

  const load = async () => {
    setLoading(true)
    try {
      const { invoke } = await import("@tauri-apps/api/tauri")
      const res = await invoke<string>('cfg_load')
      const obj = JSON.parse(res || '{}')
      setUrl(obj.DARWIN_URL || obj.NEXT_PUBLIC_DARWIN_URL || "")
      setKey(obj.DARWIN_SERVER_KEY || "")
    } catch (_) {
      // noop
    } finally {
      setLoading(false)
    }
  }

  const save = async () => {
    setLoading(true)
    try {
      const { invoke } = await import("@tauri-apps/api/tauri")
      const payload = JSON.stringify({ DARWIN_URL: url, DARWIN_SERVER_KEY: key })
      await invoke('cfg_save', { jsonPayload: payload })
    } catch (_) {
      // noop
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-2">
      <Input placeholder="DARWIN_URL (ex: https://darwin.agourakis.med.br)" value={url} onChange={(e) => setUrl(e.target.value)} />
      <Input placeholder="DARWIN_SERVER_KEY" value={key} onChange={(e) => setKey(e.target.value)} />
      <div className="flex gap-2">
        <Button size="sm" variant="outline" onClick={load} disabled={loading}>Load</Button>
        <Button size="sm" onClick={save} disabled={loading}>Save</Button>
      </div>
    </div>
  )
}
