"use client"

import * as React from "react"
import { useRouter } from "next/navigation"
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command"

export function CommandPalette() {
  const [open, setOpen] = React.useState(false)
  const router = useRouter()

  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Ctrl/Cmd+K opens
      if ((e.key === "k" || e.key === "K") && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        setOpen((v) => !v)
      }
      // g d, g p, g a
      if (e.key.toLowerCase() === "g") {
        let handler = (ev: KeyboardEvent) => {
          const k = ev.key.toLowerCase()
          if (k === "d") router.push("/discovery")
          if (k === "p") router.push("/puct")
          if (k === "a") router.push("/admin")
          window.removeEventListener("keydown", handler)
        }
        window.addEventListener("keydown", handler, { once: true })
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [router])

  return (
    <CommandDialog open={open} onOpenChange={setOpen}>
      <CommandInput placeholder="Digite um comando ou pesquise…" />
      <CommandList>
        <CommandEmpty>Nenhum resultado.</CommandEmpty>
        <CommandGroup heading="Navegação">
          <CommandItem onSelect={() => router.push("/")}>Home</CommandItem>
          <CommandItem onSelect={() => router.push("/discovery")}>Discovery</CommandItem>
          <CommandItem onSelect={() => router.push("/puct")}>PUCT</CommandItem>
          <CommandItem onSelect={() => router.push("/compare")}>Compare</CommandItem>
          <CommandItem onSelect={() => router.push("/prompt-lab")}>Prompt-Lab</CommandItem>
          <CommandItem onSelect={() => router.push("/admin")}>Admin</CommandItem>
        </CommandGroup>
      </CommandList>
    </CommandDialog>
  )
}

