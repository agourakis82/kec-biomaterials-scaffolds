"use client"

import * as React from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { darwin } from "@/lib/darwin"
import {
  Toast,
  ToastTitle,
  ToastDescription,
} from "@/components/ui/toast"

interface FeedItem {
  title: string
  source?: string
  domain?: string
  timestamp?: string
}

export default function DiscoveryPage() {
  const [loading, setLoading] = React.useState(false)
  const [toastOpen, setToastOpen] = React.useState(false)
  const [toastMsg, setToastMsg] = React.useState<string>("")
  const [items, setItems] = React.useState<FeedItem[]>([])
  const [qSource, setQSource] = React.useState("")
  const [qDomain, setQDomain] = React.useState("")

  const filtered = items.filter((i) =>
    (qSource ? (i.source ?? "").toLowerCase().includes(qSource.toLowerCase()) : true) &&
    (qDomain ? (i.domain ?? "").toLowerCase().includes(qDomain.toLowerCase()) : true)
  )

  const runDiscovery = async () => {
    setLoading(true)
    try {
      await darwin.discoveryRun(true)
      setToastMsg("Discovery disparado com sucesso")
      setToastOpen(true)
    } catch (e: any) {
      setToastMsg("Erro ao disparar discovery")
      setToastOpen(true)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-2">
        <Button onClick={runDiscovery} disabled={loading}>
          Forçar discovery agora
        </Button>
        <Input
          placeholder="Filtrar por source"
          value={qSource}
          onChange={(e) => setQSource(e.target.value)}
          className="w-48"
        />
        <Input
          placeholder="Filtrar por domínio"
          value={qDomain}
          onChange={(e) => setQDomain(e.target.value)}
          className="w-48"
        />
      </div>

      {loading && (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-24 w-full" />
          ))}
        </div>
      )}

      {!loading && (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          {filtered.length === 0 && (
            <div className="text-sm text-muted-foreground">Sem itens.</div>
          )}
          {filtered.map((it, i) => (
            <Card key={i} className="transition-colors">
              <CardHeader>
                <CardTitle className="text-base">{it.title}</CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground flex items-center justify-between">
                <div className="space-x-3">
                  {it.source && <span>source: {it.source}</span>}
                  {it.domain && <span>domain: {it.domain}</span>}
                  {it.timestamp && <span>{new Date(it.timestamp).toLocaleString()}</span>}
                </div>
                <div>
                  <Button size="sm" variant="outline" onClick={() => { setToastMsg("Ingest OK (MVP)"); setToastOpen(true) }}>Ingest to corpus</Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      <Toast open={toastOpen} onOpenChange={setToastOpen}>
        <ToastTitle>Aviso</ToastTitle>
        <ToastDescription>{toastMsg}</ToastDescription>
      </Toast>
    </div>
  )
}

