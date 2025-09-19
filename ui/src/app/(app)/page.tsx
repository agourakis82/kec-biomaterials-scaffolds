"use client"

import * as React from "react"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { AnswerCard } from "@/components/answers/AnswerCard"
import { SourcesList, type SourceItem } from "@/components/sources/SourcesList"
import { PdfDrawer } from "@/components/pdf/PdfDrawer"
import { darwin } from "@/lib/darwin"
import { useActiveProfile } from "@/hooks/useActiveProfile"

interface RagResponse {
  answer?: string
  sources?: SourceItem[]
}

export default function HomePage() {
  const [query, setQuery] = React.useState("")
  const [iterative, setIterative] = React.useState(false)
  const [loading, setLoading] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [data, setData] = React.useState<RagResponse | null>(null)
  const [hoveredIndex, setHoveredIndex] = React.useState<number | null>(null)
  const [pdfUrl, setPdfUrl] = React.useState<string | null>(null)
  const [pdfOpen, setPdfOpen] = React.useState(false)

  const profile = useActiveProfile()

  const submit = async () => {
    setLoading(true)
    setError(null)
    setData(null)
    try {
      const payload: any = {
        query,
        profile: {
          domain: profile?.domain ?? "",
          include: profile?.includeTags ?? [],
          exclude: profile?.excludeTags ?? [],
        },
      }
      const resp = iterative
        ? await darwin.ragIterative(query, payload)
        : await darwin.ragSearch(query, payload)
      setData({
        answer: resp?.answer ?? resp?.content ?? "",
        sources: resp?.sources ?? resp?.references ?? [],
      })
    } catch (e: any) {
      setError(e?.message ?? "Erro na consulta")
    } finally {
      setLoading(false)
    }
  }

  const openPreview = (s: SourceItem) => {
    if (!s.pdf_url) return
    setPdfUrl(s.pdf_url)
    setPdfOpen(true)
  }

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
      <div className="lg:col-span-2 space-y-4">
        <div className="rounded-lg border p-4">
          <label className="text-sm text-muted-foreground">Pergunta</label>
          <Textarea
            className="mt-2 min-h-[120px]"
            placeholder="Pergunte ao DARWIN…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <div className="mt-3 flex items-center justify-between">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={iterative}
                onChange={(e) => setIterative(e.target.checked)}
              />
              Iterative (ReAct)
            </label>
            <Button onClick={submit} disabled={!query || loading}>
              Consultar
            </Button>
          </div>
        </div>

        {loading && (
          <div className="space-y-3">
            <Skeleton className="h-6 w-1/3" />
            <Skeleton className="h-24 w-full" />
            <Skeleton className="h-24 w-full" />
          </div>
        )}

        {error && (
          <div className="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm">
            {error}
          </div>
        )}

        {!loading && data?.answer && (
          <AnswerCard
            answer={data.answer}
            hoveredIndex={hoveredIndex}
            onHoverIndex={setHoveredIndex}
          />
        )}
      </div>

      <div className="space-y-4">
        {loading && (
          <div className="space-y-2">
            <Skeleton className="h-6 w-24" />
            <Skeleton className="h-20 w-full" />
            <Skeleton className="h-20 w-full" />
          </div>
        )}

        {!loading && (
          <SourcesList
            sources={data?.sources ?? []}
            hoveredIndex={hoveredIndex}
            onHoverIndex={setHoveredIndex}
            onPreview={openPreview}
          />
        )}
      </div>

      <PdfDrawer url={pdfUrl} open={pdfOpen} onOpenChange={setPdfOpen} />
    </div>
  )
}
