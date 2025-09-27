"use client"

import * as React from "react"
import { Textarea } from "@/components/ui/textarea"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { AnswerCard } from "@/components/answers/AnswerCard"
import { darwin } from "@/lib/darwin"
import { diffWords } from "@/lib/diff"
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts"

export default function ComparePage() {
  const [query, setQuery] = React.useState("")
  const [loading, setLoading] = React.useState(false)
  const [rag, setRag] = React.useState<any>(null)
  const [iter, setIter] = React.useState<any>(null)
  const [error, setError] = React.useState<string | null>(null)
  const [hoverLeft, setHoverLeft] = React.useState<number | null>(null)
  const [hoverRight, setHoverRight] = React.useState<number | null>(null)

  const submit = async () => {
    setLoading(true)
    setError(null)
    setRag(null)
    setIter(null)
    try {
      const [r1, r2] = await Promise.all([
        darwin.ragSearch(query, {}),
        darwin.ragIterative(query, {}),
      ])
      setRag({ answer: r1?.answer ?? r1?.content ?? "", sources: r1?.sources ?? r1?.references ?? [] })
      setIter({ answer: r2?.answer ?? r2?.content ?? "", sources: r2?.sources ?? r2?.references ?? [] })
    } catch (e: any) {
      setError(e?.message ?? "Erro ao comparar")
    } finally {
      setLoading(false)
    }
  }

  const chartData = React.useMemo(() => {
    const counts: Record<string, number> = {}
    ;(rag?.sources ?? []).forEach((s: any) => {
      counts[s.title ?? ""] = (counts[s.title ?? ""] ?? 0) + 1
    })
    ;(iter?.sources ?? []).forEach((s: any) => {
      counts[s.title ?? ""] = (counts[s.title ?? ""] ?? 0) + 1
    })
    return Object.entries(counts).slice(0, 12).map(([name, value]) => ({ name: name.slice(0, 24), value }))
  }, [rag, iter])

  const diff = React.useMemo(() => diffWords(rag?.answer ?? "", iter?.answer ?? ""), [rag, iter])

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Comparar RAG vs Iterative</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-3 md:flex-row">
            <Textarea className="min-h-[120px] flex-1" placeholder="Pergunta" value={query} onChange={(e) => setQuery(e.target.value)} />
            <Button className="md:self-start" onClick={submit} disabled={!query || loading}>Rodar</Button>
          </div>
        </CardContent>
      </Card>

      {loading && <Skeleton className="h-40 w-full" />}
      {error && <div className="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm">{error}</div>}

      {!loading && (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          <div className="lg:col-span-1">
            <Card>
              <CardHeader>
                <CardTitle>Fontes (contagem)</CardTitle>
              </CardHeader>
              <CardContent style={{ height: 260 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData}>
                    <XAxis dataKey="name" hide />
                    <YAxis width={28} />
                    <Tooltip />
                    <Bar dataKey="value" fill="#60a5fa" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
            <Card className="mt-6">
              <CardHeader>
                <CardTitle>Diff (LCS)</CardTitle>
              </CardHeader>
              <CardContent className="prose prose-invert max-w-none text-sm">
                <p>
                  {diff.map((t, i) => (
                    <span key={i} className={t.type === "ins" ? "bg-green-600/30" : t.type === "del" ? "bg-red-600/30" : undefined}>
                      {t.text}
                    </span>
                  ))}
                </p>
              </CardContent>
            </Card>
          </div>
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>RAG</CardTitle>
              </CardHeader>
              <CardContent>
                <AnswerCard answer={rag?.answer ?? ""} hoveredIndex={hoverLeft} onHoverIndex={setHoverLeft} />
              </CardContent>
            </Card>
          </div>
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Iterative</CardTitle>
              </CardHeader>
              <CardContent>
                <AnswerCard answer={iter?.answer ?? ""} hoveredIndex={hoverRight} onHoverIndex={setHoverRight} />
              </CardContent>
            </Card>
          </div>
        </div>
      )}
    </div>
  )
}

