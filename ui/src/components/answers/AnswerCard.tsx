"use client"

import * as React from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

function splitCitations(text: string): Array<{ type: "text" | "cite"; value: string; index?: number }> {
  const parts: Array<{ type: "text" | "cite"; value: string; index?: number }> = []
  const re = /\[(\d+)\]/g
  let lastIndex = 0
  let m: RegExpExecArray | null
  while ((m = re.exec(text)) !== null) {
    if (m.index > lastIndex) parts.push({ type: "text", value: text.slice(lastIndex, m.index) })
    parts.push({ type: "cite", value: m[0], index: Number(m[1]) })
    lastIndex = m.index + m[0].length
  }
  if (lastIndex < text.length) parts.push({ type: "text", value: text.slice(lastIndex) })
  return parts
}

export function AnswerCard({
  answer,
  hoveredIndex,
  onHoverIndex,
}: {
  answer: string
  hoveredIndex?: number | null
  onHoverIndex?: (idx: number | null) => void
}) {
  const lines = answer.split(/\n{2,}/)

  return (
    <Card>
      <CardHeader>
        <CardTitle>Resposta</CardTitle>
      </CardHeader>
      <CardContent className="prose prose-invert max-w-none">
        {lines.map((para, i) => (
          <p key={i} className="leading-relaxed">
            {splitCitations(para).map((part, j) => {
              if (part.type === "text") return <React.Fragment key={j}>{part.value}</React.Fragment>
              const active = hoveredIndex === part.index
              return (
                <span
                  key={j}
                  onMouseEnter={() => onHoverIndex?.(part.index ?? null)}
                  onMouseLeave={() => onHoverIndex?.(null)}
                  className={`mx-0.5 inline-flex h-5 min-w-[20px] items-center justify-center rounded px-1 text-xs font-semibold ${
                    active ? "bg-primary text-primary-foreground" : "bg-muted text-foreground"
                  }`}
                >
                  [{part.index}]
                </span>
              )
            })}
          </p>
        ))}
      </CardContent>
    </Card>
  )
}

