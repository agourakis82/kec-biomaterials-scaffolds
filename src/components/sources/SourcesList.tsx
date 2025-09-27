"use client"

import * as React from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"

export interface SourceItem {
  title: string
  url?: string
  doi?: string
  pdf_url?: string
}

export function SourcesList({
  sources,
  hoveredIndex,
  onPreview,
  onHoverIndex,
}: {
  sources: SourceItem[]
  hoveredIndex?: number | null
  onPreview?: (src: SourceItem) => void
  onHoverIndex?: (idx: number | null) => void
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Fontes</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {sources.length === 0 && (
          <div className="text-sm text-muted-foreground">Sem fontes.</div>
        )}
        {sources.map((s, i) => (
          <div
            key={i}
            onMouseEnter={() => onHoverIndex?.(i + 1)}
            onMouseLeave={() => onHoverIndex?.(null)}
            className={`rounded-md border p-3 text-sm ${
              hoveredIndex === i + 1 ? "border-primary bg-accent" : "border-border"
            }`}
          >
            <div className="mb-2 flex items-center justify-between gap-2">
              <div className="font-medium">[{i + 1}] {s.title}</div>
              <div className="shrink-0">
                {s.pdf_url && (
                  <Button size="sm" variant="outline" onClick={() => onPreview?.(s)}>
                    Preview
                  </Button>
                )}
              </div>
            </div>
            <div className="text-xs text-muted-foreground space-x-3 break-all">
              {s.url && (
                <a href={s.url} target="_blank" rel="noreferrer" className="underline">
                  link
                </a>
              )}
              {s.doi && <span>doi: {s.doi}</span>}
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}

