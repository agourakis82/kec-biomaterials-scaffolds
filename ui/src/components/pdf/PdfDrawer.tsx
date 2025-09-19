"use client"

import * as React from "react"
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Worker, Viewer } from "@react-pdf-viewer/core"
import { defaultLayoutPlugin } from "@react-pdf-viewer/default-layout"
import "@react-pdf-viewer/core/lib/styles/index.css"
import "@react-pdf-viewer/default-layout/lib/styles/index.css"

export function PdfDrawer({
  url,
  open,
  onOpenChange,
  title = "PDF Preview",
}: {
  url: string | null
  open: boolean
  onOpenChange: (open: boolean) => void
  title?: string
}) {
  const layoutPluginInstance = React.useMemo(() => defaultLayoutPlugin(), [])

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-full sm:max-w-3xl p-0">
        <SheetHeader className="p-4">
          <SheetTitle>{title}</SheetTitle>
        </SheetHeader>
        <ScrollArea className="h-[calc(100vh-64px)]">
          {url ? (
            <div className="h-[calc(100vh-100px)]">
              <Worker workerUrl="https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.worker.min.js">
                <Viewer fileUrl={url} plugins={[layoutPluginInstance]} />
              </Worker>
            </div>
          ) : (
            <div className="p-4 text-sm text-muted-foreground">Nenhum PDF selecionado.</div>
          )}
        </ScrollArea>
      </SheetContent>
    </Sheet>
  )
}
