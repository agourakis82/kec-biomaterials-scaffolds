"use client"

import * as React from "react"
import { motion } from "framer-motion"
import { MessageSquare, Copy, Share, ThumbsUp, ThumbsDown } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"

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
  const [copied, setCopied] = React.useState(false)
  const lines = answer.split(/\n{2,}/)

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(answer)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy text: ', err)
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card className="shadow-soft hover-lift">
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <MessageSquare className="h-5 w-5 text-primary" />
              Resposta
            </CardTitle>
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={copyToClipboard}
                className="h-8 px-2"
              >
                <Copy className="h-4 w-4" />
                {copied ? "Copiado!" : "Copiar"}
              </Button>
              <Button variant="ghost" size="sm" className="h-8 px-2">
                <Share className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="prose prose-slate dark:prose-invert max-w-none">
            {lines.map((para, i) => (
              <motion.p 
                key={i} 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
                className="leading-relaxed text-base mb-4 last:mb-0"
              >
                {splitCitations(para).map((part, j) => {
                  if (part.type === "text") {
                    return <React.Fragment key={j}>{part.value}</React.Fragment>
                  }
                  const active = hoveredIndex === part.index
                  return (
                    <motion.span
                      key={j}
                      onMouseEnter={() => onHoverIndex?.(part.index ?? null)}
                      onMouseLeave={() => onHoverIndex?.(null)}
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.95 }}
                      className={`mx-1 inline-flex h-6 min-w-[24px] items-center justify-center rounded-full px-2 text-xs font-semibold cursor-pointer transition-all duration-200 ${
                        active 
                          ? "bg-primary text-primary-foreground shadow-glow" 
                          : "bg-muted hover:bg-muted-foreground/20 text-foreground"
                      }`}
                    >
                      {part.index}
                    </motion.span>
                  )
                })}
              </motion.p>
            ))}
          </div>
          
          {/* Feedback Section */}
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="flex items-center justify-between pt-4 border-t"
          >
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">Esta resposta foi útil?</span>
              <div className="flex items-center gap-1">
                <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                  <ThumbsUp className="h-4 w-4" />
                </Button>
                <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                  <ThumbsDown className="h-4 w-4" />
                </Button>
              </div>
            </div>
            <Badge variant="secondary" className="text-xs">
              Gerado por IA
            </Badge>
          </motion.div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
