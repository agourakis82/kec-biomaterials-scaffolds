"use client"

import * as React from "react"
import { motion } from "framer-motion"
import { Search, Sparkles, Zap, BookOpen, History, Star } from "lucide-react"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { AnswerCard } from "@/components/answers/AnswerCard"
import { SourcesList, type SourceItem } from "@/components/sources/SourcesList"
import { PdfDrawer } from "@/components/pdf/PdfDrawer"
import { darwin } from "@/lib/darwin"
import { useActiveProfile } from "@/hooks/useActiveProfile"

interface RagResponse {
  answer?: string
  sources?: SourceItem[]
}

const suggestedQueries = [
  "Como funcionam os scaffolds de biomateriais?",
  "Quais são as propriedades dos materiais biocompatíveis?",
  "Métodos de caracterização de biomateriais",
  "Aplicações de hidrogéis em engenharia de tecidos",
]

const recentQueries = [
  "Propriedades mecânicas de scaffolds",
  "Biocompatibilidade de polímeros",
  "Técnicas de fabricação 3D",
]

export default function HomePage() {
  const [query, setQuery] = React.useState("")
  const [iterative, setIterative] = React.useState(false)
  const [loading, setLoading] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [data, setData] = React.useState<RagResponse | null>(null)
  const [hoveredIndex, setHoveredIndex] = React.useState<number | null>(null)
  const [pdfUrl, setPdfUrl] = React.useState<string | null>(null)
  const [pdfOpen, setPdfOpen] = React.useState(false)
  const [hasSearched, setHasSearched] = React.useState(false)

  const profile = useActiveProfile()

  const submit = async () => {
    if (!query.trim()) return
    
    setLoading(true)
    setError(null)
    setData(null)
    setHasSearched(true)
    
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

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      submit()
    }
  }

  const openPreview = (s: SourceItem) => {
    if (!s.pdf_url) return
    setPdfUrl(s.pdf_url)
    setPdfOpen(true)
  }

  const selectSuggestion = (suggestion: string) => {
    setQuery(suggestion)
  }

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      {!hasSearched && (
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center space-y-6 py-12"
        >
          <div className="space-y-4">
            <motion.div
              initial={{ scale: 0.8 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.2 }}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 text-primary text-sm font-medium"
            >
              <Sparkles className="h-4 w-4" />
              Powered by Advanced RAG Technology
            </motion.div>
            <h1 className="text-4xl md:text-6xl font-bold text-gradient">
              Explore Biomaterials
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Descubra insights avançados sobre biomateriais e scaffolds com nossa plataforma de IA especializada
            </p>
          </div>
        </motion.div>
      )}

      {/* Search Section */}
      <motion.div 
        layout
        className={`${hasSearched ? 'max-w-4xl mx-auto' : 'max-w-3xl mx-auto'}`}
      >
        <Card className="shadow-soft hover-lift">
          <CardHeader className="pb-4">
            <CardTitle className="flex items-center gap-2">
              <Search className="h-5 w-5 text-primary" />
              {hasSearched ? "Nova Consulta" : "Faça sua pergunta"}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="relative">
              <Textarea
                className="min-h-[120px] resize-none pr-12 text-base"
                placeholder="Pergunte sobre biomateriais, scaffolds, propriedades mecânicas..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyPress}
              />
              <div className="absolute bottom-3 right-3 text-xs text-muted-foreground">
                ⌘+Enter para enviar
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <motion.label 
                className="flex items-center gap-2 text-sm cursor-pointer"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <input
                  type="checkbox"
                  className="h-4 w-4 rounded border-2 border-primary text-primary focus:ring-primary"
                  checked={iterative}
                  onChange={(e) => setIterative(e.target.checked)}
                />
                <Zap className="h-4 w-4" />
                Modo Iterativo (ReAct)
              </motion.label>
              
              <Button 
                onClick={submit} 
                disabled={!query.trim() || loading}
                className="px-8 shadow-glow"
                size="lg"
              >
                {loading ? (
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  >
                    <Sparkles className="h-4 w-4" />
                  </motion.div>
                ) : (
                  <>
                    <Search className="h-4 w-4 mr-2" />
                    Consultar
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Suggestions - Only show when no search has been made */}
      {!hasSearched && (
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="max-w-4xl mx-auto space-y-6"
        >
          <div className="grid md:grid-cols-2 gap-6">
            <Card className="hover-lift">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <BookOpen className="h-5 w-5 text-primary" />
                  Sugestões de Consulta
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {suggestedQueries.map((suggestion, index) => (
                  <motion.button
                    key={index}
                    onClick={() => selectSuggestion(suggestion)}
                    className="w-full text-left p-3 rounded-lg hover:bg-accent transition-colors text-sm"
                    whileHover={{ x: 4 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    {suggestion}
                  </motion.button>
                ))}
              </CardContent>
            </Card>

            <Card className="hover-lift">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <History className="h-5 w-5 text-primary" />
                  Consultas Recentes
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {recentQueries.map((recent, index) => (
                  <motion.button
                    key={index}
                    onClick={() => selectSuggestion(recent)}
                    className="w-full text-left p-3 rounded-lg hover:bg-accent transition-colors text-sm flex items-center justify-between"
                    whileHover={{ x: 4 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <span>{recent}</span>
                    <Star className="h-4 w-4 text-muted-foreground" />
                  </motion.button>
                ))}
              </CardContent>
            </Card>
          </div>
        </motion.div>
      )}

      {/* Results Section */}
      {hasSearched && (
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="grid grid-cols-1 gap-6 lg:grid-cols-3"
        >
          <div className="lg:col-span-2 space-y-6">
            {loading && (
              <Card>
                <CardContent className="p-6 space-y-4">
                  <div className="flex items-center gap-2">
                    <Skeleton className="h-5 w-5 rounded-full" />
                    <Skeleton className="h-5 w-32" />
                  </div>
                  <div className="space-y-3">
                    <Skeleton className="h-4 w-full" />
                    <Skeleton className="h-4 w-full" />
                    <Skeleton className="h-4 w-3/4" />
                  </div>
                </CardContent>
              </Card>
            )}

            {error && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
              >
                <Card className="border-destructive/50 bg-destructive/5">
                  <CardContent className="p-6">
                    <div className="flex items-center gap-2 text-destructive">
                      <span className="font-medium">Erro na consulta:</span>
                      <span>{error}</span>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )}

            {!loading && data?.answer && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <AnswerCard
                  answer={data.answer}
                  hoveredIndex={hoveredIndex}
                  onHoverIndex={setHoveredIndex}
                />
              </motion.div>
            )}
          </div>

          <div className="space-y-6">
            {loading && (
              <Card>
                <CardHeader>
                  <Skeleton className="h-6 w-24" />
                </CardHeader>
                <CardContent className="space-y-3">
                  <Skeleton className="h-20 w-full" />
                  <Skeleton className="h-20 w-full" />
                </CardContent>
              </Card>
            )}

            {!loading && (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2 }}
              >
                <SourcesList
                  sources={data?.sources ?? []}
                  hoveredIndex={hoveredIndex}
                  onHoverIndex={setHoveredIndex}
                  onPreview={openPreview}
                />
              </motion.div>
            )}
          </div>
        </motion.div>
      )}

      <PdfDrawer url={pdfUrl} open={pdfOpen} onOpenChange={setPdfOpen} />
    </div>
  )
}
