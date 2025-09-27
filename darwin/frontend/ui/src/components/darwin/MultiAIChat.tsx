"use client"

import * as React from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Send, Sparkles, Brain, Zap, MessageSquare, Settings, RefreshCw } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Skeleton } from "@/components/ui/skeleton"

interface MultiAIChatProps {
  onAIRecommendation?: (model: string) => void
  domains?: string[]
}

interface ChatMessage {
  id: string
  content: string
  role: 'user' | 'assistant'
  model?: string
  timestamp: Date
  processing?: boolean
}

interface AIModel {
  id: string
  name: string
  description: string
  strengths: string[]
  optimal_domains: string[]
  performance_score: number
}

const DEFAULT_MODELS: AIModel[] = [
  {
    id: 'gpt-4',
    name: 'ChatGPT-4',
    description: 'Raciocínio avançado e análise científica',
    strengths: ['Análise complexa', 'Síntese científica', 'Raciocínio lógico'],
    optimal_domains: ['biomaterials', 'research', 'analysis'],
    performance_score: 0.95
  },
  {
    id: 'claude-3',
    name: 'Claude-3 Sonnet',
    description: 'Especialista em documentos e literatura científica',
    strengths: ['Análise de documentos', 'Extração de insights', 'Precisão científica'],
    optimal_domains: ['literature', 'biomaterials', 'medical'],
    performance_score: 0.92
  },
  {
    id: 'gemini',
    name: 'Gemini Pro',
    description: 'Análise multimodal e dados complexos',
    strengths: ['Dados multimensionais', 'Visualização', 'Inferência estatística'],
    optimal_domains: ['data_analysis', 'visualization', 'statistics'],
    performance_score: 0.88
  }
]

export function MultiAIChat({ 
  domains = ["biomaterials"], 
  onAIRecommendation 
}: MultiAIChatProps) {
  const [messages, setMessages] = React.useState<ChatMessage[]>([])
  const [inputValue, setInputValue] = React.useState("")
  const [selectedModel, setSelectedModel] = React.useState<string>("")
  const [loading, setLoading] = React.useState(false)
  const [models, setModels] = React.useState<AIModel[]>(DEFAULT_MODELS)
  const [autoSelect, setAutoSelect] = React.useState(true)
  
  // Auto-recommend best AI model based on domain
  const recommendBestModel = React.useCallback((query: string) => {
    if (!autoSelect) return selectedModel

    // Simple heuristics for model recommendation
    const queryLower = query.toLowerCase()
    
    if (queryLower.includes('analis') || queryLower.includes('dados') || queryLower.includes('estat')) {
      return 'gemini'
    }
    
    if (queryLower.includes('literatura') || queryLower.includes('artigo') || queryLower.includes('paper')) {
      return 'claude-3'
    }
    
    if (queryLower.includes('complex') || queryLower.includes('síntese') || queryLower.includes('raciocín')) {
      return 'gpt-4'
    }

    // Default to best overall model
    const domainModel = models.find(m => 
      m.optimal_domains.some(d => domains.includes(d))
    )
    return domainModel?.id || 'gpt-4'
  }, [autoSelect, selectedModel, models, domains])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!inputValue.trim() || loading) return

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      content: inputValue.trim(),
      role: 'user',
      timestamp: new Date()
    }

    const recommendedModel = recommendBestModel(inputValue)
    const processingMessage: ChatMessage = {
      id: (Date.now() + 1).toString(),
      content: '',
      role: 'assistant',
      model: recommendedModel,
      timestamp: new Date(),
      processing: true
    }

    setMessages(prev => [...prev, userMessage, processingMessage])
    setInputValue("")
    setLoading(true)

    try {
      const response = await fetch('/api/multi-ai', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: inputValue,
          model: recommendedModel,
          context: {
            domains,
            conversation_history: messages.slice(-4) // Last 4 messages for context
          }
        }),
      })

      if (!response.ok) throw new Error('Multi-AI request failed')

      const data = await response.json()

      // Update processing message with actual response
      setMessages(prev => 
        prev.map(msg => 
          msg.id === processingMessage.id 
            ? { ...msg, content: data.response || data.answer || 'Resposta recebida', processing: false }
            : msg
        )
      )

      // Call recommendation callback
      if (onAIRecommendation) {
        onAIRecommendation(recommendedModel)
      }

    } catch (error) {
      console.error('Multi-AI Chat Error:', error)
      
      // Update processing message with error
      setMessages(prev => 
        prev.map(msg => 
          msg.id === processingMessage.id 
            ? { 
                ...msg, 
                content: 'Erro ao processar mensagem. Tente novamente.', 
                processing: false 
              }
            : msg
        )
      )
    } finally {
      setLoading(false)
    }
  }

  const getModelIcon = (modelId: string) => {
    switch (modelId) {
      case 'gpt-4': return <Sparkles className="h-4 w-4" />
      case 'claude-3': return <Brain className="h-4 w-4" />
      case 'gemini': return <Zap className="h-4 w-4" />
      default: return <MessageSquare className="h-4 w-4" />
    }
  }

  const getModelColor = (modelId: string) => {
    switch (modelId) {
      case 'gpt-4': return 'text-green-600 bg-green-50 border-green-200'
      case 'claude-3': return 'text-blue-600 bg-blue-50 border-blue-200'
      case 'gemini': return 'text-purple-600 bg-purple-50 border-purple-200'
      default: return 'text-gray-600 bg-gray-50 border-gray-200'
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-6 w-6 text-primary" />
              Multi-AI Hub DARWIN
            </CardTitle>
            <div className="flex items-center gap-2">
              <Badge variant="secondary" className="flex items-center gap-1">
                <Sparkles className="h-3 w-3" />
                {domains.join(", ")}
              </Badge>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="outline"
                    className="w-full justify-between"
                    disabled={autoSelect}
                  >
                    <span>
                      {autoSelect
                        ? "Auto-Seleção Ativa"
                        : selectedModel
                          ? models.find(m => m.id === selectedModel)?.name || "Selecionar IA"
                          : "Selecionar IA"
                      }
                    </span>
                    <Settings className="h-4 w-4 ml-2" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent className="w-80">
                  {models.map(model => (
                    <DropdownMenuItem
                      key={model.id}
                      onClick={() => setSelectedModel(model.id)}
                      className="flex items-center gap-2 p-3"
                    >
                      {getModelIcon(model.id)}
                      <div className="flex-1">
                        <div className="font-medium">{model.name}</div>
                        <div className="text-xs text-muted-foreground">{model.description}</div>
                      </div>
                      <Badge variant="outline">
                        {Math.round(model.performance_score * 100)}%
                      </Badge>
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
            <Button
              variant={autoSelect ? "default" : "outline"}
              size="sm"
              onClick={() => setAutoSelect(!autoSelect)}
              className="flex items-center gap-1"
            >
              <Settings className="h-4 w-4" />
              Auto-IA
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Messages */}
      <Card className="min-h-[400px]">
        <CardContent className="p-6">
          <div className="space-y-4 mb-4 max-h-96 overflow-y-auto">
            {messages.length === 0 && (
              <div className="text-center text-muted-foreground py-8">
                <Brain className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Converse com múltiplas IAs simultaneamente</p>
                <p className="text-sm">O DARWIN selecionará automaticamente a melhor IA para sua pergunta</p>
              </div>
            )}
            
            <AnimatePresence>
              {messages.map((message) => (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div className={`max-w-[80%] ${
                    message.role === 'user' 
                      ? 'bg-primary text-primary-foreground' 
                      : `border ${message.model ? getModelColor(message.model) : 'bg-muted'}`
                  } rounded-lg p-3`}>
                    {message.role === 'assistant' && message.model && (
                      <div className="flex items-center gap-2 mb-2">
                        {getModelIcon(message.model)}
                        <span className="text-sm font-medium">
                          {models.find(m => m.id === message.model)?.name}
                        </span>
                        {message.processing && (
                          <RefreshCw className="h-3 w-3 animate-spin" />
                        )}
                      </div>
                    )}
                    
                    {message.processing ? (
                      <div className="space-y-2">
                        <Skeleton className="h-4 w-full" />
                        <Skeleton className="h-4 w-3/4" />
                      </div>
                    ) : (
                      <p className="text-sm leading-relaxed">{message.content}</p>
                    )}
                    
                    <div className="text-xs opacity-70 mt-2">
                      {message.timestamp.toLocaleTimeString()}
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>

          {/* Input Form */}
          <form onSubmit={handleSubmit} className="flex gap-2">
            <Textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Faça uma pergunta para o DARWIN Multi-AI Hub..."
              className="min-h-[60px] resize-none"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                  handleSubmit(e)
                }
              }}
            />
            <Button 
              type="submit" 
              disabled={!inputValue.trim() || loading}
              className="px-6"
            >
              {loading ? (
                <RefreshCw className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          </form>
          
          <div className="text-xs text-muted-foreground mt-2 text-center">
            ⌘+Enter para enviar • Auto-IA selecionará o melhor modelo
          </div>
        </CardContent>
      </Card>

      {/* Model Performance */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Performance dos Modelos IA</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {models.map(model => (
              <motion.div
                key={model.id}
                whileHover={{ scale: 1.02 }}
                className={`p-4 rounded-lg border ${getModelColor(model.id)}`}
              >
                <div className="flex items-center gap-2 mb-2">
                  {getModelIcon(model.id)}
                  <span className="font-medium">{model.name}</span>
                  <Badge variant="outline">
                    {Math.round(model.performance_score * 100)}%
                  </Badge>
                </div>
                <p className="text-sm mb-3">{model.description}</p>
                <div className="space-y-1">
                  <div className="text-xs font-medium">Forças:</div>
                  <div className="flex flex-wrap gap-1">
                    {model.strengths.map(strength => (
                      <Badge key={strength} variant="secondary" className="text-xs">
                        {strength}
                      </Badge>
                    ))}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}