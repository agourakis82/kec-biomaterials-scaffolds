"use client"

import * as React from "react"
import { motion, AnimatePresence } from "framer-motion"
import { 
  Activity, Upload, Play, Pause, Download, BarChart3, 
  Zap, Timer, Cpu, TrendingUp, AlertTriangle, CheckCircle2
} from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

interface KECDashboardProps {
  scaffoldData?: any
  realTime?: boolean
}

interface KECMetrics {
  h_spectral: number
  k_forman: number
  sigma: number
  swp: number
  processing_time_ms: number
  topology_complexity: number
  percolation_threshold: number
}

interface AnalysisResult {
  id: string
  timestamp: Date
  metrics: KECMetrics
  scaffold_id: string
  performance_score: number
  optimization_suggestions: string[]
  status: "processing" | "completed" | "error"
}

const MOCK_ANALYSIS_RESULTS: AnalysisResult[] = [
  {
    id: "analysis_1",
    timestamp: new Date(),
    scaffold_id: "scaffold_3d_001",
    performance_score: 0.87,
    status: "completed",
    metrics: {
      h_spectral: 2.34,
      k_forman: 1.67,
      sigma: 0.42,
      swp: 3.21,
      processing_time_ms: 18,
      topology_complexity: 0.76,
      percolation_threshold: 0.59
    },
    optimization_suggestions: [
      "Aumentar conectividade entre poros",
      "Otimizar distribuição de tamanhos",
      "Melhorar simetria estrutural"
    ]
  },
  {
    id: "analysis_2", 
    timestamp: new Date(Date.now() - 120000),
    scaffold_id: "scaffold_3d_002",
    performance_score: 0.92,
    status: "completed",
    metrics: {
      h_spectral: 2.89,
      k_forman: 2.12,
      sigma: 0.38,
      swp: 2.87,
      processing_time_ms: 15,
      topology_complexity: 0.83,
      percolation_threshold: 0.64
    },
    optimization_suggestions: [
      "Estrutura quase ótima",
      "Considerar variações mínimas",
      "Validar experimentalmente"
    ]
  }
]

export function KECDashboard({ realTime = true }: KECDashboardProps) {
  const [analyses, setAnalyses] = React.useState<AnalysisResult[]>(MOCK_ANALYSIS_RESULTS)
  const [currentAnalysis, setCurrentAnalysis] = React.useState<AnalysisResult | null>(null)
  const [isProcessing, setIsProcessing] = React.useState(false)
  const [realTimeEnabled, setRealTimeEnabled] = React.useState(realTime)
  const [selectedTab, setSelectedTab] = React.useState("overview")

  // Simulate real-time metrics updates
  React.useEffect(() => {
    if (!realTimeEnabled || !currentAnalysis) return

    const interval = setInterval(() => {
      setCurrentAnalysis(prev => {
        if (!prev) return null
        
        return {
          ...prev,
          metrics: {
            ...prev.metrics,
            processing_time_ms: Math.max(12, prev.metrics.processing_time_ms + (Math.random() - 0.5) * 2),
            h_spectral: Math.max(0, prev.metrics.h_spectral + (Math.random() - 0.5) * 0.1),
            sigma: Math.max(0, Math.min(1, prev.metrics.sigma + (Math.random() - 0.5) * 0.02))
          }
        }
      })
    }, 1000)

    return () => clearInterval(interval)
  }, [realTimeEnabled, currentAnalysis])

  const startNewAnalysis = async () => {
    setIsProcessing(true)
    
    const newAnalysis: AnalysisResult = {
      id: `analysis_${Date.now()}`,
      timestamp: new Date(),
      scaffold_id: `scaffold_3d_${String(analyses.length + 1).padStart(3, '0')}`,
      performance_score: 0,
      status: "processing",
      metrics: {
        h_spectral: 0,
        k_forman: 0,
        sigma: 0,
        swp: 0,
        processing_time_ms: 0,
        topology_complexity: 0,
        percolation_threshold: 0
      },
      optimization_suggestions: []
    }

    setCurrentAnalysis(newAnalysis)
    setAnalyses(prev => [newAnalysis, ...prev])

    try {
      const response = await fetch('/api/kec-metrics', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scaffold_data: "mock_data",
          real_time: realTimeEnabled,
          analysis_type: "full_topology"
        }),
      })

      if (response.ok) {
        const result = await response.json()
        
        // Update analysis with results
        const completedAnalysis = {
          ...newAnalysis,
          status: "completed" as const,
          performance_score: result.performance_score || 0.85,
          metrics: result.metrics || newAnalysis.metrics,
          optimization_suggestions: result.suggestions || [
            "Análise DARWIN concluída",
            "Verificar sugestões detalhadas"
          ]
        }
        
        setCurrentAnalysis(completedAnalysis)
        setAnalyses(prev => prev.map(a => a.id === newAnalysis.id ? completedAnalysis : a))
      }
    } catch (error) {
      console.error('KEC Analysis Error:', error)
      
      // Update analysis with error
      const errorAnalysis = {
        ...newAnalysis,
        status: "error" as const
      }
      
      setCurrentAnalysis(errorAnalysis)
      setAnalyses(prev => prev.map(a => a.id === newAnalysis.id ? errorAnalysis : a))
    } finally {
      setIsProcessing(false)
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "processing": return <Activity className="h-4 w-4 animate-pulse text-blue-500" />
      case "completed": return <CheckCircle2 className="h-4 w-4 text-green-500" />
      case "error": return <AlertTriangle className="h-4 w-4 text-red-500" />
      default: return <Activity className="h-4 w-4" />
    }
  }

  const getPerformanceColor = (score: number) => {
    if (score >= 0.8) return "text-green-600 bg-green-50"
    if (score >= 0.6) return "text-yellow-600 bg-yellow-50"
    return "text-red-600 bg-red-50"
  }

  const exportResults = () => {
    const dataStr = JSON.stringify(analyses, null, 2)
    const dataBlob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(dataBlob)
    const link = document.createElement('a')
    link.href = url
    link.download = 'kec-analysis-results.json'
    link.click()
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-6 w-6 text-primary" />
              KEC Analysis Dashboard DARWIN
            </CardTitle>
            <div className="flex items-center gap-2">
              <Badge variant={realTimeEnabled ? "default" : "secondary"} className="flex items-center gap-1">
                <Zap className="h-3 w-3" />
                {realTimeEnabled ? "Real-time Ativo" : "Modo Batch"}
              </Badge>
              {currentAnalysis && (
                <Badge variant="outline" className="flex items-center gap-1">
                  <Timer className="h-3 w-3" />
                  {currentAnalysis.metrics.processing_time_ms.toFixed(1)}ms
                </Badge>
              )}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button
                onClick={startNewAnalysis}
                disabled={isProcessing}
                className="flex items-center gap-2"
              >
                {isProcessing ? (
                  <Activity className="h-4 w-4 animate-spin" />
                ) : (
                  <Play className="h-4 w-4" />
                )}
                {isProcessing ? 'Analisando...' : 'Nova Análise'}
              </Button>
              
              <Button
                variant="outline"
                onClick={() => setRealTimeEnabled(!realTimeEnabled)}
                className="flex items-center gap-2"
              >
                {realTimeEnabled ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                Real-time
              </Button>
            </div>

            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" className="flex items-center gap-1">
                <Upload className="h-4 w-4" />
                Upload Data
              </Button>
              
              <Button 
                variant="outline" 
                size="sm" 
                onClick={exportResults}
                className="flex items-center gap-1"
              >
                <Download className="h-4 w-4" />
                Export
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Current Analysis */}
      <AnimatePresence>
        {currentAnalysis && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <Card className="border-primary/50 bg-primary/5">
              <CardHeader className="pb-4">
                <div className="flex items-center justify-between">
                  <CardTitle className="flex items-center gap-2">
                    {getStatusIcon(currentAnalysis.status)}
                    Análise Atual: {currentAnalysis.scaffold_id}
                  </CardTitle>
                  <div className="flex items-center gap-2">
                    {currentAnalysis.status === "completed" && (
                      <Badge className={`${getPerformanceColor(currentAnalysis.performance_score)} border`}>
                        Score: {(currentAnalysis.performance_score * 100).toFixed(1)}%
                      </Badge>
                    )}
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                  <div className="space-y-2">
                    <div className="text-sm font-medium text-muted-foreground">H-Spectral</div>
                    <div className="text-2xl font-bold">
                      {currentAnalysis.metrics.h_spectral.toFixed(2)}
                    </div>
                    <div className="text-xs text-muted-foreground">Conectividade espectral</div>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="text-sm font-medium text-muted-foreground">K-Forman</div>
                    <div className="text-2xl font-bold text-blue-600">
                      {currentAnalysis.metrics.k_forman.toFixed(2)}
                    </div>
                    <div className="text-xs text-muted-foreground">Curvatura de Forman</div>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="text-sm font-medium text-muted-foreground">Sigma (σ)</div>
                    <div className="text-2xl font-bold text-green-600">
                      {currentAnalysis.metrics.sigma.toFixed(3)}
                    </div>
                    <div className="text-xs text-muted-foreground">Coef. agregação</div>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="text-sm font-medium text-muted-foreground">SWP</div>
                    <div className="text-2xl font-bold text-purple-600">
                      {currentAnalysis.metrics.swp.toFixed(2)}
                    </div>
                    <div className="text-xs text-muted-foreground">Small-world param.</div>
                  </div>
                </div>

                {/* Performance Indicators */}
                <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <div className="flex items-center justify-between text-sm mb-2">
                      <span>Complexidade Topológica</span>
                      <span>{(currentAnalysis.metrics.topology_complexity * 100).toFixed(1)}%</span>
                    </div>
                    <Progress value={currentAnalysis.metrics.topology_complexity * 100} />
                  </div>
                  
                  <div>
                    <div className="flex items-center justify-between text-sm mb-2">
                      <span>Limiar de Percolação</span>
                      <span>{(currentAnalysis.metrics.percolation_threshold * 100).toFixed(1)}%</span>
                    </div>
                    <Progress value={currentAnalysis.metrics.percolation_threshold * 100} />
                  </div>
                </div>

                {/* Real-time Performance */}
                <div className="mt-4 flex items-center justify-between bg-muted/50 rounded-lg p-3">
                  <div className="flex items-center gap-2">
                    <Cpu className="h-4 w-4" />
                    <span className="text-sm font-medium">Performance</span>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="text-sm">
                      <span className="text-muted-foreground">Tempo:</span>
                      <span className="ml-1 font-mono">
                        {currentAnalysis.metrics.processing_time_ms.toFixed(1)}ms
                      </span>
                    </div>
                    <Badge variant={currentAnalysis.metrics.processing_time_ms < 20 ? "default" : "secondary"}>
                      {currentAnalysis.metrics.processing_time_ms < 20 ? "Ultra-rápido" : "Normal"}
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Detailed Analysis */}
      <Tabs value={selectedTab} onValueChange={setSelectedTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Visão Geral</TabsTrigger>
          <TabsTrigger value="topology">Topologia 3D</TabsTrigger>
          <TabsTrigger value="optimization">Otimização</TabsTrigger>
          <TabsTrigger value="history">Histórico</TabsTrigger>
        </TabsList>
        
        <TabsContent value="overview" className="space-y-6">
          {/* Metrics Overview */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <TrendingUp className="h-5 w-5" />
                  Métricas Médias
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {analyses.filter(a => a.status === "completed").length > 0 ? (
                    <>
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">H-Spectral</span>
                        <span className="font-medium">
                          {(analyses
                            .filter(a => a.status === "completed")
                            .reduce((sum, a) => sum + a.metrics.h_spectral, 0) / 
                            analyses.filter(a => a.status === "completed").length
                          ).toFixed(2)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">Performance</span>
                        <span className="font-medium">
                          {(analyses
                            .filter(a => a.status === "completed")
                            .reduce((sum, a) => sum + a.performance_score, 0) / 
                            analyses.filter(a => a.status === "completed").length * 100
                          ).toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">Tempo Médio</span>
                        <span className="font-medium">
                          {(analyses
                            .filter(a => a.status === "completed")
                            .reduce((sum, a) => sum + a.metrics.processing_time_ms, 0) / 
                            analyses.filter(a => a.status === "completed").length
                          ).toFixed(1)}ms
                        </span>
                      </div>
                    </>
                  ) : (
                    <div className="text-center text-muted-foreground py-4">
                      Nenhuma análise concluída ainda
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Status do Sistema</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Análises Ativas</span>
                    <Badge variant="default">
                      {analyses.filter(a => a.status === "processing").length}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Concluídas</span>
                    <Badge variant="secondary">
                      {analyses.filter(a => a.status === "completed").length}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Real-time</span>
                    <Badge variant={realTimeEnabled ? "default" : "outline"}>
                      {realTimeEnabled ? "Ativo" : "Inativo"}
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Performance DARWIN</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">
                      {analyses.filter(a => a.status === "completed" && a.metrics.processing_time_ms < 20).length}
                    </div>
                    <div className="text-sm text-muted-foreground">Análises sub-20ms</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-primary">115k</div>
                    <div className="text-sm text-muted-foreground">Nodes/segundo</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="topology" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Visualização Topologia 3D</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="aspect-video bg-muted rounded-lg flex items-center justify-center">
                <div className="text-center text-muted-foreground">
                  <div className="text-lg font-medium mb-2">Visualização 3D</div>
                  <div className="text-sm">Scaffold topology será renderizada aqui</div>
                  <div className="text-xs mt-2">WebGL/Three.js integration</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="optimization" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Sugestões de Otimização</CardTitle>
            </CardHeader>
            <CardContent>
              {currentAnalysis?.optimization_suggestions && currentAnalysis.optimization_suggestions.length > 0 ? (
                <div className="space-y-3">
                  {currentAnalysis.optimization_suggestions.map((suggestion, index) => (
                    <div key={index} className="flex items-start gap-3 p-3 bg-muted/50 rounded-lg">
                      <div className="text-primary font-semibold">{index + 1}.</div>
                      <div className="text-sm">{suggestion}</div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center text-muted-foreground py-8">
                  Execute uma análise para ver sugestões de otimização
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="history" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Histórico de Análises</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {analyses.map((analysis) => (
                  <div
                    key={analysis.id}
                    className="flex items-center justify-between p-3 border rounded-lg hover:bg-muted/50 transition-colors cursor-pointer"
                    onClick={() => setCurrentAnalysis(analysis)}
                  >
                    <div className="flex items-center gap-3">
                      {getStatusIcon(analysis.status)}
                      <div>
                        <div className="font-medium">{analysis.scaffold_id}</div>
                        <div className="text-xs text-muted-foreground">
                          {analysis.timestamp.toLocaleString()}
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-3">
                      {analysis.status === "completed" && (
                        <Badge className={getPerformanceColor(analysis.performance_score)}>
                          {(analysis.performance_score * 100).toFixed(0)}%
                        </Badge>
                      )}
                      <div className="text-sm text-muted-foreground">
                        {analysis.metrics.processing_time_ms.toFixed(1)}ms
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}