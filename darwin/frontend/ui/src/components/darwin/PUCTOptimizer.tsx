"use client"

import * as React from "react"
import { motion, AnimatePresence } from "framer-motion"
import { 
  Cpu, Play, Pause, RotateCcw, Settings, TrendingUp, 
  Target, Layers, Zap, Timer, Award, BarChart
} from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

interface PUCTOptimizerProps {
  problemType: "scaffold" | "network" | "mathematical"
  realTimeProgress?: boolean
}

interface PUCTConfig {
  cPuct: number
  budget: number
  explorationWeight: number
  maxDepth: number
  simulationCount: number
  parallelWorkers: number
}

interface TreeNode {
  id: string
  value: number
  visits: number
  ucb_score: number
  children: TreeNode[]
  level: number
  is_expanded: boolean
  best_child?: boolean
}

interface OptimizationResult {
  id: string
  timestamp: Date
  config: PUCTConfig
  final_value: number
  total_nodes: number
  execution_time_ms: number
  nodes_per_second: number
  pareto_solutions: Array<{
    objective1: number
    objective2: number
    parameters: Record<string, number>
  }>
  convergence_data: Array<{
    iteration: number
    best_value: number
  }>
  status: "running" | "completed" | "paused" | "error"
}

const DEFAULT_CONFIG: PUCTConfig = {
  cPuct: 1.414,
  budget: 10000,
  explorationWeight: 0.7,
  maxDepth: 15,
  simulationCount: 1000,
  parallelWorkers: 4
}

const MOCK_TREE: TreeNode = {
  id: "root",
  value: 0.85,
  visits: 1000,
  ucb_score: 1.23,
  level: 0,
  is_expanded: true,
  children: [
    {
      id: "child_1",
      value: 0.82,
      visits: 400,
      ucb_score: 1.15,
      level: 1,
      is_expanded: true,
      best_child: true,
      children: [
        { id: "child_1_1", value: 0.78, visits: 200, ucb_score: 1.05, level: 2, is_expanded: false, children: [] },
        { id: "child_1_2", value: 0.80, visits: 200, ucb_score: 1.12, level: 2, is_expanded: false, children: [] }
      ]
    },
    {
      id: "child_2",
      value: 0.75,
      visits: 600,
      ucb_score: 1.08,
      level: 1,
      is_expanded: true,
      children: [
        { id: "child_2_1", value: 0.72, visits: 300, ucb_score: 0.95, level: 2, is_expanded: false, children: [] },
        { id: "child_2_2", value: 0.77, visits: 300, ucb_score: 1.02, level: 2, is_expanded: false, children: [] }
      ]
    }
  ]
}

export function PUCTOptimizer({ 
  problemType, 
  realTimeProgress = true 
}: PUCTOptimizerProps) {
  const [config, setConfig] = React.useState<PUCTConfig>(DEFAULT_CONFIG)
  const [currentRun, setCurrentRun] = React.useState<OptimizationResult | null>(null)
  const [searchTree, setSearchTree] = React.useState<TreeNode>(MOCK_TREE)
  const [isOptimizing, setIsOptimizing] = React.useState(false)
  const [isPaused, setIsPaused] = React.useState(false)
  const [selectedTab, setSelectedTab] = React.useState("config")
  const [progress, setProgress] = React.useState(0)

  // Real-time optimization progress simulation
  React.useEffect(() => {
    if (!isOptimizing || isPaused) return

    const interval = setInterval(() => {
      setProgress(prev => {
        const newProgress = Math.min(prev + Math.random() * 2, 100)
        
        if (currentRun) {
          setCurrentRun(prevRun => ({
            ...prevRun!,
            total_nodes: Math.floor((newProgress / 100) * config.budget),
            nodes_per_second: 115000 + Math.random() * 5000,
            final_value: 0.6 + (newProgress / 100) * 0.3,
            convergence_data: [
              ...prevRun!.convergence_data,
              {
                iteration: prevRun!.convergence_data.length,
                best_value: 0.6 + (newProgress / 100) * 0.3 + Math.random() * 0.05
              }
            ]
          }))
        }

        if (newProgress >= 100) {
          setIsOptimizing(false)
          if (currentRun) {
            setCurrentRun(prev => ({ ...prev!, status: "completed" }))
          }
        }

        return newProgress
      })
    }, 100)

    return () => clearInterval(interval)
  }, [isOptimizing, isPaused, config.budget, currentRun])

  const startOptimization = async () => {
    setIsOptimizing(true)
    setIsPaused(false)
    setProgress(0)

    const newRun: OptimizationResult = {
      id: `optimization_${Date.now()}`,
      timestamp: new Date(),
      config: { ...config },
      final_value: 0,
      total_nodes: 0,
      execution_time_ms: 0,
      nodes_per_second: 0,
      pareto_solutions: [
        { objective1: 0.85, objective2: 0.78, parameters: { param1: 1.2, param2: 0.8 } },
        { objective1: 0.82, objective2: 0.82, parameters: { param1: 1.1, param2: 0.9 } },
        { objective1: 0.79, objective2: 0.86, parameters: { param1: 1.0, param2: 1.0 } }
      ],
      convergence_data: [],
      status: "running"
    }

    setCurrentRun(newRun)

    try {
      const response = await fetch('/api/tree-search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          problem_type: problemType,
          config: config,
          real_time: realTimeProgress
        }),
      })

      if (response.ok) {
        const result = await response.json()
        setCurrentRun(prev => ({
          ...prev!,
          ...result,
          status: "completed"
        }))
      }
    } catch (error) {
      console.error('PUCT Optimization Error:', error)
      setCurrentRun(prev => ({ ...prev!, status: "error" }))
    }
  }

  const pauseOptimization = () => {
    setIsPaused(true)
    setCurrentRun(prev => prev ? { ...prev, status: "paused" } : null)
  }

  const resumeOptimization = () => {
    setIsPaused(false)
    setCurrentRun(prev => prev ? { ...prev, status: "running" } : null)
  }

  const resetOptimization = () => {
    setIsOptimizing(false)
    setIsPaused(false)
    setProgress(0)
    setCurrentRun(null)
  }

  const renderTreeNode = (node: TreeNode, x: number, y: number, parentX?: number, parentY?: number) => {
    const nodeSize = Math.max(8, Math.min(20, node.visits / 100))
    const color = node.best_child ? "fill-green-500" : 
                  node.ucb_score > 1.1 ? "fill-blue-500" : "fill-gray-400"

    return (
      <g key={node.id}>
        {/* Connection to parent */}
        {parentX !== undefined && parentY !== undefined && (
          <line
            x1={parentX}
            y1={parentY}
            x2={x}
            y2={y}
            stroke="#94a3b8"
            strokeWidth={node.best_child ? 3 : 1}
            className={node.best_child ? "stroke-green-500" : ""}
          />
        )}
        
        {/* Node circle */}
        <circle
          cx={x}
          cy={y}
          r={nodeSize}
          className={`${color} transition-all duration-300 cursor-pointer hover:opacity-80`}
          onClick={() => console.log('Node selected:', node)}
        />
        
        {/* Node label */}
        <text
          x={x}
          y={y - nodeSize - 5}
          textAnchor="middle"
          className="text-xs fill-current font-medium"
        >
          {node.value.toFixed(2)}
        </text>
        
        {/* Visits count */}
        <text
          x={x}
          y={y + nodeSize + 15}
          textAnchor="middle"
          className="text-xs fill-gray-600"
        >
          {node.visits}
        </text>

        {/* Render children */}
        {node.children.map((child, index) => {
          const childX = x + (index - (node.children.length - 1) / 2) * 80
          const childY = y + 80
          return renderTreeNode(child, childX, childY, x, y)
        })}
      </g>
    )
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "running": return "text-blue-600 bg-blue-50"
      case "completed": return "text-green-600 bg-green-50"
      case "paused": return "text-yellow-600 bg-yellow-50"
      case "error": return "text-red-600 bg-red-50"
      default: return "text-gray-600 bg-gray-50"
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Target className="h-6 w-6 text-primary" />
              PUCT Optimizer DARWIN
              <Badge variant="outline" className="ml-2">
                {problemType}
              </Badge>
            </CardTitle>
            
            {currentRun && (
              <div className="flex items-center gap-2">
                <Badge className={getStatusColor(currentRun.status)}>
                  {currentRun.status}
                </Badge>
                <Badge variant="secondary" className="flex items-center gap-1">
                  <Zap className="h-3 w-3" />
                  {currentRun.nodes_per_second.toLocaleString()} nodes/s
                </Badge>
              </div>
            )}
          </div>
        </CardHeader>
        
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button
                onClick={startOptimization}
                disabled={isOptimizing && !isPaused}
                className="flex items-center gap-2"
              >
                <Play className="h-4 w-4" />
                {isOptimizing ? 'Otimizando...' : 'Iniciar Otimização'}
              </Button>
              
              {isOptimizing && (
                <Button
                  variant="outline"
                  onClick={isPaused ? resumeOptimization : pauseOptimization}
                  className="flex items-center gap-2"
                >
                  {isPaused ? <Play className="h-4 w-4" /> : <Pause className="h-4 w-4" />}
                  {isPaused ? 'Continuar' : 'Pausar'}
                </Button>
              )}
              
              <Button
                variant="outline"
                onClick={resetOptimization}
                className="flex items-center gap-2"
              >
                <RotateCcw className="h-4 w-4" />
                Reset
              </Button>
            </div>

            {currentRun && (
              <div className="flex items-center gap-4">
                <div className="text-sm">
                  <span className="text-muted-foreground">Progresso:</span>
                  <span className="ml-2 font-mono">{progress.toFixed(1)}%</span>
                </div>
                <div className="text-sm">
                  <span className="text-muted-foreground">Nodes:</span>
                  <span className="ml-2 font-mono">
                    {currentRun.total_nodes.toLocaleString()}
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Progress Bar */}
          {currentRun && (
            <div className="mt-4">
              <Progress value={progress} className="h-2" />
              <div className="flex justify-between text-xs text-muted-foreground mt-1">
                <span>0</span>
                <span>{config.budget.toLocaleString()} nodes</span>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Main Content Tabs */}
      <Tabs value={selectedTab} onValueChange={setSelectedTab}>
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="config">Configuração</TabsTrigger>
          <TabsTrigger value="tree">Árvore de Busca</TabsTrigger>
          <TabsTrigger value="convergence">Convergência</TabsTrigger>
          <TabsTrigger value="pareto">Pareto Front</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
        </TabsList>

        <TabsContent value="config" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="h-5 w-5" />
                Parâmetros PUCT
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="cPuct">C-PUCT (Exploration)</Label>
                    <div className="mt-2">
                      <Slider
                        value={[config.cPuct]}
                        onValueChange={([value]) => setConfig(prev => ({ ...prev, cPuct: value }))}
                        max={3}
                        min={0.1}
                        step={0.1}
                      />
                      <div className="text-sm text-muted-foreground mt-1">
                        Atual: {config.cPuct.toFixed(1)}
                      </div>
                    </div>
                  </div>

                  <div>
                    <Label htmlFor="budget">Budget (Nodes)</Label>
                    <Input
                      type="number"
                      value={config.budget}
                      onChange={(e) => setConfig(prev => ({ ...prev, budget: parseInt(e.target.value) || 1000 }))}
                      className="mt-1"
                    />
                  </div>

                  <div>
                    <Label htmlFor="explorationWeight">Peso de Exploração</Label>
                    <div className="mt-2">
                      <Slider
                        value={[config.explorationWeight]}
                        onValueChange={([value]) => setConfig(prev => ({ ...prev, explorationWeight: value }))}
                        max={1}
                        min={0}
                        step={0.1}
                      />
                      <div className="text-sm text-muted-foreground mt-1">
                        Atual: {config.explorationWeight.toFixed(1)}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <Label htmlFor="maxDepth">Profundidade Máxima</Label>
                    <Input
                      type="number"
                      value={config.maxDepth}
                      onChange={(e) => setConfig(prev => ({ ...prev, maxDepth: parseInt(e.target.value) || 10 }))}
                      className="mt-1"
                    />
                  </div>

                  <div>
                    <Label htmlFor="simulationCount">Simulações por Node</Label>
                    <Input
                      type="number"
                      value={config.simulationCount}
                      onChange={(e) => setConfig(prev => ({ ...prev, simulationCount: parseInt(e.target.value) || 100 }))}
                      className="mt-1"
                    />
                  </div>

                  <div>
                    <Label htmlFor="parallelWorkers">Workers Paralelos</Label>
                    <Input
                      type="number"
                      value={config.parallelWorkers}
                      onChange={(e) => setConfig(prev => ({ ...prev, parallelWorkers: parseInt(e.target.value) || 1 }))}
                      className="mt-1"
                    />
                  </div>
                </div>
              </div>

              {/* Performance Estimation */}
              <div className="bg-muted/50 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-3">
                  <Cpu className="h-4 w-4" />
                  <span className="font-medium">Estimativa de Performance</span>
                </div>
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <div className="text-muted-foreground">Tempo Estimado</div>
                    <div className="font-mono">
                      {(config.budget / 115000).toFixed(2)}s
                    </div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Throughput</div>
                    <div className="font-mono">115k nodes/s</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Memory Usage</div>
                    <div className="font-mono">
                      ~{Math.round(config.budget * 0.001)}MB
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="tree" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Layers className="h-5 w-5" />
                Visualização da Árvore de Busca
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="w-full h-96 bg-muted/20 rounded-lg p-4 overflow-auto">
                <svg width="600" height="400" className="mx-auto">
                  {renderTreeNode(searchTree, 300, 50)}
                </svg>
              </div>
              
              <div className="mt-4 flex items-center justify-center gap-6 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                  <span>Melhor Caminho</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                  <span>Alto UCB</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-gray-400 rounded-full"></div>
                  <span>Explorado</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="convergence" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5" />
                Curva de Convergência
              </CardTitle>
            </CardHeader>
            <CardContent>
              {currentRun && currentRun.convergence_data.length > 0 ? (
                <div className="w-full h-64 bg-muted/20 rounded-lg p-4 flex items-center justify-center">
                  <div className="text-center text-muted-foreground">
                    <BarChart className="h-12 w-12 mx-auto mb-2 opacity-50" />
                    <div>Gráfico de convergência</div>
                    <div className="text-xs">
                      {currentRun.convergence_data.length} pontos de dados
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center text-muted-foreground py-12">
                  <div>Execute uma otimização para ver a convergência</div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="pareto" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Award className="h-5 w-5" />
                Frente de Pareto
              </CardTitle>
            </CardHeader>
            <CardContent>
              {currentRun?.pareto_solutions.length ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {currentRun.pareto_solutions.map((solution, index) => (
                      <motion.div
                        key={index}
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: index * 0.1 }}
                        className="border rounded-lg p-4 bg-gradient-to-br from-blue-50 to-purple-50"
                      >
                        <div className="flex items-center gap-2 mb-3">
                          <Award className="h-4 w-4 text-gold" />
                          <span className="font-medium">Solução {index + 1}</span>
                        </div>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Obj 1:</span>
                            <span className="font-mono">{solution.objective1.toFixed(3)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Obj 2:</span>
                            <span className="font-mono">{solution.objective2.toFixed(3)}</span>
                          </div>
                          <div className="pt-2 border-t">
                            <div className="text-xs text-muted-foreground mb-1">Parâmetros:</div>
                            {Object.entries(solution.parameters).map(([key, value]) => (
                              <div key={key} className="flex justify-between text-xs">
                                <span>{key}:</span>
                                <span className="font-mono">{value.toFixed(2)}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="text-center text-muted-foreground py-12">
                  Execute uma otimização para ver soluções Pareto-otimais
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="performance" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <Timer className="h-5 w-5" />
                  Tempo de Execução
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center">
                  <div className="text-3xl font-bold text-primary">
                    {currentRun ? (currentRun.execution_time_ms / 1000).toFixed(2) : '0.00'}
                  </div>
                  <div className="text-sm text-muted-foreground">segundos</div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <Zap className="h-5 w-5" />
                  Throughput
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center">
                  <div className="text-3xl font-bold text-green-600">
                    {currentRun ? Math.round(currentRun.nodes_per_second / 1000) : '115'}k
                  </div>
                  <div className="text-sm text-muted-foreground">nodes/segundo</div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <Target className="h-5 w-5" />
                  Melhor Valor
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center">
                  <div className="text-3xl font-bold text-purple-600">
                    {currentRun ? currentRun.final_value.toFixed(3) : '0.000'}
                  </div>
                  <div className="text-sm text-muted-foreground">função objetivo</div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}