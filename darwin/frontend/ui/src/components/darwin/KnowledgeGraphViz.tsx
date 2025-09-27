"use client"

import * as React from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Network, Search, Filter, Zap, Globe, Maximize2, Download, Play, Pause, RefreshCw } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Slider } from "@/components/ui/slider"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"

interface KnowledgeGraphProps {
  domains?: string[]
  interactive?: boolean
  layout?: "force" | "hierarchical" | "domain"
}

interface GraphNode {
  id: string
  label: string
  domain: string
  type: "concept" | "research" | "connection" | "insight"
  weight: number
  connections: number
  x?: number
  y?: number
  discovered_at?: Date
}

interface GraphEdge {
  source: string
  target: string
  weight: number
  type: "research" | "semantic" | "temporal" | "cross_domain"
  strength: number
}

interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
  domains: string[]
  stats: {
    total_concepts: number
    cross_domain_connections: number
    recent_discoveries: number
  }
}

const MOCK_GRAPH_DATA: GraphData = {
  nodes: [
    { id: "biomaterials", label: "Biomateriais", domain: "materials", type: "concept", weight: 0.9, connections: 12 },
    { id: "scaffolds", label: "Scaffolds", domain: "materials", type: "concept", weight: 0.8, connections: 8 },
    { id: "tissue_eng", label: "Eng. Tecidos", domain: "biomedical", type: "concept", weight: 0.85, connections: 15 },
    { id: "porosity", label: "Porosidade", domain: "physics", type: "research", weight: 0.7, connections: 6 },
    { id: "biocompat", label: "Biocompatibilidade", domain: "biomedical", type: "research", weight: 0.75, connections: 10 },
    { id: "polymer", label: "Polímeros", domain: "chemistry", type: "concept", weight: 0.8, connections: 9 },
    { id: "collagen", label: "Colágeno", domain: "biology", type: "research", weight: 0.72, connections: 7 },
    { id: "3d_printing", label: "Impressão 3D", domain: "technology", type: "insight", weight: 0.88, connections: 11 },
  ],
  edges: [
    { source: "biomaterials", target: "scaffolds", weight: 0.9, type: "research", strength: 0.95 },
    { source: "scaffolds", target: "tissue_eng", weight: 0.85, type: "cross_domain", strength: 0.8 },
    { source: "scaffolds", target: "porosity", weight: 0.7, type: "research", strength: 0.75 },
    { source: "tissue_eng", target: "biocompat", weight: 0.8, type: "semantic", strength: 0.85 },
    { source: "biomaterials", target: "polymer", weight: 0.75, type: "cross_domain", strength: 0.7 },
    { source: "collagen", target: "biocompat", weight: 0.6, type: "research", strength: 0.65 },
    { source: "3d_printing", target: "scaffolds", weight: 0.82, type: "temporal", strength: 0.9 },
  ],
  domains: ["materials", "biomedical", "physics", "chemistry", "biology", "technology"],
  stats: {
    total_concepts: 347,
    cross_domain_connections: 89,
    recent_discoveries: 23
  }
}

const DOMAIN_COLORS = {
  materials: "bg-blue-100 border-blue-300 text-blue-800",
  biomedical: "bg-green-100 border-green-300 text-green-800",
  physics: "bg-purple-100 border-purple-300 text-purple-800",
  chemistry: "bg-orange-100 border-orange-300 text-orange-800",
  biology: "bg-pink-100 border-pink-300 text-pink-800",
  technology: "bg-cyan-100 border-cyan-300 text-cyan-800",
}

export function KnowledgeGraphViz({ 
  domains = ["biomaterials"], 
  interactive = true, 
  layout = "force" 
}: KnowledgeGraphProps) {
  const [graphData, setGraphData] = React.useState<GraphData>(MOCK_GRAPH_DATA)
  const [searchQuery, setSearchQuery] = React.useState("")
  const [selectedDomains, setSelectedDomains] = React.useState<string[]>(domains)
  const [currentLayout, setCurrentLayout] = React.useState(layout)
  const [zoomLevel, setZoomLevel] = React.useState([1])
  const [isAnimating, setIsAnimating] = React.useState(false)
  const [selectedNode, setSelectedNode] = React.useState<GraphNode | null>(null)
  const [loading, setLoading] = React.useState(false)

  // Filter nodes based on search and domain selection
  const filteredNodes = React.useMemo(() => {
    return graphData.nodes.filter(node => {
      const matchesSearch = searchQuery === "" || 
        node.label.toLowerCase().includes(searchQuery.toLowerCase())
      const matchesDomain = selectedDomains.length === 0 || 
        selectedDomains.includes(node.domain)
      return matchesSearch && matchesDomain
    })
  }, [graphData.nodes, searchQuery, selectedDomains])

  // Calculate node positions based on layout
  const getNodePosition = React.useCallback((node: GraphNode, index: number) => {
    const centerX = 250
    const centerY = 200
    const radius = 150

    switch (currentLayout) {
      case "force":
        // Simulate force-directed layout
        const angle = (index / filteredNodes.length) * 2 * Math.PI
        return {
          x: centerX + Math.cos(angle) * radius * node.weight,
          y: centerY + Math.sin(angle) * radius * node.weight
        }
      
      case "hierarchical":
        // Hierarchical layout based on connections
        const level = Math.floor(node.connections / 5)
        const position = index % 4
        return {
          x: 100 + position * 100,
          y: 50 + level * 80
        }
      
      case "domain":
        // Group by domains
        const domainIndex = graphData.domains.indexOf(node.domain)
        const domainAngle = (domainIndex / graphData.domains.length) * 2 * Math.PI
        const nodeInDomain = filteredNodes.filter(n => n.domain === node.domain).indexOf(node)
        return {
          x: centerX + Math.cos(domainAngle) * 120 + (nodeInDomain % 3 - 1) * 30,
          y: centerY + Math.sin(domainAngle) * 120 + Math.floor(nodeInDomain / 3) * 25
        }
      
      default:
        return { x: centerX, y: centerY }
    }
  }, [currentLayout, filteredNodes, graphData.domains])

  const loadGraphData = async () => {
    setLoading(true)
    try {
      const response = await fetch(`/api/knowledge-graph?domain=${selectedDomains.join(',')}`)
      if (response.ok) {
        const data = await response.json()
        setGraphData(data)
      }
    } catch (error) {
      console.error('Knowledge Graph Load Error:', error)
    } finally {
      setLoading(false)
    }
  }

  const toggleAnimation = () => {
    setIsAnimating(!isAnimating)
  }

  const exportGraph = () => {
    const dataStr = JSON.stringify(graphData, null, 2)
    const dataBlob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(dataBlob)
    const link = document.createElement('a')
    link.href = url
    link.download = 'knowledge-graph.json'
    link.click()
  }

  const getEdgeColor = (type: string) => {
    switch (type) {
      case "cross_domain": return "stroke-purple-500"
      case "research": return "stroke-blue-500"
      case "semantic": return "stroke-green-500"
      case "temporal": return "stroke-orange-500"
      default: return "stroke-gray-400"
    }
  }

  return (
    <div className="space-y-6">
      {/* Controls */}
      <Card>
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Network className="h-6 w-6 text-primary" />
              Knowledge Graph DARWIN
            </CardTitle>
            <div className="flex items-center gap-2">
              <Badge variant="secondary" className="flex items-center gap-1">
                <Globe className="h-3 w-3" />
                {graphData.stats.total_concepts} conceitos
              </Badge>
              <Badge variant="outline" className="flex items-center gap-1">
                <Zap className="h-3 w-3" />
                {graphData.stats.cross_domain_connections} conexões
              </Badge>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Search and Filters */}
          <div className="flex gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Buscar conceitos..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>
            
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" className="flex items-center gap-2">
                  <Filter className="h-4 w-4" />
                  Domínios ({selectedDomains.length})
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent>
                {graphData.domains.map(domain => (
                  <DropdownMenuItem
                    key={domain}
                    onClick={() => {
                      setSelectedDomains(prev => 
                        prev.includes(domain) 
                          ? prev.filter(d => d !== domain)
                          : [...prev, domain]
                      )
                    }}
                    className="flex items-center justify-between"
                  >
                    <span className="capitalize">{domain}</span>
                    {selectedDomains.includes(domain) && (
                      <div className="h-2 w-2 bg-primary rounded-full" />
                    )}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline">
                  Layout: {currentLayout}
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent>
                <DropdownMenuItem onClick={() => setCurrentLayout("force")}>
                  Force-Directed
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => setCurrentLayout("hierarchical")}>
                  Hierárquico
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => setCurrentLayout("domain")}>
                  Por Domínio
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>

          {/* Controls Row */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <span className="text-sm">Zoom:</span>
                <Slider
                  value={zoomLevel}
                  onValueChange={setZoomLevel}
                  max={3}
                  min={0.5}
                  step={0.1}
                  className="w-20"
                />
              </div>
              
              <Button
                variant="outline"
                size="sm"
                onClick={toggleAnimation}
                className="flex items-center gap-1"
              >
                {isAnimating ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                {isAnimating ? 'Pausar' : 'Animar'}
              </Button>
            </div>

            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={loadGraphData}
                disabled={loading}
                className="flex items-center gap-1"
              >
                <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                Atualizar
              </Button>
              
              <Button
                variant="outline"
                size="sm"
                onClick={exportGraph}
                className="flex items-center gap-1"
              >
                <Download className="h-4 w-4" />
                Exportar
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Graph Visualization */}
      <Card className="min-h-[500px]">
        <CardContent className="p-6">
          <div className="relative w-full h-96 bg-gray-50 rounded-lg overflow-hidden">
            <svg
              width="100%"
              height="100%"
              viewBox="0 0 500 400"
              style={{ transform: `scale(${zoomLevel[0]})` }}
              className="transition-transform duration-300"
            >
              {/* Draw Edges */}
              {graphData.edges.map((edge, index) => {
                const sourceNode = filteredNodes.find(n => n.id === edge.source)
                const targetNode = filteredNodes.find(n => n.id === edge.target)
                
                if (!sourceNode || !targetNode) return null
                
                const sourcePos = getNodePosition(sourceNode, filteredNodes.indexOf(sourceNode))
                const targetPos = getNodePosition(targetNode, filteredNodes.indexOf(targetNode))
                
                return (
                  <motion.line
                    key={`edge-${index}`}
                    x1={sourcePos.x}
                    y1={sourcePos.y}
                    x2={targetPos.x}
                    y2={targetPos.y}
                    className={`${getEdgeColor(edge.type)} transition-all duration-500`}
                    strokeWidth={edge.strength * 3}
                    strokeOpacity={0.6}
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: 1, delay: index * 0.1 }}
                  />
                )
              })}
              
              {/* Draw Nodes */}
              <AnimatePresence>
                {filteredNodes.map((node, index) => {
                  const position = getNodePosition(node, index)
                  const isSelected = selectedNode?.id === node.id
                  
                  return (
                    <motion.g
                      key={node.id}
                      initial={{ scale: 0, opacity: 0 }}
                      animate={{ 
                        scale: 1, 
                        opacity: 1,
                        x: position.x,
                        y: position.y
                      }}
                      exit={{ scale: 0, opacity: 0 }}
                      transition={{ 
                        duration: 0.5, 
                        delay: index * 0.05,
                        type: "spring"
                      }}
                      className="cursor-pointer"
                      onClick={() => setSelectedNode(isSelected ? null : node)}
                      whileHover={{ scale: 1.1 }}
                    >
                      {/* Node Circle */}
                      <circle
                        r={node.weight * 20 + 10}
                        className={`${DOMAIN_COLORS[node.domain as keyof typeof DOMAIN_COLORS] || 'bg-gray-100'} 
                          transition-all duration-300 ${isSelected ? 'stroke-primary stroke-2' : 'stroke-gray-300 stroke-1'}`}
                        fill="currentColor"
                        fillOpacity={0.8}
                      />
                      
                      {/* Node Label */}
                      <text
                        textAnchor="middle"
                        dominantBaseline="middle"
                        className="text-xs font-medium fill-current pointer-events-none"
                      >
                        {node.label}
                      </text>
                      
                      {/* Connection Count Badge */}
                      <circle
                        cx={node.weight * 15 + 5}
                        cy={-(node.weight * 15 + 5)}
                        r={8}
                        className="fill-primary"
                      />
                      <text
                        x={node.weight * 15 + 5}
                        y={-(node.weight * 15 + 5)}
                        textAnchor="middle"
                        dominantBaseline="middle"
                        className="text-xs fill-white font-bold pointer-events-none"
                      >
                        {node.connections}
                      </text>
                    </motion.g>
                  )
                })}
              </AnimatePresence>
            </svg>
            
            {/* Legend */}
            <div className="absolute top-4 left-4 bg-white/90 backdrop-blur-sm rounded-lg p-3 space-y-2">
              <div className="text-xs font-medium">Domínios:</div>
              {graphData.domains.slice(0, 4).map(domain => (
                <div key={domain} className="flex items-center gap-2 text-xs">
                  <div className={`w-3 h-3 rounded-full ${DOMAIN_COLORS[domain as keyof typeof DOMAIN_COLORS]?.split(' ')[0]} border`} />
                  <span className="capitalize">{domain}</span>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Node Details */}
      <AnimatePresence>
        {selectedNode && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <div className={`w-4 h-4 rounded-full ${DOMAIN_COLORS[selectedNode.domain as keyof typeof DOMAIN_COLORS]}`} />
                  {selectedNode.label}
                  <Badge variant="outline" className="ml-2">
                    {selectedNode.type}
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <div className="text-sm font-medium text-muted-foreground">Domínio</div>
                    <div className="capitalize">{selectedNode.domain}</div>
                  </div>
                  <div>
                    <div className="text-sm font-medium text-muted-foreground">Conexões</div>
                    <div>{selectedNode.connections} links</div>
                  </div>
                  <div>
                    <div className="text-sm font-medium text-muted-foreground">Peso</div>
                    <div>{(selectedNode.weight * 100).toFixed(1)}%</div>
                  </div>
                </div>
                
                <div className="mt-4">
                  <div className="text-sm font-medium text-muted-foreground mb-2">Conexões Relacionadas</div>
                  <div className="flex flex-wrap gap-1">
                    {graphData.edges
                      .filter(edge => edge.source === selectedNode.id || edge.target === selectedNode.id)
                      .map((edge, index) => {
                        const connectedNodeId = edge.source === selectedNode.id ? edge.target : edge.source
                        const connectedNode = graphData.nodes.find(n => n.id === connectedNodeId)
                        return connectedNode ? (
                          <Badge key={index} variant="secondary" className="text-xs">
                            {connectedNode.label}
                          </Badge>
                        ) : null
                      })
                    }
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Stats Panel */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Estatísticas do Grafo</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-primary">{filteredNodes.length}</div>
              <div className="text-sm text-muted-foreground">Conceitos Visíveis</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{graphData.stats.cross_domain_connections}</div>
              <div className="text-sm text-muted-foreground">Conexões Interdisciplinares</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">{graphData.stats.recent_discoveries}</div>
              <div className="text-sm text-muted-foreground">Descobertas Recentes</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}