"use client"

import Link from "next/link"
import { motion } from "framer-motion"
import {
  Brain, Network, BarChart3, Target, Telescope,
  Zap, CheckCircle2, Activity, ArrowRight, Sparkles
} from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"

const DARWIN_FEATURES = [
  {
    id: "multi-ai",
    title: "Multi-AI Hub",
    description: "Chat unificado com ChatGPT-4, Claude-3 e Gemini Pro com seleção automática",
    icon: Brain,
    href: "/darwin/multi-ai",
    color: "text-blue-600 bg-blue-50",
    status: "active",
    performance: "3 IAs ativas"
  },
  {
    id: "knowledge-graph", 
    title: "Knowledge Graph",
    description: "Visualização interdisciplinar com conexões cross-domain em tempo real",
    icon: Network,
    href: "/darwin/knowledge-graph",
    color: "text-green-600 bg-green-50",
    status: "active",
    performance: "347 conceitos"
  },
  {
    id: "kec-metrics",
    title: "KEC Analysis",
    description: "Análise topológica de scaffolds com performance ultra-rápida (<20ms)",
    icon: BarChart3,
    href: "/darwin/kec-metrics", 
    color: "text-purple-600 bg-purple-50",
    status: "active",
    performance: "<20ms analysis"
  },
  {
    id: "tree-search",
    title: "PUCT Optimizer",
    description: "Tree search avançado com 115k nodes/segundo para otimização complexa",
    icon: Target,
    href: "/darwin/tree-search",
    color: "text-orange-600 bg-orange-50", 
    status: "active",
    performance: "115k nodes/s"
  },
  {
    id: "discovery",
    title: "Scientific Discovery",
    description: "Monitoramento de 26 RSS feeds com detecção de novelty automática",
    icon: Telescope,
    href: "/darwin/discovery",
    color: "text-pink-600 bg-pink-50",
    status: "active", 
    performance: "26 feeds RSS"
  }
]

export default function DARWINDashboard() {
  return (
    <div className="space-y-12">
      {/* Epic Hero Section */}
      <div className="text-center space-y-8">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: [0.25, 0.46, 0.45, 0.94] }}
          className="space-y-6"
        >
          <div className="darwin-status-online inline-flex items-center gap-2">
            <Sparkles className="h-4 w-4" />
            DARWIN META-RESEARCH BRAIN · Todas as 9 features ativas
          </div>
          
          <h1 className="darwin-title-epic">
            DARWIN Dashboard
          </h1>
          
          <p className="darwin-subtitle-epic">
            Centro de comando do sistema DARWIN para biomateriais e pesquisa científica.
            Multi-AI Hub, Knowledge Graph, KEC Analysis, PUCT Optimizer e Scientific Discovery
            em uma plataforma integrada épica.
          </p>
        </motion.div>

        {/* Enhanced System Status */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3, duration: 0.6, ease: [0.25, 0.46, 0.45, 0.94] }}
          className="darwin-glass-deep inline-flex items-center gap-6 p-6 rounded-2xl max-w-lg mx-auto"
        >
          <div className="flex items-center gap-3">
            <div className="darwin-neural-node w-3 h-3"></div>
            <span className="font-semibold text-lg">Sistema Online</span>
          </div>
          <div className="darwin-status-processing">
            Backend: Port 8090 ✓
          </div>
        </motion.div>
      </div>

      {/* Features Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {DARWIN_FEATURES.map((feature, index) => (
          <motion.div
            key={feature.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <Card className="group hover:shadow-lg transition-all duration-300 hover:-translate-y-1">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div className={`p-2 rounded-lg ${feature.color}`}>
                    <feature.icon className="h-6 w-6" />
                  </div>
                  <Badge variant={feature.status === "active" ? "default" : "secondary"}>
                    <Activity className="h-3 w-3 mr-1" />
                    {feature.status}
                  </Badge>
                </div>
                <CardTitle className="text-xl">{feature.title}</CardTitle>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {feature.description}
                </p>
              </CardHeader>
              
              <CardContent className="pt-0">
                <div className="space-y-4">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Performance:</span>
                    <span className="font-mono font-medium">{feature.performance}</span>
                  </div>
                  
                  <Button 
                    asChild 
                    className="w-full group-hover:bg-primary/90 transition-colors"
                  >
                    <Link href={feature.href} className="flex items-center justify-center gap-2">
                      Acessar {feature.title}
                      <ArrowRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
                    </Link>
                  </Button>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>

      {/* Performance Stats */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
      >
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="h-5 w-5" />
              Performance DARWIN em Tempo Real
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-6">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">3</div>
                <div className="text-xs text-muted-foreground">IAs Ativas</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">347</div>
                <div className="text-xs text-muted-foreground">Conceitos K-Graph</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">&lt;20</div>
                <div className="text-xs text-muted-foreground">ms KEC Analysis</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-600">115k</div>
                <div className="text-xs text-muted-foreground">nodes/s PUCT</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-pink-600">26</div>
                <div className="text-xs text-muted-foreground">RSS Feeds</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Quick Actions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
      >
        <Card>
          <CardHeader>
            <CardTitle>Ações Rápidas</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-3">
              <Button variant="outline" asChild>
                <Link href="/darwin/multi-ai">
                  <Brain className="h-4 w-4 mr-2" />
                  Nova Consulta Multi-AI
                </Link>
              </Button>
              
              <Button variant="outline" asChild>
                <Link href="/darwin/kec-metrics">
                  <BarChart3 className="h-4 w-4 mr-2" />
                  Analisar Scaffold
                </Link>
              </Button>
              
              <Button variant="outline" asChild>
                <Link href="/darwin/tree-search">
                  <Target className="h-4 w-4 mr-2" />
                  Otimizar PUCT
                </Link>
              </Button>
              
              <Button variant="outline" asChild>
                <Link href="/darwin/discovery">
                  <Telescope className="h-4 w-4 mr-2" />
                  Ver Descobertas
                </Link>
              </Button>
              
              <Button variant="outline" asChild>
                <Link href="/darwin/knowledge-graph">
                  <Network className="h-4 w-4 mr-2" />
                  Explorar K-Graph
                </Link>
              </Button>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  )
}