"use client"

import * as React from "react"
import { motion, AnimatePresence } from "framer-motion"
import { 
  Telescope, Rss, Bell, Eye, TrendingUp, ExternalLink, 
  Calendar, Filter, Search, Play, Pause, RefreshCw, Zap, Star
} from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"

interface DiscoveryDashboardProps {
  autoRefresh?: boolean
}

interface RSSFeed {
  id: string
  name: string
  domain: string
  url: string
  status: "active" | "error" | "paused"
  last_updated: Date
  articles_count: number
  novelty_score: number
}

interface Discovery {
  id: string
  title: string
  summary: string
  source_feed: string
  novelty_score: number
  cross_domain_connections: string[]
  discovered_at: Date
  keywords: string[]
  url?: string
  impact_score: number
  verification_status: "pending" | "verified" | "disputed"
}

interface Alert {
  id: string
  type: "high_novelty" | "cross_domain" | "trending" | "feed_error"
  title: string
  description: string
  discovery_id?: string
  timestamp: Date
  severity: "low" | "medium" | "high"
  dismissed: boolean
}

const MOCK_RSS_FEEDS: RSSFeed[] = [
  {
    id: "nature_biomat",
    name: "Nature Biomaterials",
    domain: "biomaterials",
    url: "https://nature.com/nmat/rss",
    status: "active",
    last_updated: new Date(),
    articles_count: 847,
    novelty_score: 0.78
  },
  {
    id: "science_materials",
    name: "Science Materials",
    domain: "materials_science",
    url: "https://science.org/materials/rss",
    status: "active", 
    last_updated: new Date(Date.now() - 300000),
    articles_count: 1203,
    novelty_score: 0.82
  },
  {
    id: "biomedicine_today",
    name: "Biomedicine Today",
    domain: "biomedical",
    url: "https://biomedtoday.com/rss",
    status: "error",
    last_updated: new Date(Date.now() - 3600000),
    articles_count: 0,
    novelty_score: 0
  }
]

const MOCK_DISCOVERIES: Discovery[] = [
  {
    id: "disc_001",
    title: "Novel Graphene-Collagen Composite for Tissue Engineering",
    summary: "Researchers developed a new composite material combining graphene and collagen that shows unprecedented biocompatibility and electrical conductivity for neural tissue applications.",
    source_feed: "nature_biomat",
    novelty_score: 0.94,
    cross_domain_connections: ["materials_science", "biomedical", "neuroscience"],
    discovered_at: new Date(Date.now() - 1800000),
    keywords: ["graphene", "collagen", "biocompatibility", "neural tissue"],
    impact_score: 8.7,
    verification_status: "verified",
    url: "https://example.com/discovery1"
  },
  {
    id: "disc_002", 
    title: "3D-Printed Vascular Networks with Self-Healing Properties",
    summary: "Breakthrough in 3D printing technology allows creation of vascular networks that can self-repair minor damage, revolutionizing organ-on-chip applications.",
    source_feed: "science_materials",
    novelty_score: 0.87,
    cross_domain_connections: ["bioengineering", "3d_printing", "vascular_biology"],
    discovered_at: new Date(Date.now() - 7200000),
    keywords: ["3d printing", "vascular", "self-healing", "organ-on-chip"],
    impact_score: 7.9,
    verification_status: "pending"
  }
]

const MOCK_ALERTS: Alert[] = [
  {
    id: "alert_001",
    type: "high_novelty",
    title: "Alta Novelty Score Detectada",
    description: "Discovery sobre Graphene-Collagen alcançou 94% de novelty",
    discovery_id: "disc_001",
    timestamp: new Date(Date.now() - 1200000),
    severity: "high",
    dismissed: false
  },
  {
    id: "alert_002",
    type: "feed_error",
    title: "Feed RSS com Erro",
    description: "Biomedicine Today não responde há 1 hora",
    timestamp: new Date(Date.now() - 3600000),
    severity: "medium",
    dismissed: false
  }
]

export function DiscoveryDashboard({ autoRefresh = true }: DiscoveryDashboardProps) {
  const [feeds, setFeeds] = React.useState<RSSFeed[]>(MOCK_RSS_FEEDS)
  const [discoveries, setDiscoveries] = React.useState<Discovery[]>(MOCK_DISCOVERIES)
  const [alerts, setAlerts] = React.useState<Alert[]>(MOCK_ALERTS)
  const [isMonitoring, setIsMonitoring] = React.useState(true)
  const [searchQuery, setSearchQuery] = React.useState("")
  const [selectedDomains, setSelectedDomains] = React.useState<string[]>([])
  const [selectedTab, setSelectedTab] = React.useState("discoveries")

  // Auto-refresh simulation
  React.useEffect(() => {
    if (!autoRefresh || !isMonitoring) return

    const interval = setInterval(async () => {
      try {
        const response = await fetch('/api/discovery?limit=10')
        if (response.ok) {
          const data = await response.json()
          if (data.discoveries) {
            setDiscoveries(data.discoveries)
          }
          if (data.feeds) {
            setFeeds(data.feeds)
          }
          if (data.alerts) {
            setAlerts(data.alerts)
          }
        }
      } catch (error) {
        console.error('Discovery Dashboard Update Error:', error)
      }
    }, 30000) // Update every 30 seconds

    return () => clearInterval(interval)
  }, [autoRefresh, isMonitoring])

  const filteredDiscoveries = React.useMemo(() => {
    return discoveries.filter(discovery => {
      const matchesSearch = searchQuery === "" || 
        discovery.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        discovery.summary.toLowerCase().includes(searchQuery.toLowerCase()) ||
        discovery.keywords.some(k => k.toLowerCase().includes(searchQuery.toLowerCase()))
      
      const matchesDomain = selectedDomains.length === 0 ||
        discovery.cross_domain_connections.some(d => selectedDomains.includes(d))

      return matchesSearch && matchesDomain
    })
  }, [discoveries, searchQuery, selectedDomains])

  const runDiscovery = async () => {
    try {
      const response = await fetch('/api/discovery', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_once: true }),
      })

      if (response.ok) {
        const result = await response.json()
        console.log('Discovery run initiated:', result)
      }
    } catch (error) {
      console.error('Discovery Run Error:', error)
    }
  }

  const dismissAlert = (alertId: string) => {
    setAlerts(prev => prev.map(alert => 
      alert.id === alertId ? { ...alert, dismissed: true } : alert
    ))
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "active": return "text-green-600 bg-green-50"
      case "error": return "text-red-600 bg-red-50"
      case "paused": return "text-yellow-600 bg-yellow-50"
      default: return "text-gray-600 bg-gray-50"
    }
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "high": return "border-red-500 bg-red-50"
      case "medium": return "border-yellow-500 bg-yellow-50"
      case "low": return "border-blue-500 bg-blue-50"
      default: return "border-gray-500 bg-gray-50"
    }
  }

  const getNoveltyColor = (score: number) => {
    if (score >= 0.8) return "text-green-600 bg-green-100"
    if (score >= 0.6) return "text-yellow-600 bg-yellow-100"
    return "text-red-600 bg-red-100"
  }

  const uniqueDomains = React.useMemo(() => {
    const domains = new Set<string>()
    discoveries.forEach(d => d.cross_domain_connections.forEach(domain => domains.add(domain)))
    feeds.forEach(f => domains.add(f.domain))
    return Array.from(domains)
  }, [discoveries, feeds])

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Telescope className="h-6 w-6 text-primary" />
              Scientific Discovery DARWIN
            </CardTitle>
            
            <div className="flex items-center gap-2">
              <Badge variant={isMonitoring ? "default" : "secondary"} className="flex items-center gap-1">
                <Rss className="h-3 w-3" />
                {feeds.filter(f => f.status === "active").length}/{feeds.length} feeds
              </Badge>
              
              <Badge variant="outline" className="flex items-center gap-1">
                <Bell className="h-3 w-3" />
                {alerts.filter(a => !a.dismissed).length} alerts
              </Badge>
            </div>
          </div>
        </CardHeader>
        
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button
                onClick={runDiscovery}
                className="flex items-center gap-2"
              >
                <Play className="h-4 w-4" />
                Executar Discovery
              </Button>
              
              <Button
                variant="outline"
                onClick={() => setIsMonitoring(!isMonitoring)}
                className="flex items-center gap-2"
              >
                {isMonitoring ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                {isMonitoring ? 'Pausar' : 'Monitorar'}
              </Button>
            </div>

            <div className="flex items-center gap-4">
              {/* Search */}
              <div className="relative">
                <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Buscar discoveries..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10 w-64"
                />
              </div>
              
              {/* Domain Filter */}
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" className="flex items-center gap-2">
                    <Filter className="h-4 w-4" />
                    Domínios ({selectedDomains.length})
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent>
                  {uniqueDomains.map(domain => (
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
                      <span className="capitalize">{domain.replace('_', ' ')}</span>
                      {selectedDomains.includes(domain) && (
                        <div className="h-2 w-2 bg-primary rounded-full" />
                      )}
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Alerts */}
      <AnimatePresence>
        {alerts.filter(a => !a.dismissed).length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <Card className="border-orange-200 bg-orange-50/50">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Bell className="h-5 w-5 text-orange-600" />
                  Alertas Ativos
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {alerts.filter(a => !a.dismissed).slice(0, 3).map(alert => (
                    <motion.div
                      key={alert.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      className={`flex items-center justify-between p-3 border rounded-lg ${getSeverityColor(alert.severity)}`}
                    >
                      <div className="flex items-center gap-3">
                        <div className="flex items-center gap-2">
                          <div className={`w-2 h-2 rounded-full ${
                            alert.severity === 'high' ? 'bg-red-500' : 
                            alert.severity === 'medium' ? 'bg-yellow-500' : 'bg-blue-500'
                          }`} />
                          <span className="font-medium text-sm">{alert.title}</span>
                        </div>
                        <div className="text-sm text-muted-foreground">
                          {alert.description}
                        </div>
                      </div>
                      
                      <div className="flex items-center gap-2">
                        <div className="text-xs text-muted-foreground">
                          {alert.timestamp.toLocaleTimeString()}
                        </div>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => dismissAlert(alert.id)}
                        >
                          ×
                        </Button>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Content */}
      <Tabs value={selectedTab} onValueChange={setSelectedTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="discoveries">Descobertas</TabsTrigger>
          <TabsTrigger value="feeds">RSS Feeds</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
          <TabsTrigger value="timeline">Timeline</TabsTrigger>
        </TabsList>

        <TabsContent value="discoveries" className="space-y-6">
          <div className="grid gap-6">
            {filteredDiscoveries.map((discovery, index) => (
              <motion.div
                key={discovery.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <Card className="hover:shadow-lg transition-all duration-300">
                  <CardHeader className="pb-3">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <CardTitle className="text-lg leading-tight mb-2">
                          {discovery.title}
                        </CardTitle>
                        <div className="flex items-center gap-2 mb-2">
                          <Badge className={`${getNoveltyColor(discovery.novelty_score)} border`}>
                            <Star className="h-3 w-3 mr-1" />
                            {(discovery.novelty_score * 100).toFixed(0)}% novelty
                          </Badge>
                          <Badge variant="outline" className="text-xs">
                            Impact: {discovery.impact_score}
                          </Badge>
                          <Badge variant={discovery.verification_status === "verified" ? "default" : "secondary"}>
                            {discovery.verification_status}
                          </Badge>
                        </div>
                      </div>
                      
                      <div className="flex items-center gap-2">
                        <div className="text-sm text-muted-foreground">
                          {discovery.discovered_at.toLocaleString()}
                        </div>
                        {discovery.url && (
                          <Button size="sm" variant="ghost" asChild>
                            <a href={discovery.url} target="_blank" rel="noopener noreferrer">
                              <ExternalLink className="h-4 w-4" />
                            </a>
                          </Button>
                        )}
                      </div>
                    </div>
                  </CardHeader>
                  
                  <CardContent>
                    <p className="text-sm text-muted-foreground mb-4 leading-relaxed">
                      {discovery.summary}
                    </p>
                    
                    <div className="space-y-3">
                      <div>
                        <div className="text-xs font-medium text-muted-foreground mb-1">
                          Keywords:
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {discovery.keywords.map(keyword => (
                            <Badge key={keyword} variant="secondary" className="text-xs">
                              {keyword}
                            </Badge>
                          ))}
                        </div>
                      </div>
                      
                      <div>
                        <div className="text-xs font-medium text-muted-foreground mb-1">
                          Cross-Domain Connections:
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {discovery.cross_domain_connections.map(domain => (
                            <Badge key={domain} variant="outline" className="text-xs">
                              {domain.replace('_', ' ')}
                            </Badge>
                          ))}
                        </div>
                      </div>
                      
                      <div className="flex items-center justify-between text-xs text-muted-foreground">
                        <span>Fonte: {feeds.find(f => f.id === discovery.source_feed)?.name}</span>
                        <span>{discovery.cross_domain_connections.length} conexões interdisciplinares</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="feeds" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {feeds.map(feed => (
              <Card key={feed.id} className="hover:shadow-md transition-all duration-300">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg">{feed.name}</CardTitle>
                    <Badge className={getStatusColor(feed.status)}>
                      {feed.status}
                    </Badge>
                  </div>
                  <div className="text-sm text-muted-foreground capitalize">
                    {feed.domain.replace('_', ' ')}
                  </div>
                </CardHeader>
                
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Artigos:</span>
                      <span className="font-medium">{feed.articles_count.toLocaleString()}</span>
                    </div>
                    
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Novelty Score:</span>
                      <Badge className={`${getNoveltyColor(feed.novelty_score)} text-xs`}>
                        {(feed.novelty_score * 100).toFixed(0)}%
                      </Badge>
                    </div>
                    
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Última Atualização:</span>
                      <span className="font-mono text-xs">
                        {feed.last_updated.toLocaleTimeString()}
                      </span>
                    </div>
                    
                    <div className="flex items-center gap-2 pt-2">
                      <Button size="sm" variant="outline" className="flex-1">
                        <RefreshCw className="h-4 w-4 mr-1" />
                        Atualizar
                      </Button>
                      <Button size="sm" variant="ghost">
                        <Eye className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Telescope className="h-5 w-5" />
                  Total Discoveries
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-primary">{discoveries.length}</div>
                <div className="text-sm text-muted-foreground">Esta semana</div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-lg flex items-center gap-2">
                  <TrendingUp className="h-5 w-5" />
                  Novelty Média
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-green-600">
                  {discoveries.length > 0 ? 
                    (discoveries.reduce((sum, d) => sum + d.novelty_score, 0) / discoveries.length * 100).toFixed(0) 
                    : 0}%
                </div>
                <div className="text-sm text-muted-foreground">Todas descobertas</div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Zap className="h-5 w-5" />
                  Conexões Cross-Domain
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-blue-600">
                  {discoveries.reduce((sum, d) => sum + d.cross_domain_connections.length, 0)}
                </div>
                <div className="text-sm text-muted-foreground">Conexões ativas</div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Rss className="h-5 w-5" />
                  Feeds Ativos
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-purple-600">
                  {feeds.filter(f => f.status === "active").length}
                </div>
                <div className="text-sm text-muted-foreground">de {feeds.length} feeds</div>
              </CardContent>
            </Card>
          </div>

          {/* Domain Distribution */}
          <Card>
            <CardHeader>
              <CardTitle>Distribuição por Domínio</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {uniqueDomains.map(domain => {
                  const count = discoveries.filter(d => 
                    d.cross_domain_connections.includes(domain)
                  ).length
                  const percentage = discoveries.length > 0 ? (count / discoveries.length) * 100 : 0
                  
                  return (
                    <div key={domain} className="flex items-center gap-4">
                      <div className="w-32 text-sm capitalize">{domain.replace('_', ' ')}</div>
                      <div className="flex-1 bg-muted rounded-full h-2">
                        <div 
                          className="bg-primary h-2 rounded-full transition-all duration-500"
                          style={{ width: `${percentage}%` }}
                        />
                      </div>
                      <div className="text-sm font-mono w-12">{count}</div>
                    </div>
                  )
                })}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="timeline" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Calendar className="h-5 w-5" />
                Timeline de Descobertas
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {discoveries
                  .sort((a, b) => b.discovered_at.getTime() - a.discovered_at.getTime())
                  .map((discovery, index) => (
                    <motion.div
                      key={discovery.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.05 }}
                      className="flex items-start gap-4 pb-4 border-b border-muted/50 last:border-0"
                    >
                      <div className="flex flex-col items-center">
                        <div className={`w-3 h-3 rounded-full ${
                          discovery.novelty_score >= 0.8 ? 'bg-green-500' : 
                          discovery.novelty_score >= 0.6 ? 'bg-yellow-500' : 'bg-blue-500'
                        }`} />
                        {index < discoveries.length - 1 && (
                          <div className="w-px h-8 bg-muted mt-2" />
                        )}
                      </div>
                      
                      <div className="flex-1 min-w-0">
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <h4 className="font-medium text-sm leading-tight mb-1">
                              {discovery.title}
                            </h4>
                            <p className="text-xs text-muted-foreground mb-2 line-clamp-2">
                              {discovery.summary}
                            </p>
                            <div className="flex items-center gap-2">
                              <Badge className={`${getNoveltyColor(discovery.novelty_score)} text-xs`}>
                                {(discovery.novelty_score * 100).toFixed(0)}%
                              </Badge>
                              <div className="text-xs text-muted-foreground">
                                {discovery.cross_domain_connections.length} conexões
                              </div>
                            </div>
                          </div>
                          <div className="text-xs text-muted-foreground ml-4 whitespace-nowrap">
                            {discovery.discovered_at.toLocaleString()}
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}