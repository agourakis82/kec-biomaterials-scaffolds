"use client"

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Brain, 
  Zap, 
  MessageSquare, 
  BarChart3, 
  Settings,
  Menu,
  X,
  Bell,
  Search,
  User,
  LogOut,
  Activity,
  TrendingUp,
  Users,
  Database,
  Cpu,
  Network,
  Play,
  Pause,
  RefreshCw,
  ChevronRight,
  Sparkles,
  Target,
  Rocket,
  Shield
} from 'lucide-react'
import { cn, formatNumber, formatDuration, apiRequest } from '@/lib/utils'
import BackendStatus from './BackendStatus'

interface User {
  id: string
  name: string
  email: string
  role: 'admin' | 'researcher' | 'guest'
  avatar?: string
  permissions: string[]
}

interface ModernDashboardProps {
  user: User
  onLogout: () => void
}

type Section = 'overview' | 'research' | 'performance' | 'chat' | 'analytics' | 'settings'

const NAVIGATION_ITEMS = [
  {
    id: 'overview' as Section,
    label: 'Dashboard',
    icon: <BarChart3 className="w-5 h-5" />,
    description: 'Visão geral do sistema',
    color: 'from-blue-500 to-cyan-500',
    instructions: 'Monitore métricas em tempo real e status do sistema'
  },
  {
    id: 'research' as Section,
    label: 'Pesquisa IA',
    icon: <Brain className="w-5 h-5" />,
    description: 'Multi-Agent Research',
    color: 'from-purple-500 to-pink-500',
    instructions: 'Inicie pesquisas colaborativas com múltiplos agentes de IA'
  },
  {
    id: 'performance' as Section,
    label: 'Performance',
    icon: <Zap className="w-5 h-5" />,
    description: 'JAX Ultra-Performance',
    color: 'from-green-500 to-emerald-500',
    instructions: 'Monitore performance JAX e otimizações em tempo real'
  },
  {
    id: 'chat' as Section,
    label: 'Chat IA',
    icon: <MessageSquare className="w-5 h-5" />,
    description: 'Conversas Inteligentes',
    color: 'from-orange-500 to-red-500',
    instructions: 'Converse com agentes especializados em diferentes domínios'
  },
  {
    id: 'analytics' as Section,
    label: 'Analytics',
    icon: <TrendingUp className="w-5 h-5" />,
    description: 'Análise Avançada',
    color: 'from-indigo-500 to-purple-500',
    instructions: 'Visualize insights e padrões nos dados de pesquisa'
  },
  {
    id: 'settings' as Section,
    label: 'Configurações',
    icon: <Settings className="w-5 h-5" />,
    description: 'Sistema e Perfil',
    color: 'from-gray-500 to-slate-500',
    instructions: 'Configure preferências e parâmetros do sistema'
  }
]

const MOCK_METRICS = {
  activeAgents: 8,
  totalQueries: 1247,
  avgResponseTime: 0.8,
  successRate: 98.5,
  systemLoad: 45,
  memoryUsage: 67,
  processingTasks: 12,
  completedToday: 89
}

export const ModernDashboard: React.FC<ModernDashboardProps> = ({
  user,
  onLogout
}) => {
  const [activeSection, setActiveSection] = useState<Section>('overview')
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [metrics, setMetrics] = useState(MOCK_METRICS)
  const [systemStatus, setSystemStatus] = useState<'online' | 'processing' | 'maintenance'>('online')

  // Simulate real-time metrics updates
  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics(prev => ({
        ...prev,
        systemLoad: Math.max(20, Math.min(80, prev.systemLoad + (Math.random() - 0.5) * 10)),
        memoryUsage: Math.max(40, Math.min(90, prev.memoryUsage + (Math.random() - 0.5) * 5)),
        avgResponseTime: Math.max(0.3, Math.min(2.0, prev.avgResponseTime + (Math.random() - 0.5) * 0.2)),
        processingTasks: Math.max(0, Math.min(25, prev.processingTasks + Math.floor((Math.random() - 0.5) * 3)))
      }))
    }, 3000)

    return () => clearInterval(interval)
  }, [])

  const handleLogout = async () => {
    try {
      await apiRequest('/api/auth/logout', { method: 'POST' })
    } catch (error) {
      console.error('Logout error:', error)
    } finally {
      onLogout()
    }
  }

  const renderOverview = () => (
    <div className="space-y-8">
      {/* Welcome Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card p-8"
      >
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold gradient-text mb-2">
              Bem-vindo, {user.name.split(' ')[0]}!
            </h1>
            <p className="text-muted-foreground text-lg">
              Sua plataforma de pesquisa inteligente está operacional
            </p>
          </div>
          <div className="flex items-center gap-4">
            <BackendStatus />
            <div className={cn(
              "status-online",
              systemStatus === 'processing' && "status-processing",
              systemStatus === 'maintenance' && "status-error"
            )}>
              <Activity className="w-4 h-4" />
              {systemStatus === 'online' && 'Sistema Online'}
              {systemStatus === 'processing' && 'Processando'}
              {systemStatus === 'maintenance' && 'Manutenção'}
            </div>
            <button className="btn-primary flex items-center gap-2">
              <Rocket className="w-4 h-4" />
              Iniciar Pesquisa
            </button>
          </div>
        </div>
      </motion.div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {[
          {
            title: 'Nova Pesquisa',
            description: 'Inicie uma pesquisa colaborativa',
            icon: <Brain className="w-8 h-8" />,
            color: 'from-purple-500 to-pink-500',
            action: () => setActiveSection('research')
          },
          {
            title: 'Chat IA',
            description: 'Converse com agentes especializados',
            icon: <MessageSquare className="w-8 h-8" />,
            color: 'from-orange-500 to-red-500',
            action: () => setActiveSection('chat')
          },
          {
            title: 'Ver Analytics',
            description: 'Analise dados e insights',
            icon: <TrendingUp className="w-8 h-8" />,
            color: 'from-indigo-500 to-purple-500',
            action: () => setActiveSection('analytics')
          }
        ].map((action, index) => (
          <motion.button
            key={action.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            onClick={action.action}
            className="modern-card text-left group"
          >
            <div className={cn(
              "w-16 h-16 rounded-2xl bg-gradient-to-r flex items-center justify-center mb-4 text-white",
              action.color
            )}>
              {action.icon}
            </div>
            <h3 className="text-xl font-bold mb-2 group-hover:gradient-text transition-all">
              {action.title}
            </h3>
            <p className="text-muted-foreground mb-4">{action.description}</p>
            <div className="flex items-center text-primary font-medium">
              Acessar <ChevronRight className="w-4 h-4 ml-1" />
            </div>
          </motion.button>
        ))}
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {[
          {
            label: 'Agentes Ativos',
            value: metrics.activeAgents,
            icon: <Users className="w-6 h-6" />,
            color: 'text-blue-500',
            change: '+2',
            trend: 'up'
          },
          {
            label: 'Consultas Hoje',
            value: formatNumber(metrics.totalQueries),
            icon: <Database className="w-6 h-6" />,
            color: 'text-green-500',
            change: '+15%',
            trend: 'up'
          },
          {
            label: 'Tempo Resposta',
            value: `${metrics.avgResponseTime.toFixed(1)}s`,
            icon: <Zap className="w-6 h-6" />,
            color: 'text-yellow-500',
            change: '-0.2s',
            trend: 'down'
          },
          {
            label: 'Taxa Sucesso',
            value: `${metrics.successRate}%`,
            icon: <Target className="w-6 h-6" />,
            color: 'text-purple-500',
            change: '+0.5%',
            trend: 'up'
          }
        ].map((metric, index) => (
          <motion.div
            key={metric.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 + index * 0.1 }}
            className="modern-card"
          >
            <div className="flex items-center justify-between mb-4">
              <div className={cn("p-3 rounded-xl bg-muted/50", metric.color)}>
                {metric.icon}
              </div>
              <span className={cn(
                "text-sm font-semibold px-2 py-1 rounded-full",
                metric.trend === 'up' ? 'text-green-600 bg-green-100 dark:bg-green-900 dark:text-green-400' : 'text-blue-600 bg-blue-100 dark:bg-blue-900 dark:text-blue-400'
              )}>
                {metric.change}
              </span>
            </div>
            <div className="text-3xl font-bold mb-2">{metric.value}</div>
            <div className="text-muted-foreground font-medium">{metric.label}</div>
          </motion.div>
        ))}
      </div>

      {/* System Status */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.7 }}
          className="glass-card p-8"
        >
          <h3 className="text-2xl font-bold mb-6 flex items-center gap-3">
            <Cpu className="w-7 h-7 text-blue-500" />
            Status do Sistema
          </h3>
          <div className="space-y-6">
            <div>
              <div className="flex justify-between text-lg font-medium mb-3">
                <span>CPU</span>
                <span>{metrics.systemLoad}%</span>
              </div>
              <div className="w-full bg-muted rounded-full h-3">
                <motion.div
                  className="bg-gradient-to-r from-blue-500 to-cyan-500 h-3 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${metrics.systemLoad}%` }}
                  transition={{ duration: 0.5 }}
                />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-lg font-medium mb-3">
                <span>Memória</span>
                <span>{metrics.memoryUsage}%</span>
              </div>
              <div className="w-full bg-muted rounded-full h-3">
                <motion.div
                  className="bg-gradient-to-r from-green-500 to-emerald-500 h-3 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${metrics.memoryUsage}%` }}
                  transition={{ duration: 0.5 }}
                />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-lg font-medium mb-3">
                <span>Tarefas Ativas</span>
                <span>{metrics.processingTasks}</span>
              </div>
              <div className="flex items-center gap-2">
                <RefreshCw className="w-4 h-4 animate-spin text-blue-500" />
                <span className="text-muted-foreground">Processando em tempo real</span>
              </div>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.8 }}
          className="glass-card p-8"
        >
          <h3 className="text-2xl font-bold mb-6 flex items-center gap-3">
            <Network className="w-7 h-7 text-purple-500" />
            Atividade Recente
          </h3>
          <div className="space-y-4">
            {[
              { action: 'Pesquisa sobre biomateriais concluída', time: '2 min atrás', status: 'success', icon: <Brain className="w-4 h-4" /> },
              { action: 'Agente de análise iniciado', time: '5 min atrás', status: 'info', icon: <Zap className="w-4 h-4" /> },
              { action: 'Chat com especialista finalizado', time: '8 min atrás', status: 'success', icon: <MessageSquare className="w-4 h-4" /> },
              { action: 'Backup automático realizado', time: '15 min atrás', status: 'info', icon: <Shield className="w-4 h-4" /> }
            ].map((activity, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.9 + index * 0.1 }}
                className="flex items-center gap-4 p-4 rounded-xl bg-muted/30 hover:bg-muted/50 transition-colors"
              >
                <div className={cn(
                  "p-2 rounded-lg",
                  activity.status === 'success' ? 'bg-green-100 text-green-600 dark:bg-green-900 dark:text-green-400' : 'bg-blue-100 text-blue-600 dark:bg-blue-900 dark:text-blue-400'
                )}>
                  {activity.icon}
                </div>
                <div className="flex-1">
                  <div className="font-medium">{activity.action}</div>
                  <div className="text-sm text-muted-foreground">{activity.time}</div>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  )

  const renderSection = () => {
    const currentItem = NAVIGATION_ITEMS.find(item => item.id === activeSection)
    
    switch (activeSection) {
      case 'overview':
        return renderOverview()
      case 'research':
        return (
          <div className="space-y-8">
            <div className="glass-card p-8 text-center">
              <div className="w-24 h-24 bg-gradient-to-r from-purple-500 to-pink-500 rounded-3xl flex items-center justify-center mx-auto mb-6">
                <Brain className="w-12 h-12 text-white" />
              </div>
              <h2 className="text-3xl font-bold mb-4">Multi-Agent Research</h2>
              <p className="text-muted-foreground text-lg mb-8 max-w-2xl mx-auto">
                Sistema de pesquisa colaborativa com múltiplos agentes de IA especializados em diferentes domínios científicos
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <button className="btn-primary text-lg px-8 py-4">
                  <Play className="w-5 h-5 mr-2" />
                  Iniciar Nova Pesquisa
                </button>
                <button className="btn-secondary text-lg px-8 py-4">
                  Ver Pesquisas Anteriores
                </button>
              </div>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {[
                { name: 'Agente Biomateriais', status: 'online', specialty: 'Materiais Avançados' },
                { name: 'Agente Química', status: 'online', specialty: 'Síntese Molecular' },
                { name: 'Agente Física', status: 'processing', specialty: 'Mecânica Quântica' },
                { name: 'Agente Medicina', status: 'online', specialty: 'Pesquisa Clínica' },
                { name: 'Agente Dados', status: 'online', specialty: 'Análise Estatística' },
                { name: 'Agente Literatura', status: 'online', specialty: 'Revisão Sistemática' }
              ].map((agent, index) => (
                <motion.div
                  key={agent.name}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="modern-card"
                >
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="font-bold text-lg">{agent.name}</h3>
                    <div className={cn(
                      "w-3 h-3 rounded-full",
                      agent.status === 'online' ? 'bg-green-500' : 'bg-yellow-500 animate-pulse'
                    )} />
                  </div>
                  <p className="text-muted-foreground mb-4">{agent.specialty}</p>
                  <button className="w-full btn-secondary">
                    Conversar
                  </button>
                </motion.div>
              ))}
            </div>
          </div>
        )
      default:
        return (
          <div className="glass-card p-12 text-center">
            <div className={cn(
              "w-24 h-24 rounded-3xl flex items-center justify-center mx-auto mb-6 bg-gradient-to-r text-white",
              currentItem?.color
            )}>
              {currentItem?.icon && React.cloneElement(currentItem.icon, { className: "w-12 h-12" })}
            </div>
            <h2 className="text-3xl font-bold mb-4">{currentItem?.label}</h2>
            <p className="text-muted-foreground text-lg mb-6 max-w-2xl mx-auto">
              {currentItem?.instructions}
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button className="btn-primary text-lg px-8 py-4">
                <Sparkles className="w-5 h-5 mr-2" />
                Começar Agora
              </button>
              <button className="btn-secondary text-lg px-8 py-4">
                Ver Documentação
              </button>
            </div>
          </div>
        )
    }
  }

  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.aside
            initial={{ x: -400, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -400, opacity: 0 }}
            transition={{ type: "spring", damping: 25, stiffness: 200 }}
            className="w-80 glass-card border-r border-border/50 flex flex-col"
          >
            {/* Sidebar Header */}
            <div className="p-6 border-b border-border/50">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-500 rounded-2xl flex items-center justify-center">
                    <Brain className="w-7 h-7 text-white" />
                  </div>
                  <div>
                    <h2 className="font-bold text-xl">DARWIN AI</h2>
                    <p className="text-sm text-muted-foreground">Research Platform</p>
                  </div>
                </div>
                <button
                  onClick={() => setSidebarOpen(false)}
                  className="w-10 h-10 rounded-xl glass-button flex items-center justify-center"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              
              {/* Search */}
              <div className="relative">
                <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                <input
                  type="text"
                  placeholder="Buscar funcionalidades..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="form-input pl-12 h-12"
                />
              </div>
            </div>

            {/* Navigation */}
            <nav className="flex-1 p-6 space-y-3">
              {NAVIGATION_ITEMS.map((item) => (
                <motion.button
                  key={item.id}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => setActiveSection(item.id)}
                  className={cn(
                    "w-full p-4 rounded-2xl text-left transition-all duration-300",
                    activeSection === item.id
                      ? "glass-card shadow-lg scale-[1.02]"
                      : "hover:bg-muted/30"
                  )}
                >
                  <div className="flex items-center gap-4">
                    <div className={cn(
                      "w-12 h-12 rounded-xl flex items-center justify-center",
                      activeSection === item.id
                        ? `bg-gradient-to-r ${item.color} text-white shadow-lg`
                        : "bg-muted/50 text-muted-foreground"
                    )}>
                      {item.icon}
                    </div>
                    <div className="flex-1">
                      <div className="font-semibold text-lg">{item.label}</div>
                      <div className="text-sm text-muted-foreground">
                        {item.description}
                      </div>
                    </div>
                  </div>
                </motion.button>
              ))}
            </nav>

            {/* User Profile */}
            <div className="p-6 border-t border-border/50">
              <div className="flex items-center gap-4 mb-4">
                <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-emerald-500 rounded-2xl flex items-center justify-center">
                  <User className="w-6 h-6 text-white" />
                </div>
                <div className="flex-1">
                  <div className="font-semibold">{user.name}</div>
                  <div className="text-sm text-muted-foreground capitalize">{user.role}</div>
                </div>
              </div>
              <button
                onClick={handleLogout}
                className="w-full btn-secondary flex items-center justify-center gap-2"
              >
                <LogOut className="w-4 h-4" />
                Sair da Plataforma
              </button>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Bar */}
        <div className="glass-card border-b border-border/50 p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              {!sidebarOpen && (
                <button
                  onClick={() => setSidebarOpen(true)}
                  className="w-12 h-12 rounded-xl glass-button flex items-center justify-center"
                >
                  <Menu className="w-6 h-6" />
                </button>
              )}
              
              <div>
                <h1 className="font-bold text-2xl">
                  {NAVIGATION_ITEMS.find(item => item.id === activeSection)?.label}
                </h1>
                <p className="text-muted-foreground text-lg">
                  {NAVIGATION_ITEMS.find(item => item.id === activeSection)?.instructions}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <button className="w-12 h-12 rounded-xl glass-button flex items-center justify-center">
                <Bell className="w-6 h-6" />
              </button>
              
              <div className="status-online">
                <Activity className="w-4 h-4" />
                Online
              </div>
              
              <div className="text-muted-foreground font-mono">
                {new Date().toLocaleTimeString('pt-BR')}
              </div>
            </div>
          </div>
        </div>

        {/* Content */}
        <main className="flex-1 overflow-auto p-8">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeSection}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              {renderSection()}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  )
}

export default ModernDashboard