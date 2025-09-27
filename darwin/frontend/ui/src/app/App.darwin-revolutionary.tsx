"use client"

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ThemeProvider } from 'next-themes'
import { 
  Zap, 
  Network, 
  MessageSquare, 
  BarChart3, 
  Settings,
  Menu,
  X,
  Sparkles,
  Atom
} from 'lucide-react'
import { DarwinLogo } from '../components/quantum/DarwinLogo'

// Import our revolutionary components
import QuantumLayout from '../components/quantum/QuantumLayout'
import ResearchTeamDashboard from '../components/quantum/ResearchTeamDashboard'
import JAXPerformanceDashboard from '../components/quantum/JAXPerformanceDashboard'
import ConsciousnessChat from '../components/quantum/ConsciousnessChat'
import QuantumAuth from '../components/quantum/QuantumAuth'
import ProtectedRoute, { RoleGuard, PermissionGuard } from '../components/quantum/ProtectedRoute'
import UserProfile from '../components/quantum/UserProfile'
import useAuth from '../hooks/useAuth'

interface User {
  id: string
  name: string
  email: string
  role: 'admin' | 'researcher' | 'guest'
  avatar?: string
  permissions: string[]
}

interface DarwinRevolutionaryAppProps {
  author?: string
}

type Section = 'research-team' | 'performance' | 'chat' | 'analytics' | 'settings'

const NAVIGATION_ITEMS = [
  {
    id: 'research-team' as Section,
    label: 'Research Team',
    icon: <DarwinLogo size={20} variant="icon" animated={false} />,
    description: 'Multi-Agent Collaboration',
    color: 'quantum-primary'
  },
  {
    id: 'performance' as Section,
    label: 'JAX Performance',
    icon: <Zap className="w-5 h-5" />,
    description: 'Ultra-Performance Monitor',
    color: 'biotech-success'
  },
  {
    id: 'chat' as Section,
    label: 'Agent Chat',
    icon: <MessageSquare className="w-5 h-5" />,
    description: 'Individual Conversations',
    color: 'consciousness-glow'
  },
  {
    id: 'analytics' as Section,
    label: 'Analytics',
    icon: <BarChart3 className="w-5 h-5" />,
    description: 'Cross-Domain Analysis',
    color: 'neural-accent'
  },
  {
    id: 'settings' as Section,
    label: 'Settings',
    icon: <Settings className="w-5 h-5" />,
    description: 'System Configuration',
    color: 'research-gold'
  }
]

export const DarwinRevolutionaryApp: React.FC<DarwinRevolutionaryAppProps> = ({
  author = "Dr. Demetrios Chiuratto Agourakis"
}) => {
  const [activeSection, setActiveSection] = useState<Section>('research-team')
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [queryClient] = useState(() => new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 5 * 60 * 1000, // 5 minutes
        refetchOnWindowFocus: false,
      },
    },
  }))

  // Authentication hook
  const { 
    user, 
    isAuthenticated, 
    isLoading: authLoading, 
    login, 
    register, 
    logout,
    updateProfile 
  } = useAuth()

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey) {
        switch (e.key) {
          case '1':
            e.preventDefault()
            setActiveSection('research-team')
            break
          case '2':
            e.preventDefault()
            setActiveSection('performance')
            break
          case '3':
            e.preventDefault()
            setActiveSection('chat')
            break
          case '4':
            e.preventDefault()
            setActiveSection('analytics')
            break
          case '5':
            e.preventDefault()
            setActiveSection('settings')
            break
          case 'b':
            e.preventDefault()
            setSidebarOpen(!sidebarOpen)
            break
        }
      }
    }

    window.addEventListener('keydown', handleKeyPress)
    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [sidebarOpen])

  // Handle authentication
  const handleLogin = async (userData: User) => {
    // The login is handled by the useAuth hook
    console.log('User logged in:', userData)
  }

  const handleRegister = async (registerData: any) => {
    const result = await register(registerData)
    if (result.success) {
      console.log('Registration successful')
    } else {
      console.error('Registration failed:', result.error)
    }
  }

  // Show loading screen while checking authentication
  if (authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          className="w-12 h-12 rounded-full border-4 border-quantum-primary border-t-transparent"
        />
      </div>
    )
  }

  // Show authentication screen if not logged in
  if (!isAuthenticated) {
    return (
      <QueryClientProvider client={queryClient}>
        <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
          <QuantumAuth 
            onLogin={handleLogin}
            onRegister={handleRegister}
            isLoading={authLoading}
          />
        </ThemeProvider>
      </QueryClientProvider>
    )
  }

  const renderActiveSection = () => {
    switch (activeSection) {
      case 'research-team':
        return (
          <ProtectedRoute 
            user={user} 
            requiredPermissions={['research', 'all']}
          >
            <ResearchTeamDashboard
              onAgentSelect={(agent) => {
                console.log('Agent selected:', agent)
              }}
              onCollaborativeQuery={(query, agents) => {
                console.log('Collaborative query:', query, agents)
              }}
            />
          </ProtectedRoute>
        )
      
      case 'performance':
        return (
          <ProtectedRoute 
            user={user} 
            requiredPermissions={['research', 'all']}
          >
            <JAXPerformanceDashboard
              onBenchmarkStart={() => {
                console.log('Benchmark started')
              }}
              onBatchProcess={(count) => {
                console.log('Batch processing:', count)
              }}
            />
          </ProtectedRoute>
        )
      
      case 'chat':
        return (
          <ProtectedRoute 
            user={user} 
            requiredPermissions={['chat', 'all']}
          >
            <div className="h-full">
              <ConsciousnessChat
                onAgentSwitch={(agent) => {
                  console.log('Agent switched:', agent)
                }}
              />
            </div>
          </ProtectedRoute>
        )
      
      case 'analytics':
        return (
          <ProtectedRoute 
            user={user} 
            requiredPermissions={['analytics', 'all']}
          >
            <div className="p-8 space-y-8">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-center space-y-4"
              >
                <h1 className="research-title">
                  🌐 Cross-Domain Analysis
                </h1>
                <p className="subtitle-elegant max-w-3xl mx-auto">
                  Interdisciplinary insights visualization and domain connection mapping 
                  for comprehensive research synthesis
                </p>
              </motion.div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {[
                  { title: 'Domain Connections', value: '247', color: 'quantum-primary' },
                  { title: 'Insight Synthesis', value: '89', color: 'neural-accent' },
                  { title: 'Research Patterns', value: '156', color: 'consciousness-glow' }
                ].map((metric, index) => (
                  <div key={index} className="consciousness-metric">
                    <div className={`text-${metric.color} mb-2`}>
                      <Network className="w-6 h-6" />
                    </div>
                    <div className="text-3xl font-quantum font-bold mb-1">
                      {metric.value}
                    </div>
                    <div className="text-sm text-muted-foreground font-neural">
                      {metric.title}
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="neural-glass p-8 text-center">
                <Atom className="w-16 h-16 text-quantum-primary mx-auto mb-4" />
                <h3 className="text-2xl font-quantum font-semibold mb-4">
                  Advanced Analytics Coming Soon
                </h3>
                <p className="text-muted-foreground font-neural max-w-2xl mx-auto">
                  Interactive knowledge graphs, domain relationship mapping, and 
                  AI-powered insight synthesis are being developed to enhance your research workflow.
                </p>
              </div>
            </div>
          </ProtectedRoute>
        )
      
      case 'settings':
        return (
          <RoleGuard user={user} allowedRoles={['admin', 'researcher']}>
            <div className="p-8 space-y-8">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-center space-y-4"
              >
                <h1 className="research-title">
                  ⚙️ System Configuration
                </h1>
                <p className="subtitle-elegant max-w-3xl mx-auto">
                  Configure your DARWIN AutoGen + JAX system for optimal performance 
                  and personalized research experience
                </p>
              </motion.div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="neural-glass p-6 space-y-4">
                  <h3 className="text-xl font-quantum font-semibold mb-4">
                    Performance Settings
                  </h3>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="font-neural">JAX Acceleration</span>
                      <div className="quantum-glass px-3 py-1 rounded-full text-sm font-neural">
                        Enabled
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="font-neural">GPU Utilization</span>
                      <div className="quantum-glass px-3 py-1 rounded-full text-sm font-neural">
                        Auto
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="font-neural">TPU Integration</span>
                      <div className="quantum-glass px-3 py-1 rounded-full text-sm font-neural">
                        Available
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="neural-glass p-6 space-y-4">
                  <h3 className="text-xl font-quantum font-semibold mb-4">
                    Agent Configuration
                  </h3>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="font-neural">Active Agents</span>
                      <div className="quantum-glass px-3 py-1 rounded-full text-sm font-neural">
                        8/8
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="font-neural">Collaboration Mode</span>
                      <div className="quantum-glass px-3 py-1 rounded-full text-sm font-neural">
                        Enhanced
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="font-neural">Response Time</span>
                      <div className="quantum-glass px-3 py-1 rounded-full text-sm font-neural">
                        Optimized
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="consciousness-glass p-8 text-center">
                <Settings className="w-16 h-16 text-research-gold mx-auto mb-4" />
                <h3 className="text-2xl font-quantum font-semibold mb-4">
                  Advanced Configuration
                </h3>
                <p className="text-muted-foreground font-neural max-w-2xl mx-auto">
                  Detailed system configuration, API endpoints management, and 
                  performance tuning options will be available in the next update.
                </p>
              </div>
            </div>
          </RoleGuard>
        )
      
      default:
        return null
    }
  }

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
        <QuantumLayout
          author={author}
          title="DARWIN Research Hub"
          subtitle="AutoGen Multi-Agent + JAX Ultra-Performance System"
        >
          <div className="flex h-screen">
            {/* Quantum Sidebar */}
            <AnimatePresence>
              {sidebarOpen && (
                <motion.aside
                  initial={{ x: -300, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  exit={{ x: -300, opacity: 0 }}
                  transition={{ type: "spring", damping: 25, stiffness: 200 }}
                  className="w-80 neural-glass border-r border-border/50 flex flex-col"
                >
                  {/* Sidebar Header */}
                  <div className="p-6 border-b border-border/50">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-3">
                        <DarwinLogo 
                          size={40} 
                          variant="icon" 
                          animated={true}
                          className="quantum-glass rounded-full p-1"
                        />
                        <div>
                          <h2 className="font-quantum font-bold text-lg">
                            DARWIN Hub
                          </h2>
                          <p className="text-xs text-muted-foreground font-neural">
                            Revolutionary Research
                          </p>
                        </div>
                      </div>
                      <button
                        onClick={() => setSidebarOpen(false)}
                        className="w-8 h-8 rounded-lg quantum-glass flex items-center justify-center hover:bg-quantum-primary/10 transition-colors"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                    
                    {/* Quick Stats */}
                    <div className="grid grid-cols-2 gap-3">
                      <div className="consciousness-glass p-3 text-center">
                        <div className="text-lg font-quantum font-bold text-quantum-primary">
                          8
                        </div>
                        <div className="text-xs text-muted-foreground font-neural">
                          Agents
                        </div>
                      </div>
                      <div className="consciousness-glass p-3 text-center">
                        <div className="text-lg font-quantum font-bold text-biotech-success">
                          1.2K
                        </div>
                        <div className="text-xs text-muted-foreground font-neural">
                          Speedup
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Navigation */}
                  <nav className="flex-1 p-6 space-y-2">
                    {NAVIGATION_ITEMS.map((item) => (
                      <motion.button
                        key={item.id}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={() => setActiveSection(item.id)}
                        className={`w-full p-4 rounded-xl text-left transition-all duration-300 ${
                          activeSection === item.id
                            ? 'quantum-glass ring-2 ring-quantum-primary/50 shadow-lg'
                            : 'hover:bg-background/50'
                        }`}
                      >
                        <div className="flex items-center gap-4">
                          <div className={`w-10 h-10 rounded-lg quantum-glass flex items-center justify-center text-${item.color}`}>
                            {item.icon}
                          </div>
                          <div className="flex-1">
                            <div className="font-quantum font-semibold">
                              {item.label}
                            </div>
                            <div className="text-sm text-muted-foreground font-neural">
                              {item.description}
                            </div>
                          </div>
                        </div>
                      </motion.button>
                    ))}
                  </nav>

                  {/* Sidebar Footer */}
                  <div className="p-6 border-t border-border/50">
                    <div className="consciousness-glass p-4 text-center">
                      <div className="text-sm font-neural text-muted-foreground mb-2">
                        System Status
                      </div>
                      <div className="flex items-center justify-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-biotech-success animate-pulse" />
                        <span className="text-sm font-quantum font-semibold text-biotech-success">
                          All Systems Operational
                        </span>
                      </div>
                    </div>
                  </div>
                </motion.aside>
              )}
            </AnimatePresence>

            {/* Main Content Area */}
            <div className="flex-1 flex flex-col overflow-hidden">
              {/* Top Bar */}
              <div className="neural-glass border-b border-border/50 p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    {!sidebarOpen && (
                      <button
                        onClick={() => setSidebarOpen(true)}
                        className="w-10 h-10 rounded-lg quantum-glass flex items-center justify-center hover:bg-quantum-primary/10 transition-colors"
                      >
                        <Menu className="w-5 h-5" />
                      </button>
                    )}
                    
                    <div>
                      <h1 className="font-quantum font-bold text-xl">
                        {NAVIGATION_ITEMS.find(item => item.id === activeSection)?.label}
                      </h1>
                      <p className="text-sm text-muted-foreground font-neural">
                        {NAVIGATION_ITEMS.find(item => item.id === activeSection)?.description}
                      </p>
                    </div>
                  </div>

                  {/* Quick Actions */}
                  <div className="flex items-center gap-4">
                    <div className="consciousness-glass px-3 py-1 rounded-full">
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-quantum-primary animate-pulse" />
                        <span className="text-sm font-neural">Live</span>
                      </div>
                    </div>
                    
                    <div className="text-sm text-muted-foreground font-data">
                      {new Date().toLocaleTimeString()}
                    </div>

                    {/* User Profile */}
                    <UserProfile 
                      user={user!}
                      onLogout={logout}
                      onUpdateProfile={updateProfile}
                    />
                  </div>
                </div>
              </div>

              {/* Content */}
              <main className="flex-1 overflow-auto">
                <AnimatePresence mode="wait">
                  <motion.div
                    key={activeSection}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                    transition={{ duration: 0.3 }}
                    className="h-full"
                  >
                    {renderActiveSection()}
                  </motion.div>
                </AnimatePresence>
              </main>
            </div>
          </div>
        </QuantumLayout>
      </ThemeProvider>
    </QueryClientProvider>
  )
}

export default DarwinRevolutionaryApp