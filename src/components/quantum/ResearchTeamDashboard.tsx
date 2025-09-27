"use client"

import React, { useState } from 'react'
import { Users, MessageSquare, Zap, Brain, Search, Play } from 'lucide-react'
import { cn } from '@/lib/utils'

interface Agent {
  id: string
  name: string
  type: string
  status: 'active' | 'idle' | 'busy'
  description: string
  capabilities: string[]
}

interface ResearchTeamDashboardProps {
  onAgentSelect: (agent: Agent) => void
  onCollaborativeQuery: (query: string, agents: Agent[]) => void
}

const mockAgents: Agent[] = [
  {
    id: '1',
    name: 'Dr. Neural',
    type: 'Neural Networks Specialist',
    status: 'active',
    description: 'Especialista em redes neurais e deep learning',
    capabilities: ['Deep Learning', 'Neural Architecture', 'Training Optimization']
  },
  {
    id: '2',
    name: 'Prof. Quantum',
    type: 'Quantum Computing Expert',
    status: 'idle',
    description: 'Especialista em computação quântica e algoritmos',
    capabilities: ['Quantum Algorithms', 'Quantum Mechanics', 'Quantum ML']
  },
  {
    id: '3',
    name: 'Dr. Bio',
    type: 'Bioinformatics Researcher',
    status: 'busy',
    description: 'Especialista em bioinformática e análise genômica',
    capabilities: ['Genomics', 'Protein Analysis', 'Drug Discovery']
  },
  {
    id: '4',
    name: 'AI Synthesizer',
    type: 'Knowledge Synthesis Agent',
    status: 'active',
    description: 'Sintetiza conhecimento de múltiplas fontes',
    capabilities: ['Knowledge Synthesis', 'Literature Review', 'Cross-domain Analysis']
  }
]

export const ResearchTeamDashboard: React.FC<ResearchTeamDashboardProps> = ({
  onAgentSelect,
  onCollaborativeQuery
}) => {
  const [selectedAgents, setSelectedAgents] = useState<Agent[]>([])
  const [query, setQuery] = useState('')

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'text-green-500'
      case 'busy':
        return 'text-yellow-500'
      case 'idle':
        return 'text-gray-500'
      default:
        return 'text-gray-500'
    }
  }

  const getStatusLabel = (status: string) => {
    switch (status) {
      case 'active':
        return 'Ativo'
      case 'busy':
        return 'Ocupado'
      case 'idle':
        return 'Inativo'
      default:
        return status
    }
  }

  const toggleAgentSelection = (agent: Agent) => {
    setSelectedAgents(prev => {
      const isSelected = prev.some(a => a.id === agent.id)
      if (isSelected) {
        return prev.filter(a => a.id !== agent.id)
      } else {
        return [...prev, agent]
      }
    })
  }

  const handleCollaborativeQuery = () => {
    if (query.trim() && selectedAgents.length > 0) {
      onCollaborativeQuery(query, selectedAgents)
      setQuery('')
    }
  }

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="text-center space-y-4">
        <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-500 via-blue-500 to-green-500 bg-clip-text text-transparent">
          🤖 Research Team Dashboard
        </h1>
        <p className="text-muted-foreground max-w-3xl mx-auto">
          Coordene agentes de IA especializados para pesquisa colaborativa e análise interdisciplinar
        </p>
      </div>

      {/* Team Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {[
          { label: 'Agentes Ativos', value: mockAgents.filter(a => a.status === 'active').length, icon: Users, color: 'text-green-500' },
          { label: 'Colaborações', value: '23', icon: MessageSquare, color: 'text-blue-500' },
          { label: 'Consultas Hoje', value: '156', icon: Search, color: 'text-purple-500' },
          { label: 'Performance', value: '98%', icon: Zap, color: 'text-yellow-500' }
        ].map((stat, index) => (
          <div key={index} className="bg-background/50 border border-border/50 rounded-lg p-6 text-center">
            <div className={cn("mb-2", stat.color)}>
              <stat.icon className="w-6 h-6 mx-auto" />
            </div>
            <div className="text-2xl font-bold mb-1">{stat.value}</div>
            <div className="text-sm text-muted-foreground">{stat.label}</div>
          </div>
        ))}
      </div>

      {/* Collaborative Query */}
      <div className="bg-background/50 border border-border/50 rounded-lg p-6">
        <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <Brain className="w-5 h-5" />
          Consulta Colaborativa
        </h3>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">
              Pergunta de Pesquisa:
            </label>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ex: Como podemos otimizar algoritmos de machine learning para análise de proteínas usando computação quântica?"
              className="w-full p-3 bg-background/50 border border-border/50 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none"
              rows={3}
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">
              Agentes Selecionados ({selectedAgents.length}):
            </label>
            <div className="flex flex-wrap gap-2">
              {selectedAgents.map(agent => (
                <span
                  key={agent.id}
                  className="px-3 py-1 bg-purple-500/20 text-purple-300 rounded-full text-sm flex items-center gap-2"
                >
                  {agent.name}
                  <button
                    onClick={() => toggleAgentSelection(agent)}
                    className="text-purple-300 hover:text-white"
                  >
                    ×
                  </button>
                </span>
              ))}
            </div>
          </div>

          <button
            onClick={handleCollaborativeQuery}
            disabled={!query.trim() || selectedAgents.length === 0}
            className={cn(
              "flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all",
              "bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600",
              "text-white disabled:opacity-50 disabled:cursor-not-allowed"
            )}
          >
            <Play className="w-4 h-4" />
            Iniciar Colaboração
          </button>
        </div>
      </div>

      {/* Agents Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-6">
        {mockAgents.map((agent) => (
          <div
            key={agent.id}
            className={cn(
              "bg-background/50 border border-border/50 rounded-lg p-6 cursor-pointer transition-all hover:bg-background/80",
              selectedAgents.some(a => a.id === agent.id) && "ring-2 ring-purple-500 bg-purple-500/10"
            )}
            onClick={() => toggleAgentSelection(agent)}
          >
            {/* Agent Header */}
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 rounded-full bg-gradient-to-r from-purple-500 to-blue-500 flex items-center justify-center">
                  <Brain className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h3 className="font-semibold">{agent.name}</h3>
                  <p className="text-sm text-muted-foreground">{agent.type}</p>
                </div>
              </div>
              
              <div className="flex items-center gap-2">
                <div className={cn("w-2 h-2 rounded-full", getStatusColor(agent.status))} />
                <span className={cn("text-xs", getStatusColor(agent.status))}>
                  {getStatusLabel(agent.status)}
                </span>
              </div>
            </div>

            {/* Agent Description */}
            <p className="text-sm text-muted-foreground mb-4">
              {agent.description}
            </p>

            {/* Capabilities */}
            <div>
              <div className="text-sm font-medium mb-2">Especialidades:</div>
              <div className="flex flex-wrap gap-1">
                {agent.capabilities.map((capability, index) => (
                  <span
                    key={index}
                    className="px-2 py-1 text-xs bg-blue-500/20 text-blue-300 rounded-full"
                  >
                    {capability}
                  </span>
                ))}
              </div>
            </div>

            {/* Action Button */}
            <button
              onClick={(e) => {
                e.stopPropagation()
                onAgentSelect(agent)
              }}
              className="w-full mt-4 py-2 px-4 bg-background/50 border border-border/50 rounded-lg hover:bg-background/80 transition-colors text-sm"
            >
              Chat Individual
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}

export default ResearchTeamDashboard
