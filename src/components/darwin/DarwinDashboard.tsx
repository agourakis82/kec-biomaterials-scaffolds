"use client"

import React from 'react'
import { GrFormSearch, GrCpu, GrAnalytics, GrCode, GrBook, GrStatusGood, GrPlay } from 'react-icons/gr'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card'
import { Button } from '../ui/button'
import { Progress } from '../ui/progress'
import { Badge } from '../ui/badge'

interface DashboardCardProps {
  title: string
  description: string
  icon: React.ComponentType<any>
  status: 'active' | 'idle' | 'processing'
  metrics?: {
    label: string
    value: string | number
  }[]
  onAction: () => void
  actionLabel: string
}

function DashboardCard({ title, description, icon: Icon, status, metrics, onAction, actionLabel }: DashboardCardProps) {
  const statusColors = {
    active: 'hsl(var(--darwin-success))',
    idle: 'hsl(var(--darwin-warning))',
    processing: 'hsl(var(--darwin-info))'
  }

  const statusLabels = {
    active: 'Active',
    idle: 'Idle', 
    processing: 'Processing'
  }

  return (
    <Card className="darwin-card hover:shadow-lg transition-all duration-300">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 rounded-full darwin-card flex items-center justify-center">
              <Icon className="text-xl" style={{ color: statusColors[status] }} />
            </div>
            <div>
              <CardTitle className="font-darwin-display text-lg">{title}</CardTitle>
              <CardDescription className="font-darwin-body text-sm">
                {description}
              </CardDescription>
            </div>
          </div>
          
          <Badge 
            className={`darwin-status-${status === 'active' ? 'success' : status === 'processing' ? 'warning' : 'error'}`}
          >
            {statusLabels[status]}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {metrics && (
          <div className="grid grid-cols-2 gap-4">
            {metrics.map((metric, index) => (
              <div key={index} className="text-center p-2 rounded-lg bg-muted/30">
                <div className="text-lg font-bold font-darwin-mono" 
                     style={{ color: statusColors[status] }}>
                  {metric.value}
                </div>
                <div className="text-xs text-muted-foreground font-darwin-body">
                  {metric.label}
                </div>
              </div>
            ))}
          </div>
        )}

        <Button 
          className="w-full darwin-button-primary font-darwin-body"
          onClick={onAction}
        >
          <GrPlay className="mr-2" />
          {actionLabel}
        </Button>
      </CardContent>
    </Card>
  )
}

interface DarwinDashboardProps {
  onSectionChange: (section: string) => void
  systemStatus: {
    ragPlus: 'active' | 'idle' | 'processing'
    memory: 'active' | 'idle' | 'processing'
    treeSearch: 'active' | 'idle' | 'processing'
    dataExplorer: 'active' | 'idle' | 'processing'
    notebooks: 'active' | 'idle' | 'processing'
  }
  metrics: {
    totalQueries: number
    activeJobs: number
    memoryEntries: number
    datasets: number
    notebooks: number
  }
}

export function DarwinDashboard({ onSectionChange, systemStatus, metrics }: DarwinDashboardProps) {
  const dashboardCards = [
    {
      id: 'rag-plus',
      title: 'RAG++ Research',
      description: 'Advanced scientific queries with iterative search',
      icon: GrFormSearch,
      status: systemStatus.ragPlus,
      metrics: [
        { label: 'Total Queries', value: metrics.totalQueries },
        { label: 'Success Rate', value: '94%' }
      ],
      actionLabel: 'Start Research'
    },
    {
      id: 'memory',
      title: 'Memory System',
      description: 'Conversation history and project continuity',
      icon: GrCpu,
      status: systemStatus.memory,
      metrics: [
        { label: 'Conversations', value: metrics.memoryEntries },
        { label: 'Continuity', value: '85%' }
      ],
      actionLabel: 'View Memory'
    },
    {
      id: 'tree-search',
      title: 'Tree Search',
      description: 'MCTS and PUCT algorithm visualization',
      icon: GrAnalytics,
      status: systemStatus.treeSearch,
      metrics: [
        { label: 'Searches', value: 47 },
        { label: 'Avg Depth', value: 8 }
      ],
      actionLabel: 'Run Search'
    },
    {
      id: 'data-explorer',
      title: 'Data Explorer',
      description: 'AG5 and HELIO scientific datasets',
      icon: GrCode,
      status: systemStatus.dataExplorer,
      metrics: [
        { label: 'Datasets', value: metrics.datasets },
        { label: 'Records', value: '125K' }
      ],
      actionLabel: 'Explore Data'
    },
    {
      id: 'notebooks',
      title: 'Notebooks',
      description: 'Jupyter notebook management and execution',
      icon: GrBook,
      status: systemStatus.notebooks,
      metrics: [
        { label: 'Notebooks', value: metrics.notebooks },
        { label: 'Running', value: 2 }
      ],
      actionLabel: 'Open Notebooks'
    }
  ]

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-3xl font-darwin-display font-bold" 
            style={{ color: 'hsl(var(--darwin-primary))' }}>
          Darwin Scientific Platform
        </h1>
        <p className="text-muted-foreground font-darwin-body">
          Advanced research tools for biomaterials and evolutionary analysis
        </p>
      </div>

      {/* System Status Overview */}
      <Card className="darwin-card">
        <CardHeader>
          <CardTitle className="font-darwin-display flex items-center space-x-2">
            <GrStatusGood style={{ color: 'hsl(var(--darwin-success))' }} />
            <span>System Status</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {Object.entries(systemStatus).map(([system, status]) => (
              <div key={system} className="text-center space-y-2">
                <div className="text-sm font-medium font-darwin-body capitalize">
                  {system.replace(/([A-Z])/g, ' $1').trim()}
                </div>
                <div className={`w-3 h-3 rounded-full mx-auto ${
                  status === 'active' ? 'bg-green-500' :
                  status === 'processing' ? 'bg-yellow-500' : 'bg-gray-400'
                }`} />
                <div className="text-xs text-muted-foreground capitalize">
                  {status}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Dashboard Cards Grid */}
      <div className="darwin-grid">
        {dashboardCards.map((card) => (
          <DashboardCard
            key={card.id}
            title={card.title}
            description={card.description}
            icon={card.icon}
            status={card.status}
            metrics={card.metrics}
            onAction={() => onSectionChange(card.id)}
            actionLabel={card.actionLabel}
          />
        ))}
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="darwin-card text-center">
          <CardContent className="p-4">
            <div className="text-2xl font-bold font-darwin-mono" 
                 style={{ color: 'hsl(var(--darwin-primary))' }}>
              {metrics.activeJobs}
            </div>
            <div className="text-sm text-muted-foreground font-darwin-body">
              Active Jobs
            </div>
          </CardContent>
        </Card>

        <Card className="darwin-card text-center">
          <CardContent className="p-4">
            <div className="text-2xl font-bold font-darwin-mono" 
                 style={{ color: 'hsl(var(--darwin-accent))' }}>
              98.5%
            </div>
            <div className="text-sm text-muted-foreground font-darwin-body">
              System Uptime
            </div>
          </CardContent>
        </Card>

        <Card className="darwin-card text-center">
          <CardContent className="p-4">
            <div className="text-2xl font-bold font-darwin-mono" 
                 style={{ color: 'hsl(var(--darwin-info))' }}>
              2.3s
            </div>
            <div className="text-sm text-muted-foreground font-darwin-body">
              Avg Response
            </div>
          </CardContent>
        </Card>

        <Card className="darwin-card text-center">
          <CardContent className="p-4">
            <div className="text-2xl font-bold font-darwin-mono" 
                 style={{ color: 'hsl(var(--darwin-memory))' }}>
              15.2GB
            </div>
            <div className="text-sm text-muted-foreground font-darwin-body">
              Data Processed
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}