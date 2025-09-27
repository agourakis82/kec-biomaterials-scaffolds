"use client"

import React, { useState } from 'react'
import { DarwinAuth } from '../components/darwin/DarwinAuth'
import { DarwinSidebar } from '../components/darwin/DarwinSidebar'
import { DarwinDashboard } from '../components/darwin/DarwinDashboard'
import { RAGPlusInterface } from '../components/darwin/RAGPlusInterface'
import { mockStore, mockQuery } from '../lib/darwinMockData'

export default function DarwinApp() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [activeSection, setActiveSection] = useState('dashboard')
  const [isLoading, setIsLoading] = useState(false)
  const [ragResults, setRagResults] = useState(mockQuery.ragPlusResults)

  // Mock authentication
  const handleLogin = async (username: string, password: string) => {
    setIsLoading(true)
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1500))
    setIsAuthenticated(true)
    setIsLoading(false)
  }

  const handleLogout = () => {
    setIsAuthenticated(false)
    setActiveSection('dashboard')
  }

  const handleRAGQuery = async (query: string, type: string) => {
    setIsLoading(true)
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 2000))
    
    // Mock response based on query
    const mockResponse = {
      query,
      answer: `Based on current research, ${query.toLowerCase()} involves several key factors. Recent studies show significant advances in methodology and applications, with particular emphasis on optimization and practical implementation.`,
      sources: mockQuery.ragPlusResults.sources,
      method: type as "rag_plus_iterative",
      retrievedDocs: Math.floor(Math.random() * 20) + 5,
      processingTime: Math.random() * 3 + 1
    }
    
    setRagResults(mockResponse)
    setIsLoading(false)
  }

  // Mock system status and metrics
  const systemStatus = {
    ragPlus: 'active' as const,
    memory: 'active' as const,
    treeSearch: 'idle' as const,
    dataExplorer: 'active' as const,
    notebooks: 'processing' as const
  }

  const metrics = {
    totalQueries: 1247,
    activeJobs: 3,
    memoryEntries: 89,
    datasets: 12,
    notebooks: 8
  }

  if (!isAuthenticated) {
    return (
      <DarwinAuth 
        onLogin={handleLogin}
        isLoading={isLoading}
      />
    )
  }

  const renderActiveSection = () => {
    switch (activeSection) {
      case 'dashboard':
        return (
          <DarwinDashboard
            onSectionChange={setActiveSection}
            systemStatus={systemStatus}
            metrics={metrics}
          />
        )
      case 'rag-plus':
        return (
          <RAGPlusInterface
            onQuery={handleRAGQuery}
            isLoading={isLoading}
            results={ragResults}
          />
        )
      case 'memory':
        return (
          <div className="p-6">
            <h1 className="text-3xl font-darwin-display font-bold mb-4" 
                style={{ color: 'hsl(var(--darwin-primary))' }}>
              Memory System
            </h1>
            <p className="text-muted-foreground font-darwin-body">
              Conversation history and project continuity dashboard coming soon...
            </p>
          </div>
        )
      case 'tree-search':
        return (
          <div className="p-6">
            <h1 className="text-3xl font-darwin-display font-bold mb-4" 
                style={{ color: 'hsl(var(--darwin-primary))' }}>
              Tree Search Playground
            </h1>
            <p className="text-muted-foreground font-darwin-body">
              MCTS and PUCT algorithm visualization coming soon...
            </p>
          </div>
        )
      case 'data-explorer':
        return (
          <div className="p-6">
            <h1 className="text-3xl font-darwin-display font-bold mb-4" 
                style={{ color: 'hsl(var(--darwin-primary))' }}>
              Data Explorer
            </h1>
            <p className="text-muted-foreground font-darwin-body">
              AG5 and HELIO dataset explorer coming soon...
            </p>
          </div>
        )
      case 'notebooks':
        return (
          <div className="p-6">
            <h1 className="text-3xl font-darwin-display font-bold mb-4" 
                style={{ color: 'hsl(var(--darwin-primary))' }}>
              Notebook Manager
            </h1>
            <p className="text-muted-foreground font-darwin-body">
              Jupyter notebook management interface coming soon...
            </p>
          </div>
        )
      default:
        return (
          <DarwinDashboard
            onSectionChange={setActiveSection}
            systemStatus={systemStatus}
            metrics={metrics}
          />
        )
    }
  }

  return (
    <div className="min-h-screen flex" style={{ background: 'var(--bg-main)' }}>
      {/* Sidebar */}
      <DarwinSidebar
        activeSection={activeSection}
        onSectionChange={setActiveSection}
        onLogout={handleLogout}
        user={mockStore.auth.user}
      />

      {/* Main Content */}
      <div className="flex-1 overflow-auto">
        {renderActiveSection()}
      </div>
    </div>
  )
}