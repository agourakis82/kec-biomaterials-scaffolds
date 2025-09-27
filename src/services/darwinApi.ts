// Darwin Backend API Integration
// This file contains all the real API calls to the Darwin backend

const DARWIN_API_BASE = process.env.NEXT_PUBLIC_DARWIN_API_URL || 'http://localhost:8090'

// Helper function for API calls
async function darwinApiCall<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<{ data?: T; error?: string; success: boolean }> {
  try {
    const token = localStorage.getItem('darwin_token')
    
    const response = await fetch(`${DARWIN_API_BASE}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...(token && { Authorization: `Bearer ${token}` }),
        ...options.headers,
      },
      ...options,
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Network error' }))
      return {
        success: false,
        error: errorData.detail || `HTTP ${response.status}: ${response.statusText}`
      }
    }

    const data = await response.json()
    return { success: true, data }
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error occurred'
    }
  }
}

// Research Team APIs
export const researchTeamApi = {
  // Get active agents
  getActiveAgents: () => darwinApiCall('/api/v1/research-team/agents'),
  
  // Start collaborative research
  startResearch: (query: string, agents: string[]) => 
    darwinApiCall('/api/v1/research-team/start', {
      method: 'POST',
      body: JSON.stringify({ query, agents })
    }),
  
  // Get research status
  getResearchStatus: (researchId: string) => 
    darwinApiCall(`/api/v1/research-team/status/${researchId}`),
  
  // Get research results
  getResearchResults: (researchId: string) => 
    darwinApiCall(`/api/v1/research-team/results/${researchId}`)
}

// JAX Performance APIs
export const performanceApi = {
  // Get system metrics
  getMetrics: () => darwinApiCall('/api/v1/ultra-performance/metrics'),
  
  // Start benchmark
  startBenchmark: (config: any) => 
    darwinApiCall('/api/v1/ultra-performance/benchmark', {
      method: 'POST',
      body: JSON.stringify(config)
    }),
  
  // Get performance history
  getPerformanceHistory: (timeRange: string) => 
    darwinApiCall(`/api/v1/ultra-performance/history?range=${timeRange}`)
}

// Multi-AI Chat APIs
export const chatApi = {
  // Get available agents
  getAgents: () => darwinApiCall('/api/v1/multi-ai/agents'),
  
  // Send message to agent
  sendMessage: (agentId: string, message: string, conversationId?: string) => 
    darwinApiCall('/api/v1/multi-ai/chat', {
      method: 'POST',
      body: JSON.stringify({ agentId, message, conversationId })
    }),
  
  // Get conversation history
  getConversation: (conversationId: string) => 
    darwinApiCall(`/api/v1/multi-ai/conversation/${conversationId}`),
  
  // Get all conversations
  getConversations: () => darwinApiCall('/api/v1/multi-ai/conversations')
}

// Discovery APIs
export const discoveryApi = {
  // Run discovery analysis
  runDiscovery: (query: string, options: any) => 
    darwinApiCall('/api/v1/discovery/run', {
      method: 'POST',
      body: JSON.stringify({ query, options })
    }),
  
  // Get discovery results
  getDiscoveryResults: (discoveryId: string) => 
    darwinApiCall(`/api/v1/discovery/results/${discoveryId}`),
  
  // Get discovery history
  getDiscoveryHistory: () => darwinApiCall('/api/v1/discovery/history')
}

// Knowledge Graph APIs
export const knowledgeGraphApi = {
  // Get graph data
  getGraph: (query?: string) => 
    darwinApiCall(`/api/v1/knowledge-graph${query ? `?query=${encodeURIComponent(query)}` : ''}`),
  
  // Add node to graph
  addNode: (nodeData: any) => 
    darwinApiCall('/api/v1/knowledge-graph/node', {
      method: 'POST',
      body: JSON.stringify(nodeData)
    }),
  
  // Add relationship
  addRelationship: (relationshipData: any) => 
    darwinApiCall('/api/v1/knowledge-graph/relationship', {
      method: 'POST',
      body: JSON.stringify(relationshipData)
    })
}

// KEC Metrics APIs
export const kecMetricsApi = {
  // Get KEC metrics
  getMetrics: () => darwinApiCall('/api/v1/kec-metrics'),
  
  // Update metrics
  updateMetrics: (metricsData: any) => 
    darwinApiCall('/api/v1/kec-metrics', {
      method: 'POST',
      body: JSON.stringify(metricsData)
    })
}

// Tree Search APIs (PUCT)
export const treeSearchApi = {
  // Run PUCT search
  runPuctSearch: (searchParams: any) => 
    darwinApiCall('/api/v1/tree-search/puct', {
      method: 'POST',
      body: JSON.stringify(searchParams)
    }),
  
  // Get search results
  getSearchResults: (searchId: string) => 
    darwinApiCall(`/api/v1/tree-search/results/${searchId}`)
}

// RAG Plus APIs
export const ragApi = {
  // Search documents
  search: (query: string, options: any) => 
    darwinApiCall('/api/v1/rag-plus/search', {
      method: 'POST',
      body: JSON.stringify({ query, ...options })
    }),
  
  // Iterative search
  iterativeSearch: (query: string, context: any) => 
    darwinApiCall('/api/v1/rag-plus/iterative', {
      method: 'POST',
      body: JSON.stringify({ query, context })
    })
}

// Health check
export const healthApi = {
  check: () => darwinApiCall('/api/v1/health')
}

export default {
  researchTeamApi,
  performanceApi,
  chatApi,
  discoveryApi,
  knowledgeGraphApi,
  kecMetricsApi,
  treeSearchApi,
  ragApi,
  healthApi
}