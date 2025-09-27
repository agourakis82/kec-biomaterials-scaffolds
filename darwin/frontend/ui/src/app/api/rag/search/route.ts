import { NextRequest, NextResponse } from 'next/server'
export const dynamic = 'force-dynamic'
import { getServerEnv, getApiUrl } from '@/lib/env.server'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const env = getServerEnv()
    
    // DARWIN RAG++ endpoint - produção: api.agourakis.med.br
    const url = getApiUrl('/api/v1/rag-plus')
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-KEY': env.DARWIN_SERVER_KEY,
      },
      body: JSON.stringify(body),
    })
    
    if (!response.ok) {
      throw new Error(`RAG search failed: ${response.status} ${response.statusText}`)
    }
    
    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('DARWIN RAG Search Error:', error)
    return NextResponse.json({
      error: 'Internal Server Error',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 })
  }
}
