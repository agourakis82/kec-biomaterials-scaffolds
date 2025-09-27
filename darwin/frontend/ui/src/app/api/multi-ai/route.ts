import { NextRequest, NextResponse } from 'next/server'
export const dynamic = 'force-dynamic'
import { getServerEnv, getApiUrl } from '@/lib/env.server'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const env = getServerEnv()
    
    // DARWIN Multi-AI Hub endpoint - produção: api.agourakis.med.br
    const url = getApiUrl('/api/v1/multi-ai/chat')
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-KEY': env.DARWIN_SERVER_KEY,
      },
      body: JSON.stringify(body),
    })
    
    if (!response.ok) {
      throw new Error(`Multi-AI Chat failed: ${response.status} ${response.statusText}`)
    }
    
    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('DARWIN Multi-AI Error:', error)
    return NextResponse.json({
      error: 'Internal Server Error',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 })
  }
}

// GET para obter modelos disponíveis
export async function GET(request: NextRequest) {
  try {
    const env = getServerEnv()
    
    const url = getApiUrl('/api/v1/multi-ai/models')
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'X-API-KEY': env.DARWIN_SERVER_KEY,
      },
    })
    
    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('DARWIN Multi-AI Models Error:', error)
    return NextResponse.json({
      error: 'Internal Server Error',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 })
  }
}