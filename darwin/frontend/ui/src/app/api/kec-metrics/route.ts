import { NextRequest, NextResponse } from 'next/server'
export const dynamic = 'force-dynamic'
import { getServerEnv, getApiUrl } from '@/lib/env.server'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const env = getServerEnv()
    
    // DARWIN KEC Analysis endpoint - produção: api.agourakis.med.br
    const url = getApiUrl('/api/v1/kec-metrics/analyze')
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-KEY': env.DARWIN_SERVER_KEY,
      },
      body: JSON.stringify(body),
    })
    
    if (!response.ok) {
      throw new Error(`KEC Analysis failed: ${response.status} ${response.statusText}`)
    }
    
    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('DARWIN KEC Metrics Error:', error)
    return NextResponse.json({
      error: 'Internal Server Error',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 })
  }
}

// GET para status/info
export async function GET(request: NextRequest) {
  try {
    const env = getServerEnv()
    
    const url = getApiUrl('/api/v1/kec-metrics/status')
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
    console.error('DARWIN KEC Metrics Status Error:', error)
    return NextResponse.json({
      error: 'Internal Server Error',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 })
  }
}