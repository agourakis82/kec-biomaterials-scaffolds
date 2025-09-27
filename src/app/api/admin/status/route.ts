import { NextRequest, NextResponse } from 'next/server'
export const dynamic = 'force-static'
import { getServerEnv } from '@/lib/env.server'

export async function GET(request: NextRequest) {
  try {
    const env = getServerEnv()
    const response = await fetch(`${env.NEXT_PUBLIC_DARWIN_URL}/openapi.json`, {
      method: 'GET',
      headers: { 'X-API-KEY': env.DARWIN_SERVER_KEY },
    })
    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 })
  }
}
