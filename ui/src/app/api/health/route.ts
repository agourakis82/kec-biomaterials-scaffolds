import { NextRequest, NextResponse } from 'next/server'
import { getServerEnv } from '@/lib/env.server'

export async function GET(request: NextRequest) {
  try {
    const env = getServerEnv()
    const response = await fetch(`${env.NEXT_PUBLIC_KEC_BIOMAT_URL}/healthz`)
    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 })
  }
}
