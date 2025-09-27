import { NextRequest, NextResponse } from 'next/server'
export const dynamic = 'force-static'
import { getServerEnv } from '@/lib/env.server'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const env = getServerEnv()
    // Adapt payload to API contract: QuickSearch expects { query, depth?, budget? }
    const payload = {
      query: body?.root ?? body?.query ?? '',
      depth: body?.depth ?? 3,
      budget: body?.budget ?? 50,
    }
    const response = await fetch(`${env.NEXT_PUBLIC_DARWIN_URL}/tree-search/quick-search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-KEY': env.DARWIN_SERVER_KEY,
      },
      body: JSON.stringify(payload),
    })
    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 })
  }
}
