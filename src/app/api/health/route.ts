import { NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET() {
  const payload = {
    status: 'ok',
    app: process.env.NEXT_PUBLIC_APP_NAME || 'darwin-ui',
    env: process.env.NEXT_PUBLIC_ENVIRONMENT || process.env.NODE_ENV || 'production',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
  };
  return NextResponse.json(payload, { status: 200 });
}

// Hint: container HEALTHCHECK expects /api/health (already configured)
