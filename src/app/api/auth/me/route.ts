import { NextRequest, NextResponse } from 'next/server'

export async function GET(request: NextRequest) {
  try {
    const token = request.cookies.get('darwin_token')?.value || 
                 request.headers.get('authorization')?.replace('Bearer ', '')

    if (!token) {
      return NextResponse.json(
        { detail: 'Token não fornecido' },
        { status: 401 }
      )
    }

    // Mock user validation - In production, validate JWT token
    const mockUser = {
      id: '1',
      name: 'Dr. Demetrios Chiuratto Agourakis',
      email: 'agourakis@medresearch.com',
      role: 'admin' as const,
      permissions: ['research', 'chat', 'analytics', 'admin', 'all']
    }

    return NextResponse.json({
      user: mockUser,
      message: 'Usuário autenticado'
    })
    
  } catch (error) {
    console.error('Auth check error:', error)
    return NextResponse.json(
      { detail: 'Erro interno do servidor' },
      { status: 500 }
    )
  }
}