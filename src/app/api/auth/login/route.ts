import { NextResponse } from "next/server"

const AUTH_COOKIE = "darwin_token"

interface Credentials {
  email?: string
  username?: string
  password?: string
}

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as Credentials
    const { email, username, password } = body

    const validUsername = process.env.APP_USERNAME ?? "agourakis"
    const validPassword = process.env.APP_PASSWORD ?? "medresearch"

    // Support both email and username login
    const loginField = email || username

    if (!loginField || !password) {
      return NextResponse.json({ 
        success: false, 
        detail: "Informe email/usuário e senha" 
      }, { status: 400 })
    }

    // Mock user data for successful login
    const mockUser = {
      id: '1',
      name: 'Dr. Demetrios Chiuratto Agourakis',
      email: email || 'agourakis@medresearch.com',
      role: 'admin' as const,
      permissions: ['research', 'chat', 'analytics', 'admin', 'all']
    }

    // Check credentials (allow demo login with any email/password or specific credentials)
    const isValidLogin = (
      (loginField === validUsername && password === validPassword) ||
      (email && password.length >= 6) // Demo mode: any email with 6+ char password
    )

    if (isValidLogin) {
      const mockToken = 'darwin_jwt_' + Date.now()
      
      const response = NextResponse.json({ 
        success: true, 
        access_token: mockToken,
        user: mockUser,
        message: "Login realizado com sucesso"
      })
      
      response.cookies.set({
        name: AUTH_COOKIE,
        value: mockToken,
        httpOnly: true,
        secure: process.env.NODE_ENV === "production",
        sameSite: "lax",
        path: "/",
        maxAge: 60 * 60 * 24, // 24 hours
      })
      
      return response
    }

    return NextResponse.json({ 
      success: false, 
      detail: "Credenciais inválidas" 
    }, { status: 401 })
    
  } catch (error) {
    console.error('Login error:', error)
    return NextResponse.json({ 
      success: false, 
      detail: "Erro interno do servidor" 
    }, { status: 500 })
  }
}
