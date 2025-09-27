import { NextResponse } from 'next/server'

const AUTH_COOKIE = "darwin_token"

interface RegisterData {
  name: string
  email: string
  password: string
  role?: 'researcher' | 'guest'
}

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as RegisterData
    const { name, email, password, role = 'guest' } = body

    // Basic validation
    if (!name || !email || !password) {
      return NextResponse.json({ 
        success: false, 
        detail: "Nome, email e senha são obrigatórios" 
      }, { status: 400 })
    }

    if (password.length < 6) {
      return NextResponse.json({ 
        success: false, 
        detail: "Senha deve ter pelo menos 6 caracteres" 
      }, { status: 400 })
    }

    // Email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
    if (!emailRegex.test(email)) {
      return NextResponse.json({ 
        success: false, 
        detail: "Email inválido" 
      }, { status: 400 })
    }

    // In production, you would:
    // 1. Check if email already exists
    // 2. Hash the password
    // 3. Save to database
    // 4. Send verification email
    
    // For now, create mock user
    const newUser = {
      id: Date.now().toString(),
      name,
      email,
      role,
      permissions: role === 'researcher' ? ['research', 'chat', 'analytics'] : ['chat']
    }

    const mockToken = 'darwin_jwt_' + Date.now()
    
    const response = NextResponse.json({ 
      success: true, 
      access_token: mockToken,
      user: newUser,
      message: "Conta criada com sucesso"
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
    
  } catch (error) {
    console.error('Registration error:', error)
    return NextResponse.json({ 
      success: false, 
      detail: "Erro interno do servidor" 
    }, { status: 500 })
  }
}