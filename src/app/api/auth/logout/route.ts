import { NextResponse } from "next/server"

const AUTH_COOKIE = "darwin_token"

export async function POST() {
  try {
    const response = NextResponse.json({ 
      success: true,
      message: 'Logout realizado com sucesso'
    })
    
    // Clear both cookies for compatibility
    response.cookies.set({
      name: "agourakis_auth",
      value: "",
      path: "/",
      expires: new Date(0),
    })
    
    response.cookies.set({
      name: AUTH_COOKIE,
      value: "",
      path: "/",
      expires: new Date(0),
    })
    
    return response
  } catch (error) {
    console.error('Logout error:', error)
    return NextResponse.json({ 
      success: false, 
      detail: 'Erro interno do servidor' 
    }, { status: 500 })
  }
}
