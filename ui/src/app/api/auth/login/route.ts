import { NextResponse } from "next/server"

const AUTH_COOKIE = "agourakis_auth"

interface Credentials {
  username?: string
  password?: string
}

export async function POST(request: Request) {
  const { username, password } = (await request.json()) as Credentials

  const validUsername = process.env.APP_USERNAME ?? "agourakis"
  const validPassword = process.env.APP_PASSWORD ?? "medresearch"

  if (!username || !password) {
    return NextResponse.json({ success: false, message: "Informe usuário e senha" }, { status: 400 })
  }

  if (username === validUsername && password === validPassword) {
    const response = NextResponse.json({ success: true, username })
    response.cookies.set({
      name: AUTH_COOKIE,
      value: username,
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      path: "/",
      maxAge: 60 * 60 * 8,
    })
    return response
  }

  return NextResponse.json({ success: false, message: "Credenciais inválidas" }, { status: 401 })
}
