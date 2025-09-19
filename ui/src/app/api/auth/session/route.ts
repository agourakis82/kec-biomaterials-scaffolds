import { NextResponse } from "next/server"
import { cookies } from "next/headers"

const AUTH_COOKIE = "agourakis_auth"

export async function GET() {
  const cookieStore = cookies()
  const session = cookieStore.get(AUTH_COOKIE)

  if (!session?.value) {
    return NextResponse.json({ authenticated: false }, { status: 401 })
  }

  return NextResponse.json({ authenticated: true, username: session.value })
}
