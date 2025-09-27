import { NextResponse } from "next/server"
import type { NextRequest } from "next/server"
import { jwtVerify } from "jose"

const PUBLIC_PATHS = ["/login", "/api/auth/login", "/api/auth/logout", "/api/auth/session"]
const SECRET_KEY = process.env.AUTH_SECRET_KEY || "your-secret-key-change-in-production"

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl

  const isLoginPage = pathname === "/login"
  const isPublicPath = PUBLIC_PATHS.some((path) => pathname.startsWith(path))
  const isStaticAsset = pathname.startsWith("/_next") || pathname.startsWith("/assets") || pathname === "/favicon.ico"

  if (isStaticAsset || isPublicPath) {
    return NextResponse.next()
  }

  // Check for token in Authorization header or cookie
  const authHeader = request.headers.get("Authorization")
  const tokenCookie = request.cookies.get("darwin_token")
  
  let token = null
  if (authHeader && authHeader.startsWith("Bearer ")) {
    token = authHeader.substring(7)
  } else if (tokenCookie) {
    token = tokenCookie.value
  }

  // For demo purposes, accept any token that looks like our mock tokens
  const hasValidToken = token && (token.includes("mock_token") || token.length > 10)

  if (isLoginPage) {
    if (hasValidToken) {
      const redirectUrl = request.nextUrl.searchParams.get("from") ?? "/"
      return NextResponse.redirect(new URL(redirectUrl, request.url))
    }
    return NextResponse.next()
  }

  if (!hasValidToken) {
    const loginUrl = new URL("/login", request.url)
    loginUrl.searchParams.set("from", pathname)
    return NextResponse.redirect(loginUrl)
  }

  return NextResponse.next()
}

export const config = {
  matcher: ["/((?!_next|assets|favicon.ico).*)"],
}
