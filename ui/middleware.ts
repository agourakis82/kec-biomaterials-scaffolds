import { NextResponse } from "next/server"
import type { NextRequest } from "next/server"

const AUTH_COOKIE = "agourakis_auth"
const PUBLIC_PATHS = ["/api/auth/login", "/api/auth/logout", "/api/auth/session"]

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl

  const isLoginPage = pathname === "/login"
  const isPublicPath = PUBLIC_PATHS.some((path) => pathname.startsWith(path))
  const isStaticAsset = pathname.startsWith("/_next") || pathname.startsWith("/assets") || pathname === "/favicon.ico"

  const sessionCookie = request.cookies.get(AUTH_COOKIE)
  const validUsername = process.env.APP_USERNAME ?? "agourakis"

  if (isStaticAsset || isPublicPath) {
    return NextResponse.next()
  }

  if (isLoginPage) {
    if (sessionCookie?.value === validUsername) {
      const redirectUrl = request.nextUrl.searchParams.get("from") ?? "/"
      return NextResponse.redirect(new URL(redirectUrl, request.url))
    }
    return NextResponse.next()
  }

  if (sessionCookie?.value === validUsername) {
    return NextResponse.next()
  }

  const loginUrl = new URL("/login", request.url)
  loginUrl.searchParams.set("from", pathname)
  return NextResponse.redirect(loginUrl)
}

export const config = {
  matcher: ["/((?!_next|assets|favicon.ico).*)"],
}
