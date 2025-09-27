import { NextResponse } from "next/server"
import type { NextRequest } from "next/server"

const PUBLIC_PATHS = ["/api/auth/login", "/api/auth/logout", "/api/auth/session", "/api/auth/me"]

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl

  const isPublicPath = PUBLIC_PATHS.some((path) => pathname.startsWith(path))
  const isStaticAsset = pathname.startsWith("/_next") || 
                       pathname.startsWith("/assets") || 
                       pathname === "/favicon.ico" ||
                       pathname.startsWith("/public")

  // Allow all static assets and API routes
  if (isStaticAsset || isPublicPath) {
    return NextResponse.next()
  }

  // For the main app, let client-side handle authentication
  // This removes the server-side redirect that was causing issues
  return NextResponse.next()
}

export const config = {
  matcher: ["/((?!_next|assets|favicon.ico|public).*)"],
}