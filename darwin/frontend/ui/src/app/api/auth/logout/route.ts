import { NextResponse } from "next/server"

const AUTH_COOKIE = "agourakis_auth"

export async function POST() {
  const response = NextResponse.json({ success: true })
  response.cookies.set({
    name: AUTH_COOKIE,
    value: "",
    path: "/",
    expires: new Date(0),
  })
  return response
}
