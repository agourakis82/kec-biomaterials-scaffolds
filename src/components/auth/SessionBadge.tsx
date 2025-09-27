"use client"

import * as React from "react"
import { Loader2 } from "lucide-react"
import { Badge } from "@/components/ui/badge"

interface SessionResponse {
  authenticated: boolean
  username?: string
}

export function SessionBadge() {
  const [session, setSession] = React.useState<SessionResponse | null>(null)
  const [loading, setLoading] = React.useState(true)

  React.useEffect(() => {
    let active = true

    const fetchSession = async () => {
      try {
        const response = await fetch("/api/auth/session", { cache: "no-store" })
        if (!response.ok) {
          if (active) setSession({ authenticated: false })
          return
        }
        const data = (await response.json()) as SessionResponse
        if (active) setSession(data)
      } catch (error) {
        if (active) setSession({ authenticated: false })
      } finally {
        if (active) setLoading(false)
      }
    }

    fetchSession()
    return () => {
      active = false
    }
  }, [])

  if (loading) {
    return (
      <Badge variant="outline" className="flex items-center gap-1">
        <Loader2 className="h-3 w-3 animate-spin" />
        Verificando
      </Badge>
    )
  }

  if (!session?.authenticated || !session?.username) {
    return null
  }

  return (
    <Badge variant="outline" className="hidden items-center gap-1 text-xs font-medium md:inline-flex">
      Sessão ativa · {session.username}
    </Badge>
  )
}
