"use client"

import * as React from "react"
import { useRouter } from "next/navigation"
import { LogOut } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useAuthStore } from "@/store/auth"

export function SignOutButton() {
  const router = useRouter()
  const { logout } = useAuthStore()
  const [loading, setLoading] = React.useState(false)

  const handleSignOut = async () => {
    setLoading(true)
    try {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8090"
      await fetch(`${backendUrl}/api/auth/logout`, { method: "POST" })
      logout()
      router.replace("/login")
    } catch (error) {
      console.error("Erro ao finalizar sessão", error)
      logout()
      router.replace("/login")
    } finally {
      setLoading(false)
    }
  }

  return (
    <Button
      variant="ghost"
      size="sm"
      className="hidden items-center gap-1 text-xs md:inline-flex"
      onClick={handleSignOut}
      disabled={loading}
    >
      <LogOut className="h-4 w-4" />
      {loading ? "Saindo..." : "Sair"}
    </Button>
  )
}
