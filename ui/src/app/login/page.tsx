"use client"

import * as React from "react"
import Link from "next/link"
import { useRouter, useSearchParams } from "next/navigation"
import { motion } from "framer-motion"
import { Shield, Sparkles } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"

export default function LoginPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const redirectTo = searchParams.get("from") ?? "/"

  const [username, setUsername] = React.useState("")
  const [password, setPassword] = React.useState("")
  const [error, setError] = React.useState<string | null>(null)
  const [loading, setLoading] = React.useState(false)

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setError(null)
    setLoading(true)

    try {
      const response = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      })

      if (response.ok) {
        setUsername("")
        setPassword("")
        router.replace(redirectTo)
        return
      }

      const payload = await response.json()
      setError(payload?.message ?? "Não foi possível autenticar")
    } catch (err) {
      setError("Erro ao comunicar com o servidor. Tente novamente.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-br from-background via-background to-primary/10 px-4 py-12">
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="w-full max-w-md"
      >
        <Card className="border-primary/20 bg-background/90 backdrop-blur">
          <CardHeader className="space-y-4 text-center">
            <div className="flex justify-center">
              <Badge variant="secondary" className="flex items-center gap-2 border border-primary/40 bg-primary/10 text-primary">
                <Shield className="h-4 w-4" />
                Acesso restrito
              </Badge>
            </div>
            <div className="space-y-2">
              <CardTitle className="text-2xl font-semibold">Agourakis Med Research</CardTitle>
              <p className="text-sm text-muted-foreground">
                Área dedicada à curadoria científica conduzida por Demetrios Chiuratto Agourakis.
              </p>
            </div>
          </CardHeader>
          <CardContent>
            <form className="space-y-4" onSubmit={handleSubmit}>
              <div className="space-y-1">
                <label htmlFor="username" className="text-sm font-medium">Usuário</label>
                <Input
                  id="username"
                  value={username}
                  autoComplete="username"
                  onChange={(event) => setUsername(event.target.value)}
                  placeholder="Informe o usuário autorizado"
                  required
                />
              </div>
              <div className="space-y-1">
                <label htmlFor="password" className="text-sm font-medium">Senha</label>
                <Input
                  id="password"
                  type="password"
                  value={password}
                  autoComplete="current-password"
                  onChange={(event) => setPassword(event.target.value)}
                  placeholder="Digite sua senha de acesso"
                  required
                />
              </div>

              {error && (
                <p className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
                  {error}
                </p>
              )}

              <Button type="submit" className="w-full" disabled={loading}>
                {loading ? "Validando credenciais..." : "Entrar"}
              </Button>
            </form>

            <div className="mt-6 space-y-2 text-center text-xs text-muted-foreground">
              <p>
                Necessita de suporte? Escreva para
                {" "}
                <Link href="mailto:contato@agourakis.med.br" className="text-primary underline-offset-2 hover:underline">
                  contato@agourakis.med.br
                </Link>
              </p>
              <p className="flex items-center justify-center gap-2">
                <Sparkles className="h-3 w-3" />
                As credenciais são definidas via variáveis de ambiente seguras.
              </p>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  )
}
