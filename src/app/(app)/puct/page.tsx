"use client"

import * as React from "react"
import dynamic from "next/dynamic"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { darwin } from "@/lib/darwin"

const PuctGraph = dynamic(() => import("@/components/puct/PuctGraph").then(m => m.PuctGraph), { ssr: false })

export default function PuctPage() {
  const [root, setRoot] = React.useState("")
  const [budget, setBudget] = React.useState<number>(50)
  const [cPuct, setCPuct] = React.useState<number>(1.4)
  const [data, setData] = React.useState<any>(null)
  const [loading, setLoading] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)

  const run = async () => {
    setLoading(true)
    setError(null)
    try {
      const resp = await darwin.puct(root, budget, cPuct)
      setData(resp)
    } catch (e: any) {
      setError(e?.message ?? "Erro no PUCT")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>PUCT</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-4">
            <Input placeholder="root" value={root} onChange={(e) => setRoot(e.target.value)} />
            <Input
              type="number"
              placeholder="budget"
              value={budget}
              onChange={(e) => setBudget(parseInt(e.target.value || "0", 10))}
            />
            <Input
              type="number"
              step="0.1"
              placeholder="c_puct"
              value={cPuct}
              onChange={(e) => setCPuct(parseFloat(e.target.value || "0"))}
            />
            <Button onClick={run} disabled={loading}>Rodar</Button>
          </div>
        </CardContent>
      </Card>

      {loading && <Skeleton className="h-[600px] w-full" />}
      {error && (
        <div className="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm">{error}</div>
      )}
      {!loading && <PuctGraph data={data} />}
    </div>
  )
}
