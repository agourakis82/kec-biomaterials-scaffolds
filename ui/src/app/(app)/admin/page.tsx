"use client"

import * as React from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { darwin } from "@/lib/darwin"
import { motion } from "framer-motion"

export default function AdminPage() {
  const [loading, setLoading] = React.useState(false)
  const [status, setStatus] = React.useState<any>(null)

  const ping = async () => {
    setLoading(true)
    try {
      const resp = await darwin.adminStatus()
      setStatus(resp)
    } finally {
      setLoading(false)
    }
  }

  const exportJsonl = () => {
    const rows = [
      { id: 1, title: "Example Source", url: "https://example.com" },
      { id: 2, title: "Another", url: "https://example.com/2" },
    ]
    const blob = new Blob(rows.map((r) => JSON.stringify(r) + "\n"), { type: "application/jsonl" } as any)
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "corpus.jsonl"
    a.click()
    URL.revokeObjectURL(url)
  }

  React.useEffect(() => {
    void ping()
  }, [])

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Status do Backend</CardTitle>
        </CardHeader>
        <CardContent>
          {loading && <Skeleton className="h-12 w-full" />}
          {!loading && (
            <div className="flex items-center justify-between">
              <div className="text-sm text-muted-foreground">
                {status ? "Online" : "Sem dados"}
              </div>
              <motion.div whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.98 }}>
                <Button onClick={ping} variant="outline">Ping</Button>
              </motion.div>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Export</CardTitle>
        </CardHeader>
        <CardContent>
          <motion.div whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.98 }}>
            <Button onClick={exportJsonl}>Export .jsonl</Button>
          </motion.div>
        </CardContent>
      </Card>
    </div>
  )
}

