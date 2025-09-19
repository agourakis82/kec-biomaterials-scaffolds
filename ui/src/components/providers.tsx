"use client"

import * as React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { ToastProvider, ToastViewport } from "@/components/ui/toast"

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = React.useState(() => new QueryClient())

  return (
    <ToastProvider>
      <QueryClientProvider client={queryClient}>
        {children}
        <ToastViewport />
      </QueryClientProvider>
    </ToastProvider>
  )
}
