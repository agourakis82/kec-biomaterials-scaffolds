"use client"

import React from 'react'
import { cn } from '@/lib/utils'

interface QuantumLayoutProps {
  children: React.ReactNode
  author?: string
  title?: string
  subtitle?: string
  className?: string
}

export const QuantumLayout: React.FC<QuantumLayoutProps> = ({
  children,
  author,
  title,
  subtitle,
  className
}) => {
  return (
    <div className={cn("min-h-screen bg-background text-foreground", className)}>
      {/* Background Effects */}
      <div className="fixed inset-0 -z-10">
        <div className="absolute inset-0 bg-gradient-to-br from-purple-900/20 via-blue-900/20 to-green-900/20" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(139,92,246,0.1),transparent_50%)]" />
      </div>

      {/* Main Content */}
      <main className="relative z-10">
        {children}
      </main>

      {/* Footer */}
      {author && (
        <footer className="fixed bottom-4 right-4 z-20">
          <div className="bg-background/80 backdrop-blur-sm border border-border/50 rounded-lg px-3 py-2">
            <p className="text-xs text-muted-foreground">
              © {new Date().getFullYear()} {author}
            </p>
          </div>
        </footer>
      )}
    </div>
  )
}

export default QuantumLayout
