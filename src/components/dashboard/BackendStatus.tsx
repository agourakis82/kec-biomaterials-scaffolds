"use client"

import React, { useState, useEffect } from 'react'
import { AlertTriangle, CheckCircle, XCircle, RefreshCw } from 'lucide-react'
import { cn } from '@/lib/utils'

interface BackendStatusProps {
  className?: string
}

export const BackendStatus: React.FC<BackendStatusProps> = ({ className }) => {
  const [status, setStatus] = useState<'checking' | 'online' | 'offline'>('checking')
  const [lastCheck, setLastCheck] = useState<Date>(new Date())

  const checkBackendStatus = async () => {
    setStatus('checking')
    try {
      const response = await fetch('/api/auth/me', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      })
      
      if (response.ok) {
        setStatus('online')
      } else {
        setStatus('offline')
      }
    } catch (error) {
      setStatus('offline')
    }
    setLastCheck(new Date())
  }

  useEffect(() => {
    checkBackendStatus()
    const interval = setInterval(checkBackendStatus, 30000) // Check every 30 seconds
    return () => clearInterval(interval)
  }, [])

  const getStatusConfig = () => {
    switch (status) {
      case 'online':
        return {
          icon: <CheckCircle className="w-4 h-4" />,
          text: 'Backend Online',
          className: 'bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 border-green-200 dark:border-green-800'
        }
      case 'offline':
        return {
          icon: <XCircle className="w-4 h-4" />,
          text: 'Modo Demo - Backend Offline',
          className: 'bg-yellow-100 dark:bg-yellow-900 text-yellow-800 dark:text-yellow-200 border-yellow-200 dark:border-yellow-800'
        }
      case 'checking':
        return {
          icon: <RefreshCw className="w-4 h-4 animate-spin" />,
          text: 'Verificando...',
          className: 'bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 border-blue-200 dark:border-blue-800'
        }
    }
  }

  const config = getStatusConfig()

  return (
    <div className={cn(
      "inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium border",
      config.className,
      className
    )}>
      {config.icon}
      <span>{config.text}</span>
      {status === 'offline' && (
        <button
          onClick={checkBackendStatus}
          className="ml-2 p-1 rounded-full hover:bg-black/10 dark:hover:bg-white/10 transition-colors"
          title="Tentar reconectar"
        >
          <RefreshCw className="w-3 h-3" />
        </button>
      )}
    </div>
  )
}

export default BackendStatus