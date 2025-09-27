"use client"

import React, { useEffect, useState } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ThemeProvider } from 'next-themes'
import ModernAuth from '../src/components/auth/ModernAuth'
import ModernDashboard from '../src/components/dashboard/ModernDashboard'
import { useAuth } from '../src/hooks/useAuth'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,
      refetchOnWindowFocus: false,
    },
  },
})

function AppContent() {
  const { user, isAuthenticated, isLoading, login, register, logout } = useAuth()
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  if (!isAuthenticated || !user) {
    return (
      <ModernAuth
        onLogin={(userData) => {
          // Login successful, auth state will be updated automatically
          console.log('Login successful:', userData)
        }}
        onRegister={(registerData) => {
          // Registration successful, auth state will be updated automatically
          console.log('Registration successful:', registerData)
        }}
        isLoading={isLoading}
      />
    )
  }

  return (
    <ModernDashboard
      user={user}
      onLogout={logout}
    />
  )
}

export default function HomePage() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider 
        attribute="class" 
        defaultTheme="light" 
        enableSystem={false}
        disableTransitionOnChange
        storageKey="darwin-theme"
      >
        <AppContent />
      </ThemeProvider>
    </QueryClientProvider>
  )
}