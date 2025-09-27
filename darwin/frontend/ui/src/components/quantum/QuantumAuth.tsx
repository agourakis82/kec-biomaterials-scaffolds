"use client"

import React, { useState } from 'react'
import { cn } from '@/lib/utils'
import { DarwinLogo } from './DarwinLogo'

interface User {
  id: string
  name: string
  email: string
  role: 'admin' | 'researcher' | 'guest'
  avatar?: string
  permissions: string[]
}

interface QuantumAuthProps {
  onLogin: (user: User) => void
  onRegister: (data: any) => void
  isLoading?: boolean
}

export const QuantumAuth: React.FC<QuantumAuthProps> = ({
  onLogin,
  onRegister,
  isLoading = false
}) => {
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    role: 'researcher' as 'researcher' | 'guest'
  })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (mode === 'login') {
      // Mock login - in real app, this would call an API
      const mockUser: User = {
        id: '1',
        name: 'Dr. Researcher',
        email: formData.email,
        role: 'researcher',
        permissions: ['research', 'chat', 'analytics', 'all']
      }
      
      // Set cookie for middleware
      document.cookie = `darwin_token=mock_token_123; path=/; max-age=86400`
      
      onLogin(mockUser)
    } else {
      // Set cookie for middleware
      document.cookie = `darwin_token=mock_token_456; path=/; max-age=86400`
      
      onRegister(formData)
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setFormData(prev => ({
      ...prev,
      [e.target.name]: e.target.value
    }))
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-purple-900/20 via-blue-900/20 to-green-900/20">
      {/* Background Effects */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(139,92,246,0.1),transparent_50%)]" />
      
      <div className="relative z-10 w-full max-w-md p-8">
        {/* Auth Card */}
        <div className="bg-background/80 backdrop-blur-sm border border-border/50 rounded-2xl p-8 shadow-2xl">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="flex justify-center mb-4">
              <DarwinLogo size={64} variant="icon" animated={true} />
            </div>
            <h1 className="text-2xl font-bold mb-2">
              DARWIN AI
            </h1>
            <p className="text-muted-foreground">
              {mode === 'login' ? 'Entre na sua conta' : 'Crie sua conta'}
            </p>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-6">
            {mode === 'register' && (
              <div>
                <label htmlFor="name" className="block text-sm font-medium mb-2">
                  Nome Completo
                </label>
                <input
                  type="text"
                  id="name"
                  name="name"
                  value={formData.name}
                  onChange={handleInputChange}
                  required
                  className="w-full px-4 py-3 bg-background/50 border border-border/50 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  placeholder="Dr. Seu Nome"
                />
              </div>
            )}

            <div>
              <label htmlFor="email" className="block text-sm font-medium mb-2">
                Email
              </label>
              <input
                type="email"
                id="email"
                name="email"
                value={formData.email}
                onChange={handleInputChange}
                required
                className="w-full px-4 py-3 bg-background/50 border border-border/50 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                placeholder="seu@email.com"
              />
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-medium mb-2">
                Senha
              </label>
              <input
                type="password"
                id="password"
                name="password"
                value={formData.password}
                onChange={handleInputChange}
                required
                className="w-full px-4 py-3 bg-background/50 border border-border/50 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                placeholder="••••••••"
              />
            </div>

            {mode === 'register' && (
              <div>
                <label htmlFor="role" className="block text-sm font-medium mb-2">
                  Tipo de Conta
                </label>
                <select
                  id="role"
                  name="role"
                  value={formData.role}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 bg-background/50 border border-border/50 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                >
                  <option value="researcher">Pesquisador</option>
                  <option value="guest">Visitante</option>
                </select>
              </div>
            )}

            <button
              type="submit"
              disabled={isLoading}
              className={cn(
                "w-full py-3 px-4 rounded-lg font-medium transition-all duration-200",
                "bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600",
                "text-white shadow-lg hover:shadow-xl",
                "disabled:opacity-50 disabled:cursor-not-allowed"
              )}
            >
              {isLoading ? (
                <div className="flex items-center justify-center">
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                  Processando...
                </div>
              ) : (
                mode === 'login' ? 'Entrar' : 'Criar Conta'
              )}
            </button>
          </form>

          {/* Toggle Mode */}
          <div className="mt-6 text-center">
            <button
              type="button"
              onClick={() => setMode(mode === 'login' ? 'register' : 'login')}
              className="text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              {mode === 'login' ? (
                <>Não tem conta? <span className="text-purple-500 font-medium">Criar conta</span></>
              ) : (
                <>Já tem conta? <span className="text-purple-500 font-medium">Fazer login</span></>
              )}
            </button>
          </div>

          {/* Demo Info */}
          <div className="mt-6 p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
            <p className="text-xs text-center text-muted-foreground">
              <strong>Demo:</strong> Use qualquer email/senha para testar
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-8">
          <p className="text-xs text-muted-foreground">
            © 2024 Dr. Demetrios Chiuratto Agourakis
          </p>
        </div>
      </div>
    </div>
  )
}

export default QuantumAuth
