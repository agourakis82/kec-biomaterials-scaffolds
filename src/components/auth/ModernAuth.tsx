"use client"

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Eye, EyeOff, Mail, Lock, User, Sparkles, Shield, ArrowRight, Brain, Zap } from 'lucide-react'
import { cn, apiRequest } from '@/lib/utils'

interface User {
  id: string
  name: string
  email: string
  role: 'admin' | 'researcher' | 'guest'
  avatar?: string
  permissions: string[]
}

interface ModernAuthProps {
  onLogin: (user: User) => void
  onRegister: (data: any) => void
  isLoading?: boolean
}

export const ModernAuth: React.FC<ModernAuthProps> = ({
  onLogin,
  onRegister,
  isLoading = false
}) => {
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [showPassword, setShowPassword] = useState(false)
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    role: 'researcher' as 'researcher' | 'guest'
  })
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setErrors({})
    setLoading(true)
    
    // Basic validation
    const newErrors: Record<string, string> = {}
    
    if (!formData.email) {
      newErrors.email = 'Email é obrigatório'
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Email inválido'
    }
    
    if (!formData.password) {
      newErrors.password = 'Senha é obrigatória'
    } else if (formData.password.length < 6) {
      newErrors.password = 'Senha deve ter pelo menos 6 caracteres'
    }
    
    if (mode === 'register' && !formData.name) {
      newErrors.name = 'Nome é obrigatório'
    }
    
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors)
      setLoading(false)
      return
    }
    
    try {
      if (mode === 'login') {
        // Try API first, fallback to mock
        try {
          const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              email: formData.email,
              password: formData.password
            })
          })

          const result = await response.json()

          if (response.ok && result.success) {
            localStorage.setItem('darwin_token', result.access_token)
            onLogin(result.user)
            return
          }
        } catch (apiError) {
          console.log('API unavailable, using offline mode')
        }

        // Fallback to mock authentication
        await new Promise(resolve => setTimeout(resolve, 1000))
        
        const mockUser: User = {
          id: '1',
          name: 'Dr. Demetrios Chiuratto Agourakis',
          email: formData.email,
          role: 'admin',
          permissions: ['research', 'chat', 'analytics', 'admin', 'all']
        }
        
        localStorage.setItem('darwin_token', 'mock_token_' + Date.now())
        onLogin(mockUser)
      } else {
        // Registration - always use mock for now
        await new Promise(resolve => setTimeout(resolve, 1000))
        
        const mockUser: User = {
          id: '2',
          name: formData.name,
          email: formData.email,
          role: formData.role,
          permissions: formData.role === 'researcher' ? ['research', 'chat', 'analytics'] : ['chat']
        }
        
        localStorage.setItem('darwin_token', 'mock_token_' + Date.now())
        onRegister(mockUser)
      }
    } catch (error) {
      setErrors({ general: 'Erro de autenticação. Tente novamente.' })
    } finally {
      setLoading(false)
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target
    setFormData(prev => ({ ...prev, [name]: value }))
    
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: '' }))
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center relative overflow-hidden p-4">
      {/* Modern Background */}
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 dark:from-slate-950 dark:via-blue-950 dark:to-indigo-950" />
        
        {/* Animated Elements */}
        <motion.div
          animate={{
            scale: [1, 1.2, 1],
            rotate: [0, 180, 360],
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            ease: "linear"
          }}
          className="absolute top-1/4 left-1/4 w-96 h-96 bg-gradient-to-r from-blue-400/10 to-purple-400/10 rounded-full blur-3xl"
        />
        <motion.div
          animate={{
            scale: [1.2, 1, 1.2],
            rotate: [360, 180, 0],
          }}
          transition={{
            duration: 25,
            repeat: Infinity,
            ease: "linear"
          }}
          className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-gradient-to-r from-purple-400/10 to-pink-400/10 rounded-full blur-3xl"
        />
      </div>

      {/* Main Content */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="relative z-10 w-full max-w-md"
      >
        {/* Auth Card */}
        <div className="glass-card p-8">
          {/* Header */}
          <div className="text-center mb-8">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
              className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-r from-blue-500 to-purple-500 rounded-3xl mb-6 shadow-2xl"
            >
              <Brain className="w-10 h-10 text-white" />
            </motion.div>
            
            <motion.h1
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
              className="text-4xl font-bold gradient-text mb-3"
            >
              DARWIN AI
            </motion.h1>
            
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 }}
              className="text-muted-foreground text-lg"
            >
              Plataforma de Pesquisa Inteligente
            </motion.p>
          </div>

          {/* Mode Toggle */}
          <div className="flex bg-muted/30 rounded-2xl p-1 mb-8">
            <button
              type="button"
              onClick={() => setMode('login')}
              className={cn(
                "flex-1 py-3 px-6 rounded-xl text-sm font-semibold transition-all duration-300",
                mode === 'login'
                  ? "bg-white dark:bg-slate-800 shadow-lg text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              Entrar
            </button>
            <button
              type="button"
              onClick={() => setMode('register')}
              className={cn(
                "flex-1 py-3 px-6 rounded-xl text-sm font-semibold transition-all duration-300",
                mode === 'register'
                  ? "bg-white dark:bg-slate-800 shadow-lg text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              Registrar
            </button>
          </div>

          {/* Form */}
          <AnimatePresence mode="wait">
            <motion.form
              key={mode}
              initial={{ opacity: 0, x: mode === 'login' ? -20 : 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: mode === 'login' ? 20 : -20 }}
              transition={{ duration: 0.3 }}
              onSubmit={handleSubmit}
              className="space-y-6"
            >
              {mode === 'register' && (
                <div>
                  <label htmlFor="name" className="block text-sm font-semibold mb-3">
                    Nome Completo
                  </label>
                  <div className="relative">
                    <User className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                    <input
                      type="text"
                      id="name"
                      name="name"
                      value={formData.name}
                      onChange={handleInputChange}
                      className={cn(
                        "form-input pl-12 h-14 text-lg",
                        errors.name && "border-red-500 focus:ring-red-500"
                      )}
                      placeholder="Dr. Seu Nome"
                    />
                  </div>
                  {errors.name && (
                    <p className="text-red-500 text-sm mt-2">{errors.name}</p>
                  )}
                </div>
              )}

              <div>
                <label htmlFor="email" className="block text-sm font-semibold mb-3">
                  Email
                </label>
                <div className="relative">
                  <Mail className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                  <input
                    type="email"
                    id="email"
                    name="email"
                    value={formData.email}
                    onChange={handleInputChange}
                    className={cn(
                      "form-input pl-12 h-14 text-lg",
                      errors.email && "border-red-500 focus:ring-red-500"
                    )}
                    placeholder="seu@email.com"
                  />
                </div>
                {errors.email && (
                  <p className="text-red-500 text-sm mt-2">{errors.email}</p>
                )}
              </div>

              <div>
                <label htmlFor="password" className="block text-sm font-semibold mb-3">
                  Senha
                </label>
                <div className="relative">
                  <Lock className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                  <input
                    type={showPassword ? "text" : "password"}
                    id="password"
                    name="password"
                    value={formData.password}
                    onChange={handleInputChange}
                    className={cn(
                      "form-input pl-12 pr-12 h-14 text-lg",
                      errors.password && "border-red-500 focus:ring-red-500"
                    )}
                    placeholder="••••••••"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-4 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                  >
                    {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                  </button>
                </div>
                {errors.password && (
                  <p className="text-red-500 text-sm mt-2">{errors.password}</p>
                )}
              </div>

              {mode === 'register' && (
                <div>
                  <label htmlFor="role" className="block text-sm font-semibold mb-3">
                    Tipo de Conta
                  </label>
                  <select
                    id="role"
                    name="role"
                    value={formData.role}
                    onChange={handleInputChange}
                    className="form-input h-14 text-lg"
                  >
                    <option value="researcher">Pesquisador</option>
                    <option value="guest">Visitante</option>
                  </select>
                </div>
              )}

              {errors.general && (
                <div className="p-4 bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800 rounded-xl">
                  <p className="text-red-600 dark:text-red-400 text-sm font-medium">
                    {errors.general}
                  </p>
                </div>
              )}

              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                type="submit"
                disabled={loading}
                className="btn-primary w-full h-14 text-lg disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? (
                  <div className="flex items-center justify-center">
                    <div className="loading-spinner mr-3" />
                    Processando...
                  </div>
                ) : (
                  <div className="flex items-center justify-center">
                    {mode === 'login' ? 'Entrar na Plataforma' : 'Criar Conta'}
                    <ArrowRight className="w-5 h-5 ml-3" />
                  </div>
                )}
              </motion.button>
            </motion.form>
          </AnimatePresence>

          {/* Demo Info */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 }}
            className="mt-8 p-6 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950/20 dark:to-purple-950/20 border border-blue-200 dark:border-blue-800 rounded-2xl"
          >
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2 bg-blue-100 dark:bg-blue-900 rounded-lg">
                <Zap className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              </div>
              <span className="text-lg font-semibold text-blue-700 dark:text-blue-300">
                Modo Demonstração
              </span>
            </div>
            <p className="text-blue-600 dark:text-blue-400 leading-relaxed">
              Use qualquer email e senha (mín. 6 caracteres) para testar a plataforma. 
              O sistema está configurado para demonstração com dados simulados.
            </p>
          </motion.div>
        </div>

        {/* Footer */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
          className="text-center mt-8"
        >
          <p className="text-muted-foreground font-medium">
            © 2024 Dr. Demetrios Chiuratto Agourakis
          </p>
          <p className="text-muted-foreground text-sm mt-2">
            AGOURAKIS MED RESEARCH - Plataforma de Pesquisa Inteligente
          </p>
        </motion.div>
      </motion.div>
    </div>
  )
}

export default ModernAuth