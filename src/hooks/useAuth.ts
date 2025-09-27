"use client"

import { useState, useEffect } from 'react'

interface User {
  id: string
  name: string
  email: string
  role: 'admin' | 'researcher' | 'guest'
  avatar?: string
  permissions: string[]
}

interface AuthState {
  user: User | null
  isAuthenticated: boolean
  isLoading: boolean
}

interface LoginData {
  email: string
  password: string
}

interface RegisterData {
  name: string
  email: string
  password: string
  role?: 'researcher' | 'guest'
}

interface AuthResult {
  success: boolean
  error?: string
  user?: User
}

export const useAuth = () => {
  const [authState, setAuthState] = useState<AuthState>({
    user: null,
    isAuthenticated: false,
    isLoading: true
  })

  // Check for existing session on mount
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const token = localStorage.getItem('darwin_token')
        if (token) {
          // Try API call first, fallback to mock if backend unavailable
          try {
            const response = await fetch('/api/auth/me', {
              headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
              }
            })

            if (response.ok) {
              const data = await response.json()
              setAuthState({
                user: data.user,
                isAuthenticated: true,
                isLoading: false
              })
              return
            }
          } catch (apiError) {
            console.log('API unavailable, using offline mode')
          }

          // Fallback to mock user for demo
          const mockUser: User = {
            id: '1',
            name: 'Dr. Demetrios Chiuratto Agourakis',
            email: 'agourakis@medresearch.com',
            role: 'admin',
            permissions: ['research', 'chat', 'analytics', 'admin', 'all']
          }
          
          setAuthState({
            user: mockUser,
            isAuthenticated: true,
            isLoading: false
          })
        } else {
          setAuthState({
            user: null,
            isAuthenticated: false,
            isLoading: false
          })
        }
      } catch (error) {
        console.error('Auth check failed:', error)
        setAuthState({
          user: null,
          isAuthenticated: false,
          isLoading: false
        })
      }
    }

    checkAuth()
  }, [])

  const login = async (data: LoginData): Promise<AuthResult> => {
    try {
      setAuthState(prev => ({ ...prev, isLoading: true }))
      
      // Try real API first, fallback to mock
      try {
        const response = await fetch('/api/auth/login', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(data)
        })

        const result = await response.json()

        if (response.ok && result.success) {
          localStorage.setItem('darwin_token', result.access_token)
          
          setAuthState({
            user: result.user,
            isAuthenticated: true,
            isLoading: false
          })

          return { success: true, user: result.user }
        }
      } catch (apiError) {
        console.log('API unavailable, using offline authentication')
      }

      // Fallback to mock authentication for demo
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      const mockUser: User = {
        id: '1',
        name: 'Dr. Demetrios Chiuratto Agourakis',
        email: data.email,
        role: 'admin',
        permissions: ['research', 'chat', 'analytics', 'admin', 'all']
      }

      localStorage.setItem('darwin_token', 'mock_token_' + Date.now())
      
      setAuthState({
        user: mockUser,
        isAuthenticated: true,
        isLoading: false
      })

      return { success: true, user: mockUser }
    } catch (error) {
      setAuthState(prev => ({ ...prev, isLoading: false }))
      return { success: false, error: 'Erro de autenticação' }
    }
  }

  const register = async (data: RegisterData): Promise<AuthResult> => {
    try {
      setAuthState(prev => ({ ...prev, isLoading: true }))
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      // Mock successful registration
      const mockUser: User = {
        id: '2',
        name: data.name,
        email: data.email,
        role: data.role || 'guest',
        permissions: data.role === 'researcher' ? ['research', 'chat'] : ['chat']
      }

      localStorage.setItem('darwin_token', 'mock_token_456')
      
      setAuthState({
        user: mockUser,
        isAuthenticated: true,
        isLoading: false
      })

      return { success: true, user: mockUser }
    } catch (error) {
      setAuthState(prev => ({ ...prev, isLoading: false }))
      return { success: false, error: 'Registration failed' }
    }
  }

  const logout = async (): Promise<void> => {
    try {
      // Call logout API to invalidate token on server
      await fetch('/api/auth/logout', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('darwin_token')}`,
          'Content-Type': 'application/json'
        }
      })
    } catch (error) {
      console.error('Logout API call failed:', error)
    } finally {
      localStorage.removeItem('darwin_token')
      setAuthState({
        user: null,
        isAuthenticated: false,
        isLoading: false
      })
    }
  }

  const updateProfile = async (updates: Partial<User>): Promise<AuthResult> => {
    try {
      if (!authState.user) {
        return { success: false, error: 'No user logged in' }
      }

      const updatedUser = { ...authState.user, ...updates }
      
      setAuthState(prev => ({
        ...prev,
        user: updatedUser
      }))

      return { success: true, user: updatedUser }
    } catch (error) {
      return { success: false, error: 'Profile update failed' }
    }
  }

  return {
    user: authState.user,
    isAuthenticated: authState.isAuthenticated,
    isLoading: authState.isLoading,
    login,
    register,
    logout,
    updateProfile
  }
}

export default useAuth
