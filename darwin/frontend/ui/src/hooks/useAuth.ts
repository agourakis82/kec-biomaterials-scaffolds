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
          // Simulate API call to validate token
          const mockUser: User = {
            id: '1',
            name: 'Dr. Researcher',
            email: 'researcher@darwin.ai',
            role: 'researcher',
            permissions: ['research', 'chat', 'analytics']
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
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      // Mock successful login
      const mockUser: User = {
        id: '1',
        name: 'Dr. Researcher',
        email: data.email,
        role: 'researcher',
        permissions: ['research', 'chat', 'analytics', 'all']
      }

      localStorage.setItem('darwin_token', 'mock_token_123')
      
      setAuthState({
        user: mockUser,
        isAuthenticated: true,
        isLoading: false
      })

      return { success: true, user: mockUser }
    } catch (error) {
      setAuthState(prev => ({ ...prev, isLoading: false }))
      return { success: false, error: 'Login failed' }
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
    localStorage.removeItem('darwin_token')
    setAuthState({
      user: null,
      isAuthenticated: false,
      isLoading: false
    })
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
