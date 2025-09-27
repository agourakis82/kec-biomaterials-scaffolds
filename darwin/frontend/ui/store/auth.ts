import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { devtools } from 'zustand/middleware'

interface AuthState {
  token: string | null
  isAuthenticated: boolean
  username: string | null
  login: (username: string, password: string) => Promise<boolean>
  logout: () => void
}

const useAuthStore = create<AuthState>()(
  devtools(
    persist(
      (set, get) => ({
        token: null,
        isAuthenticated: false,
        username: null,
        login: async (username: string, password: string) => {
          try {
            const response = await fetch('http://localhost:8000/auth/login', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ username, password }),
            })
            if (!response.ok) throw new Error('Login failed')
            const data = await response.json()
            const { access_token: token } = data
            set({ token, isAuthenticated: true, username })
            return true
          } catch (error) {
            console.error('Login error:', error)
            return false
          }
        },
        logout: () => {
          set({ token: null, isAuthenticated: false, username: null })
          localStorage.removeItem('zustand')
        },
      }),
      {
        name: 'auth-storage',
        partialize: (state) => ({ token: state.token, username: state.username }),
      }
    ),
    { name: 'AuthStore' }
  )
)

export default useAuthStore