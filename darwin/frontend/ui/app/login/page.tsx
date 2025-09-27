'use client'

import QuantumAuth from '../../src/components/quantum/QuantumAuth'
import { useRouter } from 'next/navigation'

export default function LoginPage() {
  const router = useRouter()

  const handleLogin = (user: any) => {
    // Simular login bem-sucedido
    localStorage.setItem('darwin_token', 'mock_token_123')
    const from = new URLSearchParams(window.location.search).get('from') || '/'
    router.push(from)
  }

  const handleRegister = (data: any) => {
    // Simular registro bem-sucedido
    localStorage.setItem('darwin_token', 'mock_token_456')
    const from = new URLSearchParams(window.location.search).get('from') || '/'
    router.push(from)
  }

  return (
    <QuantumAuth 
      onLogin={handleLogin}
      onRegister={handleRegister}
      isLoading={false}
    />
  )
}
