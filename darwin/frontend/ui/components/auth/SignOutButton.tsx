// ... existing code ...
import useAuthStore from '../../store/auth'
import { useRouter } from 'next/navigation'

// ... existing code ...

export function SignOutButton() {
  const router = useRouter()
  const { logout } = useAuthStore()

  const handleLogout = () => {
    logout()
    router.push('/login')
  }

  return (
    <Button variant="ghost" onClick={handleLogout}>
      Sair
    </Button>
  )
}
// ... existing code ...