"use client"

import React, { useState } from 'react'
import { User, LogOut, Settings, ChevronDown } from 'lucide-react'
import { cn } from '@/lib/utils'

interface User {
  id: string
  name: string
  email: string
  role: 'admin' | 'researcher' | 'guest'
  avatar?: string
  permissions: string[]
}

interface UserProfileProps {
  user: User
  onLogout: () => void
  onUpdateProfile: (updates: Partial<User>) => void
}

export const UserProfile: React.FC<UserProfileProps> = ({
  user,
  onLogout,
  onUpdateProfile
}) => {
  const [isOpen, setIsOpen] = useState(false)

  const getRoleColor = (role: string) => {
    switch (role) {
      case 'admin':
        return 'text-red-500'
      case 'researcher':
        return 'text-blue-500'
      case 'guest':
        return 'text-green-500'
      default:
        return 'text-gray-500'
    }
  }

  const getRoleLabel = (role: string) => {
    switch (role) {
      case 'admin':
        return 'Administrador'
      case 'researcher':
        return 'Pesquisador'
      case 'guest':
        return 'Visitante'
      default:
        return role
    }
  }

  return (
    <div className="relative">
      {/* Profile Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-3 p-2 rounded-lg bg-background/50 border border-border/50 hover:bg-background/80 transition-colors"
      >
        {/* Avatar */}
        <div className="w-8 h-8 rounded-full bg-gradient-to-r from-purple-500 to-blue-500 flex items-center justify-center">
          {user.avatar ? (
            <img
              src={user.avatar}
              alt={user.name}
              className="w-full h-full rounded-full object-cover"
            />
          ) : (
            <User className="w-4 h-4 text-white" />
          )}
        </div>

        {/* User Info */}
        <div className="text-left">
          <div className="text-sm font-medium">{user.name}</div>
          <div className={cn("text-xs", getRoleColor(user.role))}>
            {getRoleLabel(user.role)}
          </div>
        </div>

        <ChevronDown className={cn(
          "w-4 h-4 transition-transform",
          isOpen && "rotate-180"
        )} />
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-10"
            onClick={() => setIsOpen(false)}
          />

          {/* Menu */}
          <div className="absolute right-0 top-full mt-2 w-64 bg-background/90 backdrop-blur-sm border border-border/50 rounded-lg shadow-xl z-20">
            {/* User Info Header */}
            <div className="p-4 border-b border-border/50">
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 rounded-full bg-gradient-to-r from-purple-500 to-blue-500 flex items-center justify-center">
                  {user.avatar ? (
                    <img
                      src={user.avatar}
                      alt={user.name}
                      className="w-full h-full rounded-full object-cover"
                    />
                  ) : (
                    <User className="w-6 h-6 text-white" />
                  )}
                </div>
                <div>
                  <div className="font-medium">{user.name}</div>
                  <div className="text-sm text-muted-foreground">{user.email}</div>
                  <div className={cn("text-xs font-medium", getRoleColor(user.role))}>
                    {getRoleLabel(user.role)}
                  </div>
                </div>
              </div>
            </div>

            {/* Permissions */}
            <div className="p-4 border-b border-border/50">
              <div className="text-sm font-medium mb-2">Permissões:</div>
              <div className="flex flex-wrap gap-1">
                {user.permissions.map((permission) => (
                  <span
                    key={permission}
                    className="px-2 py-1 text-xs bg-purple-500/20 text-purple-300 rounded-full"
                  >
                    {permission}
                  </span>
                ))}
              </div>
            </div>

            {/* Menu Items */}
            <div className="p-2">
              <button
                onClick={() => {
                  setIsOpen(false)
                  // Handle settings
                }}
                className="w-full flex items-center gap-3 p-2 text-left hover:bg-background/50 rounded-lg transition-colors"
              >
                <Settings className="w-4 h-4" />
                <span className="text-sm">Configurações</span>
              </button>

              <button
                onClick={() => {
                  setIsOpen(false)
                  onLogout()
                }}
                className="w-full flex items-center gap-3 p-2 text-left hover:bg-red-500/10 text-red-500 rounded-lg transition-colors"
              >
                <LogOut className="w-4 h-4" />
                <span className="text-sm">Sair</span>
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

export default UserProfile
