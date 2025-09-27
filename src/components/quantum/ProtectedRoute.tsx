"use client"

import React from 'react'

interface User {
  id: string
  name: string
  email: string
  role: 'admin' | 'researcher' | 'guest'
  avatar?: string
  permissions: string[]
}

interface ProtectedRouteProps {
  children: React.ReactNode
  user: User | null
  requiredPermissions?: string[]
  fallback?: React.ReactNode
}

interface RoleGuardProps {
  children: React.ReactNode
  user: User | null
  allowedRoles: ('admin' | 'researcher' | 'guest')[]
  fallback?: React.ReactNode
}

interface PermissionGuardProps {
  children: React.ReactNode
  user: User | null
  requiredPermissions: string[]
  fallback?: React.ReactNode
}

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
  children,
  user,
  requiredPermissions = [],
  fallback = <div className="p-8 text-center text-muted-foreground">Acesso negado</div>
}) => {
  if (!user) {
    return <>{fallback}</>
  }

  // Check if user has required permissions
  if (requiredPermissions.length > 0) {
    const hasPermission = requiredPermissions.some(permission => 
      user.permissions.includes(permission) || user.permissions.includes('all')
    )
    
    if (!hasPermission) {
      return <>{fallback}</>
    }
  }

  return <>{children}</>
}

export const RoleGuard: React.FC<RoleGuardProps> = ({
  children,
  user,
  allowedRoles,
  fallback = <div className="p-8 text-center text-muted-foreground">Acesso negado</div>
}) => {
  if (!user || !allowedRoles.includes(user.role)) {
    return <>{fallback}</>
  }

  return <>{children}</>
}

export const PermissionGuard: React.FC<PermissionGuardProps> = ({
  children,
  user,
  requiredPermissions,
  fallback = <div className="p-8 text-center text-muted-foreground">Acesso negado</div>
}) => {
  if (!user) {
    return <>{fallback}</>
  }

  const hasPermission = requiredPermissions.some(permission => 
    user.permissions.includes(permission) || user.permissions.includes('all')
  )
  
  if (!hasPermission) {
    return <>{fallback}</>
  }

  return <>{children}</>
}

export default ProtectedRoute
