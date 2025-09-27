"use client"

import React, { useState } from 'react'
import { GrApple, GrFormView, GrKey } from 'react-icons/gr'
import * as Form from '@radix-ui/react-form'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card'

interface DarwinAuthProps {
  onLogin: (username: string, password: string) => void
  isLoading?: boolean
  error?: string
}

export function DarwinAuth({ onLogin, isLoading = false, error }: DarwinAuthProps) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [errors, setErrors] = useState<{username?: string, password?: string}>({})

  const validateForm = () => {
    const newErrors: {username?: string, password?: string} = {}
    
    if (!username.trim()) {
      newErrors.username = 'Username is required'
    }
    
    if (!password) {
      newErrors.password = 'Password is required'
    } else if (password.length < 6) {
      newErrors.password = 'Password must be at least 6 characters'
    }
    
    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (validateForm()) {
      onLogin(username, password)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-4" 
         style={{ background: 'var(--bg-main)' }}>
      
      {/* Background decorative elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-10 w-24 h-24 rounded-full darwin-card flex items-center justify-center opacity-20">
          <GrApple className="text-4xl" style={{ color: 'hsl(var(--darwin-primary))' }} />
        </div>
        <div className="absolute bottom-20 right-10 w-32 h-32 rounded-full darwin-card flex items-center justify-center opacity-20">
          <GrFormView className="text-5xl" style={{ color: 'hsl(var(--darwin-accent))' }} />
        </div>
      </div>

      <Card className="w-full max-w-md darwin-card">
        <CardHeader className="text-center space-y-4">
          <div className="mx-auto w-16 h-16 rounded-full darwin-card flex items-center justify-center">
            <GrApple className="text-2xl" style={{ color: 'hsl(var(--darwin-primary))' }} />
          </div>
          
          <div>
            <CardTitle className="text-2xl font-darwin-display" 
                      style={{ color: 'hsl(var(--darwin-primary))' }}>
              Darwin
            </CardTitle>
            <CardDescription className="font-darwin-body">
              Scientific Research Platform
            </CardDescription>
          </div>
        </CardHeader>

        <CardContent>
          <Form.Root onSubmit={handleSubmit}>
            <div className="space-y-4">
              <Form.Field name="username">
                <div className="space-y-2">
                  <Label htmlFor="username" className="font-darwin-body font-medium">
                    Username
                  </Label>
                  <Input
                    id="username"
                    type="text"
                    placeholder="Enter your username"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    className="darwin-input"
                    required
                  />
                  {errors.username && (
                    <Form.Message className="text-sm darwin-status-error">
                      {errors.username}
                    </Form.Message>
                  )}
                </div>
              </Form.Field>

              <Form.Field name="password">
                <div className="space-y-2">
                  <Label htmlFor="password" className="font-darwin-body font-medium">
                    Password
                  </Label>
                  <div className="relative">
                    <Input
                      id="password"
                      type="password"
                      placeholder="Enter your password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      className="darwin-input pr-10"
                      required
                    />
                    <GrKey className="absolute right-3 top-1/2 transform -translate-y-1/2 text-muted-foreground" />
                  </div>
                  {errors.password && (
                    <Form.Message className="text-sm darwin-status-error">
                      {errors.password}
                    </Form.Message>
                  )}
                </div>
              </Form.Field>

              {error && (
                <div className="darwin-status-error p-3 rounded-lg text-center">
                  {error}
                </div>
              )}

              <Form.Submit asChild>
                <Button 
                  type="submit" 
                  className="w-full darwin-button-primary font-darwin-body"
                  disabled={isLoading}
                >
                  {isLoading ? 'Signing in...' : 'Sign In'}
                </Button>
              </Form.Submit>
            </div>
          </Form.Root>

          <div className="mt-6 text-center">
            <p className="text-sm text-muted-foreground font-darwin-body">
              "It is not the strongest of the species that survives, but the one most{' '}
              <span style={{ color: 'hsl(var(--darwin-accent))' }} className="font-semibold">
                responsive to change
              </span>
              "
            </p>
            <p className="text-xs text-muted-foreground mt-2">— Charles Darwin</p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}