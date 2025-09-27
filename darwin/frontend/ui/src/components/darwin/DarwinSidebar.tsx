"use client"

import React from 'react'
import { GrApple, GrAnalytics, GrCpu, GrDashboard, GrFormSearch, GrCode, GrBook, GrUser, GrLogout } from 'react-icons/gr'
import * as NavigationMenu from '@radix-ui/react-navigation-menu'
import { Button } from '../ui/button'
import { Separator } from '../ui/separator'
import { ScrollArea } from '../ui/scroll-area'

interface DarwinSidebarProps {
  activeSection: string
  onSectionChange: (section: string) => void
  onLogout: () => void
  user?: {
    username: string
    role: string
  }
}

const navigationItems = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    icon: GrDashboard,
    description: 'Overview and status'
  },
  {
    id: 'rag-plus',
    label: 'RAG++ Research',
    icon: GrFormSearch,
    description: 'Advanced scientific queries'
  },
  {
    id: 'memory',
    label: 'Memory System',
    icon: GrCpu,
    description: 'Conversations & continuity'
  },
  {
    id: 'tree-search',
    label: 'Tree Search',
    icon: GrAnalytics,
    description: 'MCTS & PUCT algorithms'
  },
  {
    id: 'data-explorer',
    label: 'Data Explorer',
    icon: GrCode,
    description: 'AG5 & HELIO datasets'
  },
  {
    id: 'notebooks',
    label: 'Notebooks',
    icon: GrBook,
    description: 'Jupyter management'
  }
]

export function DarwinSidebar({ activeSection, onSectionChange, onLogout, user }: DarwinSidebarProps) {
  return (
    <div className="w-64 h-full darwin-card border-r flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-border">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-full darwin-card flex items-center justify-center">
            <GrApple className="text-lg" style={{ color: 'hsl(var(--darwin-primary))' }} />
          </div>
          <div>
            <h1 className="text-xl font-darwin-display font-bold" 
                style={{ color: 'hsl(var(--darwin-primary))' }}>
              Darwin
            </h1>
            <p className="text-xs text-muted-foreground font-darwin-body">
              Scientific Platform
            </p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <ScrollArea className="flex-1 p-4">
        <NavigationMenu.Root orientation="vertical" className="w-full">
          <NavigationMenu.List className="space-y-2">
            {navigationItems.map((item) => {
              const Icon = item.icon
              const isActive = activeSection === item.id
              
              return (
                <NavigationMenu.Item key={item.id}>
                  <Button
                    variant={isActive ? "default" : "ghost"}
                    className={`w-full justify-start space-x-3 h-auto p-3 font-darwin-body ${
                      isActive 
                        ? 'darwin-button-primary' 
                        : 'hover:bg-muted/50'
                    }`}
                    onClick={() => onSectionChange(item.id)}
                  >
                    <Icon className="text-base flex-shrink-0" />
                    <div className="text-left">
                      <div className="font-medium">{item.label}</div>
                      <div className={`text-xs ${
                        isActive ? 'text-primary-foreground/80' : 'text-muted-foreground'
                      }`}>
                        {item.description}
                      </div>
                    </div>
                  </Button>
                </NavigationMenu.Item>
              )
            })}
          </NavigationMenu.List>
        </NavigationMenu.Root>
      </ScrollArea>

      <Separator />

      {/* User section */}
      <div className="p-4 space-y-3">
        {user && (
          <div className="flex items-center space-x-3 p-2 rounded-lg bg-muted/30">
            <div className="w-8 h-8 rounded-full darwin-card flex items-center justify-center">
              <GrUser className="text-sm" style={{ color: 'hsl(var(--darwin-primary))' }} />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium font-darwin-body truncate">
                {user.username}
              </p>
              <p className="text-xs text-muted-foreground capitalize">
                {user.role}
              </p>
            </div>
          </div>
        )}

        <Button
          variant="ghost"
          className="w-full justify-start space-x-3 font-darwin-body text-muted-foreground hover:text-foreground"
          onClick={onLogout}
        >
          <GrLogout className="text-base" />
          <span>Sign Out</span>
        </Button>
      </div>
    </div>
  )
}