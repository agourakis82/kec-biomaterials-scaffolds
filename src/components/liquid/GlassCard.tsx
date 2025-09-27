"use client"

import * as React from "react"
import { motion } from "framer-motion"
import { cn } from "@/lib/utils"

interface GlassCardProps {
  children: React.ReactNode
  className?: string
  hover?: boolean
  float?: boolean
  delay?: number
}

export function GlassCard({ 
  children, 
  className, 
  hover = true, 
  float = false,
  delay = 0 
}: GlassCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.6, ease: "easeOut" }}
      className={cn(
        "glass-card p-6",
        hover && "hover-lift hover-glass cursor-pointer",
        float && "float",
        className
      )}
    >
      {children}
    </motion.div>
  )
}

interface GlassCardHeaderProps {
  children: React.ReactNode
  className?: string
}

export function GlassCardHeader({ children, className }: GlassCardHeaderProps) {
  return (
    <div className={cn("mb-4", className)}>
      {children}
    </div>
  )
}

interface GlassCardTitleProps {
  children: React.ReactNode
  className?: string
  gradient?: boolean
}

export function GlassCardTitle({ children, className, gradient = false }: GlassCardTitleProps) {
  return (
    <h3 className={cn(
      "text-xl font-semibold font-display",
      gradient && "text-gradient",
      className
    )}>
      {children}
    </h3>
  )
}

interface GlassCardContentProps {
  children: React.ReactNode
  className?: string
}

export function GlassCardContent({ children, className }: GlassCardContentProps) {
  return (
    <div className={cn("text-muted-foreground", className)}>
      {children}
    </div>
  )
}