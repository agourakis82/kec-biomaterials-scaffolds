"use client"

import * as React from "react"
import { motion } from "framer-motion"
import { cn } from "@/lib/utils"

interface LiquidButtonProps extends Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, 'onDrag' | 'onDragEnd' | 'onDragEnter' | 'onDragLeave' | 'onDragOver' | 'onDragStart' | 'onDrop'> {
  variant?: "liquid" | "glass" | "outline" | "darwin" | "evolution"
  size?: "sm" | "md" | "lg"
  children: React.ReactNode
  icon?: React.ReactNode
}

export function LiquidButton({ 
  variant = "liquid", 
  size = "md", 
  children, 
  icon,
  className,
  ...props 
}: LiquidButtonProps) {
  const sizeClasses = {
    sm: "px-4 py-2 text-sm",
    md: "px-6 py-3 text-base",
    lg: "px-8 py-4 text-lg"
  }

  const variantClasses = {
    liquid: "btn-liquid",
    glass: "btn-glass",
    outline: "border border-white/20 bg-transparent hover:bg-white/10 text-white",
    darwin: "btn-darwin",
    evolution: "btn-evolution"
  }

  return (
    <motion.button
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      className={cn(
        "inline-flex items-center gap-2 rounded-full font-semibold transition-all duration-300",
        "focus:outline-none focus:ring-2 focus:ring-white/20 focus:ring-offset-2 focus:ring-offset-transparent",
        sizeClasses[size],
        variantClasses[variant],
        className
      )}
      {...(props as any)}
    >
      {icon && (
        <motion.span
          initial={{ rotate: 0 }}
          whileHover={{ rotate: 360 }}
          transition={{ duration: 0.6 }}
        >
          {icon}
        </motion.span>
      )}
      {children}
    </motion.button>
  )
}