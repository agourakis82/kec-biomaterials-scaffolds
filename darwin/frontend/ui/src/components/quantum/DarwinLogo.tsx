"use client"

import React from 'react'
import { cn } from '@/lib/utils'

interface DarwinLogoProps {
  size?: number
  variant?: 'full' | 'icon' | 'text'
  animated?: boolean
  className?: string
}

export const DarwinLogo: React.FC<DarwinLogoProps> = ({
  size = 32,
  variant = 'full',
  animated = true,
  className
}) => {
  const DarwinIcon = () => (
    <div
      className={cn("relative", className)}
      style={{ width: size, height: size }}
    >
      <svg
        width={size}
        height={size}
        viewBox="0 0 100 100"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* Outer quantum ring */}
        <circle
          cx="50"
          cy="50"
          r="45"
          stroke="url(#quantumGradient)"
          strokeWidth="2"
          fill="none"
          opacity="0.6"
        />
        
        {/* Inner neural network */}
        <circle
          cx="50"
          cy="50"
          r="30"
          stroke="url(#neuralGradient)"
          strokeWidth="1.5"
          fill="none"
          opacity="0.8"
        />
        
        {/* Central core */}
        <circle
          cx="50"
          cy="50"
          r="15"
          fill="url(#coreGradient)"
          opacity="0.9"
        />
        
        {/* DNA helix representation */}
        <path
          d="M35 30 Q50 40 65 30 M35 50 Q50 40 65 50 M35 70 Q50 60 65 70"
          stroke="url(#helixGradient)"
          strokeWidth="2"
          fill="none"
          opacity="0.7"
        />
        
        {/* Quantum particles */}
        <circle cx="25" cy="25" r="2" fill="#8B5CF6" opacity="0.8" />
        <circle cx="75" cy="25" r="2" fill="#06B6D4" opacity="0.8" />
        <circle cx="25" cy="75" r="2" fill="#10B981" opacity="0.8" />
        <circle cx="75" cy="75" r="2" fill="#F59E0B" opacity="0.8" />
        
        {/* Gradients */}
        <defs>
          <linearGradient id="quantumGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#8B5CF6" />
            <stop offset="50%" stopColor="#06B6D4" />
            <stop offset="100%" stopColor="#10B981" />
          </linearGradient>
          
          <linearGradient id="neuralGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#06B6D4" />
            <stop offset="100%" stopColor="#8B5CF6" />
          </linearGradient>
          
          <radialGradient id="coreGradient" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#8B5CF6" stopOpacity="0.8" />
            <stop offset="100%" stopColor="#1E1B4B" stopOpacity="0.9" />
          </radialGradient>
          
          <linearGradient id="helixGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#F59E0B" />
            <stop offset="100%" stopColor="#EF4444" />
          </linearGradient>
        </defs>
      </svg>
    </div>
  )

  const DarwinText = () => (
    <div
      className={cn("font-bold tracking-wider", className)}
      style={{ fontSize: size * 0.4 }}
    >
      <span className="bg-gradient-to-r from-purple-500 via-blue-500 to-green-500 bg-clip-text text-transparent">
        DARWIN
      </span>
    </div>
  )

  if (variant === 'icon') {
    return <DarwinIcon />
  }

  if (variant === 'text') {
    return <DarwinText />
  }

  return (
    <div className={cn("flex items-center gap-3", className)}>
      <DarwinIcon />
      <DarwinText />
    </div>
  )
}

export default DarwinLogo
