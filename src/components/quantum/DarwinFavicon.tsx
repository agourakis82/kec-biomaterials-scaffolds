"use client"

import React from 'react'

interface DarwinFaviconProps {
  size?: number
  className?: string
}

export const DarwinFavicon: React.FC<DarwinFaviconProps> = ({
  size = 32,
  className = ''
}) => {
  return (
    <div
      className={`relative ${className}`}
      style={{ width: size, height: size }}
    >
      <svg
        width={size}
        height={size}
        viewBox="0 0 32 32"
        className="drop-shadow-sm"
      >
        <defs>
          <linearGradient id="faviconGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#8B5CF6" />
            <stop offset="50%" stopColor="#06B6D4" />
            <stop offset="100%" stopColor="#10B981" />
          </linearGradient>
        </defs>

        {/* Background Circle */}
        <circle
          cx="16"
          cy="16"
          r="15"
          fill="url(#faviconGradient)"
          opacity="0.15"
        />

        {/* Simplified D with DNA */}
        <path
          d="M8 6 L8 26 L16 26 Q24 26, 24 16 Q24 6, 16 6 Z"
          fill="none"
          stroke="url(#faviconGradient)"
          strokeWidth="2"
          strokeLinecap="round"
        />

        {/* DNA Helix */}
        <path
          d="M11 11 Q13 13, 15 11 T19 11"
          stroke="#8B5CF6"
          strokeWidth="1.5"
          fill="none"
          strokeLinecap="round"
        />
        <path
          d="M11 21 Q13 19, 15 21 T19 21"
          stroke="#10B981"
          strokeWidth="1.5"
          fill="none"
          strokeLinecap="round"
        />

        {/* Central dot */}
        <circle cx="16" cy="16" r="1.5" fill="url(#faviconGradient)" />
      </svg>
    </div>
  )
}

export default DarwinFavicon