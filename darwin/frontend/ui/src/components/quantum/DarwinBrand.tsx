"use client"

import React from 'react'
import { motion } from 'framer-motion'
import { DarwinIcon } from './DarwinLogo'

interface DarwinBrandProps {
  size?: 'sm' | 'md' | 'lg' | 'xl'
  variant?: 'horizontal' | 'vertical' | 'compact'
  animated?: boolean
  className?: string
  showSubtitle?: boolean
}

const sizeMap = {
  sm: { icon: 32, title: 'text-lg', subtitle: 'text-xs' },
  md: { icon: 48, title: 'text-2xl', subtitle: 'text-sm' },
  lg: { icon: 64, title: 'text-3xl', subtitle: 'text-base' },
  xl: { icon: 80, title: 'text-4xl', subtitle: 'text-lg' }
}

export const DarwinBrand: React.FC<DarwinBrandProps> = ({
  size = 'md',
  variant = 'horizontal',
  animated = true,
  className = '',
  showSubtitle = true
}) => {
  const { icon, title, subtitle } = sizeMap[size]

  if (variant === 'vertical') {
    return (
      <motion.div
        initial={animated ? { opacity: 0, y: 20 } : {}}
        animate={animated ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.8 }}
        className={`flex flex-col items-center gap-4 ${className}`}
      >
        <DarwinIcon size={icon} animated={animated} />
        <div className="text-center">
          <motion.h1
            initial={animated ? { opacity: 0, y: 10 } : {}}
            animate={animated ? { opacity: 1, y: 0 } : {}}
            transition={{ delay: 0.3, duration: 0.6 }}
            className={`font-quantum font-bold agourakis-signature ${title}`}
          >
            DARWIN
          </motion.h1>
          {showSubtitle && (
            <motion.p
              initial={animated ? { opacity: 0, y: 10 } : {}}
              animate={animated ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: 0.5, duration: 0.6 }}
              className={`text-muted-foreground font-neural ${subtitle}`}
            >
              Research Hub
            </motion.p>
          )}
        </div>
      </motion.div>
    )
  }

  if (variant === 'compact') {
    return (
      <motion.div
        initial={animated ? { opacity: 0, scale: 0.9 } : {}}
        animate={animated ? { opacity: 1, scale: 1 } : {}}
        transition={{ duration: 0.8 }}
        className={`flex items-center gap-2 ${className}`}
      >
        <DarwinIcon size={icon} animated={animated} />
        <motion.span
          initial={animated ? { opacity: 0, x: -10 } : {}}
          animate={animated ? { opacity: 1, x: 0 } : {}}
          transition={{ delay: 0.3, duration: 0.6 }}
          className={`font-quantum font-bold agourakis-signature ${title}`}
        >
          DARWIN
        </motion.span>
      </motion.div>
    )
  }

  // Default horizontal layout
  return (
    <motion.div
      initial={animated ? { opacity: 0, x: -20 } : {}}
      animate={animated ? { opacity: 1, x: 0 } : {}}
      transition={{ duration: 0.8 }}
      className={`flex items-center gap-4 ${className}`}
    >
      <DarwinIcon size={icon} animated={animated} />
      <div className="flex flex-col">
        <motion.h1
          initial={animated ? { opacity: 0, x: -10 } : {}}
          animate={animated ? { opacity: 1, x: 0 } : {}}
          transition={{ delay: 0.3, duration: 0.6 }}
          className={`font-quantum font-bold agourakis-signature ${title} leading-tight`}
        >
          DARWIN
        </motion.h1>
        {showSubtitle && (
          <motion.p
            initial={animated ? { opacity: 0, x: -10 } : {}}
            animate={animated ? { opacity: 1, x: 0 } : {}}
            transition={{ delay: 0.5, duration: 0.6 }}
            className={`text-muted-foreground font-neural ${subtitle} leading-tight`}
          >
            AutoGen Multi-Agent + JAX
          </motion.p>
        )}
      </div>
    </motion.div>
  )
}

// Specialized brand components
export const DarwinNavBrand: React.FC<{ className?: string }> = ({ className }) => (
  <DarwinBrand 
    size="sm" 
    variant="compact" 
    animated={false} 
    className={className}
  />
)

export const DarwinHeroBrand: React.FC<{ className?: string }> = ({ className }) => (
  <DarwinBrand 
    size="xl" 
    variant="vertical" 
    animated={true} 
    showSubtitle={true}
    className={className}
  />
)

export const DarwinFooterBrand: React.FC<{ className?: string }> = ({ className }) => (
  <DarwinBrand 
    size="sm" 
    variant="horizontal" 
    animated={false} 
    showSubtitle={false}
    className={className}
  />
)

export default DarwinBrand