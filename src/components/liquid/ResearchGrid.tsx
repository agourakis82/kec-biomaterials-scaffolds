"use client"

import * as React from "react"
import { motion } from "framer-motion"
import { Leaf, Dna, Bot, Target, ArrowRight, TreePine } from "lucide-react"
import { GlassCard, GlassCardHeader, GlassCardTitle, GlassCardContent } from "./GlassCard"
import { LiquidButton } from "./LiquidButton"

interface ResearchItem {
  id: string
  title: string
  description: string
  icon: React.ReactNode
  image: string
  category: string
  progress: number
  year: string
}

const darwinResearch: ResearchItem[] = [
  {
    id: "natural-selection",
    title: "Natural Selection",
    description: "The fundamental mechanism by which species evolve and adapt to their environments through differential survival and reproduction.",
    icon: <Leaf size={24} />,
    image: "https://images.unsplash.com/photo-1715529407889-645486f64728?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NTAwNDR8MHwxfHNlYXJjaHw3fHx0cmVlJTIwYm90YW5pY2FsJTIwdmludGFnZSUyMHNjaWVudGlmaWN8ZW58MHwxfHxibGFja19hbmRfd2hpdGV8MTc1ODM2MTAyMHww&ixlib=rb-4.1.0&q=85",
    category: "Evolutionary Theory",
    progress: 100,
    year: "1859"
  },
  {
    id: "origin-species",
    title: "Origin of Species",
    description: "Revolutionary work explaining how all species descended from common ancestors through the process of natural selection.",
    icon: <TreePine size={24} />,
    image: "https://images.unsplash.com/photo-1614600144476-41603fd271c3?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NTAwNDR8MHwxfHNlYXJjaHw0fHx0cmVlJTIwYm90YW5pY2FsJTIwdmludGFnZSUyMHNjaWVudGlmaWN8ZW58MHwxfHxibGFja19hbmRfd2hpdGV8MTc1ODM2MTAyMHww&ixlib=rb-4.1.0&q=85",
    category: "Biological Sciences",
    progress: 100,
    year: "1859"
  },
  {
    id: "descent-man",
    title: "Descent of Man",
    description: "Groundbreaking exploration of human evolution and the role of sexual selection in shaping species characteristics.",
    icon: <Bot size={24} />,
    image: "https://images.unsplash.com/photo-1649375949029-3aa3f46b46b1?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NTAwNDR8MHwxfHNlYXJjaHw4fHx0cmVlJTIwYm90YW5pY2FsJTIwdmludGFnZSUyMHNjaWVudGlmaWN8ZW58MHwxfHxibGFja19hbmRfd2hpdGV8MTc1ODM2MTAyMHww&ixlib=rb-4.1.0&q=85",
    category: "Human Evolution",
    progress: 100,
    year: "1871"
  },
  {
    id: "voyage-beagle",
    title: "Voyage of the Beagle",
    description: "Transformative journey that provided crucial observations leading to the development of evolutionary theory.",
    icon: <Target size={24} />,
    image: "https://images.unsplash.com/photo-1715529407889-645486f64728?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NTAwNDR8MHwxfHNlYXJjaHw3fHx0cmVlJTIwYm90YW5pY2FsJTIwdmludGFnZSUyMHNjaWVudGlmaWN8ZW58MHwxfHxibGFja19hbmRfd2hpdGV8MTc1ODM2MTAyMHww&ixlib=rb-4.1.0&q=85",
    category: "Scientific Expedition",
    progress: 100,
    year: "1831-1836"
  }
]

export function ResearchGrid() {
  const [selectedItem, setSelectedItem] = React.useState<string | null>(null)

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
      {darwinResearch.map((item, index) => (
        <motion.div
          key={item.id}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1, duration: 0.6 }}
          onHoverStart={() => setSelectedItem(item.id)}
          onHoverEnd={() => setSelectedItem(null)}
        >
          <div 
            className={`glass-manuscript relative overflow-hidden transition-all duration-500 p-6 ${
              selectedItem === item.id ? 'scale-105 shadow-darwin-lg' : ''
            }`}
          >
            {/* Background Image with Vintage Overlay */}
            <div className="absolute inset-0 opacity-10">
              <img 
                src={item.image} 
                alt={`${item.title} - Europeana on Unsplash`}
                className="w-full h-full object-cover"
                style={{ width: '100%', height: '300px' }}
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent" />
            </div>

            {/* Vintage manuscript lines effect */}
            <div className="absolute inset-0 opacity-5 pointer-events-none">
              <div className="w-full h-full" style={{
                backgroundImage: `repeating-linear-gradient(
                  0deg,
                  transparent,
                  transparent 23px,
                  hsl(var(--darwin-vintage)) 23px,
                  hsl(var(--darwin-vintage)) 24px
                )`
              }} />
            </div>

            {/* Content */}
            <div className="relative z-10">
              <GlassCardHeader>
                <div className="flex items-center justify-between mb-4">
                  <motion.div
                    animate={{ 
                      rotate: selectedItem === item.id ? 360 : 0,
                      scale: selectedItem === item.id ? 1.1 : 1
                    }}
                    transition={{ duration: 0.8 }}
                    className="p-3 rounded-full manuscript-texture border vintage-border"
                  >
                    <span className="text-darwin">
                      {item.icon}
                    </span>
                  </motion.div>
                  <div className="text-right">
                    <span className="text-xs font-manuscript text-vintage bg-darwin-parchment px-3 py-1 rounded-full border vintage-border">
                      {item.category}
                    </span>
                    <div className="text-xs text-manuscript mt-1 font-mono">
                      Est. {item.year}
                    </div>
                  </div>
                </div>
                <GlassCardTitle className="text-darwin font-manuscript text-2xl">
                  {item.title}
                </GlassCardTitle>
              </GlassCardHeader>

              <GlassCardContent className="text-vintage mb-6 font-manuscript leading-relaxed">
                {item.description}
              </GlassCardContent>

              {/* Progress Bar - showing completion */}
              <div className="mb-6">
                <div className="flex justify-between text-sm text-vintage mb-2 font-manuscript">
                  <span>Research Status</span>
                  <span>Complete</span>
                </div>
                <div className="w-full bg-darwin-parchment rounded-full h-2 border vintage-border">
                  <motion.div
                    className="h-2 rounded-full darwin-gradient"
                    initial={{ width: 0 }}
                    animate={{ width: `${item.progress}%` }}
                    transition={{ delay: index * 0.1 + 0.5, duration: 1.5 }}
                  />
                </div>
              </div>

              <LiquidButton 
                variant="glass" 
                size="sm"
                className="w-full justify-center btn-darwin text-darwin font-manuscript"
                icon={<ArrowRight size={16} />}
              >
                Explore Theory
              </LiquidButton>
            </div>

            {/* Decorative corner elements */}
            <div className="absolute top-4 left-4 w-8 h-8 border-l-2 border-t-2 border-darwin-vintage opacity-30" />
            <div className="absolute top-4 right-4 w-8 h-8 border-r-2 border-t-2 border-darwin-vintage opacity-30" />
            <div className="absolute bottom-4 left-4 w-8 h-8 border-l-2 border-b-2 border-darwin-vintage opacity-30" />
            <div className="absolute bottom-4 right-4 w-8 h-8 border-r-2 border-b-2 border-darwin-vintage opacity-30" />
          </div>
        </motion.div>
      ))}
    </div>
  )
}