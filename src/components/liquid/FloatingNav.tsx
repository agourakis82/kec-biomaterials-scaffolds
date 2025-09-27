"use client"

import * as React from "react"
import { motion, useScroll, useTransform } from "framer-motion"
import { Home, Leaf, Microscope, BookOpen, Mail, Bot } from "lucide-react"
import { cn } from "@/lib/utils"

interface NavItem {
  icon: React.ReactNode
  label: string
  href: string
}

const navItems: NavItem[] = [
  { icon: <Home size={20} />, label: "Home", href: "#home" },
  { icon: <Leaf size={20} />, label: "Evolution", href: "#evolution" },
  { icon: <Microscope size={20} />, label: "Research", href: "#research" },
  { icon: <BookOpen size={20} />, label: "Origin", href: "#origin" },
  { icon: <Mail size={20} />, label: "Contact", href: "#contact" },
]

export function FloatingNav() {
  const [activeSection, setActiveSection] = React.useState("home")
  const { scrollY } = useScroll()
  const opacity = useTransform(scrollY, [0, 100], [0, 1])
  const scale = useTransform(scrollY, [0, 100], [0.8, 1])

  React.useEffect(() => {
    const handleScroll = () => {
      const sections = navItems.map(item => item.href.slice(1))
      const currentSection = sections.find(section => {
        const element = document.getElementById(section)
        if (element) {
          const rect = element.getBoundingClientRect()
          return rect.top <= 100 && rect.bottom >= 100
        }
        return false
      })
      if (currentSection) {
        setActiveSection(currentSection)
      }
    }

    window.addEventListener("scroll", handleScroll)
    return () => window.removeEventListener("scroll", handleScroll)
  }, [])

  const scrollToSection = (href: string) => {
    const element = document.getElementById(href.slice(1))
    if (element) {
      element.scrollIntoView({ behavior: "smooth" })
    }
  }

  return (
    <motion.nav
      style={{ opacity, scale }}
      className="fixed top-6 left-1/2 -translate-x-1/2 z-50"
    >
      <div className="glass-nav px-6 py-3">
        <div className="flex items-center gap-2">
          {navItems.map((item, index) => (
            <motion.button
              key={item.href}
              onClick={() => scrollToSection(item.href)}
              className={cn(
                "relative px-4 py-2 rounded-full transition-all duration-300",
                "hover:bg-white/10 hover:backdrop-blur-md",
                activeSection === item.href.slice(1) && "bg-white/20"
              )}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <div className="flex items-center gap-2">
                <span className={cn(
                  "transition-colors duration-300",
                  activeSection === item.href.slice(1) 
                    ? "text-white" 
                    : "text-white/70 hover:text-white"
                )}>
                  {item.icon}
                </span>
                <span className={cn(
                  "text-sm font-medium transition-all duration-300",
                  activeSection === item.href.slice(1) 
                    ? "text-white opacity-100" 
                    : "text-white/70 opacity-0 w-0 overflow-hidden hover:opacity-100 hover:w-auto"
                )}>
                  {item.label}
                </span>
              </div>
              {activeSection === item.href.slice(1) && (
                <motion.div
                  layoutId="activeIndicator"
                  className="absolute inset-0 bg-gradient-to-r from-purple-500/30 to-blue-500/30 rounded-full"
                  transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                />
              )}
            </motion.button>
          ))}
        </div>
      </div>
    </motion.nav>
  )
}