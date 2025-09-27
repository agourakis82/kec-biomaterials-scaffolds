"use client"

import * as React from "react"
import { motion, useScroll, useTransform } from "framer-motion"
import { 
  Sparkles, 
  ArrowRight, 
  Leaf, 
  Microscope, 
  Dna, 
  TreePine,
  Mail,
  Github,
  Linkedin,
  ExternalLink,
  Bot,
  BookOpen
} from "lucide-react"
import { FloatingNav } from "@/components/liquid/FloatingNav"
import { GlassCard, GlassCardHeader, GlassCardTitle, GlassCardContent } from "@/components/liquid/GlassCard"
import { LiquidButton } from "@/components/liquid/LiquidButton"
import { ResearchGrid } from "@/components/liquid/ResearchGrid"

export default function DarwinPage() {
  const { scrollY } = useScroll()
  const heroY = useTransform(scrollY, [0, 500], [0, -150])
  const heroOpacity = useTransform(scrollY, [0, 300], [1, 0])

  return (
    <div className="min-h-screen manuscript-texture overflow-x-hidden">
      <FloatingNav />
      
      {/* Hero Section */}
      <section id="home" className="relative min-h-screen flex items-center justify-center">
        {/* Animated Background with Darwin's manuscript */}
        <div className="absolute inset-0">
          <motion.div
            style={{ y: heroY }}
            className="absolute inset-0 opacity-20"
          >
            {/* Using the provided Darwin images as data URIs would be ideal, but since we can't access them directly, 
                we'll use a placeholder that represents the vintage scientific aesthetic */}
            <div className="w-full h-full bg-gradient-to-br from-darwin-parchment via-darwin-manuscript to-darwin-sepia" />
          </motion.div>
          <div className="absolute inset-0 bg-gradient-to-b from-transparent via-background/30 to-background" />
        </div>

        {/* Floating Scientific Elements */}
        <div className="absolute inset-0 overflow-hidden">
          <motion.div
            animate={{ 
              y: [0, -25, 0],
              rotate: [0, 3, 0]
            }}
            transition={{ 
              duration: 12, 
              repeat: Infinity, 
              ease: "easeInOut" 
            }}
            className="absolute top-20 left-10 w-24 h-24 rounded-full manuscript-texture border vintage-border backdrop-blur-sm flex items-center justify-center"
          >
            <Leaf className="text-darwin" size={32} />
          </motion.div>
          <motion.div
            animate={{ 
              y: [0, 35, 0],
              rotate: [0, -8, 0]
            }}
            transition={{ 
              duration: 15, 
              repeat: Infinity, 
              ease: "easeInOut",
              delay: 3
            }}
            className="absolute top-40 right-20 w-32 h-32 rounded-full manuscript-texture border vintage-border backdrop-blur-sm flex items-center justify-center"
          >
            <TreePine className="text-vintage" size={40} />
          </motion.div>
          <motion.div
            animate={{ 
              y: [0, -20, 0],
              x: [0, 15, 0]
            }}
            transition={{ 
              duration: 18, 
              repeat: Infinity, 
              ease: "easeInOut",
              delay: 6
            }}
            className="absolute bottom-40 left-1/4 w-20 h-20 rounded-full manuscript-texture border vintage-border backdrop-blur-sm flex items-center justify-center"
          >
            <Bot className="text-darwin" size={28} />
          </motion.div>
        </div>

        {/* Hero Content */}
        <motion.div
          style={{ opacity: heroOpacity }}
          className="relative z-10 text-center px-6 max-w-6xl mx-auto"
        >
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="mb-8"
          >
            <div className="inline-flex items-center gap-3 px-8 py-4 rounded-full glass-manuscript mb-8">
              <Sparkles className="h-6 w-6 text-darwin" />
              <span className="text-darwin font-manuscript font-semibold text-lg">Darwin</span>
              <span className="text-vintage">·</span>
              <span className="text-vintage font-manuscript">Evolution & Natural Selection</span>
            </div>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="text-6xl md:text-8xl lg:text-9xl font-bold font-display mb-8"
          >
            <span className="darwin-gradient-text">
              Charles
            </span>
            <br />
            <span className="text-darwin font-manuscript">
              Darwin
            </span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="text-2xl md:text-3xl text-vintage mb-12 max-w-4xl mx-auto leading-relaxed font-manuscript"
          >
            "It is not the strongest of the species that survives, nor the most intelligent, 
            but the one most <span className="darwin-gradient-text font-semibold">responsive to change</span>"
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="flex flex-col sm:flex-row gap-6 justify-center items-center"
          >
            <LiquidButton 
              variant="evolution"
              size="lg" 
              icon={<Leaf size={20} />}
              onClick={() => document.getElementById('evolution')?.scrollIntoView({ behavior: 'smooth' })}
            >
              Explore Evolution
            </LiquidButton>
            <LiquidButton 
              variant="darwin" 
              size="lg"
              icon={<BookOpen size={20} />}
            >
              Origin of Species
            </LiquidButton>
          </motion.div>
        </motion.div>

        {/* Scroll Indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5 }}
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
        >
          <motion.div
            animate={{ y: [0, 10, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="w-6 h-10 border-2 border-vintage rounded-full flex justify-center"
          >
            <motion.div
              animate={{ y: [0, 12, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
              className="w-1 h-3 bg-vintage rounded-full mt-2"
            />
          </motion.div>
        </motion.div>
      </section>

      {/* Evolution Section */}
      <section id="evolution" className="section-padding bg-gradient-to-r from-darwin-parchment/20 to-darwin-manuscript/20">
        <div className="container-wide mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-5xl md:text-7xl font-bold font-display mb-6">
              <span className="darwin-gradient-text">Theory of</span>
              <br />
              <span className="text-darwin font-manuscript">Evolution</span>
            </h2>
            <p className="text-xl text-vintage max-w-3xl mx-auto font-manuscript leading-relaxed">
              The revolutionary scientific theory that explains the diversity of life through 
              natural selection and common descent.
            </p>
          </motion.div>

          <ResearchGrid />
        </div>
      </section>

      {/* Research Section */}
      <section id="research" className="section-padding">
        <div className="container-wide mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-5xl md:text-7xl font-bold font-display mb-6">
              <span className="text-darwin font-manuscript">Scientific</span>
              <br />
              <span className="darwin-gradient-text">Contributions</span>
            </h2>
            <p className="text-xl text-vintage max-w-3xl mx-auto font-manuscript">
              Darwin's groundbreaking research that forever changed our understanding of life on Earth.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                icon: <Leaf size={32} />,
                title: "Natural Selection",
                description: "The mechanism by which organisms with favorable traits survive and reproduce more successfully.",
                discoveries: ["Survival of the fittest", "Adaptive radiation", "Speciation"]
              },
              {
                icon: <TreePine size={32} />,
                title: "Common Descent",
                description: "All species descended from common ancestors through branching evolutionary processes.",
                discoveries: ["Tree of life", "Phylogenetic relationships", "Fossil evidence"]
              },
              {
                icon: <Bot size={32} />,
                title: "Human Evolution",
                description: "Humans evolved from earlier primates through the same evolutionary processes as other species.",
                discoveries: ["Sexual selection", "Emotional expressions", "Moral sentiments"]
              }
            ].map((research, index) => (
              <div key={index} className="glass-manuscript p-8 hover-manuscript">
                <GlassCardHeader>
                  <motion.div
                    whileHover={{ scale: 1.1, rotate: 360 }}
                    transition={{ duration: 0.8 }}
                    className="w-20 h-20 rounded-full manuscript-texture border vintage-border flex items-center justify-center mb-6 mx-auto"
                  >
                    <span className="text-darwin">
                      {research.icon}
                    </span>
                  </motion.div>
                  <GlassCardTitle className="text-darwin font-manuscript text-center text-2xl">
                    {research.title}
                  </GlassCardTitle>
                </GlassCardHeader>
                <GlassCardContent className="text-vintage mb-6 font-manuscript text-center leading-relaxed">
                  {research.description}
                </GlassCardContent>
                <div className="space-y-3">
                  {research.discoveries.map((discovery, discoveryIndex) => (
                    <motion.div
                      key={discoveryIndex}
                      initial={{ opacity: 0, x: -20 }}
                      whileInView={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.2 + discoveryIndex * 0.1 }}
                      className="flex items-center gap-3 text-sm text-vintage font-manuscript"
                    >
                      <div className="w-2 h-2 rounded-full darwin-gradient" />
                      {discovery}
                    </motion.div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Origin Section */}
      <section id="origin" className="section-padding bg-gradient-to-t from-darwin-manuscript/30 to-transparent">
        <div className="container-wide mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-5xl md:text-7xl font-bold font-display mb-6">
              <span className="darwin-gradient-text font-manuscript">On the Origin</span>
              <br />
              <span className="text-darwin font-manuscript">of Species</span>
            </h2>
            <p className="text-xl text-vintage max-w-3xl mx-auto font-manuscript">
              The masterwork that introduced the theory of evolution by natural selection to the world.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
            <div className="glass-manuscript p-8">
              <div className="mb-6">
                <span className="text-sm font-manuscript text-vintage bg-darwin-parchment px-4 py-2 rounded-full border vintage-border">
                  Published 1859
                </span>
              </div>
              <h3 className="text-3xl font-bold text-darwin font-manuscript mb-6">
                "On the Origin of Species by Means of Natural Selection"
              </h3>
              <p className="text-vintage mb-6 font-manuscript leading-relaxed text-lg">
                Darwin's revolutionary work presented compelling evidence for evolution through natural selection, 
                fundamentally changing our understanding of life's diversity and humanity's place in nature.
              </p>
              <div className="space-y-4">
                {[
                  "Variation under domestication",
                  "Variation under nature", 
                  "Struggle for existence",
                  "Natural selection",
                  "Laws of variation",
                  "Geographical distribution"
                ].map((chapter, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="flex items-center gap-3 text-vintage font-manuscript"
                  >
                    <div className="w-1.5 h-1.5 rounded-full darwin-gradient" />
                    <span className="text-sm">Chapter {index + 1}: {chapter}</span>
                  </motion.div>
                ))}
              </div>
            </div>

            <div className="glass-manuscript p-8">
              <h3 className="text-2xl font-bold text-darwin font-manuscript mb-6">
                Key Insights
              </h3>
              <div className="space-y-6">
                {[
                  {
                    quote: "I have called this principle, by which each slight variation, if useful, is preserved, by the term Natural Selection.",
                    context: "Defining natural selection"
                  },
                  {
                    quote: "It is not the strongest of the species that survives, but the one most responsive to change.",
                    context: "On adaptation"
                  },
                  {
                    quote: "There is grandeur in this view of life, with its several powers, having been originally breathed into a few forms or into one.",
                    context: "Concluding thoughts"
                  }
                ].map((insight, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.2 }}
                    className="border-l-4 border-darwin-vintage pl-4"
                  >
                    <p className="text-vintage font-manuscript italic mb-2 leading-relaxed">
                      "{insight.quote}"
                    </p>
                    <p className="text-sm text-manuscript font-manuscript">
                      — {insight.context}
                    </p>
                  </motion.div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Contact Section */}
      <section id="contact" className="section-padding bg-gradient-to-t from-background to-transparent">
        <div className="container-narrow mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="mb-16"
          >
            <h2 className="text-5xl md:text-7xl font-bold font-display mb-6">
              <span className="text-darwin font-manuscript">Explore</span>
              <br />
              <span className="darwin-gradient-text">Darwin's Legacy</span>
            </h2>
            <p className="text-xl text-vintage max-w-2xl mx-auto font-manuscript">
              Discover more about Charles Darwin's revolutionary contributions to science 
              and their lasting impact on our understanding of life.
            </p>
          </motion.div>

          <div className="glass-manuscript max-w-2xl mx-auto p-8">
            <div className="text-center">
              <motion.div
                whileHover={{ scale: 1.05 }}
                className="w-24 h-24 rounded-full manuscript-texture border vintage-border flex items-center justify-center mx-auto mb-8"
              >
                <BookOpen size={40} className="text-darwin" />
              </motion.div>
              <h3 className="text-3xl font-bold text-darwin font-manuscript mb-4">
                Charles Robert Darwin
              </h3>
              <p className="text-vintage mb-2 font-manuscript text-lg">
                Naturalist & Evolutionary Biologist
              </p>
              <p className="text-manuscript mb-8 font-manuscript">
                1809 - 1882
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <LiquidButton variant="evolution" icon={<ExternalLink size={20} />}>
                  Read Origin of Species
                </LiquidButton>
                <div className="flex gap-4 justify-center">
                  <LiquidButton variant="darwin" size="sm" icon={<Github size={20} />}>
                    Archive
                  </LiquidButton>
                  <LiquidButton variant="darwin" size="sm" icon={<Microscope size={20} />}>
                    Research
                  </LiquidButton>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-darwin-vintage/30 py-8 bg-darwin-parchment/10">
        <div className="container-wide mx-auto px-6">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="text-vintage text-sm font-manuscript">
              © 2024 Darwin - Celebrating Scientific Discovery. Educational purposes.
            </div>
            <div className="flex items-center gap-4 mt-4 md:mt-0">
              <span className="text-manuscript text-sm font-manuscript">Inspired by</span>
              <span className="darwin-gradient-text font-semibold font-manuscript">Natural Selection</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}