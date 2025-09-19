import Image from "next/image"
import Link from "next/link"
import { SettingsSheet } from "@/components/settings/SettingsSheet"
import { CommandPalette } from "@/components/command/CommandPalette"
import { ThemeToggle } from "@/components/ui/theme-toggle"

export default function AppLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="relative h-10 w-10 overflow-hidden rounded-full border-2 border-primary/20 hover-lift">
              <Image src="/assets/charles_darwin.jpg" alt="Darwin" fill className="object-cover" />
            </div>
            <Link href="/" className="text-2xl font-bold text-gradient hover:opacity-90 transition-opacity">
              DARWIN RAG++
            </Link>
          </div>
          <nav className="flex items-center gap-6 text-sm">
            <Link href="/" className="relative px-3 py-2 rounded-md hover:bg-accent transition-colors">
              Home
            </Link>
            <Link href="/discovery" className="relative px-3 py-2 rounded-md hover:bg-accent transition-colors">
              Discovery
            </Link>
            <Link href="/puct" className="relative px-3 py-2 rounded-md hover:bg-accent transition-colors">
              PUCT
            </Link>
            <Link href="/compare" className="relative px-3 py-2 rounded-md hover:bg-accent transition-colors">
              Compare
            </Link>
            <Link href="/prompt-lab" className="relative px-3 py-2 rounded-md hover:bg-accent transition-colors">
              Prompt‑Lab
            </Link>
            <Link href="/admin" className="relative px-3 py-2 rounded-md hover:bg-accent transition-colors">
              Admin
            </Link>
            <div className="flex items-center gap-2 pl-2 border-l">
              <ThemeToggle />
              <SettingsSheet />
            </div>
          </nav>
        </div>
      </header>
      <main className="container mx-auto px-4 py-8 animate-fade-in">
        {children}
      </main>
      <CommandPalette />
    </div>
  )
}
