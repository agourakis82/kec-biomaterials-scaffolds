import Image from "next/image"
import Link from "next/link"
import { SettingsSheet } from "@/components/settings/SettingsSheet"
import { CommandPalette } from "@/components/command/CommandPalette"

export default function AppLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="min-h-screen bg-background">
      <header className="border-b">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="relative h-10 w-10 overflow-hidden rounded-full border">
              <Image src="/assets/charles_darwin.jpg" alt="Darwin" fill className="object-cover" />
            </div>
            <Link href="/" className="text-2xl font-bold hover:opacity-90">DARWIN RAG++</Link>
          </div>
          <nav className="flex items-center gap-4 text-sm">
            <Link href="/" className="hover:underline">Home</Link>
            <Link href="/discovery" className="hover:underline">Discovery</Link>
            <Link href="/puct" className="hover:underline">PUCT</Link>
            <Link href="/compare" className="hover:underline">Compare</Link>
            <Link href="/prompt-lab" className="hover:underline">Prompt‑Lab</Link>
            <Link href="/admin" className="hover:underline">Admin</Link>
            <div className="pl-2"><SettingsSheet /></div>
          </nav>
        </div>
      </header>
      <main className="container mx-auto px-4 py-8">
        {children}
      </main>
      <CommandPalette />
    </div>
  )
}
