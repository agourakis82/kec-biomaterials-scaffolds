import type { Metadata } from 'next'
import '@/app/globals.css'
import { Providers } from '@/components/providers'

export const metadata: Metadata = {
  title: 'Darwin - Evolution & Natural Selection',
  description: 'Explore Charles Darwin\'s revolutionary theories on evolution and natural selection',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen bg-main-gradient font-sans antialiased">
        <Providers>
          {children}
        </Providers>
      </body>
    </html>
  )
}