import type { Metadata } from 'next'
import { Inter, Space_Grotesk, Playfair_Display, JetBrains_Mono } from 'next/font/google'
import { PathnameLogger } from '@/components/PathnameLogger'
import './globals.css'

const inter = Inter({ 
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap'
})

const spaceGrotesk = Space_Grotesk({ 
  subsets: ['latin'],
  variable: '--font-space-grotesk',
  display: 'swap'
})

const playfairDisplay = Playfair_Display({ 
  subsets: ['latin'],
  variable: '--font-playfair-display',
  display: 'swap'
})

const jetbrainsMono = JetBrains_Mono({ 
  subsets: ['latin'],
  variable: '--font-jetbrains-mono',
  display: 'swap'
})

export const metadata: Metadata = {
  metadataBase: new URL('https://darwin.agourakis.med.br'),
  title: 'DARWIN AI - Plataforma de Pesquisa Inteligente',
  description: 'Acelere sua Descoberta Científica com IA. DARWIN é a sua plataforma de IA para orquestração de agentes, pesquisa avançada e análise de dados em escala.',
  keywords: [
    'DARWIN',
    'IA',
    'Inteligência Artificial',
    'AutoGen',
    'GroupChat',
    'Vertex AI',
    'BigQuery',
    'JAX',
    'Ultra-Performance',
    'Pesquisa Científica',
    'Agentes Autônomos',
    'Dr. Demetrios Chiuratto Agourakis',
    'AGOURAKIS MED RESEARCH'
  ],
  authors: [
    {
      name: 'Dr. Demetrios Chiuratto Agourakis',
      url: 'https://agourakis.med.br'
    }
  ],
  creator: 'Dr. Demetrios Chiuratto Agourakis',
  publisher: 'AGOURAKIS MED RESEARCH',
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  openGraph: {
    type: 'website',
    locale: 'pt_BR',
    url: 'https://darwin.agourakis.med.br',
    title: 'DARWIN AI - Plataforma de Pesquisa Inteligente',
    description: 'Acelere sua Descoberta Científica com IA. DARWIN é a sua plataforma de IA para orquestração de agentes, pesquisa avançada e análise de dados em escala.',
    siteName: 'DARWIN AI',
    images: [
      {
        url: '/og-image.jpg',
        width: 1200,
        height: 630,
        alt: 'DARWIN AI - Plataforma de Pesquisa Inteligente',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'DARWIN AI - Plataforma de Pesquisa Inteligente',
    description: 'Acelere sua Descoberta Científica com IA. DARWIN é a sua plataforma de IA para orquestração de agentes, pesquisa avançada e análise de dados em escala.',
    images: ['/twitter-image.jpg'],
    creator: '@AgourakisMed',
  },
  manifest: '/manifest.json',
  icons: {
    icon: [
      { url: '/favicon-16x16.png', sizes: '16x16', type: 'image/png' },
      { url: '/favicon-32x32.png', sizes: '32x32', type: 'image/png' },
    ],
    apple: [
      { url: '/apple-touch-icon.png', sizes: '180x180', type: 'image/png' },
    ],
    other: [
      {
        rel: 'mask-icon',
        url: '/safari-pinned-tab.svg',
        color: '#8B5CF6',
      },
    ],
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {

  return (
    <html
      lang="pt-BR"
      className={`${inter.variable} ${spaceGrotesk.variable} ${playfairDisplay.variable} ${jetbrainsMono.variable}`}
      suppressHydrationWarning
    >
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <meta name="format-detection" content="telephone=no" />
        <meta name="msapplication-TileColor" content="#3B82F6" />
        <meta name="theme-color" content="#ffffff" />
      </head>
      <body className="antialiased" suppressHydrationWarning>
        {children}
        <PathnameLogger />
      </body>
    </html>
  )
}