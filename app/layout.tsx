import type { Metadata } from 'next'
import { Inter, JetBrains_Mono } from 'next/font/google'
import '../src/app/globals.css'

const inter = Inter({ 
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap'
})

const jetbrainsMono = JetBrains_Mono({ 
  subsets: ['latin'],
  variable: '--font-jetbrains-mono',
  display: 'swap'
})

export const metadata: Metadata = {
  title: 'DARWIN AI - Plataforma de Pesquisa Inteligente',
  description: 'Acelere sua Descoberta Científica com IA. DARWIN é a sua plataforma de IA para orquestração de agentes, pesquisa avançada e análise de dados em escala.',
  keywords: [
    'DARWIN',
    'IA',
    'Inteligência Artificial',
    'Pesquisa Científica',
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
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html
      lang="pt-BR"
      className={`${inter.variable} ${jetbrainsMono.variable}`}
      suppressHydrationWarning
    >
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <meta name="format-detection" content="telephone=no" />
        <meta name="theme-color" content="#3B82F6" />
      </head>
      <body className="antialiased" suppressHydrationWarning>
        {children}
      </body>
    </html>
  )
}