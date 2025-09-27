import { Metadata } from "next"
import { KECDashboard } from "@/components/darwin/KECDashboard"

export const metadata: Metadata = {
  title: "KEC Analysis DARWIN | Agourakis Med Research",
  description: "Análise topológica real-time de scaffolds biomédicos com métricas H-spectral, K-Forman e performance sub-20ms.",
}

export default function KECMetricsPage() {
  return (
    <div className="space-y-8">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold text-gradient">
          KEC Analysis DARWIN
        </h1>
        <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
          Análise topológica avançada de scaffolds com métricas KEC em tempo real. 
          Performance ultra-rápida (&lt;20ms) com visualização 3D e otimizações automáticas.
        </p>
      </div>

      <KECDashboard 
        realTime={true}
      />
    </div>
  )
}