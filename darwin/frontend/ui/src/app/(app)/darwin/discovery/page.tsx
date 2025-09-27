import { Metadata } from "next"
import { DiscoveryDashboard } from "@/components/darwin/DiscoveryDashboard"

export const metadata: Metadata = {
  title: "Scientific Discovery DARWIN | Agourakis Med Research",
  description: "Monitoramento em tempo real de 26 RSS feeds científicos com análise de novelty e descobertas cross-domain em biomateriais.",
}

export default function DiscoveryPage() {
  return (
    <div className="space-y-8">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold text-gradient">
          Scientific Discovery DARWIN
        </h1>
        <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
          Monitoramento científico em tempo real de 26 RSS feeds especializados. 
          Detecte descobertas inovadoras, conexões cross-domain e insights emergentes automaticamente.
        </p>
      </div>

      <DiscoveryDashboard 
        autoRefresh={true}
      />
    </div>
  )
}