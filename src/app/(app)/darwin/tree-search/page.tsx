import { Metadata } from "next"
import { PUCTOptimizer } from "@/components/darwin/PUCTOptimizer"

export const metadata: Metadata = {
  title: "PUCT Optimizer DARWIN | Agourakis Med Research",
  description: "Otimização tree search PUCT com performance 115k nodes/segundo para problemas complexos de biomateriais e scaffolds.",
}

export default function TreeSearchPage() {
  return (
    <div className="space-y-8">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold text-gradient">
          PUCT Optimizer DARWIN
        </h1>
        <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
          Otimização avançada usando tree search PUCT com performance ultra-alta (115k nodes/s). 
          Encontre soluções Pareto-otimais para problemas complexos de scaffolds e redes biológicas.
        </p>
      </div>

      <PUCTOptimizer 
        problemType="scaffold"
        realTimeProgress={true}
      />
    </div>
  )
}