import { Metadata } from "next"
import { KnowledgeGraphViz } from "@/components/darwin/KnowledgeGraphViz"

export const metadata: Metadata = {
  title: "Knowledge Graph DARWIN | Agourakis Med Research",
  description: "Visualização interativa do grafo de conhecimento interdisciplinar com conexões cross-domain em biomateriais.",
}

export default function KnowledgeGraphPage() {
  return (
    <div className="space-y-8">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold text-gradient">
          Knowledge Graph DARWIN
        </h1>
        <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
          Explore conexões interdisciplinares em tempo real. Visualize como conceitos de biomateriais, 
          engenharia de tecidos e ciências correlatas se conectam no ecossistema científico.
        </p>
      </div>

      <KnowledgeGraphViz 
        domains={["biomaterials", "biomedical", "materials_science"]}
        interactive={true}
        layout="domain"
      />
    </div>
  )
}