import { Metadata } from "next"
import { MultiAIChat } from "@/components/darwin/MultiAIChat"

export const metadata: Metadata = {
  title: "Multi-AI Hub DARWIN | Agourakis Med Research",
  description: "Chat unificado com múltiplas IAs com seleção automática do melhor modelo para biomateriais e pesquisa científica.",
}

export default function MultiAIPage() {
  return (
    <div className="space-y-8">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold text-gradient">
          Multi-AI Hub DARWIN
        </h1>
        <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
          Converse simultaneamente com ChatGPT-4, Claude-3 e Gemini Pro. 
          O DARWIN seleciona automaticamente a melhor IA para sua pergunta científica.
        </p>
      </div>

      <MultiAIChat
        domains={["biomaterials", "research", "analysis"]}
      />
    </div>
  )
}