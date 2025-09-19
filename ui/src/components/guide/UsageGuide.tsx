"use client"

import { motion } from "framer-motion"
import { Compass, FileText, Settings, Sparkles, Zap } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

const guideItems = [
  {
    title: "Consultas Inteligentes",
    icon: Sparkles,
    highlight: "Busque com precisão",
    steps: [
      "Digite a pergunta no campo principal usando termos claros e objetivos.",
      "Pressione ⌘/Ctrl + Enter ou clique em 'Consultar' para enviar.",
      "Analise a resposta gerada e utilize o botão 'Copiar' para salvar insights importantes.",
    ],
  },
  {
    title: "Modo Iterativo (ReAct)",
    icon: Zap,
    highlight: "Investigue em camadas",
    steps: [
      "Ative o modo Iterativo para permitir que o agente consulte múltiplas fontes.",
      "Acompanhe como os passos do raciocínio se refletem nas citações numeradas.",
      "Desative quando quiser uma resposta direta e mais rápida.",
    ],
  },
  {
    title: "Fontes, PDFs e citações",
    icon: FileText,
    highlight: "Aprofunde-se nas evidências",
    steps: [
      "Passe o cursor sobre os indicadores para destacar a referência correspondente.",
      "Clique em uma fonte para abrir a pré-visualização do PDF no painel lateral.",
      "Use o botão 'Gerado por IA' para lembrar que a resposta precisa de validação clínica.",
    ],
  },
  {
    title: "Perfis e Configurações",
    icon: Settings,
    highlight: "Personalize seu contexto",
    steps: [
      "Abra o painel de configurações para alternar domínios e filtros especializados.",
      "Utilize a paleta de comandos (⌘/Ctrl + K) para acessar ações rápidas.",
      "Alterne entre temas claro e escuro para trabalhar com conforto em qualquer ambiente.",
    ],
  },
]

export function UsageGuide() {
  return (
    <section id="guia-de-uso" className="relative overflow-hidden rounded-3xl border bg-background/60 py-12">
      <motion.div
        className="absolute inset-x-0 -top-40 flex justify-center opacity-20"
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 0.2 }}
        transition={{ duration: 1.5 }}
      >
        <Compass className="h-48 w-48 text-primary" />
      </motion.div>
      <div className="container mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="mx-auto mb-12 max-w-3xl text-center space-y-3"
        >
          <Badge variant="secondary" className="mx-auto w-fit border border-primary/40 bg-primary/10 text-primary">
            Guia de Uso
          </Badge>
          <h2 className="text-3xl font-bold">Navegue pela plataforma com confiança</h2>
          <p className="text-muted-foreground">
            Cada função foi desenhada para apoiar a pesquisa clínica em biomateriais. Siga as orientações abaixo para extrair
            o máximo do app Agourakis Med Research.
          </p>
        </motion.div>

        <div className="grid gap-6 md:grid-cols-2">
          {guideItems.map((item, index) => {
            const Icon = item.icon
            return (
              <motion.div
                key={item.title}
                initial={{ opacity: 0, y: 12 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1, duration: 0.5 }}
              >
                <Card className="h-full border-primary/20 bg-background/80 backdrop-blur">
                  <CardHeader className="space-y-3">
                    <div className="flex items-center gap-3">
                      <span className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/10 text-primary">
                        <Icon className="h-6 w-6" />
                      </span>
                      <div>
                        <CardTitle className="text-lg">{item.title}</CardTitle>
                        <p className="text-xs font-medium uppercase tracking-wide text-primary/80">
                          {item.highlight}
                        </p>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <ol className="space-y-3 pl-5 text-sm text-muted-foreground marker:text-primary">
                      {item.steps.map((step, idx) => (
                        <li key={idx} className="leading-relaxed">
                          {step}
                        </li>
                      ))}
                    </ol>
                  </CardContent>
                </Card>
              </motion.div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
