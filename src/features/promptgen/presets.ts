import type { PromptPreset } from "./types"

export const PRESETS: PromptPreset[] = [
  {
    id: "ragpp",
    name: "RAG++",
    params: {
      TASK: "Responder de forma objetiva com citações [n]",
      STYLE: "Tom técnico, conciso; bullets quando útil",
      k: 8,
      depth: 2,
      iters: 0,
    },
  },
  {
    id: "deep150",
    name: "Deep Research 150",
    params: {
      TASK: "Pesquisa profunda e síntese com plano e verificação cruzada",
      STYLE: "Detalhado, etapas claras, notas de confiança",
      iters: 5,
      k: 12,
      depth: 3,
    },
  },
  {
    id: "react",
    name: "ReAct",
    params: {
      TASK: "Raciocínio iterativo com ferramentas, registrar ‘Thought/Action/Observation’",
      STYLE: "Explícito, transparente, passos pequenos",
      iters: 3,
      k: 6,
    },
  },
  {
    id: "puct",
    name: "PUCT",
    params: {
      TASK: "Explorar árvore de decisões orientada por valor/visitas (PUCT)",
      STYLE: "Sumário do melhor caminho + alternativas",
      c_puct: 1.4,
      depth: 3,
    },
  },
]

