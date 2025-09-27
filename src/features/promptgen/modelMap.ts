import type { ModelId } from "./types"

export const MODEL_OPTIONS: { id: ModelId; name: string }[] = [
  { id: "gpt-5", name: "GPT-5" },
  { id: "gpt-5-pro", name: "GPT-5 Pro" },
  { id: "gemini-2-pro", name: "Gemini 2.0 Pro" },
]

export function openInChat(model: ModelId, system: string, user: string) {
  // Fallback: copy to clipboard; deep links vary per provider.
  const text = `System\n-----\n${system}\n\nUser\n----\n${user}`
  void navigator.clipboard.writeText(text)
  alert("Prompt copiado para a área de transferência.")
}

