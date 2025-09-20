# KEC_BIOMAT — Memória resumida (colar no ChatGPT e fixar)
- **Escopo fixo**: outputs = Entropia (H), Curvatura (κ), Coerência (σ/ϕ). Não alterar objetivos do projeto aprovado.
- **H (novo cálculo)**: entropia **espectral** (von Neumann) do Laplaciano normalizado; substitui/acompanha Shannon.
- **κ (eficiência)**: **Forman 2-complex** + **Ollivier–Ricci aproximada** (Sinkhorn/limites) com **amostragem** de arestas.
- **σ/ϕ**: manter **σ** (Humphries & Gurney) + **SWP** (Small-World Propensity) para controlar densidade.
- **σ_Q (opcional)**: coerência “quântica” = S(diag ρ) − S(ρ) (usar como feature adicional, após validação).
- **Modelagem**: Random Forest baseline; **GNN** paralela/híbrida (embedding) com nested CV, ablações e XAI.
- **Qualidade**: reprodutibilidade (<5%), escalabilidade (ORC aprox + amostragem), ΔAUROC/ΔR² significativo vs baseline.
- **Arquivos**: ver `docs/ADENDO_Metodologico_KEC2.0.md` e `configs/kec_config.yaml`.
