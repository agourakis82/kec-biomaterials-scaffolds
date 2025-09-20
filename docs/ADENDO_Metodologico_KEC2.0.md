# Adendo Metodológico (Set/2025) — Pipeline KEC 2.0 (Compatível com o Projeto Aprovado)

> **Escopo e Invariantes.** Atualização **exclusivamente computacional** do pipeline KEC para análise de
scaffolds porosos. Mantêm-se os **mesmos outputs** e interpretações: **Entropia (H)**, **Curvatura (κ)**,
**Coerência (σ/ϕ)** e diâmetro de percolação como preditores teciduais. As mudanças visam **robustez,
escalabilidade** e **validade externa** nível Q1, sem alterar objetivos científicos.

## 1) Visão Geral do Pipeline Evoluído
- μCT & pré-processamento: filtro bilateral/adaptativo (preserva borda); fontes reais priorizadas.
- Segmentação multi-escala: Otsu/limiar local (baseline); *proxy* CNN por patches (opcional) p/ microporos.
- Extração de rede porosa: PoreSpy/OpenPNM (watershed + esfera máxima); throats ponderados (>= v3.5).
- Grafo ponderado: pesos multi-atributo (diâmetro, comprimento, condutância proxy).
- **Métricas KEC (núcleo)**
  - **H (Entropia)**: **entropia espectral (von Neumann)** do Laplaciano normalizado (*substitui/acompanha* Shannon).
  - **κ (Curvatura)**: **Forman 2-complex** (rápida) + **Ollivier–Ricci** aproximada (Sinkhorn/limites) com **amostragem**.
  - **σ/ϕ (Small‑world)**: σ de Humphries & Gurney + SWP (Small‑World Propensity).
  - **σ_Q (opcional)**: coerência “quântica” = S(diag ρ) − S(ρ), com ρ a *network density matrix* (NDM).
- Modelagem & Validação: RF como baseline explícito; **GNN** paralela/híbrida (PyTorch) com *nested CV* e XAI.

## 2) Definições Operacionais
### 2.1 Entropia (H_espectral)
- Construir ρ a partir do **Laplaciano normalizado**; autovalores λᵢ → S(ρ)=−∑ pᵢ log pᵢ, pᵢ=λᵢ/∑λᵢ.
- Escalabilidade: **Lanczos** (eigs) com k≪N; fallback via estimativa estocástica de traço.

### 2.2 Curvatura (κ)
- **Forman (FRC 2-complex)**: fórmula combinatória (O(|E|)); relatar média e quantis (p95/p05).
- **Ollivier–Ricci (ORC)**: GraphRicciCurvature (Sinkhorn); **amostrar 10–30%** das arestas (ou limites inferiores).

### 2.3 Coerência (σ/ϕ) e σ_Q (opcional)
- **σ**: (C/C_rand)/(L/L_rand). **SWP** para reduzir viés de densidade.
- **σ_Q**: **S(diag ρ) − S(ρ)** como *feature* adicional (ativar apenas após validação convergente).

## 3) Plano de Validação (Q1)
- Sensibilidade: variação de limiar/ruído; *tiles* aleatórios; *error bars* de KEC.
- Robustez KEC: bootstrap de nós/arestas; **amostragem** para ORC.
- Generalização: RF (baseline) vs **GNN** (embedding ou preditor); *nested CV*, *stratified by material*.
- Nulos: *edge‑swap* (Maslov–Sneppen) preservando grau; testar desvio significativo.
- Critérios (DoD): variação <5% entre execuções; tempo máximo por grafo; ΔAUROC/ΔR² significativo ou não‑inferior.

## 4) Decisões de Implementação
- **H_espectral**: SciPy `eigsh` (k=64; tol 1e-8); fallback por traço estocástico.
- **κ**: FRC 2‑complex + ORC Sinkhorn (GraphRicciCurvature) com **amostragem** de arestas.
- **σ/ϕ**: NetworkX (σ) + implementação SWP (documentada).
- **σ_Q**: derivada do cálculo espectral (custo marginal).
- **GNN**: GCN/GAT simples; *dropout* 0.2–0.5; atributos de nó (grau, volume/área); aresta (diâmetro/condutância).

## 5) Ameaças & Mitigações
- Dependência de segmentação → stress‑tests; *tiles*; intervalos KEC reportados.
- Custo ORC → FRC + ORC aproximada/amostrada; relatar *runtime*.
- n pequeno p/ GNN → RF baseline; GNN como embedding até ampliar dataset.
- Colinearidade → VIF/PCov; importâncias estáveis (bootstrap).

## 6) Checklist Q1–Q10 (curto)
Precisão; Fontes Q1; Terminologia; Modularidade; Reprodutibilidade; Limitações; Critérios; Impacto; Ética/Dados; Alinhamento com outputs originais.

---

**Changelog (Set/2025)** — Mantidos outputs (**H/κ/σ/ϕ**); **H_espectral** (novo cálculo); **κ** com FRC+ORC‑aprox;
**σ/ϕ** mantidos; **σ_Q** opcional; RF baseline + GNN paralela/híbrida.
