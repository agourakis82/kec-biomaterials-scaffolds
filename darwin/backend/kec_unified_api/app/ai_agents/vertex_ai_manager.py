"""Vertex AI Manager - Integração Revolucionária com Gemini 1.5 Pro

Este módulo centraliza a lógica para configurar e interagir com os modelos
de IA da Vertex AI, incluindo o Gemini 1.5 Pro e modelos fine-tuned como o Med-Gemini.
"""

from typing import Dict, Any
from ..config.settings import settings

def get_vertex_ai_llm_config(model_name: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
    """
    Cria a configuração do LLM para modelos da Vertex AI (Gemini).
    """
    # Garante que as configurações do projeto e da localização estão disponíveis
    if not settings.gcp_project_id or not settings.gcp_location:
        raise ValueError("GCP_PROJECT_ID e GCP_LOCATION devem estar definidos nas configurações.")

    # Mapeia os nomes dos modelos para os identificadores completos da Vertex AI
    model_mapping = {
        "gemini-1.5-pro": f"projects/{settings.gcp_project_id}/locations/{settings.gcp_location}/publishers/google/models/gemini-1.5-pro-preview-0409",
        "med-gemini": f"projects/{settings.gcp_project_id}/locations/{settings.gcp_location}/endpoints/med_gemini_tuned_endpoint",  # Exemplo de endpoint de modelo fine-tuned
    }

    # Usa o nome do modelo para encontrar o identificador completo
    resolved_model_name = model_mapping.get(model_name, model_name)

    return {
        "model": resolved_model_name,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }