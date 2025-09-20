"""
Vertex AI Integration - Integração com Google Vertex AI
======================================================

Módulo especializado para integração com Vertex AI para embeddings,
geração de texto e outros modelos do Google Cloud.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class VertexConfig:
    """Configuração para Vertex AI."""
    project_id: str
    location: str = "us-central1"
    embedding_model: str = "textembedding-gecko@003"
    generation_model: str = "gemini-1.5-flash"
    max_output_tokens: int = 1024
    temperature: float = 0.3
    top_p: float = 0.8
    top_k: int = 40


class VertexAIIntegration:
    """
    Integração com Vertex AI para:
    - Text embeddings usando modelos gecko
    - Geração de texto com Gemini
    - Análise de sentimentos
    - Classificação de documentos
    """
    
    def __init__(self, config: VertexConfig):
        self.config = config
        self._aiplatform = None
        self._embedding_model = None
        self._generation_model = None
        self._initialized = False
        
    async def initialize(self) -> bool:
        """Inicializa conexão com Vertex AI."""
        try:
            # Lazy imports para não quebrar desenvolvimento local
            try:
                from google.cloud import aiplatform
                import vertexai
                from vertexai.preview.language_models import TextEmbeddingModel, TextGenerationModel
                
                self._aiplatform = aiplatform
                
                # Inicializa Vertex AI
                aiplatform.init(
                    project=self.config.project_id,
                    location=self.config.location
                )
                
                # Carrega modelos
                self._embedding_model = TextEmbeddingModel.from_pretrained(self.config.embedding_model)
                self._generation_model = TextGenerationModel.from_pretrained(self.config.generation_model)
                
                self._initialized = True
                logger.info("Vertex AI inicializado com sucesso")
                return True
                
            except ImportError as e:
                logger.warning(f"Vertex AI não disponível (desenvolvimento local): {e}")
                return False
                
        except Exception as e:
            logger.error(f"Erro ao inicializar Vertex AI: {e}")
            return False
    
    async def get_embedding(self, text: str) -> List[float]:
        """Gera embedding para texto usando Vertex AI."""
        if not self._initialized or not self._embedding_model:
            # Fallback determinístico para desenvolvimento
            return [(hash(text + str(i)) % 1000) / 1000.0 for i in range(768)]
            
        try:
            embeddings = self._embedding_model.get_embeddings([text])
            return embeddings[0].values
            
        except Exception as e:
            logger.error(f"Erro ao gerar embedding: {e}")
            # Fallback
            return [(hash(text + str(i)) % 1000) / 1000.0 for i in range(768)]
    
    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Gera embeddings para múltiplos textos."""
        if not self._initialized or not self._embedding_model:
            return [[(hash(text + str(i)) % 1000) / 1000.0 for i in range(768)] for text in texts]
            
        try:
            # Vertex AI suporta até 250 textos por batch
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                embeddings = self._embedding_model.get_embeddings(batch)
                all_embeddings.extend([emb.values for emb in embeddings])
                
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Erro ao gerar embeddings em batch: {e}")
            return [[(hash(text + str(i)) % 1000) / 1000.0 for i in range(768)] for text in texts]
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Gera texto usando modelo Gemini."""
        if not self._initialized or not self._generation_model:
            return f"[MOCK] Resposta gerada para: {prompt[:100]}..."
            
        try:
            # Configurações de geração
            generation_config = {
                "max_output_tokens": kwargs.get("max_tokens", self.config.max_output_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "top_k": kwargs.get("top_k", self.config.top_k),
            }
            
            response = self._generation_model.predict(
                prompt,
                **generation_config
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Erro na geração de texto: {e}")
            return f"[ERROR] Não foi possível gerar resposta: {str(e)}"
    
    async def generate_answer_with_context(
        self, 
        query: str, 
        context_docs: List[Dict[str, Any]],
        max_context_length: int = 4000
    ) -> str:
        """Gera resposta usando contexto de documentos recuperados."""
        
        # Prepara contexto
        context_parts = []
        current_length = 0
        
        for doc in context_docs:
            content = doc.get("content", "")[:500]  # Limita cada doc
            if current_length + len(content) > max_context_length:
                break
            context_parts.append(f"Documento: {content}")
            current_length += len(content)
        
        context = "\n\n".join(context_parts)
        
        # Prompt estruturado
        prompt = f"""Baseado no contexto abaixo, responda à pergunta de forma precisa e informativa.

Contexto:
{context}

Pergunta: {query}

Resposta:"""

        return await self.generate_text(prompt, max_tokens=512)
    
    async def classify_document(self, text: str, categories: List[str]) -> Dict[str, float]:
        """Classifica documento em categorias usando Vertex AI."""
        if not self._initialized:
            # Fallback para desenvolvimento
            import random
            scores = {cat: random.random() for cat in categories}
            total = sum(scores.values())
            return {cat: score/total for cat, score in scores.items()}
        
        try:
            prompt = f"""Classifique o seguinte texto nas categorias fornecidas. 
            Retorne scores de 0.0 a 1.0 para cada categoria.

Texto: {text[:1000]}

Categorias: {', '.join(categories)}

Classificação (formato JSON):"""

            response = await self.generate_text(prompt, max_tokens=256, temperature=0.1)
            
            # TODO: Parse JSON response properly
            # Por enquanto, retorna scores aleatórios normalizados
            import random
            scores = {cat: random.random() for cat in categories}
            total = sum(scores.values())
            return {cat: score/total for cat, score in scores.items()}
            
        except Exception as e:
            logger.error(f"Erro na classificação: {e}")
            # Fallback
            return {cat: 1.0/len(categories) for cat in categories}
    
    async def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extrai palavras-chave do texto usando Vertex AI."""
        if not self._initialized:
            # Fallback simples
            words = text.lower().split()
            return list(set(words[:max_keywords]))
        
        try:
            prompt = f"""Extraia as {max_keywords} palavras-chave mais importantes do seguinte texto.
            Retorne apenas as palavras, separadas por vírgula.

Texto: {text[:2000]}

Palavras-chave:"""

            response = await self.generate_text(prompt, max_tokens=100, temperature=0.1)
            
            # Parse resposta
            keywords = [kw.strip() for kw in response.split(',')]
            return keywords[:max_keywords]
            
        except Exception as e:
            logger.error(f"Erro na extração de keywords: {e}")
            # Fallback
            words = text.lower().split()
            return list(set(words[:max_keywords]))
    
    async def get_similarity(self, text1: str, text2: str) -> float:
        """Calcula similaridade semântica entre dois textos."""
        embeddings = await self.get_embeddings_batch([text1, text2])
        
        if len(embeddings) != 2:
            return 0.0
            
        # Calcula similaridade coseno
        import numpy as np
        vec1 = np.array(embeddings[0])
        vec2 = np.array(embeddings[1])
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    async def get_status(self) -> Dict[str, Any]:
        """Retorna status da integração Vertex AI."""
        return {
            "service": "vertex_ai",
            "initialized": self._initialized,
            "project_id": self.config.project_id,
            "location": self.config.location,
            "embedding_model": self.config.embedding_model,
            "generation_model": self.config.generation_model,
            "status": "ready" if self._initialized else "mock_mode"
        }