"""ChatGPT Client - Integração especializada com OpenAI para biomateriais e STEM."""

import asyncio
import openai
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from ...core.logging import get_logger
from ...models.multi_ai_models import AIModel, ChatMessage, ScientificDomain

logger = get_logger("multi_ai.chatgpt")


class ChatGPTClient:
    """Cliente especializado para ChatGPT com otimizações para biomateriais e STEM."""
    
    def __init__(self, api_key: str):
        """
        Inicializar cliente ChatGPT.
        
        Args:
            api_key: OpenAI API key
        """
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.models = [
            AIModel.GPT_4_TURBO,
            AIModel.GPT_4,
            AIModel.GPT_3_5_TURBO
        ]
        self.enabled = False
        self.specialties = [
            ScientificDomain.BIOMATERIALS,
            ScientificDomain.SCAFFOLD_DESIGN,
            ScientificDomain.MATERIALS_ENGINEERING,
            ScientificDomain.CODE_GENERATION,
            ScientificDomain.DEBUGGING,
            ScientificDomain.ARCHITECTURE
        ]
        
    async def initialize(self):
        """Inicializar cliente ChatGPT."""
        try:
            logger.info("Initializing ChatGPT client...")
            
            # Test API connection
            await self._test_connection()
            self.enabled = True
            
            logger.info("✅ ChatGPT client initialized successfully")
            logger.info(f"Available models: {[model.value for model in self.models]}")
            logger.info(f"Specialties: {[domain.value for domain in self.specialties]}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChatGPT client: {e}")
            self.enabled = False
            raise
    
    async def shutdown(self):
        """Shutdown cliente ChatGPT."""
        logger.info("Shutting down ChatGPT client...")
        self.enabled = False
        await self.client.close()
        logger.info("✅ ChatGPT client shutdown")
    
    async def chat(self, 
                   messages: List[Dict[str, str]], 
                   model: AIModel = AIModel.GPT_4_TURBO,
                   domain: Optional[ScientificDomain] = None,
                   temperature: float = 0.7,
                   max_tokens: Optional[int] = None,
                   **kwargs) -> Dict[str, Any]:
        """
        Chat completions com ChatGPT otimizado para domínios específicos.
        
        Args:
            messages: Lista de mensagens
            model: Modelo específico a usar
            domain: Domínio científico (para otimizações)
            temperature: Criatividade (0-2)
            max_tokens: Máximo de tokens de resposta
            **kwargs: Parâmetros adicionais
        """
        if not self.enabled:
            raise RuntimeError("ChatGPT client not initialized")
        
        try:
            start_time = datetime.now(timezone.utc)
            
            # Domain-specific optimization
            optimized_messages = self._optimize_for_domain(messages, domain)
            
            # Model-specific parameters
            chat_params = self._get_model_params(model, domain, temperature, max_tokens)
            
            # Make API call
            logger.info(f"ChatGPT request: {model.value} for domain {domain}")
            
            response = await self.client.chat.completions.create(
                model=model.value,
                messages=optimized_messages,
                **chat_params
            )
            
            # Process response
            result = self._process_response(response, start_time, domain)
            
            logger.info(f"ChatGPT response: {len(result['content'])} chars, "
                       f"{result['usage']['total_tokens']} tokens, "
                       f"{result['latency_ms']:.1f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"ChatGPT chat error: {e}")
            raise
    
    def _optimize_for_domain(self, messages: List[Dict[str, str]], 
                           domain: Optional[ScientificDomain]) -> List[Dict[str, str]]:
        """Otimizar mensagens para domínio específico."""
        
        optimized = messages.copy()
        
        # Add domain-specific system message if not present
        if domain and (not messages or messages[0].get("role") != "system"):
            system_message = self._get_domain_system_message(domain)
            optimized.insert(0, {"role": "system", "content": system_message})
        
        return optimized
    
    def _get_domain_system_message(self, domain: ScientificDomain) -> str:
        """Obter mensagem de sistema otimizada para domínio."""
        
        domain_prompts = {
            ScientificDomain.BIOMATERIALS: """You are a leading biomaterials expert with deep knowledge in:
- Biocompatible materials (ceramics, polymers, composites)
- Tissue engineering and regenerative medicine
- Cell-material interactions and bioactivity
- Material characterization and testing
- Regulatory aspects (FDA, ISO standards)

Focus on practical, evidence-based solutions with attention to safety and biocompatibility.""",

            ScientificDomain.SCAFFOLD_DESIGN: """You are a specialist in scaffold design and fabrication with expertise in:
- 3D printing and additive manufacturing
- Porous architecture optimization
- Mechanical property tuning
- Surface modification techniques
- Degradation kinetics modeling

Provide detailed technical guidance with consideration for manufacturing feasibility.""",

            ScientificDomain.MATERIALS_ENGINEERING: """You are a materials engineering expert specializing in:
- Structure-property relationships
- Processing-microstructure-property correlations
- Advanced characterization techniques
- Failure analysis and quality control
- Materials selection and optimization

Give precise technical analysis with quantitative recommendations where possible.""",

            ScientificDomain.CODE_GENERATION: """You are an expert software engineer with focus on:
- Clean, maintainable code architecture
- Performance optimization
- Best practices and design patterns
- Testing and documentation
- Modern development frameworks

Write high-quality, well-commented code with error handling and scalability in mind.""",

            ScientificDomain.DEBUGGING: """You are a debugging specialist with skills in:
- Systematic error analysis
- Performance profiling and optimization
- Security vulnerability assessment
- Cross-platform compatibility
- Testing and validation strategies

Provide step-by-step debugging approaches with preventive measures.""",

            ScientificDomain.ARCHITECTURE: """You are a software architect with expertise in:
- Scalable system design
- Microservices and distributed systems
- Cloud-native architectures
- Security and compliance
- Performance and reliability patterns

Design robust, scalable solutions following industry best practices."""
        }
        
        return domain_prompts.get(domain, 
            "You are a helpful AI assistant with expertise across multiple scientific and technical domains.")
    
    def _get_model_params(self, model: AIModel, domain: Optional[ScientificDomain],
                         temperature: float, max_tokens: Optional[int]) -> Dict[str, Any]:
        """Obter parâmetros otimizados por modelo."""
        
        params = {
            "temperature": temperature,
            "top_p": 0.95,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
        
        # Model-specific optimizations
        if model == AIModel.GPT_4_TURBO:
            params.update({
                "max_tokens": max_tokens or 4000,
                "top_p": 0.95,
                "temperature": min(temperature, 1.0)  # GPT-4 works better with lower temp
            })
        elif model == AIModel.GPT_4:
            params.update({
                "max_tokens": max_tokens or 2000,
                "temperature": min(temperature, 0.9)
            })
        else:  # GPT-3.5-Turbo
            params.update({
                "max_tokens": max_tokens or 1500,
                "temperature": temperature
            })
        
        # Domain-specific adjustments
        if domain in [ScientificDomain.MATHEMATICAL_PROOFS, ScientificDomain.CODE_GENERATION]:
            params["temperature"] = min(params["temperature"], 0.3)  # More deterministic
        elif domain in [ScientificDomain.BIOMATERIALS, ScientificDomain.MATERIALS_ENGINEERING]:
            params["temperature"] = min(params["temperature"], 0.7)  # Balanced creativity
        
        return params
    
    def _process_response(self, response: Any, start_time: datetime, 
                         domain: Optional[ScientificDomain]) -> Dict[str, Any]:
        """Processar resposta da API."""
        
        end_time = datetime.now(timezone.utc)
        latency_ms = (end_time - start_time).total_seconds() * 1000
        
        message = response.choices[0].message
        usage = response.usage
        
        # Extract content
        content = message.content or ""
        
        # Domain-specific post-processing
        if domain == ScientificDomain.CODE_GENERATION:
            content = self._enhance_code_response(content)
        elif domain in [ScientificDomain.BIOMATERIALS, ScientificDomain.SCAFFOLD_DESIGN]:
            content = self._enhance_scientific_response(content)
        
        return {
            "content": content,
            "role": message.role,
            "model": response.model,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            },
            "latency_ms": latency_ms,
            "timestamp": end_time.isoformat(),
            "domain": domain.value if domain else None,
            "finish_reason": response.choices[0].finish_reason
        }
    
    def _enhance_code_response(self, content: str) -> str:
        """Aprimorar resposta de código."""
        # Add helpful headers if code is detected
        if "```" in content and "def " in content:
            if not content.startswith("Here's") and not content.startswith("This"):
                content = "Here's the implemented code:\n\n" + content
        
        return content
    
    def _enhance_scientific_response(self, content: str) -> str:
        """Aprimorar resposta científica."""
        # Add safety note for biomaterials if not present
        if "biocompat" in content.lower() and "safety" not in content.lower():
            content += "\n\n⚠️ **Safety Note**: Always validate biocompatibility through appropriate testing protocols before clinical applications."
        
        return content
    
    async def _test_connection(self):
        """Testar conexão com API."""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            logger.info("ChatGPT API connection test successful")
            return True
        except Exception as e:
            logger.error(f"ChatGPT API connection test failed: {e}")
            raise
    
    def get_available_models(self) -> List[AIModel]:
        """Obter modelos disponíveis."""
        return self.models.copy()
    
    def get_specialties(self) -> List[ScientificDomain]:
        """Obter domínios de especialidade."""
        return self.specialties.copy()
    
    def is_specialized_for(self, domain: ScientificDomain) -> bool:
        """Verificar se é especializado para domínio."""
        return domain in self.specialties
    
    async def estimate_cost(self, messages: List[Dict[str, str]], 
                           model: AIModel = AIModel.GPT_4_TURBO) -> float:
        """Estimar custo de request."""
        # Rough token estimation
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        estimated_tokens = total_chars // 4  # Rough approximation
        
        # Cost per 1K tokens (as of 2024)
        cost_per_1k = {
            AIModel.GPT_4_TURBO: 0.01,
            AIModel.GPT_4: 0.03,
            AIModel.GPT_3_5_TURBO: 0.002
        }
        
        rate = cost_per_1k.get(model, 0.01)
        return (estimated_tokens / 1000) * rate
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check do cliente."""
        health_status = {
            "healthy": self.enabled,
            "status": "operational" if self.enabled else "offline",
            "available_models": [model.value for model in self.models],
            "specialties": [domain.value for domain in self.specialties],
            "last_check": datetime.now(timezone.utc).isoformat()
        }
        
        if self.enabled:
            try:
                await self._test_connection()
                health_status["api_connection"] = "healthy"
            except Exception as e:
                health_status["api_connection"] = f"error: {str(e)}"
                health_status["healthy"] = False
        
        return health_status


__all__ = ["ChatGPTClient"]