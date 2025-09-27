"""Gemini Client - Integração especializada com Google AI para research e academic writing."""

import asyncio
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from ...core.logging import get_logger
from ...models.multi_ai_models import AIModel, ChatMessage, ScientificDomain

logger = get_logger("multi_ai.gemini")


class GeminiClient:
    """Cliente especializado para Gemini com otimizações para research e Google integration."""
    
    def __init__(self, api_key: str):
        """
        Inicializar cliente Gemini.
        
        Args:
            api_key: Google AI API key
        """
        genai.configure(api_key=api_key)
        self.models = [
            AIModel.GEMINI_PRO,
            AIModel.GEMINI_PRO_VISION
        ]
        self.enabled = False
        self.specialties = [
            ScientificDomain.LITERATURE_SEARCH,
            ScientificDomain.RESEARCH_SYNTHESIS,
            ScientificDomain.ACADEMIC_WRITING,
            ScientificDomain.INTERDISCIPLINARY
        ]
        self._clients = {}
        
    async def initialize(self):
        """Inicializar cliente Gemini."""
        try:
            logger.info("Initializing Gemini client...")
            
            # Initialize model clients
            for model in self.models:
                self._clients[model] = genai.GenerativeModel(model.value)
            
            # Test API connection
            await self._test_connection()
            self.enabled = True
            
            logger.info("✅ Gemini client initialized successfully")
            logger.info(f"Available models: {[model.value for model in self.models]}")
            logger.info(f"Specialties: {[domain.value for domain in self.specialties]}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            self.enabled = False
            raise
    
    async def shutdown(self):
        """Shutdown cliente Gemini."""
        logger.info("Shutting down Gemini client...")
        self.enabled = False
        self._clients.clear()
        logger.info("✅ Gemini client shutdown")
    
    async def chat(self, 
                   messages: List[Dict[str, str]], 
                   model: AIModel = AIModel.GEMINI_PRO,
                   domain: Optional[ScientificDomain] = None,
                   temperature: float = 0.7,
                   max_tokens: Optional[int] = None,
                   **kwargs) -> Dict[str, Any]:
        """
        Chat com Gemini otimizado para research e academic writing.
        
        Args:
            messages: Lista de mensagens
            model: Modelo específico a usar
            domain: Domínio científico (para otimizações)
            temperature: Criatividade (0-2)
            max_tokens: Máximo de tokens de resposta
            **kwargs: Parâmetros adicionais
        """
        if not self.enabled:
            raise RuntimeError("Gemini client not initialized")
        
        if model not in self._clients:
            raise ValueError(f"Model {model} not available")
        
        try:
            start_time = datetime.now(timezone.utc)
            
            # Get model client
            client = self._clients[model]
            
            # Domain-specific optimization
            enhanced_prompt = self._optimize_for_domain(messages, domain)
            
            # Configure generation parameters
            generation_config = self._get_generation_config(model, domain, temperature, max_tokens)
            
            # Safety settings for research content
            safety_settings = self._get_safety_settings(domain)
            
            # Make API call
            logger.info(f"Gemini request: {model.value} for domain {domain}")
            
            # Convert to single prompt (Gemini uses different format)
            prompt = self._messages_to_prompt(enhanced_prompt)
            
            response = await asyncio.to_thread(
                client.generate_content,
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Process response
            result = self._process_response(response, start_time, model, domain)
            
            logger.info(f"Gemini response: {len(result['content'])} chars, "
                       f"~{result['estimated_tokens']} tokens, "
                       f"{result['latency_ms']:.1f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Gemini chat error: {e}")
            raise
    
    def _optimize_for_domain(self, messages: List[Dict[str, str]], 
                           domain: Optional[ScientificDomain]) -> List[Dict[str, str]]:
        """Otimizar mensagens para domínio específico."""
        
        optimized = messages.copy()
        
        # Add domain-specific system context if not present
        if domain and (not messages or messages[0].get("role") != "system"):
            system_message = self._get_domain_system_message(domain)
            optimized.insert(0, {"role": "system", "content": system_message})
        
        return optimized
    
    def _get_domain_system_message(self, domain: ScientificDomain) -> str:
        """Obter mensagem de sistema otimizada para domínio."""
        
        domain_prompts = {
            ScientificDomain.LITERATURE_SEARCH: """You are an expert research librarian and information scientist with advanced skills in:

**Literature Search Excellence:**
- Advanced search strategies across academic databases (PubMed, Google Scholar, Web of Science, Scopus)
- Boolean search operators, MeSH terms, and controlled vocabularies
- Citation analysis and bibliometrics
- Systematic review and meta-analysis methodologies

**Database Expertise:**
- PubMed/MEDLINE for biomedical literature
- IEEE Xplore for engineering and computer science
- arXiv for preprints and cutting-edge research
- Cochrane Library for evidence-based medicine
- Discipline-specific databases (ChemSpider, MathSciNet, etc.)

**Search Optimization:**
- Query refinement and iteration strategies
- Handling information overload and result filtering
- Identifying seminal papers and key researchers
- Tracking research trends and emerging topics

**Quality Assessment:**
- Journal impact factors and citation metrics
- Peer review quality indicators
- Predatory journal identification
- Research methodology evaluation

Provide comprehensive search strategies with specific database recommendations, query suggestions, and quality filtering criteria. Include practical tips for efficient literature discovery.""",

            ScientificDomain.RESEARCH_SYNTHESIS: """You are a research synthesis expert with deep expertise in:

**Synthesis Methodologies:**
- Systematic reviews and meta-analyses
- Narrative reviews and scoping studies
- Rapid reviews and umbrella reviews
- Qualitative synthesis methods (meta-ethnography, framework synthesis)

**Data Extraction & Analysis:**
- Standardized data extraction forms
- Quality assessment tools (PRISMA, GRADE, Newcastle-Ottawa)
- Statistical meta-analysis techniques
- Handling heterogeneity and bias assessment

**Knowledge Integration:**
- Identifying convergent and divergent findings
- Synthesizing quantitative and qualitative evidence
- Cross-disciplinary integration methods
- Gap analysis and future research directions

**Presentation & Communication:**
- Evidence tables and forest plots
- PRISMA flow diagrams
- Summary of findings tables
- Plain language summaries for diverse audiences

**Research Translation:**
- From research to practice recommendations
- Policy implications and implementation strategies
- Stakeholder engagement and knowledge mobilization
- Uncertainty communication and limitations

Synthesize complex research landscapes into coherent, actionable insights while maintaining methodological rigor and transparency about limitations.""",

            ScientificDomain.ACADEMIC_WRITING: """You are an academic writing expert with comprehensive knowledge in:

**Writing Excellence:**
- Clear, precise scientific communication
- Argument structure and logical flow
- Academic voice and tone
- Discipline-specific writing conventions

**Research Paper Structure:**
- IMRaD format (Introduction, Methods, Results, Discussion)
- Literature reviews and theoretical frameworks
- Methodology sections and research design
- Results presentation and data visualization
- Discussion sections and implication development

**Citation & Referencing:**
- Multiple citation styles (APA, MLA, Chicago, Vancouver, Harvard)
- Reference management and software tools
- Proper attribution and avoiding plagiarism
- Citation analysis and strategic referencing

**Publication Process:**
- Journal selection and submission strategies
- Peer review process and reviewer responses
- Manuscript revision and resubmission
- Open access publishing and preprint servers

**Research Communication:**
- Grant writing and funding applications
- Conference abstracts and presentations
- Policy briefs and stakeholder summaries
- Research dissemination strategies

**Quality Enhancement:**
- Clarity and conciseness optimization
- Avoiding common writing pitfalls
- Proofreading and editing techniques
- Collaborative writing and version control

Support authors in creating high-impact scholarly work that meets rigorous academic standards while being accessible to intended audiences.""",

            ScientificDomain.INTERDISCIPLINARY: """You are an interdisciplinary research facilitator with expertise in:

**Cross-Disciplinary Integration:**
- Bridging different theoretical frameworks and methodologies
- Translating concepts across disciplinary boundaries
- Identifying convergent themes and complementary approaches
- Managing epistemological and methodological differences

**Knowledge Synthesis:**
- Multi-perspective analysis and triangulation
- Systems thinking and complexity approaches
- Network analysis and boundary crossing
- Innovation through disciplinary fusion

**Collaboration Facilitation:**
- Team science and collaborative research methods
- Communication across disciplinary languages
- Conflict resolution and consensus building
- Shared understanding development

**Research Design:**
- Mixed-methods approaches
- Participatory and community-based research
- Transdisciplinary research frameworks
- Integration of quantitative and qualitative methods

**Application Domains:**
- Sustainability and environmental challenges
- Health and social issues
- Technology and society interactions
- Complex adaptive systems

**Innovation & Translation:**
- Knowledge mobilization across sectors
- Stakeholder engagement strategies
- Policy-relevant research development
- Real-world impact assessment

Excel at finding connections between seemingly disparate fields, facilitating meaningful collaboration, and generating novel insights through interdisciplinary synthesis."""
        }
        
        return domain_prompts.get(domain, 
            "You are Gemini, Google's helpful AI assistant with broad knowledge and strong research capabilities. You excel at finding, synthesizing, and communicating complex information.")
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Converter mensagens para formato de prompt único do Gemini."""
        
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System Instructions: {content}\n")
            elif role == "user":
                prompt_parts.append(f"User: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")
        
        return "\n".join(prompt_parts)
    
    def _get_generation_config(self, model: AIModel, domain: Optional[ScientificDomain],
                             temperature: float, max_tokens: Optional[int]) -> genai.types.GenerationConfig:
        """Obter configuração de geração otimizada."""
        
        config_params = {
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 40
        }
        
        # Model-specific optimizations
        if model == AIModel.GEMINI_PRO:
            config_params.update({
                "max_output_tokens": max_tokens or 2048,
                "temperature": min(temperature, 1.5)  # Gemini supports up to 2.0
            })
        elif model == AIModel.GEMINI_PRO_VISION:
            config_params.update({
                "max_output_tokens": max_tokens or 2048,
                "temperature": temperature
            })
        
        # Domain-specific adjustments
        if domain in [ScientificDomain.LITERATURE_SEARCH, ScientificDomain.RESEARCH_SYNTHESIS]:
            config_params["temperature"] = min(config_params["temperature"], 0.8)  # More focused
        elif domain == ScientificDomain.ACADEMIC_WRITING:
            config_params["temperature"] = min(config_params["temperature"], 0.9)  # Balanced creativity
        elif domain == ScientificDomain.INTERDISCIPLINARY:
            config_params["temperature"] = min(config_params["temperature"], 1.0)  # Creative connections
        
        return genai.types.GenerationConfig(**config_params)
    
    def _get_safety_settings(self, domain: Optional[ScientificDomain]) -> List[Dict[str, Any]]:
        """Obter configurações de segurança para conteúdo acadêmico."""
        
        # Relaxed safety for academic content
        return [
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH
            },
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH
            },
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH
            },
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH
            }
        ]
    
    def _process_response(self, response: Any, start_time: datetime, 
                         model: AIModel, domain: Optional[ScientificDomain]) -> Dict[str, Any]:
        """Processar resposta da API."""
        
        end_time = datetime.now(timezone.utc)
        latency_ms = (end_time - start_time).total_seconds() * 1000
        
        # Extract content
        content = ""
        if response.text:
            content = response.text
        
        # Domain-specific post-processing
        if domain == ScientificDomain.LITERATURE_SEARCH:
            content = self._enhance_literature_response(content)
        elif domain == ScientificDomain.RESEARCH_SYNTHESIS:
            content = self._enhance_synthesis_response(content)
        elif domain == ScientificDomain.ACADEMIC_WRITING:
            content = self._enhance_writing_response(content)
        elif domain == ScientificDomain.INTERDISCIPLINARY:
            content = self._enhance_interdisciplinary_response(content)
        
        # Estimate tokens (Gemini doesn't provide token usage in response)
        estimated_tokens = len(content.split()) * 1.3  # Rough estimation
        
        return {
            "content": content,
            "role": "assistant",
            "model": model.value,
            "usage": {
                "input_tokens": 0,  # Not provided by Gemini
                "output_tokens": int(estimated_tokens),
                "total_tokens": int(estimated_tokens)
            },
            "estimated_tokens": int(estimated_tokens),
            "latency_ms": latency_ms,
            "timestamp": end_time.isoformat(),
            "domain": domain.value if domain else None,
            "finish_reason": "stop"  # Gemini doesn't provide this explicitly
        }
    
    def _enhance_literature_response(self, content: str) -> str:
        """Aprimorar resposta de busca literatura."""
        # Add search strategy reminder if not present
        if "search" in content.lower() and "database" not in content.lower():
            content += "\n\n🔍 **Search Tip**: Remember to use multiple databases and refine your search terms based on initial results."
        
        return content
    
    def _enhance_synthesis_response(self, content: str) -> str:
        """Aprimorar resposta de síntese de pesquisa."""
        # Add methodological note if discussing synthesis
        if "synthesis" in content.lower() or "meta-analysis" in content.lower():
            if "prisma" not in content.lower():
                content += "\n\n📊 **Methodology Note**: Consider following PRISMA guidelines for systematic reviews and ensure appropriate quality assessment of included studies."
        
        return content
    
    def _enhance_writing_response(self, content: str) -> str:
        """Aprimorar resposta de escrita acadêmica."""
        # Add citation reminder if discussing writing
        if any(keyword in content.lower() for keyword in ["writing", "paper", "manuscript"]):
            if "citation" not in content.lower() and "reference" not in content.lower():
                content += "\n\n📚 **Citation Reminder**: Ensure proper citation of all sources and follow your target journal's style guidelines."
        
        return content
    
    def _enhance_interdisciplinary_response(self, content: str) -> str:
        """Aprimorar resposta interdisciplinar."""
        # Add integration note
        if "disciplin" in content.lower():
            if "integration" not in content.lower():
                content += "\n\n🔗 **Integration Note**: Consider how different disciplinary perspectives can be meaningfully integrated rather than simply juxtaposed."
        
        return content
    
    async def _test_connection(self):
        """Testar conexão com API."""
        try:
            model = genai.GenerativeModel("gemini-pro")
            response = await asyncio.to_thread(
                model.generate_content,
                "Hello",
                generation_config=genai.types.GenerationConfig(max_output_tokens=10)
            )
            logger.info("Gemini API connection test successful")
            return True
        except Exception as e:
            logger.error(f"Gemini API connection test failed: {e}")
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
                           model: AIModel = AIModel.GEMINI_PRO) -> float:
        """Estimar custo de request."""
        # Rough token estimation
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        estimated_tokens = total_chars // 4  # Rough approximation
        
        # Cost per 1K tokens (as of 2024) - Gemini is very cost-effective
        cost_per_1k = {
            AIModel.GEMINI_PRO: 0.00075,
            AIModel.GEMINI_PRO_VISION: 0.00075
        }
        
        rate = cost_per_1k.get(model, 0.00075)
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


__all__ = ["GeminiClient"]