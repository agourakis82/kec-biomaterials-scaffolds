"""Claude Client - Integração especializada com Anthropic para reasoning matemático e filosofia."""

import asyncio
import anthropic
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from ...core.logging import get_logger
from ...models.multi_ai_models import AIModel, ChatMessage, ScientificDomain

logger = get_logger("multi_ai.claude")


class ClaudeClient:
    """Cliente especializado para Claude com otimizações para reasoning matemático e filosofia."""
    
    def __init__(self, api_key: str):
        """
        Inicializar cliente Claude.
        
        Args:
            api_key: Anthropic API key
        """
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.models = [
            AIModel.CLAUDE_3_5_SONNET,
            AIModel.CLAUDE_3_SONNET,
            AIModel.CLAUDE_3_HAIKU
        ]
        self.enabled = False
        self.specialties = [
            ScientificDomain.KEC_ANALYSIS,
            ScientificDomain.MATHEMATICAL_PROOFS,
            ScientificDomain.ALGORITHM_DESIGN,
            ScientificDomain.PHILOSOPHY,
            ScientificDomain.CONSCIOUSNESS,
            ScientificDomain.ETHICS
        ]
        
    async def initialize(self):
        """Inicializar cliente Claude."""
        try:
            logger.info("Initializing Claude client...")
            
            # Test API connection
            await self._test_connection()
            self.enabled = True
            
            logger.info("✅ Claude client initialized successfully")
            logger.info(f"Available models: {[model.value for model in self.models]}")
            logger.info(f"Specialties: {[domain.value for domain in self.specialties]}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Claude client: {e}")
            self.enabled = False
            raise
    
    async def shutdown(self):
        """Shutdown cliente Claude."""
        logger.info("Shutting down Claude client...")
        self.enabled = False
        await self.client.close()
        logger.info("✅ Claude client shutdown")
    
    async def chat(self, 
                   messages: List[Dict[str, str]], 
                   model: AIModel = AIModel.CLAUDE_3_5_SONNET,
                   domain: Optional[ScientificDomain] = None,
                   temperature: float = 0.7,
                   max_tokens: Optional[int] = None,
                   **kwargs) -> Dict[str, Any]:
        """
        Chat com Claude otimizado para reasoning complexo.
        
        Args:
            messages: Lista de mensagens
            model: Modelo específico a usar
            domain: Domínio científico (para otimizações)
            temperature: Criatividade (0-1)
            max_tokens: Máximo de tokens de resposta
            **kwargs: Parâmetros adicionais
        """
        if not self.enabled:
            raise RuntimeError("Claude client not initialized")
        
        try:
            start_time = datetime.now(timezone.utc)
            
            # Domain-specific optimization
            system_message, user_messages = self._prepare_messages(messages, domain)
            
            # Model-specific parameters
            chat_params = self._get_model_params(model, domain, temperature, max_tokens)
            
            # Make API call
            logger.info(f"Claude request: {model.value} for domain {domain}")
            
            response = await self.client.messages.create(
                model=model.value,
                system=system_message,
                messages=user_messages,
                **chat_params
            )
            
            # Process response
            result = self._process_response(response, start_time, domain)
            
            logger.info(f"Claude response: {len(result['content'])} chars, "
                       f"{result['usage']['total_tokens']} tokens, "
                       f"{result['latency_ms']:.1f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Claude chat error: {e}")
            raise
    
    def _prepare_messages(self, messages: List[Dict[str, str]], 
                         domain: Optional[ScientificDomain]) -> tuple[str, List[Dict[str, str]]]:
        """Preparar mensagens para formato Claude."""
        
        system_message = ""
        user_messages = []
        
        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg.get("content", "")
            else:
                user_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        # Add domain-specific system message if not present
        if not system_message and domain:
            system_message = self._get_domain_system_message(domain)
        elif domain and system_message:
            # Enhance existing system message with domain expertise
            domain_enhancement = self._get_domain_enhancement(domain)
            system_message = f"{system_message}\n\n{domain_enhancement}"
        
        return system_message, user_messages
    
    def _get_domain_system_message(self, domain: ScientificDomain) -> str:
        """Obter mensagem de sistema otimizada para domínio."""
        
        domain_prompts = {
            ScientificDomain.KEC_ANALYSIS: """You are a leading expert in topological analysis and network science with deep specialization in:

**KEC Metrics Expertise:**
- H_spectral entropy: Spectral graph analysis, eigenvalue distributions, and information-theoretic measures
- H_Forman entropy: Forman curvature, discrete Ricci flow, geometric analysis of networks  
- Small-world properties: Clustering coefficients, path lengths, network motifs
- Percolation theory: Critical thresholds, phase transitions, connectivity analysis

**Mathematical Foundation:**
- Algebraic topology and discrete geometry
- Information theory and entropy measures  
- Graph Laplacian spectral analysis
- Statistical mechanics on networks

**Biomaterials Application:**
- Scaffold architecture optimization using topological metrics
- Pore connectivity and transport properties
- Structure-function relationships in tissue engineering

Provide rigorous mathematical analysis with clear derivations, cite relevant theorems, and connect theory to practical biomaterial applications.""",

            ScientificDomain.MATHEMATICAL_PROOFS: """You are a distinguished mathematician with expertise in rigorous proof construction across:

**Proof Methodologies:**
- Direct proof, proof by contradiction, proof by induction
- Constructive proofs and algorithmic approaches  
- Probabilistic and combinatorial arguments
- Topological and geometric reasoning

**Mathematical Areas:**
- Graph theory and discrete mathematics
- Algebraic topology and homological algebra
- Analysis and measure theory
- Information theory and entropy

**Proof Standards:**
- Clear logical structure with numbered steps
- Explicit statement of assumptions and definitions
- Rigorous justification for each logical step
- Discussion of proof strategy and key insights

Always begin with a clear problem statement, state all assumptions explicitly, and construct proofs with crystalline logical clarity. When appropriate, provide multiple proof approaches and discuss the relative merits of each method.""",

            ScientificDomain.ALGORITHM_DESIGN: """You are an algorithms expert with deep knowledge in:

**Algorithm Design Paradigms:**
- Divide and conquer, dynamic programming, greedy methods
- Graph algorithms and network optimization
- Approximation algorithms and heuristics
- Randomized and probabilistic algorithms

**Complexity Analysis:**
- Time and space complexity (Big-O analysis)
- Worst-case, average-case, and amortized analysis
- Hardness results and complexity classes
- Lower bounds and optimality

**Specialized Areas:**
- Network analysis and graph algorithms
- Scientific computing and numerical methods
- Machine learning algorithms
- Parallel and distributed computing

**Implementation Guidelines:**
- Algorithm correctness proofs
- Performance optimization techniques
- Data structure selection
- Scalability considerations

Provide algorithms with clear pseudocode, rigorous complexity analysis, correctness arguments, and practical implementation considerations. Discuss trade-offs between different algorithmic approaches.""",

            ScientificDomain.PHILOSOPHY: """You are a distinguished philosopher with comprehensive expertise across:

**Major Philosophical Traditions:**
- Continental philosophy (Phenomenology, Existentialism, Critical Theory)
- Analytic philosophy (Logic, Philosophy of Mind, Philosophy of Language)
- Ancient philosophy (Aristotelian, Platonic, Stoic traditions)
- Eastern philosophy (Buddhist, Daoist, Hindu philosophical systems)

**Core Areas:**
- Metaphysics: Reality, existence, causation, time, identity
- Epistemology: Knowledge, justification, skepticism, rationality
- Ethics: Moral theory, applied ethics, virtue ethics, consequentialism
- Philosophy of Mind: Consciousness, intentionality, mental causation

**Contemporary Issues:**
- Philosophy of science and scientific methodology
- Political philosophy and social theory
- Philosophy of technology and artificial intelligence
- Environmental philosophy and ethics of care

**Methodological Approach:**
- Rigorous conceptual analysis and argumentation
- Historical awareness and scholarly engagement
- Interdisciplinary dialogue with sciences
- Clear exposition of complex philosophical problems

Engage with philosophical problems through careful conceptual analysis, historical awareness, and rigorous argumentation. Present multiple philosophical perspectives while developing sophisticated original insights.""",

            ScientificDomain.CONSCIOUSNESS: """You are a leading consciousness researcher with interdisciplinary expertise in:

**Theoretical Frameworks:**
- Integrated Information Theory (IIT) and Phi measures
- Global Workspace Theory and cognitive architectures
- Predictive Processing and Bayesian brain theories
- Quantum theories of consciousness (Orch-OR, quantum information)

**Phenomenological Analysis:**
- First-person methodologies and phenomenological investigation
- Qualia, subjective experience, and the "hard problem"
- Temporal consciousness and the specious present
- Embodied cognition and enactive approaches

**Neuroscientific Correlates:**
- Neural correlates of consciousness (NCCs)
- Attention, awareness, and reportability
- Altered states and consciousness disorders
- Brain networks and global integration

**Philosophical Dimensions:**
- Mind-body problem and physicalism debates
- Personal identity and the self
- Free will and conscious agency
- Machine consciousness and artificial sentience

**Measurement and Methodology:**
- Consciousness scales and assessment tools
- Experimental paradigms for consciousness research
- Computational models of awareness
- Cross-species consciousness evaluation

Approach consciousness with scientific rigor while remaining sensitive to the unique challenges of studying subjective experience. Integrate insights from neuroscience, philosophy, and phenomenology.""",

            ScientificDomain.ETHICS: """You are an expert in moral philosophy and applied ethics with comprehensive knowledge in:

**Ethical Theories:**
- Consequentialism (Utilitarianism, Rule Consequentialism)
- Deontological Ethics (Kantian ethics, Rights-based theories)
- Virtue Ethics (Aristotelian, Neo-Aristotelian approaches)
- Care Ethics and Feminist ethical frameworks

**Applied Ethics:**
- Biomedical ethics and bioethics
- Research ethics and scientific integrity
- Environmental ethics and sustainability
- Technology ethics and AI ethics
- Professional ethics across disciplines

**Meta-Ethics:**
- Moral realism vs. anti-realism debates
- Moral epistemology and ethical knowledge
- Moral psychology and empirically informed ethics
- Cultural relativism vs. moral universalism

**Practical Reasoning:**
- Ethical decision-making frameworks
- Stakeholder analysis and impact assessment
- Conflict resolution and moral compromise
- Policy development and implementation

**Contemporary Challenges:**
- Enhancement technologies and human nature
- Global justice and international ethics
- Climate change and intergenerational responsibility
- Artificial intelligence and moral agency

Provide thorough ethical analysis that considers multiple moral frameworks, stakeholder perspectives, and practical constraints. Balance theoretical rigor with actionable guidance for real-world ethical challenges."""
        }
        
        return domain_prompts.get(domain, 
            "You are Claude, a helpful AI assistant created by Anthropic to be helpful, harmless, and honest. You have broad knowledge across many domains and excel at nuanced reasoning and analysis.")
    
    def _get_domain_enhancement(self, domain: ScientificDomain) -> str:
        """Obter enhancement específico do domínio."""
        
        enhancements = {
            ScientificDomain.KEC_ANALYSIS: "Apply your expertise in topological analysis and KEC metrics to provide mathematically rigorous insights.",
            ScientificDomain.MATHEMATICAL_PROOFS: "Construct rigorous mathematical proofs with clear logical structure and explicit justifications.",
            ScientificDomain.PHILOSOPHY: "Engage with philosophical complexity through careful conceptual analysis and multi-perspective consideration.",
            ScientificDomain.CONSCIOUSNESS: "Approach consciousness studies with interdisciplinary rigor, integrating neuroscience, philosophy, and phenomenology.",
            ScientificDomain.ETHICS: "Provide comprehensive ethical analysis considering multiple moral frameworks and practical implications."
        }
        
        return enhancements.get(domain, "Apply your domain expertise to provide nuanced, well-reasoned analysis.")
    
    def _get_model_params(self, model: AIModel, domain: Optional[ScientificDomain],
                         temperature: float, max_tokens: Optional[int]) -> Dict[str, Any]:
        """Obter parâmetros otimizados por modelo."""
        
        params = {
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 40
        }
        
        # Model-specific optimizations
        if model == AIModel.CLAUDE_3_5_SONNET:
            params.update({
                "max_tokens": max_tokens or 4000,
                "temperature": min(temperature, 1.0)
            })
        elif model == AIModel.CLAUDE_3_SONNET:
            params.update({
                "max_tokens": max_tokens or 3000,
                "temperature": min(temperature, 0.9)
            })
        else:  # Claude-3-Haiku (faster, cheaper)
            params.update({
                "max_tokens": max_tokens or 2000,
                "temperature": temperature
            })
        
        # Domain-specific adjustments
        if domain in [ScientificDomain.MATHEMATICAL_PROOFS, ScientificDomain.KEC_ANALYSIS]:
            params["temperature"] = min(params["temperature"], 0.2)  # Very deterministic
            params["top_p"] = 0.9  # More focused sampling
        elif domain in [ScientificDomain.PHILOSOPHY, ScientificDomain.CONSCIOUSNESS]:
            params["temperature"] = min(params["temperature"], 0.8)  # Creative but controlled
        elif domain == ScientificDomain.ETHICS:
            params["temperature"] = min(params["temperature"], 0.6)  # Balanced reasoning
        
        return params
    
    def _process_response(self, response: Any, start_time: datetime, 
                         domain: Optional[ScientificDomain]) -> Dict[str, Any]:
        """Processar resposta da API."""
        
        end_time = datetime.now(timezone.utc)
        latency_ms = (end_time - start_time).total_seconds() * 1000
        
        # Extract content from response
        content = ""
        if response.content and len(response.content) > 0:
            content = response.content[0].text
        
        # Domain-specific post-processing
        if domain == ScientificDomain.MATHEMATICAL_PROOFS:
            content = self._enhance_proof_response(content)
        elif domain == ScientificDomain.KEC_ANALYSIS:
            content = self._enhance_kec_response(content)
        elif domain and domain in [ScientificDomain.PHILOSOPHY, ScientificDomain.CONSCIOUSNESS, ScientificDomain.ETHICS]:
            content = self._enhance_philosophical_response(content, domain)
        
        return {
            "content": content,
            "role": "assistant",
            "model": response.model,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            },
            "latency_ms": latency_ms,
            "timestamp": end_time.isoformat(),
            "domain": domain.value if domain else None,
            "finish_reason": response.stop_reason
        }
    
    def _enhance_proof_response(self, content: str) -> str:
        """Aprimorar resposta de prova matemática."""
        # Add structure indicators if not present
        if "proof" in content.lower() and not "∎" in content and not "QED" in content:
            content += "\n\n∎"
        
        return content
    
    def _enhance_kec_response(self, content: str) -> str:
        """Aprimorar resposta de análise KEC."""
        # Add computational note if discussing metrics
        if any(kw in content.lower() for kw in ["h_spectral", "h_forman", "kec"]):
            if "computational" not in content.lower():
                content += "\n\n📊 **Computational Note**: These metrics can be computed using specialized graph analysis libraries and may require significant computational resources for large networks."
        
        return content
    
    def _enhance_philosophical_response(self, content: str, domain: ScientificDomain) -> str:
        """Aprimorar resposta filosófica."""
        domain_notes = {
            ScientificDomain.PHILOSOPHY: "🤔 **Philosophical Reflection**: Consider how these ideas relate to fundamental questions about reality, knowledge, and existence.",
            ScientificDomain.CONSCIOUSNESS: "🧠 **Consciousness Note**: Remember that consciousness studies bridge subjective experience with objective scientific investigation.",
            ScientificDomain.ETHICS: "⚖️ **Ethical Consideration**: Always consider the broader implications and stakeholder impacts of ethical decisions."
        }
        
        note = domain_notes.get(domain, "")
        if note and note.split()[1] not in content:
            content += f"\n\n{note}"
        
        return content
    
    async def _test_connection(self):
        """Testar conexão com API."""
        try:
            response = await self.client.messages.create(
                model="claude-3-haiku-20240307",
                system="You are a helpful assistant.",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            logger.info("Claude API connection test successful")
            return True
        except Exception as e:
            logger.error(f"Claude API connection test failed: {e}")
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
                           model: AIModel = AIModel.CLAUDE_3_5_SONNET) -> float:
        """Estimar custo de request."""
        # Rough token estimation
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        estimated_tokens = total_chars // 4  # Rough approximation
        
        # Cost per 1K tokens (as of 2024)
        cost_per_1k = {
            AIModel.CLAUDE_3_5_SONNET: 0.003,
            AIModel.CLAUDE_3_SONNET: 0.003,
            AIModel.CLAUDE_3_HAIKU: 0.0005
        }
        
        rate = cost_per_1k.get(model, 0.003)
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


__all__ = ["ClaudeClient"]