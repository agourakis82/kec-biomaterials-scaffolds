"""Configuration settings for DARWIN META-RESEARCH BRAIN unified backend.

Consolidated configuration system supporting FastAPI + MCP Server hybrid architecture
with domain-specific configurations for all research areas.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
import yaml


class MultiAIConfig(BaseSettings):
    """Multi-AI provider configurations."""
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=None)
    openai_model: str = Field(default="gpt-4-turbo")
    openai_max_tokens: int = Field(default=4096)
    
    # Anthropic Configuration
    anthropic_api_key: Optional[str] = Field(default=None)
    anthropic_model: str = Field(default="claude-3-sonnet-20240229")
    anthropic_max_tokens: int = Field(default=4096)
    
    # Google/Vertex AI Configuration
    gcp_project_id: Optional[str] = Field(default=None)
    gcp_region: str = Field(default="us-central1")
    vertex_ai_model: str = Field(default="gemini-1.5-pro")
    
    # Cohere Configuration
    cohere_api_key: Optional[str] = Field(default=None)
    cohere_model: str = Field(default="command-r-plus")
    
    # Mistral Configuration
    mistral_api_key: Optional[str] = Field(default=None)
    mistral_model: str = Field(default="mistral-large-latest")
    
    # Default provider for multi-AI routing
    default_provider: str = Field(default="openai")
    
    class Config:
        env_prefix = "MULTI_AI_"
        case_sensitive = False


class DatabaseConfig(BaseSettings):
    """Database and storage configurations."""
    
    # PostgreSQL
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="darwin_brain")
    postgres_user: str = Field(default="postgres")
    postgres_password: str = Field(default="")
    
    # Redis Cache
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_password: Optional[str] = Field(default=None)
    
    # BigQuery (GCP)
    bq_dataset: Optional[str] = Field(default=None)
    bq_table: Optional[str] = Field(default=None)
    bq_location: str = Field(default="US")
    
    # ChromaDB Vector Store
    chroma_host: str = Field(default="localhost")
    chroma_port: int = Field(default=8000)
    chroma_collection: str = Field(default="darwin_knowledge")
    
    # Vector Backend Selection
    vector_backend: str = Field(default="chroma")  # chroma, gcp_bq, etc.
    
    class Config:
        env_prefix = "DB_"
        case_sensitive = False


class DomainConfig(BaseSettings):
    """Domain-specific research configurations."""
    
    # Biomaterials Domain
    biomaterials_enabled: bool = Field(default=True)
    kec_alpha_threshold: float = Field(default=0.05)
    percolation_threshold: float = Field(default=0.593)
    
    # Neuroscience Domain
    neuroscience_enabled: bool = Field(default=True)
    eeg_sample_rate: int = Field(default=1000)
    mne_verbose: str = Field(default="WARNING")
    
    # Philosophy Domain
    philosophy_enabled: bool = Field(default=True)
    logic_engine: str = Field(default="sympy")  # sympy, prolog, pyke
    reasoning_depth: int = Field(default=5)
    
    # Quantum Mechanics Domain
    quantum_enabled: bool = Field(default=True)
    quantum_backend: str = Field(default="qiskit")  # qiskit, cirq, pennylane
    quantum_simulator: str = Field(default="aer_simulator")
    max_qubits: int = Field(default=32)
    
    # Psychiatry Domain
    psychiatry_enabled: bool = Field(default=True)
    clinical_model_path: Optional[str] = Field(default=None)
    psychopy_backend: str = Field(default="pyglet")
    
    class Config:
        env_prefix = "DOMAIN_"
        case_sensitive = False


class MCPConfig(BaseSettings):
    """MCP Server configuration."""
    
    # MCP Server Settings
    mcp_enabled: bool = Field(default=True)
    mcp_name: str = Field(default="darwin-meta-research-brain")
    mcp_version: str = Field(default="1.0.0")
    mcp_description: str = Field(default="DARWIN Meta-Research Brain - Unified Scientific Discovery Platform")
    
    # MCP Tool Configuration
    mcp_tools_enabled: List[str] = Field(default=[
        "kec_metrics", "rag_plus", "tree_search", "discovery",
        "score_contracts", "multi_ai_chat", "philosophy_reasoning",
        "quantum_simulation", "psychiatry_assessment"
    ])
    
    # MCP Resource Configuration
    mcp_resources_enabled: bool = Field(default=True)
    mcp_resource_templates: bool = Field(default=True)
    
    class Config:
        env_prefix = "MCP_"
        case_sensitive = False


class SecurityConfig(BaseSettings):
    """Security and authentication configuration."""
    
    # API Authentication
    api_key_required: bool = Field(default=False)
    api_key: Optional[str] = Field(default=None)
    jwt_secret_key: str = Field(default="your-secret-key-change-in-production")
    jwt_algorithm: str = Field(default="HS256")
    jwt_access_token_expire_minutes: int = Field(default=1440)
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_requests_per_minute: int = Field(default=60)
    rate_limit_burst_size: int = Field(default=10)
    
    class Config:
        env_prefix = "SECURITY_"
        case_sensitive = False


class MonitoringConfig(BaseSettings):
    """Monitoring and observability configuration."""
    
    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")  # json, structured, simple
    log_file: Optional[str] = Field(default=None)
    
    # Metrics
    metrics_enabled: bool = Field(default=True)
    prometheus_port: int = Field(default=9090)
    
    # Tracing
    tracing_enabled: bool = Field(default=False)
    jaeger_endpoint: Optional[str] = Field(default=None)
    
    # Sentry Error Tracking
    sentry_dsn: Optional[str] = Field(default=None)
    sentry_environment: str = Field(default="development")
    
    class Config:
        env_prefix = "MONITORING_"
        case_sensitive = False


class DiscoveryConfig(BaseSettings):
    """Scientific discovery and RSS feed configuration."""
    
    # Discovery Settings
    discovery_enabled: bool = Field(default=True)
    discovery_interval: int = Field(default=3600)  # seconds
    novelty_threshold: float = Field(default=0.7)
    max_articles_per_run: int = Field(default=100)
    
    # RSS Feeds
    rss_feeds: List[str] = Field(default=[
        "https://arxiv.org/rss/cond-mat.mtrl-sci",  # Materials Science
        "https://arxiv.org/rss/q-bio",              # Quantitative Biology
        "https://arxiv.org/rss/physics.bio-ph",     # Biological Physics
        "https://arxiv.org/rss/quant-ph",           # Quantum Physics
        "https://arxiv.org/rss/cs.AI",              # Artificial Intelligence
        "https://www.nature.com/nature.rss",        # Nature
        "https://www.science.org/rss/current.xml"   # Science Magazine
    ])
    
    # Content Processing
    content_extraction_enabled: bool = Field(default=True)
    full_text_extraction: bool = Field(default=True)
    
    class Config:
        env_prefix = "DISCOVERY_"
        case_sensitive = False


class HybridAIConfig(BaseModel):
    """Hybrid AI routing and toggles loaded optionally from YAML."""
    local_first: bool = Field(default=True)
    allow_web_bursts: bool = Field(default=True)
    request_timeout: int = Field(default=30)
    retries: int = Field(default=2)
    routing_weights: Dict[str, int] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

class Settings(BaseSettings):
    """Main configuration class consolidating all settings."""
    
    # Application Settings
    app_name: str = Field(default="DARWIN Meta-Research Brain")
    app_version: str = Field(default="1.0.0")
    app_description: str = Field(default="Unified Scientific Discovery Platform with Multi-Domain Research Capabilities")
    
    # Environment
    env: str = Field(default="development")
    debug: bool = Field(default=False)
    
    # Server Settings
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    reload: bool = Field(default=False)
    workers: int = Field(default=1)
    
    # CORS
    cors_enabled: bool = Field(default=True)
    cors_origins: List[str] = Field(default=["*"])
    cors_methods: List[str] = Field(default=["*"])
    cors_headers: List[str] = Field(default=["*"])
    
    # Static Files
    static_files_enabled: bool = Field(default=True)
    static_path: str = Field(default="static")
    
    # Paths
    base_path: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    data_path: Path = Field(default_factory=lambda: Path("data"))
    logs_path: Path = Field(default_factory=lambda: Path("logs"))
    
    # Sub-configurations
    multi_ai: MultiAIConfig = MultiAIConfig()
    database: DatabaseConfig = DatabaseConfig()
    domains: DomainConfig = DomainConfig()
    mcp: MCPConfig = MCPConfig()
    security: SecurityConfig = SecurityConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    discovery: DiscoveryConfig = DiscoveryConfig()
    hybrid: HybridAIConfig = HybridAIConfig()
    
    # Feature Flags
    features_enabled: Dict[str, bool] = Field(default_factory=lambda: {
        "multi_ai_hub": True,
        "knowledge_graph": True,
        "cross_domain_discovery": True,
        "mcp_server": True,
        "real_time_processing": True,
        "advanced_analytics": True,
        "experimental_features": False
    })
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @validator('cors_methods', 'cors_headers', pre=True)
    def parse_cors_lists(cls, v):
        """Parse CORS methods/headers from string or list."""
        if isinstance(v, str):
            s = v.strip()
            if s == "*":
                return ["*"]
            return [item.strip() for item in s.split(",") if item.strip()]
        return v
    
    # Removing validator for rss_feeds as it's in DiscoveryConfig, not Settings
    
    @property
    def database_url(self) -> str:
        """Generate PostgreSQL database URL."""
        return f"postgresql://{self.database.postgres_user}:{self.database.postgres_password}@{self.database.postgres_host}:{self.database.postgres_port}/{self.database.postgres_db}"
    
    @property
    def redis_url(self) -> str:
        """Generate Redis connection URL."""
        auth = f":{self.database.redis_password}@" if self.database.redis_password else ""
        return f"redis://{auth}{self.database.redis_host}:{self.database.redis_port}/{self.database.redis_db}"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.env.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.env.lower() == "development"
    
    def get_domain_config(self, domain: str) -> Dict[str, Any]:
        """Get configuration for specific research domain."""
        domain_configs = {
            "biomaterials": {
                "enabled": self.domains.biomaterials_enabled,
                "kec_threshold": self.domains.kec_alpha_threshold,
                "percolation_threshold": self.domains.percolation_threshold
            },
            "neuroscience": {
                "enabled": self.domains.neuroscience_enabled,
                "eeg_sample_rate": self.domains.eeg_sample_rate,
                "mne_verbose": self.domains.mne_verbose
            },
            "philosophy": {
                "enabled": self.domains.philosophy_enabled,
                "logic_engine": self.domains.logic_engine,
                "reasoning_depth": self.domains.reasoning_depth
            },
            "quantum_mechanics": {
                "enabled": self.domains.quantum_enabled,
                "backend": self.domains.quantum_backend,
                "simulator": self.domains.quantum_simulator,
                "max_qubits": self.domains.max_qubits
            },
            "psychiatry": {
                "enabled": self.domains.psychiatry_enabled,
                "model_path": self.domains.clinical_model_path,
                "psychopy_backend": self.domains.psychopy_backend
            }
        }
        return domain_configs.get(domain, {})
    
    def get_ai_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for specific AI provider."""
        provider_configs = {
            "openai": {
                "api_key": self.multi_ai.openai_api_key,
                "model": self.multi_ai.openai_model,
                "max_tokens": self.multi_ai.openai_max_tokens
            },
            "anthropic": {
                "api_key": self.multi_ai.anthropic_api_key,
                "model": self.multi_ai.anthropic_model,
                "max_tokens": self.multi_ai.anthropic_max_tokens
            },
            "vertex_ai": {
                "project_id": self.multi_ai.gcp_project_id,
                "region": self.multi_ai.gcp_region,
                "model": self.multi_ai.vertex_ai_model
            },
            "cohere": {
                "api_key": self.multi_ai.cohere_api_key,
                "model": self.multi_ai.cohere_model
            },
            "mistral": {
                "api_key": self.multi_ai.mistral_api_key,
                "model": self.multi_ai.mistral_model
            }
        }
        return provider_configs.get(provider, {})
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        use_enum_values = True
# Global settings instance
settings = Settings()

# Optional YAML overrides for hybrid AI toggles and features
_cfg_path = os.getenv("CONFIG_FILE")
if _cfg_path and os.path.exists(_cfg_path):
    try:
        with open(_cfg_path, "r", encoding="utf-8") as _f:
            _cfg = yaml.safe_load(_f) or {}
        _hy = _cfg.get("hybrid_ai") or {}
        if isinstance(_hy, dict):
            if "local_first" in _hy:
                settings.hybrid.local_first = bool(_hy["local_first"])
            if "allow_web_bursts" in _hy:
                settings.hybrid.allow_web_bursts = bool(_hy["allow_web_bursts"])
            if "default_provider" in _hy:
                settings.multi_ai.default_provider = str(_hy["default_provider"])
            _to = _hy.get("timeouts") or {}
            if "request_timeout" in _to:
                settings.hybrid.request_timeout = int(_to["request_timeout"])
            _re = _hy.get("retries") or {}
            if "max_attempts" in _re:
                settings.hybrid.retries = int(_re["max_attempts"])
            _rw = None
            if isinstance(_hy.get("routing"), dict):
                _rw = _hy["routing"].get("weights")
            if isinstance(_rw, dict):
                settings.hybrid.routing_weights = {str(k): int(v) for k, v in _rw.items()}
        _features = _cfg.get("features") or {}
        if isinstance(_features, dict):
            settings.features_enabled.update({str(k): bool(v) for k, v in _features.items()})
    except Exception:
        # ignore YAML errors to avoid breaking startup
        pass


def get_settings() -> Settings:
    """Get application settings instance."""
    return settings


# Export commonly used configurations
__all__ = [
    "Settings",
    "HybridAIConfig",
    "MultiAIConfig",
    "DatabaseConfig",
    "DomainConfig",
    "MCPConfig",
    "SecurityConfig",
    "MonitoringConfig",
    "DiscoveryConfig",
    "settings",
    "get_settings"
]