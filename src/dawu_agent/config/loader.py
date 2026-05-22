"""Enterprise configuration loader with 7-level priority and source tracking."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelProfile(BaseSettings):
    """Single model profile configuration (unified format)."""

    model_config = SettingsConfigDict(extra="ignore")

    api_key: str = ""
    base_url: str = ""
    name: str = "claude-3-5-sonnet-20241022"
    api_type: str = "anthropic"  # anthropic | openai | azure | local
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout_seconds: int = 120
    max_retries: int = 3
    retry_backoff_base: float = 2.0
    prompt_caching: bool = True
    cache_minimum_tokens: int = 1024

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v

    @property
    def provider(self) -> str:
        """Backward-compatible provider name."""
        return self.api_type


class LLMConfig(BaseSettings):
    """LLM configuration with multi-model profile support."""

    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")

    # Default model (primary) - will be populated from env vars in __init__
    default: ModelProfile = Field(default_factory=ModelProfile)

    # Fallback model
    fallback: ModelProfile | None = None

    # Named profiles for scenario-based routing
    profiles: dict[str, ModelProfile] = Field(default_factory=dict)

    # Legacy fields for backward compatibility
    provider: str = "anthropic"
    model: str = "claude-3-5-sonnet-20241022"
    fallback_provider: str | None = None
    fallback_model: str | None = None
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout_seconds: int = 120
    max_retries: int = 3
    retry_backoff_base: float = 2.0
    prompt_caching: bool = True
    cache_minimum_tokens: int = 1024

    def model_post_init(self, __context: Any) -> None:
        """Load unified env vars into default profile if not already set."""
        # Load default profile from unified env vars
        if not self.default.api_key:
            self.default = ModelProfile(
                api_key=os.getenv("MODEL_API_KEY", ""),
                base_url=os.getenv("MODEL_BASE_URL", ""),
                name=os.getenv("MODEL_NAME", "claude-3-5-sonnet-20241022"),
                api_type=os.getenv("MODEL_API_TYPE", "anthropic"),
                temperature=float(os.getenv("MODEL_TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("MODEL_MAX_TOKENS", "4096")),
            )
        # Load fallback profile
        if not self.fallback and os.getenv("FALLBACK_MODEL_API_KEY"):
            self.fallback = ModelProfile(
                api_key=os.getenv("FALLBACK_MODEL_API_KEY", ""),
                base_url=os.getenv("FALLBACK_MODEL_BASE_URL", ""),
                name=os.getenv("FALLBACK_MODEL_NAME", ""),
                api_type=os.getenv("FALLBACK_MODEL_API_TYPE", "openai"),
            )
        # Load named profiles from env vars with PROFILE_NAME__FIELD format
        self._load_profile_env_vars()

    def _load_profile_env_vars(self) -> None:
        """Load profiles from environment variables using PROFILE_NAME__FIELD format."""
        for key, value in os.environ.items():
            if "__" in key and key.endswith("_MODEL_API_KEY"):
                # Extract profile name: FAST__MODEL_API_KEY -> fast
                profile_name = key.replace("_MODEL_API_KEY", "").lower().rstrip('_')
                if profile_name in self.profiles:
                    continue
                self.profiles[profile_name] = ModelProfile(
                    api_key=value,
                    base_url=os.getenv(f"{profile_name.upper()}__MODEL_BASE_URL", ""),
                    name=os.getenv(f"{profile_name.upper()}__MODEL_NAME", ""),
                    api_type=os.getenv(f"{profile_name.upper()}__MODEL_API_TYPE", "openai"),
                )

    def get_profile(self, name: str | None = None) -> ModelProfile:
        """Get model profile by name. Returns default if not found."""
        if name is None or name == "default":
            if self.default.api_key:
                return self.default
            # Legacy fallback
            return ModelProfile(
                api_key=os.getenv("ANTHROPIC_API_KEY", ""),
                base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"),
                name=self.model,
                api_type=self.provider,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout_seconds=self.timeout_seconds,
                max_retries=self.max_retries,
                retry_backoff_base=self.retry_backoff_base,
                prompt_caching=self.prompt_caching,
                cache_minimum_tokens=self.cache_minimum_tokens,
            )

        if name in self.profiles:
            return self.profiles[name]

        raise ValueError(f"Unknown model profile: {name}")

    def get_fallback(self) -> ModelProfile | None:
        """Get fallback model profile."""
        if self.fallback:
            return self.fallback
        if self.fallback_provider and self.fallback_model:
            return ModelProfile(
                api_key=os.getenv("OPENAI_API_KEY", ""),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                name=self.fallback_model,
                api_type=self.fallback_provider,
            )
        return None


class AgentConfig(BaseSettings):
    """Agent behavior configuration."""

    model_config = SettingsConfigDict(env_prefix="AGENT_", extra="ignore")

    name: str = "dawu-agent"
    description: str = "Enterprise AI Agent"
    max_turns: int = 50
    max_tokens_per_session: int = 100000
    system_prompt_path: str = "config/agents/default.md"
    streaming: bool = True

    # Scenario-based model routing
    scenario_models: dict[str, str] = Field(default_factory=lambda: {
        "default": "default",
        "fast": "fast",
        "complex": "power",
        "vision": "vision",
        "local": "local",
    })


class SandboxConfig(BaseSettings):
    """Sandbox isolation configuration."""

    model_config = SettingsConfigDict(env_prefix="SANDBOX_", extra="ignore")

    enabled: bool = True
    isolation_level: str = "docker"  # none | path | docker | firecracker
    allowed_paths: list[str] = Field(default_factory=lambda: ["./workspace"])
    denied_patterns: list[str] = Field(default_factory=list)
    docker_image: str = "dawu-sandbox:latest"
    resource_limits: dict[str, str] = Field(default_factory=dict)


class MemoryConfig(BaseSettings):
    """Memory and context management configuration."""

    model_config = SettingsConfigDict(env_prefix="MEMORY_", extra="ignore")

    enabled: bool = True
    vector_backend: str = "chromadb"
    collection_name: str = "dawu_memories"
    embedding_model: str = "text-embedding-3-small"
    similarity_threshold: float = 0.75
    max_results: int = 5
    compression_enabled: bool = True


class LoggingConfig(BaseSettings):
    """Logging and observability configuration."""

    model_config = SettingsConfigDict(env_prefix="LOGGING_", extra="ignore")

    level: str = "INFO"
    format: str = "json"
    opentelemetry_enabled: bool = True
    opentelemetry_endpoint: str = "http://localhost:4317"
    prometheus_enabled: bool = True
    prometheus_port: int = 9090


class MultiAgentConfig(BaseSettings):
    """Multi-agent coordination configuration."""

    model_config = SettingsConfigDict(env_prefix="MULTI_AGENT_", extra="ignore")

    enabled: bool = True
    max_sub_agents: int = 5
    task_timeout: int = 300
    communication_mode: str = "hierarchical"


class Settings(BaseSettings):
    """Root settings with 7-level priority support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment
    env: str = Field(default="development", alias="ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Sub-configs
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    multi_agent: MultiAgentConfig = Field(default_factory=MultiAgentConfig)

    # Feature flags
    enable_multi_agent: bool = Field(default=True, alias="ENABLE_MULTI_AGENT")
    enable_vector_memory: bool = Field(default=True, alias="ENABLE_VECTOR_MEMORY")
    enable_sandbox: bool = Field(default=True, alias="ENABLE_SANDBOX")
    enable_audit_log: bool = Field(default=True, alias="ENABLE_AUDIT_LOG")


class ConfigLoader:
    """Loads configuration from multiple sources with priority hierarchy.

    Priority (highest to lowest):
    1. CLI arguments
    2. Feature flags
    3. Policy rules
    4. Managed config (remote)
    5. Local overrides
    6. Project defaults (YAML files)
    7. User global settings
    """

    def __init__(self, config_dir: str | Path = "config") -> None:
        self.config_dir = Path(config_dir)
        self._settings: Settings | None = None
        self._source_trace: dict[str, str] = {}

    def load(self, env: str | None = None) -> Settings:
        """Load settings from all sources, merging by priority."""
        env = env or os.getenv("ENV", "development")

        # Explicitly load .env file from project root
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path, override=True)

        # Start with base settings (picks up .env automatically)
        settings = Settings()

        # Load environment-specific YAML
        yaml_path = self.config_dir / f"{env}.yaml"
        if yaml_path.exists():
            yaml_data = self._load_yaml(yaml_path)
            settings = self._merge_yaml(settings, yaml_data)
            self._source_trace["yaml"] = str(yaml_path)

        # Load local overrides if present
        local_yaml = self.config_dir / "local.yaml"
        if local_yaml.exists():
            local_data = self._load_yaml(local_yaml)
            settings = self._merge_yaml(settings, local_data)
            self._source_trace["local"] = str(local_yaml)

        self._settings = settings
        return settings

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        """Load and parse YAML file with environment variable interpolation."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return self._interpolate_env_vars(data)

    def _interpolate_env_vars(self, obj: Any) -> Any:
        """Recursively interpolate ${VAR} and ${VAR:-default} patterns."""
        import re

        if isinstance(obj, str):
            pattern = re.compile(r'\$\{([^}]+)\}')

            def replacer(match):
                var_expr = match.group(1)
                if ':-' in var_expr:
                    var_name, default = var_expr.split(':-', 1)
                    return os.getenv(var_name, default)
                return os.getenv(var_expr, match.group(0))

            return pattern.sub(replacer, obj)
        elif isinstance(obj, dict):
            return {k: self._interpolate_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._interpolate_env_vars(item) for item in obj]
        return obj

    def _merge_yaml(self, settings: Settings, data: dict[str, Any]) -> Settings:
        """Merge YAML data into settings (shallow merge for nested configs)."""
        for key, value in data.items():
            if hasattr(settings, key) and isinstance(value, dict):
                current = getattr(settings, key)
                # Handle nested Pydantic models by reconstructing them
                if isinstance(current, BaseSettings):
                    current_dict = current.model_dump()
                    current_dict.update(value)
                    try:
                        new_current = current.__class__(**current_dict)
                        setattr(settings, key, new_current)
                        self._source_trace[key] = "yaml"
                    except Exception:
                        # Fallback: direct attribute assignment
                        for sub_key, sub_value in value.items():
                            if hasattr(current, sub_key):
                                setattr(current, sub_key, sub_value)
                                self._source_trace[f"{key}.{sub_key}"] = "yaml"
                else:
                    for sub_key, sub_value in value.items():
                        if hasattr(current, sub_key):
                            setattr(current, sub_key, sub_value)
                            self._source_trace[f"{key}.{sub_key}"] = "yaml"
            elif hasattr(settings, key):
                setattr(settings, key, value)
                self._source_trace[key] = "yaml"
        return settings

    def get_source_trace(self) -> dict[str, str]:
        """Return trace of where each config value originated."""
        return self._source_trace.copy()

    @property
    def settings(self) -> Settings:
        """Get loaded settings (raises if not loaded)."""
        if self._settings is None:
            raise RuntimeError("Settings not loaded. Call load() first.")
        return self._settings
