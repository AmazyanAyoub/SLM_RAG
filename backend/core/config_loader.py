# YAML/Env loader
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class LLMConfig(BaseModel):
    provider: str
    model_name: str
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 1024
    base_url: Optional[str] = None
    api_key: Optional[str] = None

class RetrievalConfig(BaseModel):
    embedder_model: str = Field(alias="embedder_name")
    vector_store_host: str
    vector_store_port: int
    vector_store_collection: str
    vector_store_api_key: Optional[str] = None
    
    # Phase 1.5 & Phase 4 flags
    enable_late_interaction: bool = False
    enable_graph: bool = False

class AppConfig(BaseModel):
    project_name: str
    fast_llm: LLMConfig
    smart_llm: LLMConfig
    retrieval: RetrievalConfig
    
    @classmethod
    def load(cls, config_path: str = "configs/base.yaml") -> "AppConfig":
        """
        Loads the YAML config and injects environment variables.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"❌ Config file not found at: {path.absolute()}")

        with open(path, "r") as f:
            raw_config = yaml.safe_load(f)

        # 1. Extract Project Info
        project_name = raw_config.get("project", {}).get("name", "SLM_RAG")

        # 2. Extract LLM Configs
        llm_section = raw_config.get("llm", {})
        
        fast_conf = LLMConfig(
            provider=llm_section["fast"]["provider"],
            model_name=llm_section["fast"]["model_name"],
            base_url=llm_section["fast"].get("base_url"),
            temperature=llm_section["fast"].get("temperature", 0.1),
            # Ollama usually doesn't need an API key, but we allow it
            api_key=None 
        )

        smart_conf = LLMConfig(
            provider=llm_section["smart"]["provider"],
            model_name=llm_section["smart"]["model_name"],
            base_url=llm_section["smart"].get("base_url"),
            temperature=llm_section["smart"].get("temperature", 0.1),
            api_key=os.getenv("GROQ_API_KEY") if llm_section["smart"]["provider"] == "groq" else None
        )

        # 3. Extract Retrieval Config
        ret_section = raw_config.get("retrieval", {})
        qdrant_section = ret_section.get("vector_store", {})
        
        retrieval_conf = RetrievalConfig(
            embedder_name=ret_section["embedder"]["model_name"],
            vector_store_host=qdrant_section["host"],
            vector_store_port=qdrant_section["port"],
            vector_store_collection=qdrant_section["collection"],
            vector_store_api_key=os.getenv("QDRANT_API_KEY"), # INJECTED FROM ENV
            enable_late_interaction=ret_section.get("late_interaction", {}).get("enabled", False),
            enable_graph=ret_section.get("graph", {}).get("enabled", False)
        )

        return cls(
            project_name=project_name,
            fast_llm=fast_conf,
            smart_llm=smart_conf,
            retrieval=retrieval_conf
        )

# Global Config Object
try:
    settings = AppConfig.load()
    print(f"✅ Configuration Loaded: {settings.project_name}")
except Exception as e:
    print(f"⚠️ Config Load Failed (ignore this if running unit tests): {e}")
    settings = None