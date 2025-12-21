from langchain_ollama import ChatOllama
# from langchain_groq import ChatGroq
from backend.core.config_loader import settings

class LLMFactory:
    @staticmethod
    def get_fast_llm():
        """
        Returns the configured Fast LLM (e.g., qwen3:4b).
        """
        config = settings.fast_llm
        
        if config.provider == "ollama":
            return ChatOllama(
                model=config.model_name,
                base_url=config.base_url,
                temperature=config.temperature,
                # keep_alive="5m" # Optional
            )
        # elif config.provider == "groq":
        #      return ChatGroq(
        #         model_name=config.model_name,
        #         temperature=config.temperature,
        #         api_key=config.api_key
        #     )
        else:
            raise ValueError(f"Unsupported Fast Provider: {config.provider}")

    @staticmethod
    def get_smart_llm():
        """
        Returns the configured Smart LLM (e.g., qwen3:8b).
        """
        config = settings.smart_llm
        
        if config.provider == "ollama":
            return ChatOllama(
                model=config.model_name,
                base_url=config.base_url,
                temperature=config.temperature,
            )
        # elif config.provider == "groq":
        #     ...
        else:
            raise ValueError(f"Unsupported Smart Provider: {config.provider}")