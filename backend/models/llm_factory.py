from langchain_ollama import ChatOllama
# from langchain_groq import ChatGroq
from backend.core.config_loader import settings

class LLMFactory:
    @staticmethod
    def get_student_llm():
        """
        Returns the configured Student LLM (e.g., Llama 3 8B via Ollama).
        """
        config = settings.student_llm
        
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
            raise ValueError(f"Unsupported Student Provider: {config.provider}")

    # @staticmethod
    # def get_teacher_llm():
    #     """
    #     Returns the configured Teacher LLM (e.g., Llama 3 70B via Groq).
    #     """
    #     config = settings.teacher_llm
        
    #     if config.provider == "groq":
    #         return ChatGroq(
    #             model_name=config.model_name,
    #             temperature=config.temperature,
    #             api_key=config.api_key
    #         )
    #     else:
    #          # Fallback or other providers
    #         raise ValueError(f"Unsupported Teacher Provider: {config.provider}")