import os
from typing import Optional, Dict, Any
import logging

# LangChain imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama 
from worker.config.models import MochiWorkerConfig, LLMConfigEntry

class LLMServiceError(Exception):
    """Custom exception for LLMService errors."""
    pass

class LLMService:
    """Provides instances of LangChain LLMs based on named configuration profiles."""

    def __init__(self, config: MochiWorkerConfig, logger_instance: Optional[logging.Logger] = None):
        """
        Initializes the LLMService.

        Args:
            config: The MochiWorkerConfig instance containing llm_profiles.
            logger_instance: Optional logger instance.
        """
        if not isinstance(config, MochiWorkerConfig):
            raise LLMServiceError("LLMService must be initialized with a MochiWorkerConfig instance.")
        self.config = config
        self.logger = logger_instance or logging.getLogger(f"mochi.{self.__class__.__name__}")
        self.logger.info("LLMService initialized.")

    def _get_api_key(self, provider: str, configured_key: Optional[str]) -> Optional[str]:
        """Gets API key, preferring configured_key, then environment variable."""
        if configured_key:
            return configured_key
        
        env_var_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY", 
            "azure": "AZURE_OPENAI_API_KEY"
        }
        env_var_name = env_var_map.get(provider.lower())
        if env_var_name:
            key_from_env = os.getenv(env_var_name)
            if key_from_env:
                self.logger.debug(f"Using API key from environment variable {env_var_name} for {provider}.")
                return key_from_env
        return None

    def get_llm(self, profile_name: str) -> BaseChatModel:
        """
        Retrieves and initializes an LLM instance based on the given profile name.

        Args:
            profile_name: The name of the LLM profile defined in MochiWorkerConfig.llm_profiles.

        Returns:
            An initialized LangChain BaseChatModel instance.

        Raises:
            LLMServiceError: If the profile is not found, the provider is unsupported,
                             or instantiation fails.
        """
        self.logger.debug(f"Attempting to get LLM for profile: '{profile_name}'")
        llm_config_entry = self.config.get_llm_profile(profile_name)

        if not llm_config_entry:
            raise LLMServiceError(f"LLM profile '{profile_name}' not found in configuration.")

        provider = llm_config_entry.provider.lower()
        model_name = llm_config_entry.model
        api_key = self._get_api_key(provider, llm_config_entry.api_key)
        base_url = llm_config_entry.base_url
        temperature = llm_config_entry.temperature
        max_tokens = llm_config_entry.max_tokens
        azure_api_version = llm_config_entry.azure_api_version
        azure_deployment = llm_config_entry.azure_deployment
        additional_params = llm_config_entry.additional_params.copy()

        try:
            if provider == "anthropic":
                self.logger.debug(f"Instantiating ChatAnthropic for model: {model_name}")
                client_kwargs = {}
                if temperature is not None: client_kwargs['temperature'] = temperature
                if max_tokens is not None: client_kwargs['max_tokens'] = max_tokens 
                if base_url: client_kwargs['base_url'] = base_url
                client_kwargs.update(additional_params)
                if 'max_tokens' in client_kwargs and 'max_tokens_to_sample' not in client_kwargs:
                    client_kwargs['max_tokens_to_sample'] = client_kwargs.pop('max_tokens')
                
                return ChatAnthropic(
                    model=model_name,
                    api_key=api_key,
                    **client_kwargs
                )
            elif provider == "openai":
                self.logger.debug(f"Instantiating ChatOpenAI for model: {model_name}")
                client_kwargs = {}
                if temperature is not None: client_kwargs['temperature'] = temperature
                if max_tokens is not None: client_kwargs['max_tokens'] = max_tokens
                if base_url: client_kwargs['base_url'] = base_url
                client_kwargs.update(additional_params)
                return ChatOpenAI(
                    model=model_name,
                    api_key=api_key,
                    **client_kwargs
                )
            elif provider == "azure":
                self.logger.debug(f"Instantiating AzureChatOpenAI for deployment: {azure_deployment}")
                if not azure_api_version or not azure_deployment:
                    raise LLMServiceError("Azure provider requires azure_api_version and azure_deployment to be set in the LLM profile.")
                client_kwargs = {}
                if temperature is not None: client_kwargs['temperature'] = temperature
                if max_tokens is not None: client_kwargs['max_tokens'] = max_tokens 
                client_kwargs.update(additional_params)
                return ChatOpenAI(
                    model=model_name,
                    api_key=api_key,
                    api_version=azure_api_version,
                    azure_deployment=azure_deployment,
                    azure_endpoint=base_url,
                    **client_kwargs
                )
            elif provider == "google":
                self.logger.debug(f"Instantiating ChatGoogleGenerativeAI for model: {model_name}")
                client_kwargs = {}
                if temperature is not None: client_kwargs['temperature'] = temperature
                if max_tokens is not None: client_kwargs['max_tokens'] = max_tokens
                client_kwargs.update(additional_params)
                return ChatGoogleGenerativeAI(
                    model=model_name,
                    api_key=api_key,
                    **client_kwargs
                )
            elif provider == "ollama":
                self.logger.debug(f"Instantiating ChatOllama for model: {model_name} at base_url: {base_url}")
                if not base_url:
                    base_url = "http://localhost:11434"
                    self.logger.debug(f"No base_url for Ollama, defaulting to {base_url}")
                client_kwargs = {}
                if temperature is not None: client_kwargs['temperature'] = temperature
                if max_tokens is not None: 
                    if 'options' not in additional_params: additional_params['options'] = {}
                    additional_params['options']['num_predict'] = max_tokens
                client_kwargs.update(additional_params)
                return ChatOllama(
                    model=model_name,
                    base_url=base_url,
                    **client_kwargs
                )
            else:
                raise LLMServiceError(f"Unsupported LLM provider: '{provider}' for profile '{profile_name}'. Supported: anthropic, openai, azure, ollama.")
        
        except ImportError as e_import:
            self.logger.error(f"ImportError for provider {provider}: {e_import}. Ensure necessary langchain packages are installed (e.g., langchain-openai, langchain-anthropic).", exc_info=True)
            raise LLMServiceError(f"Missing langchain package for provider '{provider}'. Details: {e_import}") from e_import
        except Exception as e:
            self.logger.error(f"Failed to instantiate LLM for profile '{profile_name}' (provider: {provider}, model: {model_name}): {e}", exc_info=True)
            raise LLMServiceError(f"LLM instantiation failed for profile '{profile_name}'. Error: {e}") from e
