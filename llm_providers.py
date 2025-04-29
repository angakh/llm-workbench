"""
llm_providers.py - LLM Integration Framework for ModelComparator

This module implements a flexible and extensible framework for connecting to various
Large Language Model (LLM) providers, allowing the application to leverage different
AI models based on availability, performance needs, or user preference.

Key components:
1. Base Class:
   - LLMProvider: Abstract base class defining the common interface for all providers
     with methods for listing available models and generating responses.

2. Provider Implementations:
   - OllamaProvider: Connects to locally-running Ollama models
   - OpenAIProvider: Integrates with OpenAI's API
   - AnthropicProvider: Connects to Anthropic's API for Claude models

3. Manager Class:
   - ProviderManager: Orchestrates provider registration, selection, and usage,
     providing a unified interface for the application to interact with any LLM
     regardless of the backend service.

Key functionalities:
- Provider registration and discovery
- Model listing for each provider
- Unified prompt formatting and response generation
- Configuration loading from environment variables or config files
- Error handling and logging for API interactions
- Multi-model response generation

The architecture follows a plugin-style design pattern where new LLM providers can be
added by implementing the LLMProvider interface and registering with the ProviderManager.
"""
import os
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Union
import requests
import logging
import importlib.util
from concurrent.futures import ThreadPoolExecutor

# Check if OpenAI is available, handle gracefully if not
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI package not available. OpenAI provider will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMProvider:
    """Base class for LLM providers"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the LLM provider with configuration.
        
        Args:
            config: Configuration dictionary for the provider
        """
        self.config = config or {}
        self.name = "base"
        self.enabled = True
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models for this provider.
        
        Returns:
            List of model names
        """
        return []
    
    def generate_response(self, prompt: str, model: str, 
                         temperature: float = 0.7, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The prompt template
            model: The model to use
            temperature: Temperature parameter for generation
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary with response information
        """
        raise NotImplementedError("Subclasses must implement generate_response")
    
    def format_message(self, prompt: str) -> str:
        """
        Format the message for the model.
        
        Args:
            prompt: The prompt template
            
        Returns:
            Formatted message
        """
        return prompt
        

class OllamaProvider(LLMProvider):
    """Provider for Ollama models"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the Ollama provider.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.name = "ollama"
        self.base_url = self.config.get("api_url", "http://localhost:11434")
        self.enabled = self.check_availability()
    
    def check_availability(self) -> bool:
        """
        Check if Ollama is available.
        
        Returns:
            True if available, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models from Ollama.
        
        Returns:
            List of model names
        """
        if not self.enabled:
            return []
            
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                return models
            else:
                logger.error(f"Failed to get Ollama models: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error getting Ollama models: {e}")
            return []
    
    def generate_response(self, prompt: str, model: str = "llama3", 
                         temperature: float = 0.7, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Generate a response using Ollama.
        
        Args:
            prompt: The prompt template
            model: The model to use (e.g., "llama3")
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Response information dictionary
        """
        start_time = time.time()
        
        try:
            formatted_message = self.format_message(prompt)
            
            payload = {
                "model": model,
                "prompt": formatted_message,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "text": data.get("response", "No response generated"),
                    "model": model,
                    "provider": self.name,
                    "time": elapsed_time,
                    "error": None
                }
            else:
                error_msg = f"Ollama error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {
                    "text": "",
                    "model": model,
                    "provider": self.name,
                    "time": elapsed_time,
                    "error": error_msg
                }
                
        except Exception as e:
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.error(f"Error generating response with Ollama: {e}")
            return {
                "text": "",
                "model": model,
                "provider": self.name,
                "time": elapsed_time,
                "error": str(e)
            }


class OpenAIProvider(LLMProvider):
    """Provider for OpenAI models"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the OpenAI provider.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.name = "openai"
        self.api_key = self.config.get("api_key", os.environ.get("OPENAI_API_KEY", ""))
        self.enabled = OPENAI_AVAILABLE and bool(self.api_key)
        self.client = None
        if self.enabled:
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {e}")
                self.enabled = False
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models from OpenAI.
        
        Returns:
            List of model names
        """
        if not self.enabled or not self.client:
            return []
            
        try:
            models = self.client.models.list()
            model_names = [model.id for model in models.data]
            # Filter to just the chat models
            chat_models = [m for m in model_names if any(name in m for name in ["gpt", "GPT"])]
            return chat_models
        except Exception as e:
            logger.error(f"Error getting OpenAI models: {e}")
            return []
    
    def generate_response(self, prompt: str, model: str = "gpt-4o", 
                         temperature: float = 0.7, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Generate a response using OpenAI.
        
        Args:
            prompt: The prompt template
            model: The model to use (e.g., "gpt-4o")
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Response information dictionary
        """
        if not self.enabled or not self.client:
            return {
                "text": "",
                "model": model,
                "provider": self.name,
                "time": 0,
                "error": "OpenAI client not initialized"
            }
            
        start_time = time.time()
        
        try:
            system_prompt = """You are a helpful AI assistant. 
            Provide clear, concise, and accurate responses to the user's questions."""
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            return {
                "text": response.choices[0].message.content,
                "model": model,
                "provider": self.name,
                "time": elapsed_time,
                "error": None
            }
            
        except Exception as e:
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.error(f"Error generating response with OpenAI: {e}")
            return {
                "text": "",
                "model": model,
                "provider": self.name,
                "time": elapsed_time,
                "error": str(e)
            }


class AnthropicProvider(LLMProvider):
    """Provider for Anthropic Claude models"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the Anthropic provider.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.name = "anthropic"
        self.api_key = self.config.get("api_key", os.environ.get("ANTHROPIC_API_KEY", ""))
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.default_model = "claude-3-opus-20240229"
        self.enabled = bool(self.api_key)
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available Claude models.
        
        Returns:
            List of model names
        """
        if not self.enabled:
            return []
            
        # Anthropic doesn't have a models list endpoint, so we hardcode the models
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229", 
            "claude-3-haiku-20240307",
            "claude-3.5-sonnet-20240620",
            "claude-3-5-sonnet-20240620",
            "claude-2.1", 
            "claude-2.0"
        ]
    
    def generate_response(self, prompt: str, model: str = None, 
                         temperature: float = 0.7, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Generate a response using Anthropic Claude.
        
        Args:
            prompt: The prompt template
            model: The model to use
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Response information dictionary
        """
        if not self.enabled:
            return {
                "text": "",
                "model": model or self.default_model,
                "provider": self.name,
                "time": 0,
                "error": "Anthropic API key not set"
            }
            
        start_time = time.time()
        
        try:
            model = model or self.default_model
            
            system_prompt = """You are a helpful AI assistant.
            Provide clear, concise, and accurate responses to the user's questions."""
            
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "system": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload
            )
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "text": data.get("content", [{"text": "No response generated"}])[0]["text"],
                    "model": model,
                    "provider": self.name,
                    "time": elapsed_time,
                    "error": None
                }
            else:
                error_msg = f"Anthropic error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {
                    "text": "",
                    "model": model,
                    "provider": self.name,
                    "time": elapsed_time,
                    "error": error_msg
                }
                
        except Exception as e:
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.error(f"Error generating response with Anthropic: {e}")
            return {
                "text": "",
                "model": model or self.default_model,
                "provider": self.name,
                "time": elapsed_time,
                "error": str(e)
            }


class ProviderManager:
    """Manager for LLM providers"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the provider manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.providers = {}
        self.config = {}
        self.selected_models = {}  # Dict to track selected models for each provider
        
        # Load configuration if provided
        if config_path:
            self.load_config(config_path)
        
        # Register default providers
        self.register_default_providers()
    
    def load_config(self, config_path: str):
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to the configuration file
        """
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self.config = {}
    
    def register_default_providers(self):
        """Register the default LLM providers."""
        # Configure Ollama provider
        ollama_config = self.config.get("ollama", {})
        if not ollama_config:
            ollama_config = {"api_url": "http://localhost:11434"}
        self.providers["ollama"] = OllamaProvider(ollama_config)
        
        # Configure OpenAI provider if available
        if OPENAI_AVAILABLE:
            openai_config = self.config.get("openai", {})
            if not openai_config:
                openai_config = {"api_key": os.environ.get("OPENAI_API_KEY", "")}
            self.providers["openai"] = OpenAIProvider(openai_config)
        
        # Configure Anthropic provider
        anthropic_config = self.config.get("anthropic", {})
        if not anthropic_config:
            anthropic_config = {"api_key": os.environ.get("ANTHROPIC_API_KEY", "")}
        self.providers["anthropic"] = AnthropicProvider(anthropic_config)
    
    def register_provider(self, provider_name: str, provider_instance: LLMProvider):
        """
        Register a new provider.
        
        Args:
            provider_name: Name of the provider
            provider_instance: Provider instance
        """
        self.providers[provider_name] = provider_instance
    
    def get_provider(self, provider_name: str) -> Optional[LLMProvider]:
        """
        Get a provider by name.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Provider instance, or None if not found
        """
        return self.providers.get(provider_name)
    
    def get_available_providers(self) -> List[str]:
        """
        Get list of available and enabled provider names.
        
        Returns:
            List of provider names
        """
        return [name for name, provider in self.providers.items() if provider.enabled]
    
    def get_provider_models(self, provider_name: str) -> List[str]:
        """
        Get available models for a specific provider.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            List of model names
        """
        provider = self.get_provider(provider_name)
        if provider and provider.enabled:
            return provider.get_available_models()
        return []
    
    def get_all_models(self) -> Dict[str, List[str]]:
        """
        Get all available models from all providers.
        
        Returns:
            Dictionary mapping provider names to lists of model names
        """
        models = {}
        for provider_name, provider in self.providers.items():
            if provider.enabled:
                try:
                    provider_models = provider.get_available_models()
                    if provider_models:
                        models[provider_name] = provider_models
                except Exception as e:
                    logger.error(f"Error getting models for {provider_name}: {e}")
        
        return models
    
    def select_model(self, provider_name: str, model_name: str, selected: bool = True):
        """
        Select or deselect a model for use.
        
        Args:
            provider_name: Name of the provider
            model_name: Name of the model
            selected: Whether the model is selected (True) or deselected (False)
        """
        if provider_name not in self.selected_models:
            self.selected_models[provider_name] = set()
            
        if selected:
            self.selected_models[provider_name].add(model_name)
        elif model_name in self.selected_models[provider_name]:
            self.selected_models[provider_name].remove(model_name)
    
    def get_selected_models(self) -> Dict[str, List[str]]:
        """
        Get all currently selected models grouped by provider.
        
        Returns:
            Dictionary mapping provider names to lists of selected model names
        """
        return {
            provider: list(models) 
            for provider, models in self.selected_models.items() 
            if models
        }
    
    def generate_response(self, prompt: str, provider_name: str, model_name: str,
                         temperature: float = 0.7, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Generate a response using a specific provider and model.
        
        Args:
            prompt: The prompt text
            provider_name: Name of the provider
            model_name: Name of the model to use
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Response information dictionary
        """
        provider = self.get_provider(provider_name)
        if not provider or not provider.enabled:
            return {
                "text": "",
                "model": model_name,
                "provider": provider_name,
                "time": 0,
                "error": f"Provider '{provider_name}' not found or not enabled"
            }
        
        try:
            return provider.generate_response(
                prompt=prompt,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            return {
                "text": "",
                "model": model_name,
                "provider": provider_name,
                "time": 0,
                "error": error_msg
            }
    
    def generate_responses_from_all_selected(self, prompt: str, temperature: float = 0.7,
                                           max_tokens: int = 1000) -> List[Dict[str, Any]]:
        """
        Generate responses from all selected models.
        
        Args:
            prompt: The prompt text
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            List of response information dictionaries
        """
        responses = []
        
        with ThreadPoolExecutor() as executor:
            future_to_model = {}
            
            for provider_name, model_names in self.get_selected_models().items():
                provider = self.get_provider(provider_name)
                if not provider or not provider.enabled:
                    continue
                    
                for model_name in model_names:
                    future = executor.submit(
                        self.generate_response,
                        prompt=prompt,
                        provider_name=provider_name,
                        model_name=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    future_to_model[(provider_name, model_name)] = future
            
            for (provider_name, model_name), future in future_to_model.items():
                try:
                    response = future.result()
                    responses.append(response)
                except Exception as e:
                    logger.error(f"Error getting response from {provider_name}/{model_name}: {e}")
                    responses.append({
                        "text": "",
                        "model": model_name,
                        "provider": provider_name,
                        "time": 0,
                        "error": str(e)
                    })
        
        return responses