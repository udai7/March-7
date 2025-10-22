"""
LLM Client abstraction for interacting with Ollama API
"""
import requests
import time
from typing import List, Optional, Dict, Any
import config


class LLMClient:
    """Client for interacting with Ollama LLM service"""
    
    def __init__(
        self,
        model_name: str = config.LLM_MODEL,
        base_url: str = config.LLM_BASE_URL,
        temperature: float = config.LLM_TEMPERATURE,
        max_tokens: int = config.LLM_MAX_TOKENS,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize LLM client
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for Ollama API
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # API endpoints
        self.generate_endpoint = f"{self.base_url}/api/generate"
        self.tags_endpoint = f"{self.base_url}/api/tags"
    
    def check_availability(self) -> bool:
        """
        Check if LLM service is available and model is loaded
        
        Returns:
            True if service is available, False otherwise
        """
        try:
            response = requests.get(self.tags_endpoint, timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                # Check if our model is available
                model_names = [m.get('name', '') for m in models]
                return any(self.model_name in name for name in model_names)
            return False
        except requests.exceptions.RequestException:
            return False
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> str:
        """
        Generate text from prompt with error handling and retries
        
        Args:
            prompt: Input prompt for generation
            max_tokens: Override default max tokens
            temperature: Override default temperature
            stream: Whether to stream the response
            
        Returns:
            Generated text response
            
        Raises:
            RuntimeError: If generation fails after all retries
        """
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.generate_endpoint,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', '').strip()
                else:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    
            except requests.exceptions.Timeout:
                last_error = "Request timeout"
            except requests.exceptions.ConnectionError:
                last_error = "Connection error - is Ollama running?"
            except requests.exceptions.RequestException as e:
                last_error = f"Request error: {str(e)}"
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
            
            # Exponential backoff before retry
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2 ** attempt)
                time.sleep(delay)
        
        # All retries failed
        raise RuntimeError(
            f"Failed to generate response after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )
    
    def generate_with_context(
        self,
        query: str,
        context: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate response using RAG pattern with retrieved context
        
        Args:
            query: User query
            context: List of retrieved context documents
            max_tokens: Override default max tokens
            temperature: Override default temperature
            
        Returns:
            Generated response incorporating context
        """
        # Build RAG prompt
        context_text = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(context)])
        
        rag_prompt = f"""You are a CO₂ reduction advisor helping users reduce their carbon footprint.

Use the following context to answer the user's query. Base your recommendations on the provided information.

{context_text}

User Query: {query}

Provide a helpful, actionable response with specific recommendations:"""
        
        return self.generate(
            prompt=rag_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
