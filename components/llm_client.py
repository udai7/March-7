"""
LLM Client abstraction for interacting with Ollama API or Hugging Face Inference API
"""
import requests
import time
from typing import List, Optional, Dict, Any
import config


class LLMClient:
    """Client for interacting with LLM services (Ollama, Hugging Face, or Groq)"""
    
    def __init__(
        self,
        provider: str = config.LLM_PROVIDER,
        model_name: str = config.LLM_MODEL,
        base_url: str = config.LLM_BASE_URL,
        api_key: str = None,
        temperature: float = config.LLM_TEMPERATURE,
        max_tokens: int = config.LLM_MAX_TOKENS,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize LLM client
        
        Args:
            provider: "ollama", "huggingface", or "groq"
            model_name: Name of the model to use
            base_url: Base URL for Ollama API (ignored for HF/Groq)
            api_key: API key (HF or Groq, ignored for Ollama)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Set API key based on provider
        if api_key:
            self.api_key = api_key
        elif self.provider == "huggingface":
            self.api_key = config.HUGGINGFACE_API_KEY
        elif self.provider == "groq":
            self.api_key = config.GROQ_API_KEY
        else:
            self.api_key = None
        
        # Set up endpoints based on provider
        if self.provider == "ollama":
            self.generate_endpoint = f"{self.base_url}/api/generate"
            self.tags_endpoint = f"{self.base_url}/api/tags"
        elif self.provider == "huggingface":
            self.generate_endpoint = f"https://api-inference.huggingface.co/models/{self.model_name}"
        elif self.provider == "groq":
            self.generate_endpoint = "https://api.groq.com/openai/v1/chat/completions"
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'ollama', 'huggingface', or 'groq'")
    
    def check_availability(self) -> bool:
        """
        Check if LLM service is available and model is loaded
        
        Returns:
            True if service is available, False otherwise
        """
        if self.provider in ["huggingface", "groq"]:
            # For HF/Groq, just check if we have an API key
            return bool(self.api_key)
        
        # For Ollama
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
            stream: Whether to stream the response (Ollama only)
            
        Returns:
            Generated text response
            
        Raises:
            RuntimeError: If generation fails after all retries
        """
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        
        if self.provider == "huggingface":
            return self._generate_huggingface(prompt, max_tokens, temperature)
        elif self.provider == "groq":
            return self._generate_groq(prompt, max_tokens, temperature)
        else:
            return self._generate_ollama(prompt, max_tokens, temperature, stream)
    
    def _generate_ollama(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stream: bool
    ) -> str:
        """Generate using Ollama API"""
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
                    timeout=60
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
            
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2 ** attempt)
                time.sleep(delay)
        
        raise RuntimeError(
            f"Failed to generate response after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )
    
    def _generate_huggingface(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate using Hugging Face Inference API"""
        if not self.api_key:
            raise RuntimeError("Hugging Face API key not provided. Set HUGGINGFACE_API_KEY in config.")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False
            }
        }
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.generate_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=30  # HF is faster
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # HF returns a list of generated texts
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get('generated_text', '').strip()
                    elif isinstance(result, dict):
                        return result.get('generated_text', '').strip()
                    return str(result).strip()
                elif response.status_code == 503:
                    # Model is loading, wait and retry
                    last_error = "Model is loading, please wait..."
                    time.sleep(20)  # Wait for model to load
                    continue
                else:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    
            except requests.exceptions.Timeout:
                last_error = "Request timeout"
            except requests.exceptions.RequestException as e:
                last_error = f"Request error: {str(e)}"
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
            
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2 ** attempt)
                time.sleep(delay)
        
        raise RuntimeError(
            f"Failed to generate response after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )
    
    def _generate_groq(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate using Groq API (OpenAI-compatible, extremely fast)"""
        if not self.api_key:
            raise RuntimeError("Groq API key not provided. Get free key from console.groq.com")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful CO₂ reduction advisor. Provide concise, actionable advice."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 1,
            "stream": False
        }
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.generate_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=10  # Groq is very fast
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        return result["choices"][0]["message"]["content"].strip()
                    return ""
                else:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    
            except requests.exceptions.Timeout:
                last_error = "Request timeout"
            except requests.exceptions.RequestException as e:
                last_error = f"Request error: {str(e)}"
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
            
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2 ** attempt)
                time.sleep(delay)
        
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
