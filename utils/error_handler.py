"""
Error handling utilities for CO2 Reduction AI Agent
"""
import time
import traceback
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import logging


class ErrorHandler:
    """Centralized error handling for the application"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize error handler
        
        Args:
            logger: Logger instance for error logging
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def handle_file_upload_error(self, error: Exception, filename: Optional[str] = None) -> str:
        """
        Handle file upload errors and return user-friendly message
        
        Args:
            error: The exception that occurred
            filename: Name of the file being uploaded
            
        Returns:
            User-friendly error message
        """
        context = {"filename": filename, "error_type": type(error).__name__}
        self.log_error(error, context)
        
        error_msg = str(error).lower()
        
        # File format errors
        if "extension" in error_msg or "format" in error_msg:
            return (
                "❌ Invalid file format. Please upload a CSV or Excel file (.csv, .xlsx, .xls).\n\n"
                f"File: {filename or 'Unknown'}"
            )
        
        # File size errors
        if "size" in error_msg or "large" in error_msg:
            return (
                "❌ File is too large. Maximum file size is 10 MB.\n\n"
                f"File: {filename or 'Unknown'}"
            )
        
        # Missing columns errors
        if "column" in error_msg or "missing" in error_msg:
            return (
                "❌ Invalid file structure. Your file is missing required columns.\n\n"
                "Required columns:\n"
                "• Activity\n"
                "• Avg_CO2_Emission(kg/day) or Avg_CO2_Emission\n"
                "• Category\n\n"
                f"Error details: {str(error)}"
            )
        
        # Empty file errors
        if "empty" in error_msg or "no data" in error_msg:
            return (
                "❌ The uploaded file is empty or contains no valid data.\n\n"
                f"File: {filename or 'Unknown'}"
            )
        
        # Parsing errors
        if "parse" in error_msg or "read" in error_msg:
            return (
                "❌ Unable to read the file. Please ensure it's a valid CSV or Excel file.\n\n"
                f"File: {filename or 'Unknown'}\n"
                f"Error: {str(error)}"
            )
        
        # Generic file error
        return (
            "❌ An error occurred while processing your file.\n\n"
            f"File: {filename or 'Unknown'}\n"
            f"Error: {str(error)}\n\n"
            "Please check that your file is a valid CSV or Excel file with the required columns."
        )
    
    def handle_llm_error(
        self,
        error: Exception,
        retry_func: Optional[Callable] = None,
        max_retries: int = 3,
        initial_delay: float = 1.0
    ) -> str:
        """
        Handle LLM service errors with retry logic and exponential backoff
        
        Args:
            error: The exception that occurred
            retry_func: Optional function to retry
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds before first retry
            
        Returns:
            User-friendly error message or result from successful retry
        """
        context = {
            "error_type": type(error).__name__,
            "max_retries": max_retries
        }
        self.log_error(error, context)
        
        error_msg = str(error).lower()
        
        # Connection errors - attempt retry if function provided
        if retry_func and ("connection" in error_msg or "timeout" in error_msg):
            self.logger.info(f"Attempting to retry LLM request with exponential backoff")
            
            for attempt in range(max_retries):
                try:
                    delay = initial_delay * (2 ** attempt)
                    self.logger.debug(f"Retry attempt {attempt + 1}/{max_retries} after {delay}s delay")
                    time.sleep(delay)
                    
                    result = retry_func()
                    self.logger.info(f"Retry successful on attempt {attempt + 1}")
                    return result
                    
                except Exception as retry_error:
                    self.logger.warning(f"Retry attempt {attempt + 1} failed: {str(retry_error)}")
                    if attempt == max_retries - 1:
                        # Last attempt failed
                        context["retry_attempts"] = max_retries
                        self.log_error(retry_error, context)
        
        # Connection/availability errors
        if "connection" in error_msg or "refused" in error_msg:
            return (
                "❌ Unable to connect to the AI service.\n\n"
                "Please ensure:\n"
                "• Ollama is running (start with: ollama serve)\n"
                "• The model is available (check with: ollama list)\n"
                "• The service URL is correct in configuration\n\n"
                f"Error: {str(error)}"
            )
        
        # Timeout errors
        if "timeout" in error_msg:
            return (
                "⏱️ The AI service took too long to respond.\n\n"
                "This might happen if:\n"
                "• The model is still loading\n"
                "• The query is too complex\n"
                "• The system is under heavy load\n\n"
                "Please try again in a moment."
            )
        
        # Model not found errors
        if "model" in error_msg and ("not found" in error_msg or "unavailable" in error_msg):
            return (
                "❌ The AI model is not available.\n\n"
                "Please ensure the model is installed:\n"
                "• Run: ollama pull llama3\n"
                "• Or: ollama pull mistral\n\n"
                f"Error: {str(error)}"
            )
        
        # Rate limiting errors
        if "rate" in error_msg or "limit" in error_msg or "429" in error_msg:
            return (
                "⚠️ Too many requests. Please wait a moment and try again.\n\n"
                "The AI service is temporarily limiting requests."
            )
        
        # Generic LLM error
        return (
            "❌ An error occurred while generating the response.\n\n"
            f"Error: {str(error)}\n\n"
            "Please try again. If the problem persists, check that the AI service is running correctly."
        )
    
    def handle_vector_store_error(self, error: Exception, operation: Optional[str] = None) -> str:
        """
        Handle vector store/database errors
        
        Args:
            error: The exception that occurred
            operation: The operation being performed (e.g., 'search', 'add', 'initialize')
            
        Returns:
            User-friendly error message
        """
        context = {
            "error_type": type(error).__name__,
            "operation": operation
        }
        self.log_error(error, context)
        
        error_msg = str(error).lower()
        
        # Collection not found errors
        if "collection" in error_msg and ("not found" in error_msg or "does not exist" in error_msg):
            return (
                "❌ Knowledge base not initialized.\n\n"
                "The sustainability tips database needs to be set up. "
                "This should happen automatically on first run.\n\n"
                f"Operation: {operation or 'Unknown'}\n"
                f"Error: {str(error)}"
            )
        
        # Permission/access errors
        if "permission" in error_msg or "access" in error_msg:
            return (
                "❌ Database access error.\n\n"
                "Unable to access the vector database. Please check file permissions.\n\n"
                f"Operation: {operation or 'Unknown'}\n"
                f"Error: {str(error)}"
            )
        
        # Disk space errors
        if "disk" in error_msg or "space" in error_msg:
            return (
                "❌ Insufficient disk space.\n\n"
                "The system is running low on disk space. Please free up some space and try again.\n\n"
                f"Operation: {operation or 'Unknown'}"
            )
        
        # Embedding errors
        if "embedding" in error_msg or "dimension" in error_msg:
            return (
                "❌ Error generating embeddings.\n\n"
                "There was a problem creating text embeddings for semantic search.\n\n"
                f"Operation: {operation or 'Unknown'}\n"
                f"Error: {str(error)}"
            )
        
        # Search/query errors
        if operation == "search" or "query" in error_msg:
            return (
                "❌ Error searching knowledge base.\n\n"
                "Unable to retrieve relevant information from the database.\n\n"
                f"Error: {str(error)}\n\n"
                "The system will continue with limited context."
            )
        
        # Generic vector store error
        return (
            "❌ Database error occurred.\n\n"
            f"Operation: {operation or 'Unknown'}\n"
            f"Error: {str(error)}\n\n"
            "Please try again. If the problem persists, the database may need to be reinitialized."
        )
    
    def log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        include_traceback: bool = True
    ) -> None:
        """
        Log error with context information
        
        Args:
            error: The exception to log
            context: Additional context information
            include_traceback: Whether to include full traceback
        """
        context = context or {}
        
        # Build error message
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Format context
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        
        # Log the error
        log_message = f"{error_type}: {error_msg}"
        if context_str:
            log_message += f" | Context: {context_str}"
        
        self.logger.error(log_message)
        
        # Log traceback if requested
        if include_traceback:
            tb = traceback.format_exc()
            self.logger.debug(f"Traceback:\n{tb}")
    
    def handle_data_validation_error(
        self,
        error: Exception,
        data_type: str = "dataset"
    ) -> str:
        """
        Handle data validation errors
        
        Args:
            error: The exception that occurred
            data_type: Type of data being validated
            
        Returns:
            User-friendly error message
        """
        context = {
            "error_type": type(error).__name__,
            "data_type": data_type
        }
        self.log_error(error, context)
        
        error_msg = str(error).lower()
        
        # Schema validation errors
        if "schema" in error_msg or "column" in error_msg:
            return (
                f"❌ Invalid {data_type} structure.\n\n"
                f"Error: {str(error)}\n\n"
                "Please ensure your data has the required columns with correct names."
            )
        
        # Value validation errors
        if "value" in error_msg or "invalid" in error_msg or "range" in error_msg:
            return (
                f"❌ Invalid data values in {data_type}.\n\n"
                f"Error: {str(error)}\n\n"
                "Please check that:\n"
                "• Emission values are positive numbers\n"
                "• Categories are valid (Transport, Household, Food, Lifestyle)\n"
                "• All required fields are filled"
            )
        
        # Type errors
        if "type" in error_msg or "dtype" in error_msg:
            return (
                f"❌ Data type error in {data_type}.\n\n"
                f"Error: {str(error)}\n\n"
                "Please ensure emission values are numbers, not text."
            )
        
        # Generic validation error
        return (
            f"❌ Validation error in {data_type}.\n\n"
            f"Error: {str(error)}\n\n"
            "Please check your data format and try again."
        )
    
    def handle_calculation_error(
        self,
        error: Exception,
        calculation_type: str = "emission"
    ) -> str:
        """
        Handle calculation errors
        
        Args:
            error: The exception that occurred
            calculation_type: Type of calculation being performed
            
        Returns:
            User-friendly error message
        """
        context = {
            "error_type": type(error).__name__,
            "calculation_type": calculation_type
        }
        self.log_error(error, context)
        
        return (
            f"❌ Error calculating {calculation_type}.\n\n"
            f"Error: {str(error)}\n\n"
            "Please check your input data and try again."
        )


# Convenience function for creating error handler with logger
def create_error_handler(logger: Optional[logging.Logger] = None) -> ErrorHandler:
    """
    Create an ErrorHandler instance
    
    Args:
        logger: Optional logger instance
        
    Returns:
        ErrorHandler instance
    """
    return ErrorHandler(logger=logger)
