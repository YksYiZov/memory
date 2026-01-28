"""
LLM Client Factory

This module provides factory functions to create appropriate LLM clients
based on deployment/model names.
"""

import os
from typing import Optional, Union
from memu.llm import OpenAIClient, BaseLLMClient


def create_llm_client(
    chat_deployment: str,
    azure_endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    use_entra_id: bool = False,
    api_version: str = "2024-02-01",
    **kwargs
) -> BaseLLMClient:
    """
    Create OpenAI LLM client
    
    Args:
        chat_deployment: Model name (e.g., 'gpt-4o-mini', 'gpt-4', etc.)
        azure_endpoint: Not used (kept for compatibility)
        api_key: OpenAI API key (if not set, reads from OPENAI_API_KEY env var)
        use_entra_id: Not used (kept for compatibility)
        api_version: Not used (kept for compatibility)
        **kwargs: Additional client-specific parameters
        
    Returns:
        BaseLLMClient: OpenAI client instance
    """
    
    # Use OpenAI client
    client_params = {
        "model": chat_deployment,
        "api_key": api_key,
        **kwargs
    }
    
    return OpenAIClient(**client_params)


def get_default_client_params(chat_deployment: str) -> dict:
    """
    Get default parameters for OpenAI client
    
    Args:
        chat_deployment: Model name
        
    Returns:
        dict: Default parameters for the client
    """
    return {
        "model": chat_deployment,
    } 