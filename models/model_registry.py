import requests
from typing import List, Optional
import os
from openai import OpenAI
from utils.helper_functions import load_config
import traceback
import yaml
import streamlit as st

class ModelRegistry:
    @staticmethod
    def get_available_models(server: str) -> List[str]:
        """Get available models for a server"""
        try:
            if server == "openai":
                return ModelRegistry.fetch_openai_models()
            elif server == "groq":
                return ModelRegistry.fetch_groq_models()
            elif server == "claude":
                return ModelRegistry.fetch_claude_models()
            elif server == "ollama":
                return ModelRegistry.fetch_ollama_models()
            elif server == "gemini":
                return ModelRegistry.fetch_gemini_models()
            return []
        except Exception as e:
            print(f"Error getting models for {server}: {e}")
            return []

    @staticmethod
    def fetch_openai_models() -> List[str]:
        """Fetch available models from OpenAI API"""
        try:
            # Get API key from Streamlit secrets
            api_key = st.secrets.api_keys.openai
            
            if not api_key:
                print("No OpenAI API key found in Streamlit secrets")
                return []
            
            print(f"Found OpenAI API key starting with: {api_key[:6]}...")
        
            client = OpenAI(api_key=api_key)
            models = client.models.list()
            
            # Filter for only chat models
            chat_models = [
                model.id for model in models 
                if "gpt" in model.id.lower()
            ]
            
            print(f"Found OpenAI chat models: {chat_models}")
            return sorted(chat_models)
        
        except Exception as e:
            print(f"Error fetching OpenAI models: {e}")
            traceback.print_exc()
            return []

    @staticmethod
    def fetch_groq_models() -> List[str]:
        """Fetch available models from Groq API"""
        try:
            api_key = st.secrets.api_keys.groq
            if not api_key:
                print("No Groq API key found in Streamlit secrets")
                return []
                
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get("https://api.groq.com/v1/models", headers=headers)
            return [model["id"] for model in response.json()["data"]]
        except Exception as e:
            print(f"Error fetching Groq models: {e}")
            return []

    @staticmethod
    def fetch_ollama_models() -> List[str]:
        """Fetch available models from local Ollama instance"""
        try:
            response = requests.get("http://localhost:11434/api/tags")
            return [model["name"] for model in response.json()["models"]]
        except Exception as e:
            print(f"Error fetching Ollama models: {e}")
            return []

    @staticmethod
    def fetch_claude_models() -> List[str]:
        """Fetch available Claude models"""
        try:
            api_key = st.secrets.api_keys.claude
            if not api_key:
                print("No Claude API key found in Streamlit secrets")
                return []
                
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            }
            response = requests.get("https://api.anthropic.com/v1/models", headers=headers)
            return [model["id"] for model in response.json()["models"]]
        except Exception as e:
            print(f"Error fetching Claude models: {e}")
            return []

    @staticmethod
    def fetch_gemini_models() -> List[str]:
        """Fetch available Gemini models"""
        try:
            api_key = st.secrets.api_keys.gemini
            if not api_key:
                print("No Gemini API key found in Streamlit secrets")
                return []
            
            # Gemini models are fixed for now
            return ["gemini-pro", "gemini-pro-vision"]
        except Exception as e:
            print(f"Error fetching Gemini models: {e}")
            return []
