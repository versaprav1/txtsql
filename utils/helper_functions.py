import os
import yaml
import json
import logging
from datetime import datetime
import traceback
from typing import Dict, Any, Optional, List
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from Streamlit secrets.
    The config_path parameter is kept for backward compatibility but is no longer used.
    """
    try:
        # Get API keys from Streamlit secrets
        config = {
            'OPENAI_API_KEY': st.secrets.api_keys.openai,
            'CLAUDE_API_KEY': st.secrets.api_keys.claude,
            'GEMINI_API_KEY': st.secrets.api_keys.gemini,
            'GROQ_API_KEY': st.secrets.api_keys.groq,
            'timestamp': datetime.now().isoformat()
        }
        
        # Set environment variables
        for key, value in config.items():
            if key != 'timestamp':
                os.environ[key] = value
                
        return config
    except Exception as e:
        logger.error(f"Error loading config from Streamlit secrets: {str(e)}")
        return {}
