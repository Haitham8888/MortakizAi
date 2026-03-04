# ===============================================================================
#                          MortakizAi Configuration
# ===============================================================================

import os

# ===============================================================================
#                              Server Settings
# ===============================================================================

# Project Name
PROJECT_NAME = "MortakizAi"

# Host and Port configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8080

# Logging level (debug, info, warning, error)
LOG_LEVEL = "error"


# ===============================================================================
#                              GPU Settings
# ===============================================================================

# Number of GPUs to use (None = use all available, 1 = single GPU, etc.)
MAX_GPU_COUNT = None

# Enable CUDA memory allocation optimization
CUDA_MEMORY_OPTIMIZATION = True

# VRAM threshold (GB) to prefer bfloat16 over 4-bit quantization
VRAM_THRESHOLD_FOR_BFLOAT16 = 30


# ===============================================================================
#                              Local Model Settings
# ===============================================================================

# List of paths to search for the model (in priority order)
MODEL_PATH_OPTIONS = [
    "./models_cache/models--Qwen--Qwen2.5-Coder-7B-Instruct",
    "./qwen-coder"
]

# Display name for the model in the UI/API
MODEL_DISPLAY_NAME = "qwen2.5-coder-7b"


# ===============================================================================
#                              Generation Settings
# ===============================================================================

# Maximum number of tokens to generate per request
MAX_NEW_TOKENS = 1024

# Temperature (0.0 = deterministic, 1.0 = highly creative)
TEMPERATURE = 0.7

# Enable random sampling
DO_SAMPLE = True

# Top-P (nucleus sampling) - set to None to disable
TOP_P = None

# Top-K - set to None to disable  
TOP_K = None

# Repetition penalty (1.0 = no penalty)
REPETITION_PENALTY = 1.0

# Generation stop tokens
STOP_TOKENS = ["<|im_end|>", "<|endoftext|>", "###"]


# ===============================================================================
#                              OpenRouter Settings (Optional)
# ===============================================================================

# Use OpenRouter cloud API instead of the local model
USE_OPENROUTER = False

# API Key (recommened to use OPENROUTER_API_KEY environment variable)
OPENROUTER_API_KEY = os.environ.get(
    "OPENROUTER_API_KEY", 
    "sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
)

# API endpoint URL
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Model ID to use on OpenRouter
OPENROUTER_MODEL_ID = "qwen/qwen3-coder-30b-a3b-instruct"


# ===============================================================================
#                              System Prompt Settings
# ===============================================================================

# Default system instruction for the AI
DEFAULT_SYSTEM_PROMPT = """You are a helpful coding assistant. Always preserve proper code formatting, indentation, and whitespace in your responses."""


# ===============================================================================
#                              Files and Storage Settings
# ===============================================================================

# Static files directory for the frontend
STATIC_FOLDER = "static"

# Data storage directory
DATA_FOLDER = "data"

# Chat history file name
HISTORY_FILE = "history.json"

# Maximum length for uploaded file content (characters)
MAX_FILE_CONTENT_LENGTH = 12000


# ===============================================================================
#                              Advanced Options
# ===============================================================================

# Enable trust_remote_code for model loading
TRUST_REMOTE_CODE = True

# Apply Mistral-specific regex fixes to the tokenizer
FIX_MISTRAL_REGEX = True

# Clear CUDA cache after each request to optimize memory usage
CLEANUP_MEMORY_AFTER_REQUEST = True
