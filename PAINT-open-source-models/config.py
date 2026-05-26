"""
Model registry and dataset-language configuration for PAINT inference experiments.
"""

import os

# ---------------------------------------------------------------------------
# Model registry
#
# The five open-weight models reported in the paper. The short_name keys
# match the per-model output folders (model_outputs/<short_name>/) and the
# MODEL_DIRS map in src/paint/ensemble.py, so inference output drops straight
# into the evaluation pipeline.
#
# Each entry: short_name -> {hf_id, vllm_extra_args, is_thinking, user_message}
#   hf_id:            HuggingFace model ID served via vLLM.
#   vllm_extra_args:  extra CLI flags passed to `vllm serve`. 4-bit
#                     bitsandbytes is only needed for the 32B model on a
#                     single ~40GB GPU; the ≤8B models run in bf16 as-is
#                     (add the same two flags if your GPU is smaller).
#   is_thinking:      True if the model emits <think> reasoning blocks.
#   user_message:     user-turn text; "/think" enables Qwen3 thinking mode.
# ---------------------------------------------------------------------------
MODELS = {
    "deepseek-r1-qwen-32b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "vllm_extra_args": [
            "--reasoning-parser", "deepseek_r1",
            "--quantization", "bitsandbytes",
            "--load-format", "bitsandbytes",
        ],
        "is_thinking": True,
        "user_message": "Please analyze the text above.",
    },
    "qwen3-8b": {
        "hf_id": "Qwen/Qwen3-8B",
        "vllm_extra_args": [],
        "is_thinking": True,
        "user_message": "/think",                 # triggers thinking mode in Qwen3
    },
    "gemma4-e4b": {
        "hf_id": "google/gemma-4-E4B",
        "vllm_extra_args": [],
        "is_thinking": False,
        "user_message": "Please analyze the text above.",
    },
    "llama-3.1-8b": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "vllm_extra_args": [],
        "is_thinking": False,
        "user_message": "Please analyze the text above.",
    },
    "medgemma-1.5": {
        "hf_id": "google/medgemma-4b-it",
        "vllm_extra_args": [],
        "is_thinking": False,
        "user_message": "Please analyze the text above.",
    },
}

# ---------------------------------------------------------------------------
# Dataset registry
# Each entry: short_name -> {csv_path, lang_code}
# lang_code maps to the SYSTEM_PROMPT_<LANG> variables in the prompt files
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    "Brazil": {
        "csv_path": os.path.join(BASE_DIR, "dataset", "Brazil.csv"),
        "lang_code": "BZ",
    },
    "Japan": {
        "csv_path": os.path.join(BASE_DIR, "dataset", "Japan_T.csv"),
        "lang_code": "JP",
    },
    "Spain": {
        "csv_path": os.path.join(BASE_DIR, "dataset", "Spain.csv"),
        "lang_code": "ES",
    },
    "UK": {
        "csv_path": os.path.join(BASE_DIR, "dataset", "UK_T.csv"),
        "lang_code": "EN",
    },
}

# ---------------------------------------------------------------------------
# Prompt file paths
# ---------------------------------------------------------------------------
PROMPT_FILES = {
    "demographics": os.path.join(BASE_DIR, "prompts", "demographic-extraction-system-prompt.txt"),
    "pain":         os.path.join(BASE_DIR, "prompts", "pain-extraction-system-prompt.txt"),
}

# ---------------------------------------------------------------------------
# Generation parameters
# ---------------------------------------------------------------------------
GENERATION_CONFIG = {
    "max_tokens": 1024,
    "temperature": 0.0,   # deterministic output for reproducibility
    "top_p": 1.0,
}

# ---------------------------------------------------------------------------
# vLLM server settings
# ---------------------------------------------------------------------------
VLLM_HOST = "http://localhost"
VLLM_PORT = 8000
VLLM_STARTUP_TIMEOUT = 300        # seconds to wait for server to be ready
VLLM_HEALTH_CHECK_INTERVAL = 5   # seconds between health checks

# Concurrency: number of simultaneous in-flight requests to vLLM
CONCURRENCY = 32

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(BASE_DIR, "results")
