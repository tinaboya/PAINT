#!/usr/bin/env bash
# =============================================================================
# PAINT open-source-model inference — local setup
# =============================================================================
# Installs dependencies, authenticates with HuggingFace, and runs a sanity
# check. Intended to be run locally on a machine with a CUDA GPU (the 30B
# models are served with vLLM + 4-bit bitsandbytes quantization).
#
# Usage:
#   bash setup.sh
#   python run_inference.py            # run all models on every dataset
#   python run_inference.py --dry-run  # verify config without loading models
# =============================================================================

set -euo pipefail

echo "=== PAINT inference setup ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no CUDA GPU detected')"
echo ""

# ---------------------------------------------------------------------------
# 1. Python dependencies
# ---------------------------------------------------------------------------
echo "--- Installing Python dependencies ---"
pip install --upgrade pip
pip install \
  "vllm>=0.6.0" \
  "transformers>=4.45.0" \
  "huggingface_hub>=0.24" \
  "accelerate>=0.30" \
  "aiohttp>=3.9" \
  "pandas>=2.0" \
  "tqdm>=4.66"

# ---------------------------------------------------------------------------
# 2. HuggingFace authentication (needed for gated models, e.g. Gemma)
# ---------------------------------------------------------------------------
echo ""
echo "--- HuggingFace authentication ---"
echo "Some models are gated; you need a token from https://huggingface.co/settings/tokens"
echo ""
if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set. Either export it before running inference:"
  echo "  export HF_TOKEN=hf_your_token_here"
  echo "or log in interactively:"
  echo "  huggingface-cli login"
else
  echo "HF_TOKEN is set. Logging in…"
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
  echo "HuggingFace login successful."
fi

# ---------------------------------------------------------------------------
# 3. (Optional) pre-download model weights to separate download from inference
# ---------------------------------------------------------------------------
echo ""
echo "--- Optional: pre-download model weights ---"
echo "  huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
echo "  huggingface-cli download Qwen/Qwen3-8B"
echo "  huggingface-cli download meta-llama/Llama-3.1-8B-Instruct"
echo "  huggingface-cli download google/gemma-4-E4B"
echo "  huggingface-cli download google/medgemma-4b-it"
echo "Weights are cached under \$HF_HOME (default ~/.cache/huggingface/hub/)."

# ---------------------------------------------------------------------------
# 4. Results directory
# ---------------------------------------------------------------------------
mkdir -p results
echo ""
echo "--- Results directory ready: $(pwd)/results ---"

# ---------------------------------------------------------------------------
# 5. Sanity check
# ---------------------------------------------------------------------------
echo ""
echo "--- Sanity checks ---"
python - <<'EOF'
import vllm, transformers, aiohttp, pandas, tqdm
print(f"vllm         {vllm.__version__}")
print(f"transformers {transformers.__version__}")
print(f"aiohttp      {aiohttp.__version__}")
print(f"pandas       {pandas.__version__}")
print(f"tqdm         {tqdm.__version__}")
print("All imports OK.")
EOF

echo ""
echo "=== Setup complete ==="
echo "Run inference with:"
echo "  python run_inference.py                       # all 5 models, all datasets"
echo "  python run_inference.py --model deepseek-r1-qwen-32b"
echo "  python run_inference.py --model gemma4-e4b --dataset Brazil"
echo "  python run_inference.py --dry-run             # verify config only"
