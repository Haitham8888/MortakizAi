import sys
import subprocess
from pathlib import Path

# ===============================================================================
#                                 CONFIGURATION
# ===============================================================================
# This section allows you to customize the AI server settings based on your 
# current hardware (e.g., 16GB Single Card vs. Dual H100 Setup).

# 1. Model Selection & Path:
# - Qwen 0.5B: Extremely fast. Fits perfectly in 16GB.
# - Qwen 7B: High quality. Recommended for 16GB cards.
# - Qwen 32B+: Requires H100 or multi-GPU (GPU_COUNT > 1).
MODEL_BASE_DIR = "/home/administrator/Continue/models_cache/models--Qwen--Qwen2.5-Coder-0.5B-Instruct"

# 2. Model Alias (Access Point):
# Keep as "q3" to match VS Code extension settings.
MODEL_ALIAS = "q3"

# 3. Context Length (Memory consumption increases with context):
# - 16384 (16k): RECOMMENDED for 16GB VRAM. Stable and handles most files.
# - 32768 (32k): Can be unstable on 16GB if GPU is busy.
# - 128000 (128k): Only for H100/Multi-GPU setups.
CONTEXT_LENGTH = 16384

# 4. GPU Memory Utilization (Safety Buffer):
# - 0.80: RECOMMENDED for 16GB cards. It leaves 20% VRAM headroom for the 
#   "sampler warmup" phase, preventing CUDA Out of Memory crashes at startup.
# - 0.90+: Only use on dedicated servers with no other running processes.
GPU_UTILIZATION = 0.8

# 5. Tensor Parallel (Multi-GPU setup):
# - 1: Use for a single GPU card (Standard 16GB setup).
# - 2+: Set this for H100 nodes or multi-card servers.
GPU_COUNT = 1

# ===============================================================================

def get_model_path(search_dir):
    """
    Search for a config.json file within the specified directory to identify 
    the actual model directory.
    """
    print(f"Searching for model configuration in: {search_dir}...")
    for config_path in Path(search_dir).rglob('config.json'):
        return str(config_path.parent)
    return None

def main():
    # Allow command line overrides, but use internal settings as defaults
    base_dir = sys.argv[1] if len(sys.argv) > 1 else MODEL_BASE_DIR
    model_name = sys.argv[2] if len(sys.argv) > 2 else MODEL_ALIAS
    context_length = sys.argv[3] if len(sys.argv) > 3 else str(CONTEXT_LENGTH)
    gpu_usage = sys.argv[4] if len(sys.argv) > 4 else str(GPU_UTILIZATION)
    gpu_count = str(GPU_COUNT)

    # Locate the definitive model path
    final_model_dir = get_model_path(base_dir)

    if not final_model_dir:
        print(f"Error: Could not locate a valid model directory at: {base_dir}")
        print("Please check the MODEL_BASE_DIR path in the script configuration.")
        sys.exit(1)

    print(f"--- vLLM Server Configuration ---")
    print(f"Model Path:     {final_model_dir}")
    print(f"Model Alias:    {model_name}")
    print(f"Context Length: {context_length}")
    print(f"GPU Memory:     {gpu_usage}")
    print(f"GPU Count:      {gpu_count} (Tensor Parallel)")
    print(f"----------------------------------")
    print("\nStarting vLLM server... Press Ctrl+C to stop.")

    # Construct the vLLM command
    server_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", final_model_dir,
        "--served-model-name", model_name,
        "--host", "0.0.0.0",
        "--port", "8888",
        "--max-model-len", context_length,
        "--gpu-memory-utilization", gpu_usage,
        "--tensor-parallel-size", gpu_count,
        "--trust-remote-code"
    ]

    try:
        # Launch the API server
        subprocess.run(server_cmd)
    except KeyboardInterrupt:
        print("\nStopping server. Process terminated by user.")

if __name__ == "__main__":
    main()
