import sys
import subprocess
import os
import logging
from pathlib import Path
from datetime import datetime

# ==============================================================================
#  CONFIGURATION
# ==============================================================================
MODEL_BASE_DIR = "/home/administrator/Continue/models_cache/models--Qwen--Qwen2.5-Coder-0.5B-Instruct"
MODEL_ALIAS = "q3"
CONTEXT_LENGTH = "16384"
GPU_UTILIZATION = "0.8"
GPU_COUNT = "auto"  # "auto" = detect via nvidia-smi, or set "1", "2", etc.
PORT = "8888"
LOG_FILE = "vllm_server.log"
# ==============================================================================


def setup_logging(log_path):
    """Configure logging to both file and console."""
    logger = logging.getLogger("vllm_launcher")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def detect_gpu_count(log):
    """
    Detect available GPU count via nvidia-smi only.
    We intentionally avoid torch.cuda.device_count() because it initializes
    CUDA prematurely, which breaks vLLM's multiprocess executor during
    tensor-parallel startup (gpu_worker.py:232 init_device).
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            count = len(result.stdout.strip().splitlines())
            log.info("Detected %d GPU(s) via nvidia-smi.", count)
            return count
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.warning("nvidia-smi failed: %s", e)

    log.warning("Could not detect GPUs. Defaulting to 1.")
    return 1


def get_gpu_info(log):
    """Return GPU details string for logging."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        log.warning("Failed to query GPU info: %s", e)
    return "N/A"


def get_model_path(search_dir, log):
    """Locate the model snapshot directory containing config.json."""
    log.info("Searching for model in: %s", search_dir)
    for config_path in Path(search_dir).rglob("config.json"):
        model_dir = str(config_path.parent)
        log.info("Found model at: %s", model_dir)
        return model_dir
    return None


def main():
    # -- Setup logging --
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, LOG_FILE)
    log = setup_logging(log_path)

    log.info("=" * 60)
    log.info("vLLM Launcher started at %s", datetime.now().isoformat())
    log.info("=" * 60)

    # -- Parse arguments --
    base_dir = sys.argv[1] if len(sys.argv) > 1 else MODEL_BASE_DIR
    model_name = sys.argv[2] if len(sys.argv) > 2 else MODEL_ALIAS
    context_length = sys.argv[3] if len(sys.argv) > 3 else CONTEXT_LENGTH
    gpu_usage = sys.argv[4] if len(sys.argv) > 4 else GPU_UTILIZATION
    gpu_count_cfg = sys.argv[5] if len(sys.argv) > 5 else GPU_COUNT

    # -- Detect GPU count --
    detected = detect_gpu_count(log)

    if gpu_count_cfg.lower() == "auto":
        gpu_count = detected
    else:
        requested = int(gpu_count_cfg)
        if requested > detected:
            log.warning(
                "Requested %d GPUs but only %d available. Clamping.",
                requested, detected
            )
            gpu_count = detected
        else:
            gpu_count = requested

    # -- Locate model --
    final_model_dir = get_model_path(base_dir, log)
    if not final_model_dir:
        log.error("No valid model found at: %s", base_dir)
        sys.exit(1)

    # -- Log GPU info --
    gpu_info = get_gpu_info(log)

    log.info("-" * 60)
    log.info("Model Path      : %s", final_model_dir)
    log.info("Model Alias     : %s", model_name)
    log.info("Context Length  : %s", context_length)
    log.info("GPU Utilization : %s", gpu_usage)
    log.info("Tensor Parallel : %d", gpu_count)
    log.info("Port            : %s", PORT)
    log.info("-" * 60)
    for line in gpu_info.splitlines():
        log.info("GPU: %s", line.strip())
    log.info("-" * 60)

    # ==================================================================
    #  Environment variables to fix NCCL errors on VM with multi-GPU.
    #
    #  Root cause from the traceback:
    #    pynccl_wrapper.py:373 -> RuntimeError: NCCL error: unhandled cuda error
    #    pynccl.py:139         -> ncclCommInitRank failed
    #
    #  All subsequent errors (WorkerProc failed, EngineCore failed,
    #  Engine core initialization failed, etc.) are consequences of
    #  this single root cause.
    # ==================================================================
    env = os.environ.copy()

    # Fix: Disable NCCL cuMem allocator (known bug causes "unhandled cuda error")
    env["NCCL_CUMEM_ENABLE"] = "0"

    # Fix: Disable P2P (not available in VM environments)
    env["NCCL_P2P_DISABLE"] = "1"

    # Fix: Disable InfiniBand (not available)
    env["NCCL_IB_DISABLE"] = "1"

    # Fix: Disable GPU Direct RDMA (not available in VM)
    env["NCCL_NET_GDR_LEVEL"] = "0"

    # Fix: Prevent NCCL from erroring on disabled P2P
    env["NCCL_IGNORE_DISABLED_P2P"] = "1"

    # Fix: Enable shared memory transport (fallback when P2P is off)
    env["NCCL_SHM_DISABLE"] = "0"

    # Fix: Use localhost for NCCL socket (single-machine multi-GPU)
    env["NCCL_SOCKET_IFNAME"] = "lo"

    # Fix: Blocking wait prevents silent hangs on NCCL failures
    env["TORCH_NCCL_BLOCKING_WAIT"] = "1"

    # Fix: Use spawn (fork + CUDA = broken multiprocess executor)
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    # Fix: Suppress "Reducing Torch parallelism from N to 1" warning
    env["OMP_NUM_THREADS"] = "1"

    # Fix: Explicit GPU assignment to prevent device index conflicts
    gpu_ids = ",".join(str(i) for i in range(gpu_count))
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids

    # Logging level for NCCL (set to INFO or TRACE for debugging)
    env["NCCL_DEBUG"] = "WARN"

    # -- Log all relevant env vars --
    log.info("Environment variables:")
    for key in sorted(env.keys()):
        if any(key.startswith(p) for p in
               ["NCCL_", "CUDA_", "VLLM_", "TORCH_NCCL", "OMP_"]):
            log.info("  %s=%s", key, env[key])

    # ==================================================================
    #  Build server command
    # ==================================================================
    server_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(final_model_dir),
        "--served-model-name", str(model_name),
        "--host", "0.0.0.0",
        "--port", PORT,
        "--max-model-len", str(context_length),
        "--gpu-memory-utilization", str(gpu_usage),
        "--tensor-parallel-size", str(gpu_count),
        "--trust-remote-code",
        "--enforce-eager",
        "--disable-custom-all-reduce",
    ]

    log.info("Server command: %s", " ".join(server_cmd))
    log.info("Starting vLLM server. Press Ctrl+C to stop.")

    # -- Run --
    try:
        process = subprocess.run(server_cmd, env=env)
        if process.returncode != 0:
            log.error("vLLM exited with code %d", process.returncode)
            sys.exit(process.returncode)
    except KeyboardInterrupt:
        log.info("Server stopped by user.")
    except Exception as e:
        log.exception("Unexpected error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()