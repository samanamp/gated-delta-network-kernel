"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import BenchmarkConfig, Solution, TraceSet

"""Main benchmark orchestration class."""



import logging
# from collections import defaultdict
# from typing import List, Set, Tuple

# from flashinfer_bench.compile import BuilderRegistry
# from flashinfer_bench.data import EvaluationStatus, Trace, TraceSet, Workload

# from flashinfer_bench.bench.runner import IsolatedRunner, PersistentRunner

logger = logging.getLogger(__name__)

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.12").apt_install("wget")
    .run_commands(
        "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get install -y nsight-compute-2025.4.1",
        'echo "export PATH=/opt/nvidia/nsight-compute/2025.4.1:$PATH" >> /etc/bash.bashrc',
    )
    .env({"PATH": "/opt/nvidia/nsight-compute/2025.4.1:/usr/local/bin:/usr/bin:/bin"})
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)


# @app.function(image=image, gpu="A100-40GB:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(solution: Solution, config: BenchmarkConfig = None) -> dict:
    """Run benchmark on Modal B200 and return results."""

    import subprocess

# ===============TEST ncu availability===================
#     # Simple CUDA kernel via PyTorch
#     script = """
# import torch

# x = torch.randn(1024, 1024, device='cuda')
# # warmup
# for _ in range(3):
#     y = torch.mm(x, x)
# torch.cuda.synchronize()

# # profiled
# y = torch.mm(x, x)
# torch.cuda.synchronize()
#     """

#     result = subprocess.run(
#         [
#             "ncu",
#             "--set", "basic",
#             "--target-processes", "all",
#             "--launch-count", "1",
#             "python3", "-c", script,
#         ],
#         capture_output=True,
#         text=True,
#         timeout=60,
#     )
#     print("STDOUT:", result.stdout[-2000:] if result.stdout else "")
#     print("STDERR:", result.stderr[-2000:] if result.stderr else "")
#     print("Return code:", result.returncode)
# ===============TEST ncu availability===================

    if config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    workload = workloads[0].workload
    # workload_str = workload.model_dump_json()
    # Workload.model_validate_json(workload_str)

    import os

    tmpdir = "/dev/shm/ncu_run"
    os.makedirs(tmpdir, exist_ok=True)

    # Write workload.json and solution.json into data-dir
    with open(os.path.join(tmpdir, "workload.json"), "w") as f:
        f.write(workload.model_dump_json())

    with open(os.path.join(tmpdir, "solution.json"), "w") as f:
        f.write(solution.model_dump_json())
    
    with open(os.path.join(tmpdir, "definition.json"), "w") as f:
        f.write(definition.model_dump_json())

    # ===============Get all kernels===================
    # result = subprocess.run(
    #     [
    #         "ncu",
    #         "--set", "none",
    #         "--target-processes", "all",
    #         "--launch-count", "100",
    #         "--print-summary", "per-kernel",
    #         "python3", "-m", "flashinfer_bench.agents._solution_runner",
    #         "--data-dir", tmpdir,
    #         "--trace-set-path", TRACE_SET_PATH,
    #     ],
    #     capture_output=True,
    #     text=True,
    #     timeout=120,
    # )
    # print(result.stdout)

    # ===============Get target kernel===================
    result = subprocess.run(
        [
            "ncu",
            "--set", "detailed",
            # "--set", "basic",
            "--target-processes", "all",
            # "--print-summary", "per-kernel",
            "--launch-count", "1",
            "--kernel-name", "gdn_kernel",
            "python3", "-m", "flashinfer_bench.agents._solution_runner",
            "--data-dir", tmpdir,
            "--trace-set-path", TRACE_SET_PATH,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    # print("STDOUT:", result.stdout[-3000:])
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr[-3000:])
    print("Return code:", result.returncode)

    return



def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()


@app.local_entrypoint()
def main():
    """Pack solution and run benchmark on Modal."""
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning benchmark on Modal B200...")
    results = run_benchmark.remote(solution)

    if not results:
        print("No results returned!")
        return

    print_results(results)
