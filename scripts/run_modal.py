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
from collections import defaultdict
from typing import List, Set, Tuple

from flashinfer_bench.compile import BuilderRegistry
from flashinfer_bench.data import EvaluationStatus, Trace, TraceSet, Workload

from flashinfer_bench.bench.runner import IsolatedRunner, PersistentRunner

logger = logging.getLogger(__name__)


class Benchmark:
    """Benchmark execution engine for FlashInfer-Bench kernel solutions.

    It runs the solutions against the workloads, and stores the results back to the trace set.
    This class manages the GPU resources and will allocate multiple processes to run the solutions
    in parallel.
    """

    def __init__(self, trace_set: TraceSet, config: BenchmarkConfig = None) -> None:
        """Initialize the Benchmark with a TraceSet and configuration.

        Parameters
        ----------
        trace_set : TraceSet
            The dataset containing definitions, solutions, and workloads to benchmark.
        config : BenchmarkConfig, optional
            Configuration parameters for benchmark execution, by default BenchmarkConfig().

        Raises
        ------
        ValueError
            If log_level is not one of the valid logging levels.
        """
        # Dataset and configuration
        self._trace_set = trace_set
        self._config = config if config is not None else BenchmarkConfig()

        # Setup registry
        self._registry = BuilderRegistry.get_instance()

        # Create runner
        if self._config.use_isolated_runner:
            self._runner = IsolatedRunner(self._config.log_dir)
        else:
            self._runner = PersistentRunner(self._config.log_dir)

    def get_trace_set(self) -> TraceSet:
        """Get the TraceSet associated with this benchmark.

        Returns
        -------
        TraceSet
            The TraceSet containing definitions, solutions, and workloads.
        """
        return self._trace_set

    def run_all(self, dump_traces: bool = True, resume: bool = False) -> TraceSet:
        """Run benchmark for all solutions in the trace set.

        Parameters
        ----------
        dump_traces : bool, optional
            If True, store traces to the trace set and in the disk.
        resume : bool, optional
            If True, skip solutions that have already been evaluated for each workload.

        Returns
        -------
        TraceSet
            A new TraceSet containing the original data plus the execution traces
            from this benchmark run. The traces are organized by definition name.
        """
        result_traces: List[Trace] = []

        definitions_to_run = self._trace_set.definitions.items()
        if self._config.definitions is not None:
            definitions_to_run = [
                (name, definition)
                for name, definition in definitions_to_run
                if name in self._config.definitions
            ]
            provided_defs = set(self._config.definitions)
            existing_defs = set(self._trace_set.definitions.keys())
            missing_defs = provided_defs - existing_defs
            if missing_defs:
                logger.warning(f"Definitions not found in trace set: {sorted(missing_defs)}")

        for def_name, definition in definitions_to_run:
            sols = self._trace_set.solutions.get(def_name, [])
            if not sols:
                logger.warning(f"No solutions found for def={def_name}, skipping definition")
                continue

            if self._config.solutions is not None:
                sols = [s for s in sols if s.name in self._config.solutions]
                if not sols:
                    logger.info(f"No matching solutions for def={def_name} after filtering")
                    continue

            logger.info(f"Processing definition: {def_name} with {len(sols)} solutions")

            existing_traces: Set[Tuple[str, str]] = set()  # (workload_uuid, solution_name)
            if resume:
                existing_def_traces = self._trace_set.traces.get(def_name, [])
                for trace in existing_def_traces:
                    if trace.solution and trace.evaluation:
                        existing_traces.add((trace.workload.uuid, trace.solution))
                if existing_traces:
                    logger.info(f"Found {len(existing_traces)} existing traces for def={def_name}")

            workloads = self._trace_set.workloads.get(def_name, [])
            def_traces: List[Trace] = []

            for wl_trace in workloads:
                workload = wl_trace.workload

                sols_to_run = sols
                if resume:
                    sols_to_run = [
                        s for s in sols if (workload.uuid, s.name) not in existing_traces
                    ]

                if not sols_to_run:
                    logger.info(f"All solutions already evaluated for workload {workload.uuid}")
                    continue
                
                
                try:
                    results = self._runner.run_workload(
                        definition, workload, sols_to_run, self._config, self._trace_set.root
                    )
                except RuntimeError as e:
                    logger.error(f"Failed to run workload {workload.uuid}: {e}", exc_info=True)
                    continue

                for sol_name, ev in results.items():
                    trace = Trace(
                        definition=def_name, workload=workload, solution=sol_name, evaluation=ev
                    )

                    result_traces.append(trace)
                    def_traces.append(trace)
                    print(results[sol_name].log)
                    if ev.status == EvaluationStatus.PASSED:
                        logger.info(
                            f"Solution '{sol_name}' for workload {workload.uuid}: PASSED with "
                            f"{ev.performance.speedup_factor:.2f}x speedup"
                        )
                    else:
                        logger.warning(
                            f"Solution '{sol_name}' for workload {workload.uuid}: {ev.status.value}"
                        )

            if dump_traces and def_traces:
                self._trace_set.add_traces(def_traces)
                logger.info(f"Saved {len(def_traces)} traces for definition {def_name}")

        traces_by_def = defaultdict(list)
        for trace in result_traces:
            traces_by_def[trace.definition].append(trace)

        if self._config.solutions is not None:
            provided_sols = set(self._config.solutions)
            existing_sols = set()
            for sols_list in self._trace_set.solutions.values():
                existing_sols.update(s.name for s in sols_list)
            missing_sols = provided_sols - existing_sols
            if missing_sols:
                logger.warning(f"Solutions not found in trace set: {sorted(missing_sols)}")

        # Create a new TraceSet with the results
        result_trace_set = TraceSet(
            root=self._trace_set.root,
            definitions=self._trace_set.definitions.copy(),
            solutions=self._trace_set.solutions.copy(),
            workloads=self._trace_set.workloads.copy(),
            traces=dict(traces_by_def),
        )

        return result_trace_set

    def close(self) -> None:
        """Release all resources held by the benchmark runner."""
        self._runner.close()

# ======================================================
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

    if config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    # print(TRACE_SET_PATH)
    # print(trace_set.workloads)
    # target_definition = "gdn_decode_qk4_v8_d128_k_last"

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")


    # print(workloads[0].model_dump_json(indent=2))

    from flashinfer_bench.agents import flashinfer_bench_run_ncu
    # workload_str="/data/workloads/gdn/gdn_decode_qk4_v8_d128_k_last.jsonl"
    # path2 = Path(workload_str)
    # print(path2.read_text())
    # workload = Workload.model_validate_json(workloads[0])
    workload = workloads[0].workload
    workload_str = workload.model_dump_json()
    Workload.model_validate_json(workload_str)

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

    # Check what the runner expects
    help_result = subprocess.run(
        ["python3", "-m", "flashinfer_bench.agents._solution_runner", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    print("Runner help:", help_result.stdout)

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

    # Run under ncu
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

    # output = flashinfer_bench_run_ncu(
    #     trace_set_path=TRACE_SET_PATH,
    #     solution=solution,
    #     workload=workload,
    #     set="detailed",
    #     page="details",
    #     # kernel_name=".*",
    #     timeout=120,
    # )
    # print(output)
    # print("donne@@@")

    return

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][trace.workload.uuid] = entry

    return results


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
