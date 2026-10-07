import platform
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "M57-Jean"
OUTPUT_PATH = RESULTS_DIR / "table9_runtime_memory.csv"

STAGES = [
    ("Data normalization", ROOT / "src" / "normalization" / "memory_normalizer.py"),
    ("Temporal feature engineering", ROOT / "src" / "temporal" / "temporal_features.py"),
    ("Temporal anomaly detection", ROOT / "src" / "anomaly" / "temporal_isolation_forest.py"),
    ("Knowledge graph construction", ROOT / "src" / "graph" / "temporal_graph.py"),
    ("Graph anomaly detection", ROOT / "src" / "anomaly" / "graph_isolation_forest.py"),
    ("SHAP explanation", ROOT / "src" / "explainability" / "temporal_shap.py"),
]


def measure_stage(name, script):
    print(f"\n=== {name} ===")
    print(f"Script: {script}")

    start = time.perf_counter()

    process = subprocess.Popen(
        [sys.executable, str(script)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    ps_process = psutil.Process(process.pid)
    peak_memory = 0

    while process.poll() is None:
        try:
            memory = ps_process.memory_info().rss
            peak_memory = max(peak_memory, memory)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        time.sleep(0.05)

    return_code = process.wait()
    elapsed = time.perf_counter() - start

    result = {
        "stage": name,
        "script": str(script.relative_to(ROOT)),
        "runtime_seconds": round(elapsed, 3),
        "peak_memory_mb": round(peak_memory / (1024 * 1024), 2),
        "return_code": return_code,
        "status": "PASS" if return_code == 0 else "FAIL",
    }

    print(
        f"Runtime: {result['runtime_seconds']} s | "
        f"Peak memory: {result['peak_memory_mb']} MB | "
        f"Return code: {return_code} | "
        f"Status: {result['status']}"
    )

    return result


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("ForensiXplain Table 9 Runtime and Memory Benchmark")
    print("=" * 60)
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU: {platform.processor()}")
    print(f"psutil: {psutil.__version__}")

    results = []

    for name, script in STAGES:
        if not script.exists():
            print(f"\nMISSING: {script}")
            results.append({
                "stage": name,
                "script": str(script.relative_to(ROOT)),
                "runtime_seconds": None,
                "peak_memory_mb": None,
                "return_code": None,
                "status": "MISSING",
            })
            continue

        results.append(measure_stage(name, script))

    df = pd.DataFrame(results)

    df["python_version"] = platform.python_version()
    df["platform"] = platform.platform()
    df["cpu"] = platform.processor()
    df["psutil_version"] = psutil.__version__

    df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 60)
    print("TABLE 9 BENCHMARK COMPLETE")
    print(f"Output: {OUTPUT_PATH}")
    print("\nResults:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
