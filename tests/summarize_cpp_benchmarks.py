"""Summarize matched Python/native CPU artifacts, optionally make a chart."""

import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("artifacts/cpp-scheduler"))
    parser.add_argument(
        "--output", type=Path, default=Path("docs/benchmarks/cpp-scheduler")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def read(name):
        return json.loads((args.input / (name + ".json")).read_text(encoding="utf-8"))

    def stage(case, key, statistic="p50"):
        return statistics.median(
            r["summary"][key][statistic] for r in case["measurements"]
        )

    result = dict(
        baseline_commit="fe7907086dfcdeabb8a1949c70b36eb0cd4ec3c6",
        cpu_metadata_only=True,
        request_retirement_separate=True,
        lifecycle_includes_native_ingestion_and_retirement=True,
        core=[],
        diagnostic=[],
        queues=[],
        decode=read("decode")["cases"],
    )
    py, cpp = read("cpu-python-core"), read("cpu-cpp-core")
    assert (
        py["request_retirement_measured_separately"]
        and cpp["request_retirement_measured_separately"]
    )
    for old, new in zip(py["cases"], cpp["cases"]):
        row = dict(
            total_tokens=old["total_tokens"],
            cached_tokens=old["cached_tokens"],
            query_tokens=old["uncached_tokens"],
            samples_per_round=old["samples_per_round"],
            rounds=old["rounds"],
            warmups=old["warmups"],
        )
        for backend, case in [("python", old), ("cpp", new)]:
            row[backend] = {
                key: dict(
                    p50=stage(case, key),
                    p95=stage(case, key, "p95"),
                    mean=stage(case, key, "mean"),
                    round_p50=[r["summary"][key]["p50"] for r in case["measurements"]],
                )
                for key in case["measurements"][0]["summary"]
            }
        row["schedule_speedup"] = stage(old, "schedule_wall_ms") / stage(
            new, "schedule_wall_ms"
        )
        row["lifecycle_speedup"] = stage(old, "lifecycle_ms") / stage(
            new, "lifecycle_ms"
        )
        row["schedule_reduction_pct"] = (1 - 1 / row["schedule_speedup"]) * 100
        result["core"].append(row)
    for old, new in zip(
        read("cpu-python-diagnostic")["cases"], read("cpu-cpp-diagnostic")["cases"]
    ):
        result["diagnostic"].append(
            dict(
                total_tokens=old["total_tokens"],
                query_tokens=old["uncached_tokens"],
                cached_tokens=old["cached_tokens"],
                pool_blocks=old["pool_blocks"],
                python_p50_ms=stage(old, "schedule_wall_ms"),
                cpp_p50_ms=stage(new, "schedule_wall_ms"),
            )
        )
    for old, new in zip(
        read("cpu-python-supplemental")["queue_probe"],
        read("cpu-cpp-supplemental")["queue_probe"],
    ):
        assert (
            old["previous_request_retirement_outside_clock"]
            and new["previous_request_retirement_outside_clock"]
        )
        assert set(old["observed_first_batch_sizes"]) == set(
            new["observed_first_batch_sizes"]
        )
        result["queues"].append(
            dict(
                waiting_requests=old["waiting_requests"],
                actual_batch_sizes=sorted(set(new["observed_first_batch_sizes"])),
                python=old["first_call_ms"],
                cpp=new["first_call_ms"],
            )
        )
    for backend in ["python", "cpp"]:
        data = read("model-" + backend + "-short")
        result.setdefault("single_gpu_model_reference_checks", {})[backend] = len(
            data["saved_reference_matches"]
        )
    result["two_gpu_100k_performance_retested"] = False
    if (args.input / "model-cpp.json").exists():
        model_py, model_cpp = read("model-python"), read("model-cpp")
        py_samples = [r for rnd in model_py["rounds"] for r in rnd]
        cpp_samples = [r for rnd in model_cpp["rounds"] for r in rnd]
        assert len(py_samples) == len(cpp_samples) == 90
        assert [r["token_ids"] for r in py_samples] == [
            r["token_ids"] for r in cpp_samples
        ]
        result["single_gpu_100k_engine"] = dict(
            single_gpu=True,
            placement="local",
            http=False,
            total_tokens=102400,
            cached_tokens=101376,
            query_tokens=1024,
            samples_per_backend=90,
            input_kind="pretokenized; fixed prefix list reused; no JSON/tokenizer",
            python=model_py["summary"],
            cpp=model_cpp["summary"],
            matching_generated_token_ids=True,
        )
    (args.output / "metrics-summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "core"}, indent=2))
    for r in result["core"]:
        print(
            r["total_tokens"],
            r["python"]["schedule_wall_ms"]["p50"],
            r["cpp"]["schedule_wall_ms"]["p50"],
            r["schedule_speedup"],
            r["lifecycle_speedup"],
        )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib to generate the optional chart.")
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout="constrained")
    xs = list(range(len(result["core"])))
    for ax, key, title in [
        (axes[0], "schedule_wall_ms", "Schedule: prefix matching + allocation"),
        (axes[1], "lifecycle_ms", "Lifecycle: includes native ingestion + cleanup"),
    ]:
        for backend, offset, color in [
            ("python", -0.18, "#3778bc"),
            ("cpp", 0.18, "#429670"),
        ]:
            values = [r[backend][key]["p50"] for r in result["core"]]
            bars = ax.bar(
                [x + offset for x in xs], values, 0.34, label=backend, color=color
            )
            ax.bar_label(
                bars, labels=[f"{v:.3f}" for v in values], fontsize=8, padding=3
            )
        ax.set(
            xticks=xs,
            xticklabels=["50Ki", "75Ki", "100Ki"],
            xlabel="Input tokens (+1Ki uncached)",
            ylabel="Median latency (ms)",
            title=title,
        )
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        ax.legend()
    pools = [r for r in result["diagnostic"] if r["pool_blocks"] > 512]
    for key, label, color in [
        ("python_p50_ms", "python", "#3778bc"),
        ("cpp_p50_ms", "cpp", "#429670"),
    ]:
        axes[2].plot(
            [r["pool_blocks"] for r in pools],
            [r[key] for r in pools],
            marker="o",
            label=label,
            color=color,
        )
    axes[2].set(
        xlabel="Metadata pool blocks (CPU only)",
        ylabel="Schedule median (ms)",
        title="100Ki input, +1Ki query: pool scaling",
    )
    axes[2].legend()
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("SwiftCache Python vs C++17 scheduler — matched CPU benchmarks")
    fig.savefig(args.output / "performance.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
