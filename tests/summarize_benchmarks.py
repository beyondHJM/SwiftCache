import json, statistics, pathlib, math, argparse

parser = argparse.ArgumentParser(
    description="Summarize the raw SwiftCache phase2 artifacts and optionally plot."
)
parser.add_argument(
    "--input", type=pathlib.Path, default=pathlib.Path("artifacts/phase2")
)
parser.add_argument(
    "--output", type=pathlib.Path, default=pathlib.Path("docs/benchmarks")
)
args = parser.parse_args()
p = args.input
out = args.output
out.mkdir(exist_ok=True)


def read(name):
    return json.loads((p / (name + ".json")).read_text())


def stats(xs):
    ordered = sorted(xs)
    return {
        "p50": statistics.median(xs),
        "p95": ordered[math.ceil(0.95 * len(xs)) - 1],
        "max": max(xs),
    }


summary = {"cpu": [], "gpu": [], "api": {}, "concurrency": {}, "suffix": {}}
for old, new in zip(
    read("cpu-baseline-core")["cases"], read("cpu-final-core")["cases"]
):

    def cstat(c, key):
        return statistics.median(m["summary"][key]["p50"] for m in c["measurements"])

    a, b = cstat(old, "schedule_wall_ms"), cstat(new, "schedule_wall_ms")
    summary["cpu"].append(
        dict(
            tokens=new["total_tokens"],
            baseline_ms=a,
            final_ms=b,
            reduction_pct=(a - b) / a * 100,
            final_p95_median_round_ms=statistics.median(
                m["summary"]["schedule_wall_ms"]["p95"] for m in new["measurements"]
            ),
            prefix_allocate_ms=cstat(new, "prefix_allocate_ms"),
            postprocess_ms=cstat(new, "postprocess_ms"),
            construct_ms=cstat(new, "construct_ms"),
        )
    )
summary["cpu_diagnostic"] = []
for old, new in zip(
    read("cpu-baseline-diagnostic")["cases"], read("cpu-final-diagnostic")["cases"]
):
    summary["cpu_diagnostic"].append(
        dict(
            tokens=new["total_tokens"],
            pool_blocks=new["pool_blocks"],
            uncached_tokens=new["uncached_tokens"],
            baseline_ms=old["median_round_p50_ms"],
            final_ms=new["median_round_p50_ms"],
        )
    )
summary["cpu_supplemental"] = {
    variant: read("cpu-" + variant + "-supplemental")
    for variant in ["baseline", "final"]
}
for data in summary["cpu_supplemental"].values():
    data["cases"] = [
        {k: v for k, v in c.items() if k != "measurements"} for c in data["cases"]
    ]
for placement in ["local", "remote"]:
    d = read("gpu-" + placement)
    for case in d["cases"]:
        raw = [s for rnd in case["rounds"] for s in rnd["samples"]]
        metrics = [r["metrics"]["batches"][0] for r in raw]
        row = dict(
            placement=placement,
            total_tokens=case["total_tokens"],
            cached_tokens=case["cached_tokens"],
            samples=len(raw),
        )
        for key in [
            "schedule_wall_ms",
            "prefix_match_allocate_ms",
            "model_gpu_span_ms",
            "model_and_sample_wall_ms",
            "kv_load_span_sum_ms",
            "kv_store_span_sum_ms",
            "prepare_wall_ms",
        ]:
            row[key] = stats([m[key] for m in metrics])
        row["client_wall_ms"] = stats([r["client_wall_ms"] for r in raw])
        summary["gpu"].append(row)
    summary["api"][placement] = read("openai-" + placement + "-long")["summary"]
    summary["concurrency"][placement] = d["concurrency"]
    summary["suffix"][placement] = [
        {k: v for k, v in c.items() if k != "raw"}
        for c in read("gpu-" + placement + "-suffix")["cases"]
    ]
    f = p / ("arrivals-" + placement + ".json")
    if f.exists():
        summary.setdefault("arrivals", {})[placement] = [
            {k: v for k, v in c.items() if k != "raw"}
            for c in json.loads(f.read_text())["cases"]
        ]
(out / "metrics-summary.json").write_text(json.dumps(summary, indent=2) + "\n", newline="\n")
print(json.dumps(summary, indent=2))
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    xs = [c["tokens"] // 1024 for c in summary["cpu"]]
    for key, label in [("baseline_ms", "Baseline"), ("final_ms", "Optimized")]:
        axes[0].plot(xs, [c[key] for c in summary["cpu"]], marker="o", label=label)
    axes[0].set(
        xlabel="Input tokens (Ki)",
        ylabel="Scheduler median (ms)",
        title="CPU: ~99% prefix hit, +1Ki query",
    )
    axes[0].legend()
    for placement in ["local", "remote"]:
        rows = [r for r in summary["gpu"] if r["placement"] == placement]
        axes[1].plot(
            xs,
            [r["model_gpu_span_ms"]["p50"] for r in rows],
            marker="o",
            label=placement + " KV",
        )
    axes[1].set(
        xlabel="Input tokens (Ki)",
        ylabel="GPU model span (ms)",
        title="Real Qwen3-8B on A100 PCIe",
    )
    axes[1].legend()
    labels = ["Scheduler", "Text preprocessing", "API first content"]
    a = summary["api"]["local"]
    vals = [
        next(
            r
            for r in summary["gpu"]
            if r["placement"] == "local" and r["total_tokens"] == 102400
        )["schedule_wall_ms"]["p50"],
        a["server_text_preprocess_ms"]["p50"],
        a["first_content_ms"]["p50"],
    ]
    axes[2].bar(labels, vals, color=["#3979c5", "#dd9440", "#4b9971"])
    axes[2].set(ylabel="Median latency (ms)", title="100Ki context, local KV")
    axes[2].tick_params(axis="x", labelsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("SwiftCache: measured scheduling, GPU and API costs (2026-11-08)")
    fig.savefig(out / "performance.png", dpi=180)
    plt.close(fig)
except ImportError as exc:
    print("Plot dependency unavailable:", exc)
