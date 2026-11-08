# Qwen3-8B 最小 Aqua baseline 验证

验证日期：2026-11-08。服务器：Cloud-A100-233。
工作目录：`/root/hujianmin/SwiftCache`，分支：`codex/aqua-baseline`。
机器可读结果见 [results/validation-summary.json](results/validation-summary.json)。

## 配置

使用物理 GPU **0、2**，各加载一份完整 BF16 Qwen3-8B；两张卡均为 A100 PCIe 40GB，
支持 CUDA peer access，没有 NVLink。consumer 和 producer 的计算、迁移串行执行。

| 项目 | 本次设置 |
| --- | --- |
| 模型 | `/root/llm-resource/Models/Qwen3-8B` |
| Python / PyTorch / Transformers | 3.11.15 / 2.6.0+cu124 / 4.51.0 |
| 注意力实现 | HF SDPA |
| producer 出借额度 | 512 MiB |
| CPU backing 上限 | 2048 MiB |
| prefill chunk / token budget / decode slots | 128 / 256 / 1 |
| context 上限 | 8192 tokens |
| OpenAI Python SDK | 3.46.0 |

## 通过的检查

| 检查 | 结果 |
| --- | --- |
| 数学、中文翻译、代码提取 | 3 个请求均与同权重 HF `generate(do_sample=False)` 的 token IDs 完全一致；分别得到 `4`、`缓存工作正常。`、`SILVER-731` |
| consumer 并发突发 | 4 请求，每条 346 个输入 tokens、12 个输出 tokens；抢占/恢复后仍与各自的独立 HF 输出一致 |
| 前缀命中 | 命中 512 个 tokens，补算 22 个输入 tokens；输出 `ALPHA-42`，与完整重算一致 |
| 手动回收与重新出借 | 3 轮 GPU→pinned DRAM→GPU；迁移前后张量逐元素一致，回收后前缀仍可正确恢复 |
| 自动回收 | producer 真实提交 10 个推理请求，触发高水位回收；先迁移并确认，再运行 producer；consumer 从 DRAM 恢复，producer 输出也与独立 HF 一致 |
| 额度耗尽 | 新张量落到 pinned CPU，未在回收窗口继续借出 GPU 容量 |
| 取消 | 取消活跃请求后，其余请求正常完成，请求快照全部释放 |
| HTTP | 官方 OpenAI SDK 对两个模型别名、Chat/Completions、缓存 usage 和 4 请求异步并发调用通过；不支持的参数返回 400 |
| 正常退出 | allocations、providers、mappings 全部为空，coordinator 退出 |
| 启动中途 SIGTERM | 启动器正常退出，coordinator 子进程退出、端口释放 |
| CPU 单元测试 | 5 passed；覆盖线程并发、1000 次随机操作、回收确认顺序、共享租约释放及 CFS 分区 |

完整 GPU 轨迹记录了 95 次抢占、94 次恢复、7 次从 DRAM 恢复，
97 次写入 peer GPU、3 次直接写入 DRAM，以及 4 组完成确认的回收/返还。
其中一个 `failed` 事件对应有意取消的请求。

测试结束后 GPU 0、2 均恢复为约 4 MiB 占用，没有保留推理服务。

## 小样本耗时

4 请求 consumer 突发的 TTFT 为 1512.82、1635.75、1750.11、1865.61 ms，
中位数 **1692.93 ms**；请求总耗时中位数 **6826.61 ms**。
这些数据包含本版串行 forward、完整 KV 打包/迁移、同步和调度开销。
单次小样本仅用于确认程序可以完成工作，不能据此计算稳定 p99、吞吐提升或与原论文的性能差距。

## 复现与记录

在仓库根目录执行：

```bash
.venv/bin/python -m pytest -q baselines/aqua/tests
.venv/bin/python -m baselines.aqua.run --gpus 0,2 --scenario validate \
  --output artifacts/aqua/validation.json
```

HTTP 需要按 [README](README.md) 先用 `--prefix-file` 启动服务，再执行：

```bash
.venv/bin/python baselines/aqua/tests/validate_http.py \
  --prefix-file artifacts/aqua/http-prefix.json \
  --output artifacts/aqua/http-validation.json
```

原始 JSON、事件轨迹、环境、服务日志和退出账本保存在服务器的 `artifacts/aqua/`，
不将原始大日志和下载的上游仓库提交到 Git。
结果摘要记录原始文件的 SHA-256，便于检查归档是否一致。
上游 Aqua 的 9 个来源文件已核对原始哈希；vendor 仅调整包路径与空白，许可证完整保留。

这版实现固定两卡配对和出借额度，复现的是 Aqua 的可回收内存借用与基本调度机制。
没有原论文的 placer/profiler、优化 vLLM batch 或 FlexGen 权重流式执行。
用于机制对照时，应标为 **Aqua-minimal-HF/Qwen3-8B**，并标注与 SwiftCache 的执行内核差异。
