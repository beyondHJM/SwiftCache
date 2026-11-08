# Aqua minimal baseline for Qwen3-8B

这是 ASPLOS 2025 [Aqua](https://arxiv.org/html/2407.21255v3) 的最小可运行适配版。
使用官方 Aqua Tensor/manager 保存推理状态，以 Transformers 4.51 的原生 Qwen3 实现执行模型。
本目录不调用 SwiftCache 的 scheduler、分配器、扩缩容窗口或 Layer Stream Cache。

代码来源、版本和许可见 [THIRD_PARTY.md](THIRD_PARTY.md) 与 [upstream.lock.json](upstream.lock.json)。
双卡实测配置、通过的检查与耗时见 [VALIDATION.md](VALIDATION.md)。

## 实现范围

- 两张 GPU 固定配对；默认各加载一份完整 BF16 Qwen3-8B，分别作为 consumer 和 producer。
- consumer 的非活跃 KV 聚合为连续 Aqua Tensor，优先存入 producer GPU；额度不足时存入 pinned CPU DRAM。
- producer 请求回收后，拒绝新的借用；consumer 只在推理迭代边界迁移 GPU 张量到 DRAM，释放物理引用后确认。
- producer 等待所有确认后使用回收的额度。低负载时重新提供额度，CPU 快照可重新迁回 producer GPU。
- producer 的队列阈值沿用公开 LLM informer 的高水位 10、低水位 2；本版等本地活跃 KV 释放后重新出借。
- 消费者采用论文 §7 的 prefill/decode 分区与分阶段最少工作量选择；无远端容量、快照落到 DRAM 时退回 FCFS。
- 前缀缓存使用显式 ID 和完整 token 前缀校验，保存完整 KV 与 greedy 下一 token，回收后可从 DRAM 恢复。
- 取消与完成均释放请求快照；主机 swap 上限不足时保留当前请求继续执行，不丢弃 KV 强行抢占。

为了最小化实现，逻辑 batch 中的 HF forward **逐条执行**，consumer 每次只驻留一个请求的活跃 KV。
producer、consumer 的 forward 与迁移也由同一执行线程串行调度，以明确推理安全边界。
没有原 vLLM 的 CUDA 批处理/gather-scatter 优化，也没有 Aqua-profiler、Gurobi Aqua-placer 或 FlexGen layer paging。
配对、出借额度与工作集上限由命令行配置。用于机制与功能 baseline，不将本实现的绝对吞吐称为原论文 Aqua 的复现结果。

## 环境

在 SwiftCache 仓库根目录使用已有 CUDA `.venv`：

```bash
.venv/bin/python -m pip install -r baselines/aqua/requirements.txt
```

已验证 Python 3.11、PyTorch 2.6.0+cu124、Transformers 4.51.0。
不需要重新编译 vLLM、修改 Qwen 权重或编译 SwiftCache C++ scheduler。
需要两张有足够空闲显存的 CUDA GPU；验证使用 GPU 0、2（A100 PCIe 40GB），支持 peer access，没有 NVLink。

## 最小运行与验证

```bash
cd /root/hujianmin/SwiftCache
.venv/bin/python -m baselines.aqua.run \
  --model /root/llm-resource/Models/Qwen3-8B \
  --gpus 0,2 \
  --scenario validate \
  --output artifacts/aqua/validation.json
```

验证运行会加载两个 Qwen3-8B，比较 HF greedy 输出，构造抢占、前缀命中、三轮 GPU→DRAM→GPU 回收/返还、
CPU fallback、producer 真实 10 请求突发与取消路径。结果和事件序列保存到 JSON。
退出时 `.final-ledger.json` 应没有任何 allocation 或 provider。

只生成一条请求：

```bash
.venv/bin/python -m baselines.aqua.run --gpus 0,2 \
  --scenario generate --prompt 'What is 2 + 2? Answer only with the number.' \
  --max-tokens 16 --output artifacts/aqua/generate.json
```

默认两份模型用于 producer/consumer 对照。添加 `--no-producer-model` 可将第二张卡仅作为内存 provider，
该模式不代表“双模型共置”实验。

## 作为服务运行

```bash
.venv/bin/python -m baselines.aqua.run \
  --model /root/llm-resource/Models/Qwen3-8B --gpus 0,2 \
  --scenario serve --port 18100 --coordinator-port 18101 \
  --output artifacts/aqua/server.json
```

启动器自动管理自己的 coordinator 子进程，服务监听 `127.0.0.1`。
访问 `/health` 判断就绪，`/metrics` 查看队列、存储位置、回收与迁移事件。
`Ctrl+C` 或向启动器发送 `SIGTERM` 会关闭推理循环、释放快照和停止 coordinator。

最小 OpenAI 风格接口支持模型列表、非流式文本 Chat Completions 和 Completions：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:18100/v1", api_key="local")
result = client.chat.completions.create(
    model="qwen3-8b-consumer",
    messages=[{"role": "user", "content": "What is 2 + 2?"}],
    temperature=0,
    max_tokens=16,
)
print(result.choices[0].message.content)
```

另一个别名为 `qwen3-8b-producer`。同时提交 producer 请求可触发真实队列压力回收。
仅支持 greedy `temperature=0, top_p=1, n=1`；stream、tools、结构化输出等未实现选项返回错误。
可用 `max_tokens` 或 `max_completion_tokens`，不能同时指定。此最小服务没有认证，供本机实验使用。

## 前缀缓存

Python 工作负载可在提交请求/启动后台循环前调用 `engine.cache_prefix(token_ids, "prefix-id")`，
再用 `engine.submit(full_token_ids, max_tokens=32, prefix_id="prefix-id")`。
请求输入必须确实以这些 token 开头，不会按字符串猜测或截断长度。

服务也可通过启动参数预热前缀：

```json
[{"id": "document", "token_ids": [1000, 1001, 1002]}]
```

将真实前缀 token IDs 写入 JSON，再添加 `--prefix-file /path/to/prefixes.json` 启动服务。
也接受 `{"id":"document","prompt":"prefix text"}`，由当前 tokenizer 编码。
通过补全接口传完整 token IDs，避免文本拼接后 tokenizer 边界变化：

```python
result = client.completions.create(
    model="qwen3-8b-consumer",
    prompt=prefix_ids + suffix_ids,
    max_tokens=16,
    temperature=0,
    extra_body={"prefix_id": "document"},
)
print(result.usage.prompt_tokens_details.cached_tokens)
```

## 参数与边界

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `--gpus` | `0,2` | consumer、producer 的物理 GPU ID；两个 ID 必须不同 |
| `--offer-mib` | 512 | producer 提供的 Aqua 张量额度 |
| `--host-swap-mib` | 2048 | 所有存活快照的 pinned CPU backing 上限 |
| `--chunk-size` | 128 | 单次 prefill chunk 上限 |
| `--token-budget` | 256 | 每次逻辑 batch 的 token 预算 |
| `--decode-slots` | 1 | 逻辑 batch 中预留的 decode slots |
| `--max-model-len` | 8192 | 输入与输出的总长度上限，默认受原模型配置约束 |
| `--ignore-model-max-length` | 关闭 | 显式覆盖原上下文上限，不修改 RoPE scaling |

活跃请求的完整 KV 必须能放入 consumer GPU；本版没有 LSC 或逐层执行，因此长度覆盖不保证物理容量或长文本准确率。
大前缀还需相应提高 CPU swap 与 provider 额度，不能沿用默认 512 MiB 宣称已测 100K。
官方 Aqua 维护 CPU backing；本实现的 KV 写入使用选中的 GPU backing 做 GPU 间 copy，CPU fallback 才写 DRAM。
动态分配、backing 初始化、打包和全局同步均未做性能优化。

批量回迁后多个快照可能共享一个 coordinator allocation；关闭部分成员不会提前释放整份 reservation。
只有最后成员关闭或整组迁回 CPU 后才释放额度，避免重复借出仍被使用的存储。

## 测试与对比

```bash
.venv/bin/python -m pytest -q baselines/aqua/tests
```

CPU 测试覆盖线程并发分配、1000 次随机回收、禁止提前确认、重复释放、容量不超卖和 CFS 分区。
真实 GPU 验证用上面的 `--scenario validate`；HTTP 验证脚本见 `tests/validate_http.py`，需要已预热的 prefix JSON 与运行中的服务。

每个请求的结果包含 token IDs、TTFT、总耗时、输出 token 时间戳和缓存命中数。
完整轨迹在 `events` 和 coordinator ledger 中；验证样本少，不作为稳定 p99 或论文性能结论。
比较 SwiftCache 时应统一权重、dtype、输入 token IDs、输出长度和硬件；本版 HF/SDPA 与 SwiftCache 的执行内核不同，需分别标注。
