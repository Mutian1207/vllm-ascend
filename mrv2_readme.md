# MRv2 Triton 离线整网 Profile（T0–T9）

每个 T 场景都在 `scenarios.json` 里明确写了：

- 测什么输入和模型配置；
- 传给 `LLM` / `SamplingParams` 的参数；
- 预期覆盖哪些 kernel；
- 是否需要特殊模型或多卡。

先看清楚再执行，不会启动模型：

```bash
python run_offline.py --list
python run_offline.py --describe T1
python run_offline.py --scene T1 --dry-run
```

执行 Torch Profile：

```bash
python run_offline.py --scene T0 --profile torch
python run_offline.py --scene T1 --profile torch
```

执行 msprof（统计 NPU Device Task 时延建议用这个）：

```bash
bash run_msprof.sh T0
bash run_msprof.sh T1
```

T0–T4 使用默认 `/home/data/Qwen3-0.6B` 可直接执行。T5–T9 分别需要 VL
模型与图片、双卡 DCP、EAGLE/MTP、DFlash/DSpark、Mamba/Hybrid 模型，填好
环境变量并把对应场景的 `skip` 改成 `false` 后执行。

脚本流程是：创建一个 `LLM` → 各 case warmup → `start_profile()` → 各 case
重复正式推理 → `stop_profile()`。T0 包含 prefill-heavy 和 decode-heavy 两个
case；其他场景的参数直接在 JSON 中可见。

表格和 JSON 中的“覆盖算子”是预期验收清单，最终必须在 Profile 里逐个确认：

- kernel 出现：记录调用次数、累计 Task Duration 和 DeviceTaskShare；
- helper 不会独立 launch：归入其父 kernel；
- Ascend 替换实现：记录实际等价 kernel 名；
- 条件未满足：标记“未覆盖”，不能用普通 Dense 结果代替。
