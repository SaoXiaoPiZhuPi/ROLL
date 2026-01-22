# SgLangPdDisStrategy 使用说明

## 概述

`SgLangPdDisStrategy` 是支持 Prefill-Decoding 解耦的 SGLang 推理策略。它将推理过程分为两个独立的阶段：
- **Prefill 阶段**：处理输入 prompt 的预填充
- **Decoding 阶段**：生成输出 token

## 主要特性

1. **双引擎架构**：同时启动 prefill 和 decoding 两个独立的 engine
2. **灵活的 TP/DP 配置**：
   - `prefill_tp_size`：prefill 阶段的 tensor parallel size
   - `decode_tp_size`：decoding 阶段的 tensor parallel size
   - 默认值为 `device_count() / 2`
3. **自动 DP 分配**：
   - `dp_size = world_size / 2`
   - 前 `dp_size` 个 rank 运行 prefill engine
   - 后 `dp_size` 个 rank 运行 decode engine
4. **内置 Router**：在 rank 0 上自动启动负载均衡器，管理 prefill 和 decode worker

## 配置示例

在 YAML 配置文件中设置策略：

```yaml
strategy_args:
  strategy_name: "sglang_pd_dis"
  strategy_config:
    # SGLang 基础配置
    mem_fraction_static: 0.9
    log_level: "info"
    disable_custom_all_reduce: true
    
    # PD 解耦配置
    prefill_tp_size: 4  # 可选，默认为 device_count() / 2
    decode_tp_size: 4   # 可选，默认为 device_count() / 2
```

## Worker 分配

假设有 8 个 GPU（world_size = 8）：

- **Prefill Workers** (rank 0-3):
  - 运行 prefill engine
  - TP size = 4 (默认)
  - 端口: 30000, 30500, 31000, 31500

- **Decode Workers** (rank 4-7):
  - 运行 decode engine
  - TP size = 4 (默认)
  - 端口: 32000, 32500, 33000, 33500

- **Router** (在 rank 0 上):
  - 自动启动 MiniLoadBalancer
  - 管理 prefill 和 decode worker 的请求分发

## 端口分配规则

```
base_port = 30000

Prefill workers:
  rank 0: 30000 + 0 * 500 = 30000
  rank 1: 30000 + 1 * 500 = 30500
  rank 2: 30000 + 2 * 500 = 31000
  rank 3: 30000 + 3 * 500 = 31500

Decode workers:
  rank 4: 30000 + (0 + 4) * 500 = 32000
  rank 5: 30000 + (1 + 4) * 500 = 32500
  rank 6: 30000 + (2 + 4) * 500 = 33000
  rank 7: 30000 + (3 + 4) * 500 = 33500
```

## 使用场景

适用于需要高吞吐量推理的场景，特别是：
- 长文本生成
- 高并发请求
- 需要分离 prefill 和 decoding 计算资源的场景

## 注意事项

1. **World Size 要求**：world_size 必须是偶数（因为 dp_size = world_size / 2）
2. **GPU 分配**：确保每个 worker 有足够的 GPU 资源
3. **端口冲突**：确保配置的端口范围内没有被其他服务占用
4. **内存管理**：prefill 和 decode engine 都会占用 GPU 内存，需要合理分配

## 与 SgLangStrategy 的区别

| 特性 | SgLangStrategy | SgLangPdDisStrategy |
|------|----------------|---------------------|
| Engine 数量 | 1 | 2 (prefill + decode) |
| TP 配置 | 单一 tensor_parallel_size | prefill_tp_size + decode_tp_size |
| Worker 分配 | 所有 rank 运行相同 engine | 前 half 运行 prefill，后 half 运行 decode |
| Router | 无 | 内置 MiniLoadBalancer |
| 适用场景 | 通用推理 | 高吞吐量、长文本生成 |

## 日志输出

初始化时会输出详细的配置信息：

```
[sglang_pd_dis][local]: global_rank=0 dp_rank=0 dp_size=4 tp_size=4 is_prefill=True is_decode=False
[sglang_pd_dis] Initialized prefill engine on rank 0
[sglang_pd_dis][router] Initializing router with prefill URLs: ['http://127.0.0.1:30000', ...]
[sglang_pd_dis][router] Initializing router with decode URLs: ['http://127.0.0.1:32000', ...]
[sglang_pd_dis][router] Router started successfully
```

## 参数同步

策略支持所有标准的参数同步操作：
- `setup_collective_group`: 设置通信组
- `broadcast_parameter`: 广播参数
- `broadcast_bucket`: 广播参数桶
- `update_parameter`: 更新参数
- `update_parameter_in_bucket`: 批量更新参数

这些操作会同时应用到 prefill 和 decode engine（如果存在）。

## 状态管理

支持模型状态的加载和卸载：
- `load_states`: 恢复模型到 GPU
- `offload_states`: 卸载模型以释放内存

两个 engine 的状态会独立管理。