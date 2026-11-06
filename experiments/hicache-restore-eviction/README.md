# HiCache 回载正确性与淘汰策略评估

本实验集聚焦于 SGLang HiCache 的 **KV 回载正确性**与 **GPU 缓存淘汰策略评估**。通过写穿备份、容量压力驱逐、Host 回载及 KV 指纹比对验证数据恢复正确性，并在统一工作负载下比较 LRU、LFU、SLRU、Priority 的命中表现和端到端请求时延。

本实验目录为 `experiments/hicache-restore-eviction/`，与同事的 `hicache/`、`radix-tree/` 并列。T1～T4 是本目录内部编号，对应早期开发记录中的 T7～T9。

## 实验内容

| 实验 | 脚本 | 验证目标 |
|---|---|---|
| T1 写穿备份 | `exp/t1_hicache_write_through.py` | 唯一命名空间首次请求冷启动，GPU→Host 备份提交与完成可关联 |
| T2 回载正确性 | `exp/t2_hicache_load_back.py` | 压力首次 GPU/Host 零命中；目标 GPU miss、Host 命中、回载完成及 KV 指纹一致 |
| T3 淘汰审计 | `exp/t3_eviction_policy.py` | 实际 GPU 淘汰节点符合指定策略，且未淘汰被锁定节点 |
| T4 性能对比 | `exp/t4_policy_benchmark.py` | 统一访问序列下比较各策略的命中占比、端到端时延和热点保护表现 |

## 目录结构

```text
hicache-restore-eviction/
├── README.md
├── run_hicache.sh       # 已启动服务的测试入口
├── exp/                # 实验脚本与专项说明
├── lib/                # 启动、环境配置和观测钩子
└── patches/            # HCU MLA H2D JIT 绕行补丁
```

## 环境与源码

当前配置面向 GLM-5.2 INT4 w4a8、8 卡 HCU、TP=8、BF16 KV、64-token 页。其他模型、设备或容量需要重新核对后端支持和压力大小。

默认使用仓库的 `sglang-das/python`，可通过 `M1_REPO` 指向实际源码的 `python/` 目录。启动前核对 `lib/env.sh` 中的模型、节点、IP 和网卡配置，并用 `--mode plan` 检查最终启动命令。公开提交前清理实际节点配置和运行数据。

HCU MLA 回载 JIT 兼容补丁见 [应用与撤销说明](patches/README.md)。补丁仅在两处 H2D 分支判断中绕开 HCU JIT，作为报告中 Invalid address access / KERNEL VMFault 的绕行措施，不代表已经修复内核根因。保存补丁文件不会自动修改源码，应用后需要重启服务。

## 启动服务器

进入本实验目录，以下以 T2 为例：

```bash
cd /path/to/repository/experiments/hicache-restore-eviction

M1_HICACHE_VERIFY=1 M1_EVICTION_VERIFY=0 bash lib/serve.sh \
  --hicache on --hc-ratio 4 --hc-policy write_through \
  --hc-layout page_first --hc-io kernel \
  --policy lru --radix on --kv bf16 --page 64 \
  --cap 16384 --ctx 32768 --chunk 8192 \
  --port 31000 --src repo --obs 1 --root /workspace/kv-runtime
```

末尾添加 `--mode plan` 可仅预览。不同实验按下表调整启动命令开头的变量，切换模式需要重启自己的测试服务：

| 模式 | M1_HICACHE_VERIFY | M1_EVICTION_VERIFY |
|---|---:|---:|
| T1 备份 | 0 | 0 |
| T2 回载指纹 | 1 | 0 |
| T4 性能 | 0 | 0 |
| T3 审计 | 0 | 1 |

全部模式保留 `--obs 1`。指纹验证涉及 GPU 同步和 CPU 数据复制，候选审计也有开销，性能测试关闭这两项。

## 运行实验

服务就绪后另开终端，在同一实验目录执行。以下命令分别对应不同启动模式，不要在一个模式下直接连跑全部测试：

```bash
export SERVER_HOST=127.0.0.1  # 替换为实际服务地址

# T1：服务按 T1 模式启动
bash run_hicache.sh --only t1 --host "$SERVER_HOST"

# T2：服务按 T2 模式启动
bash run_hicache.sh --only t2 --host "$SERVER_HOST" \
  --page 64 --prompt-len 4096 --pressure-len 8192 \
  --max-rounds 6 --expected-ranks 8 --wait-seconds 60

# T4 性能：服务按性能模式、LRU 策略启动
bash run_hicache.sh --only t4 --host "$SERVER_HOST" \
  --policy lru --page 64 --prompt-len 4096 \
  --expected-ranks 8 --cycles 10 --repeats 3

# T3 审计：服务按审计模式、LRU 策略启动
bash run_hicache.sh --only t3 --host "$SERVER_HOST" \
  --policy lru --expected-ranks 8
```

入口脚本不自动启动、停止或重启服务。默认读取当前目录 `.state/last_run`；可用 `--run-dir /path/to/run` 指定运行目录，用 `--port` 修改端口。

比较其他策略时，先将服务启动参数 `--policy` 改为 `lfu`、`slru` 或 `priority` 并重启，再同步修改测试参数。客户端的 `--policy` 不会切换服务策略。

**T4 性能测试会清缓存，仅在空闲专用测试服务上运行。**

## 请求设计与指标

T1 默认一条 4096-token 输入，生成一个 token。T2 先建立目标 KV 基线及备份，再发送压力请求和目标探测；`--max-rounds 6` 是压力轮数上限，满足条件后提前结束，并非固定六轮。

T4 各策略使用相同输入序列和参数：

| 场景 | 目的 | cycles=10 时测量请求数/次重复 |
|---|---|---:|
| hot_scan | 热点与一次性扫描混合，观察抗扫描能力 | 160 |
| phase_shift | 热点随阶段切换，观察适应新热点的能力 | 160 |
| cyclic_over_capacity | 工作集超 GPU 容量，观察循环淘汰与 Host 回载 | 180 |
| priority_scan | 优先级请求与扫描混合，观察优先级保护 | 160 |
| protected_hot_vs_recent_scan | 建立旧热点后扫描新前缀，再探测热点是否仍在 GPU | 60 |

`cycles=10` 表示场景内十个访问周期，周期间不清缓存；`repeats=3` 表示独立重复三次，各场景每次清缓存并重新预热。默认每次重复合计 720 条测量请求、54 条预热请求，预热不计入测量汇总。

| 指标 | 含义 |
|---|---|
| latency_mean_ms / P50 / P95 | 客户端请求端到端时延，包含请求路径上的计算和回载等开销，不是单独 H2D 耗时或 TTFT |
| gpu_hit_fraction / host_hit_fraction | 输入 token 命中占比，不等同于命中请求数比例；4032/4096=98.4375% |
| groups | GPU 命中、Host 命中、冷请求及热点请求的分组统计 |
| hot_probes | 热点探测请求的汇总统计；单个 hot_probe_* 字段需结合明细确认对应请求 |

## 证据与结论边界

T1 关联同 PID、同节点的备份提交和完成，当前不强制所有 rank 均形成链路。T2 要求全部预期 rank 的压力首匹配零命中，以及目标 GPU miss、Host 命中、回载完成和 SHA-256 指纹一致；覆盖支持池接口下的有效 KV 页、层及 DSA index cache。

回载正确性依据 KV 指纹，不依据生成文本是否完全一致。Host 有副本不代表发生回载，必须结合 GPU miss 和回载完成事件。

淘汰审计验证选择是否符合策略，性能测试比较给定场景下的效果。HiCache 的 Host 回载可能缩小时延差距，即使 GPU 命中比例明显不同。不能据此要求某一种策略在所有场景都最好。

本实验不覆盖远端 L3/Mooncake、多实例路由或完整 API 命名空间隔离。串行短输出请求不代表并发吞吐；一次重复不能支持统计显著的性能排名。

## 输出与协作提交

结果写入 `.state/last_run` 对应目录，包括实验 JSON、artifacts、observer 事件和运行摘要。T4 性能结果为 `t4_policy_benchmark_<policy>.json`，详细请求和分组数据用于复核结论。重跑前归档同名结果，避免覆盖。

建议在独立分支中提交本实验目录和明确说明的源码差异，保留同事原有目录。不要提交 `.state/`、运行日志、缓存或模型文件。
