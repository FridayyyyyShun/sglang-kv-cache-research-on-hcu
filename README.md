# kv-research

围绕大模型推理 **KV Cache 优化** 的研究仓库：以「前缀复用的正确性与可归因」为地基，逐步扩展到多 P 实例间的缓存复用与路由、分层缓存（HiCache）、远端 KV 存储与传输（Mooncake），以及淘汰、容量与压缩策略。

> 当前仓库内容是课题的第一阶段：Radix Tree 基础验证（T1–T6）。下文「研究路线」中的其余方向均为规划中，尚未实现。

## 课题聚焦

本课题主攻 **KV 复用与多实例（多 P）协同**，副攻 **成本感知淘汰**；容量 / 传输 / 压缩按实测瓶颈择机展开。原因：复用与路由直接决定「少算多少历史」与「请求发给谁」，是端到端收益最直接的来源，也是后续所有优化的切口。

| 方向 | 本期聚焦点（收敛后） | 可验收目标（候选门槛） | 验收成果形态 |
| --- | --- | --- | --- |
| 复用（主攻） | 多轮 / 共享前缀的「正确且充分」复用，先把基础做对 | 兼容前端达到页对齐复用上限；避免的重算 token 可量化；零错误跨版本复用 | 复用正确性报告 + 逐轮 TTFT 曲线 + 失效归因表 |
| 多 P 路由（主攻） | 联合「缓存命中 + 排队负载」的成本感知路由 | 长 / 短请求 TTFT P95/P99 改善；有效吞吐提升；命中预测准确率可报告 | 路由策略实现 + 对照 / 消融 + 命中-负载权衡曲线 |
| 淘汰（副攻） | 成本 + 概率感知保留，替代纯 LRU | 压力场景重算时间 ↓≥15%；短请求 P99 劣化 ≤5%；策略开销 ≤2% 调度时间 | 淘汰策略 + oracle 对照 + 公平性报告 |
| 容量 | 分层预算与低比特 KV（按瓶颈） | 固定显存下有效工作集扩大且总延迟不倒退 | 字节账本 + 分层容量对照 |
| 传输 | 恢复关键路径 + PD 批量（按瓶颈） | 恢复总时间、暴露通信等待下降，命中真正转化为收益 | 分段 trace + 传输改进对照 |
| 压缩 | 增量压缩 + 缓存失效协同（最后） | 质量容差内总 token / 成本下降，含摘要与回读开销 | 质量集 + 端到端成本对照 |

表中为**候选验收门槛**，用于约束实验设计与统计口径，不代表已达成的结论。

## 研究路线

### 阶段一 · Radix Tree 基础验证（当前，T1–T6）

先把「复用是否正确」这件事做实：用运行时观测钩子记录 `UnifiedRadixCache` 的 match / insert / evict / 锁事件与结构化树快照，再用黑盒请求断言 `cached_tokens`，让每一次 miss 都能归因、每一份缓存账本都自洽。

| 实验 | 检查内容 |
| --- | --- |
| T1 匹配与分裂 | 前缀命中长度、共享前缀分叉是否落在页边界、树节点账本自洽 |
| T2 共享引用与锁 | 并发时 protected > 0、结束后锁引用归零、不驱逐活跃页 |
| T3 页边界 | 不同输入长度下的命中页数与页对齐语义 |
| T4 多轮与恢复 | 多轮累积命中、未完成请求增量命中、flush 后树真正清零 |
| T5 miss 归因 | 把每次 miss 归到 cold / page_tail / salt_mismatch / capacity，不允许无法解释的 miss |
| T6 ready / 取消 / 释放 | 锁归零、tokens = evictable + protected、物理页无泄漏、取消请求不污染缓存 |

脚本见 `experiments/radix-tree/`，运行方式见下文「运行」。

### 后续方向（规划中）

| 方向 | 拟解决的问题 | 现有基础 |
| --- | --- | --- |
| 多 P 实例 cache 复用与路由 | PD 分离下多个 Prefill 实例之间的缓存共享、命中预测，以及「缓存命中 + 排队负载」联合调度，避免命中率高但排队更慢 | `experimental/sgl-router`（含 `sgl-kv-indexer`）、阶段一的事件与快照观测方法 |
| HiCache 分层缓存 | 显存 / 主机内存 / SSD 分层放置、写回与载入时序，扩大固定显存下的有效工作集 | `benchmark/hicache/`、`test/registered/hicache/`、`mem_cache/storage/*` |
| Mooncake 远端 KV 存储与传输 | 跨实例 KV 搬运的恢复关键路径与 PD 批量，压低暴露的通信等待，让「命中」真正转化为端到端收益 | `mem_cache/storage/mooncake_store/`、`storage/nixl/`、`storage/hf3fs/` |
| 成本感知淘汰 | 用成本 + 概率替换纯 LRU，在容量压力下保护高价值前缀，并给出 oracle 对照与公平性证据 | `mem_cache/evict_policy.py`、`serve.sh --policy` |
| 容量 / 低比特 KV / 压缩 | 分层预算、低比特 KV 与增量压缩，按实测瓶颈择机展开 | `serve.sh --kv bf16\|fp8`、`--cap`、`--ctx` |

以上方向都以阶段一的观测与归因方法为底座：先把「命中 / 未命中 / 重算 / 传输」的账目做清楚，再谈优化收益。新增实验将按主题放入 `experiments/` 下的独立目录。

## 仓库结构

- `sglang-das/`：SGLang 完整源码树（第三方代码，许可见文末）。实验运行时的 `PYTHONPATH` 指向 `sglang-das/python`。该目录为本地归档，可不随公开仓库发布；此时用 `M1_REPO` 指向任意一份 SGLang v0.5.18 源码树即可。
- `experiments/radix-tree/`：阶段一的启动、观测与 T1–T6 实验脚本，参见该目录的 `README.md`。
- `experiments/hicache/`：HiCache 分层缓存机制实验 M1–M6（L1 device / L2 host / L3 storage），沿用 `radix-tree` 的六个主题并叠加 L2/L3 状态、锁与生命周期证据；用 `lib/serve.sh` 启动，`run_m1.sh` 运行，说明见该目录的 `README.md` 与 `docs/`。

## 环境信息说明

本仓库已隐去原实验集群的节点 IP、网卡名、主机别名与远端路径。`lib/env.sh` 只保留占位节点 `example`（RFC 5737 文档地址，不可路由），实际地址通过 `M1_HOST` / `M1_IFNAME` 传入；客户端默认连接 `127.0.0.1:31000`。硬件型号、模型规格与 NCCL/ROCm 调优参数予以保留——它们是复现实验所必需，且不含可定位信息。源码树内少量 CI 配置、workflow 与测试注释中的内部主机地址、镜像仓库与容器名也已替换为保留占位（`*.example.invalid`、`192.0.2.0/24`）；通用示例地址（如 `10.0.0.x`）与公开镜像源保持原样。

<!--
## 来源与核验范围

由于实验机 SSH 暂不可达，此副本由本机保存的 SGLang v0.5.18 完整源码树和 radix-tree 实验脚本组装，实验记录中的远端路径已隐去。本地源码的 `radix_cache.py`、`evict_policy.py` 和 `utils.py` 未作改动，与实验保存的远端 SHA256 完全一致；其他文件尚未与远端当前目录逐一核验。为便于公开发布，CI 配置与测试注释中的内部地址已做占位替换（共 32 个文件），因此本副本不再是远端目录的逐字节镜像；恢复访问后应比较整树，再作为远端源码的精确归档。
-->

## 运行（阶段一实验）

在具有 GLM-5.2-w4a8 模型、8 卡 HYGON/ROCm 环境和项目依赖的机器中：

```bash
cd experiments/radix-tree
export M1_HOST=<目标机器 IP>
M1_MODEL_PATH=/path/to/GLM-5.2-w4a8 bash lib/serve.sh --node example
# 另一个终端
bash run_m1.sh --host "$M1_HOST"
```

实验脚本默认使用 `../../sglang-das/python`，可用 `M1_REPO=/other/sglang/python` 覆盖。更多节点别名可自行添加到 `lib/env.sh`。

## 第三方源码与许可

`sglang-das/` 来自 [sgl-project/sglang](https://github.com/sgl-project/sglang)，按 Apache-2.0 许可发布，许可证见 `sglang-das/LICENSE`，第三方声明见 `sglang-das/THIRD_PARTY_NOTICES.md`。本仓库的实验脚本为在此之上的研究性补充。
