# sglang-kv-cache-research-on-hcu

面向 GLM-5.3（W4A8）在 HCC / DCU 环境下的 SGLang KV Cache 研究仓库。

当前分支：`feat/glm-kv-opt`

## 目录结构

```
.
├── README.md
├── scripts/                  # PD 分离部署的基线启动脚本（本分支新增）
│   ├── p.sh                  # Prefill 端
│   ├── d.sh                  # Decode 端
│   └── router.sh             # Router（PD 分离调度）
└── sglang-das/               # 来自 InJuly7/sglang-das 的 feat/support_1M_context 分支快照
```

## 基础环境

| 项目 | 值 |
| --- | --- |
| 容器镜像 | `tencent-hygon-glm5.2_w4a8-sglang:0.5.18-ubuntu22.04-dtk2604-py3.10-20261029_update` |
| 登录节点 | `apply_test@10.41.103.4` |
| 脚本原始路径 | `/public/home/apply_test/maas/lvzhen/kv-research/glm5.3/scripts/baseline/`（现同步到本仓库 `scripts/`） |
| 模型路径 | `/model/GLM-5.3-Channel-INT4-w4a8` |
| DTK 环境 | `/opt/dtk-26.04-blas-nightly-1019/env.sh`（脚本内自动 source） |
| 每节点卡数 | 8 张 DCU（`HIP_VISIBLE_DEVICES=0..7`） |
| IB 设备 | `mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7` |

脚本为原环境基线脚本的直接拷贝，其中 `PYTHONPATH` 仍指向集群上的
`/workspace/kv-research/glm5.3/sglang-das_songge/python`。部署本仓库后，
请按实际落地路径修改 `p.sh` / `d.sh` 中的 `PYTHONPATH`，使其指向本仓库的
`sglang-das/python`，否则可能加载到错误版本的 sglang。

### 节点别名（脚本内映射）

| 别名 | IP | 网络接口 |
| --- | --- | --- |
| S1 | 10.41.103.12 | `enp10s0f4u2c2` |
| S2 | 10.41.103.10 | `enp10s0f4u2` |
| S3 | 10.41.101.164 | `eth0` |
| S4 | 10.41.103.4 | `enp10s0f4u2` |
| S7 | 10.41.103.8 | `enp10s0f4u2` |

## 运行方式

准备工作：把本仓库放到各节点一致的路径（例如 `/workspace/kv-research/glm5.3`），
确认模型路径、DTK 环境、`PYTHONPATH` 正确，且 `sglang-router` 在 `PATH` 中
（仅 router 节点需要）。

### 1. 启动 Prefill 端（P 节点）

```bash
# 在 S1 节点执行；参数为节点别名，也可用 PREFILL_NODE=s1 指定
bash scripts/p.sh s1
```

- 实际默认节点为 `s1`（脚本帮助文本里写的 `s2=default` 是旧注释）。
- 监听 `10.41.103.12:30000`，RCCL 分布式端口 `6000`，Mooncake bootstrap 端口 `8299`。
- 主要参数：TP=8 / EP=8 / CP=8，`--disaggregation-mode prefill`，
  `--disaggregation-transfer-backend mooncake`，KV cache 为 `fp8_e4m3`，
  context length 262144。
- KV 复用观测：由 `KV_EXPERIMENT_ID`（默认 `b1-c64-mem080`）决定输出文件
  `/workspace/kv-research/glm5.3/scripts/baseline/logs/<ID>.kv.jsonl`。
- 日志：`scripts/logs/YYYYMMDD/prefill-cp8_*.log`。

### 2. 启动 Decode 端（D 节点）

```bash
# 在 S2 节点执行；也可用 DECODE_NODE=s2 指定
bash scripts/d.sh s2
```

- 监听 `10.41.103.10:30023`，dist-init 指向自身 `6000` 端口。
- 主要参数：TP=8 / DP=8 / EP=8，`--disaggregation-mode decode`，
  Mooncake 传输，`--enable-dp-attention`，`--enable-dp-lm-head`。
- 日志：`scripts/logs/YYYYMMDD/decode-dp8ep8_*.log`。

### 3. 启动 Router

```bash
# 在任意可同时访问 P/D 节点的机器上执行
P_NODE=s1 D_NODE=s2 bash scripts/router.sh
```

- 默认监听 `0.0.0.0:30016`，策略 `cache_aware`；Prometheus 指标端口 `29018`。
- P/D 地址默认由节点别名推导，也可显式覆盖：

  ```bash
  PREFILL_URL=http://10.41.103.12:30000 \
  DECODE_URL=http://10.41.103.10:30023 \
  ROUTER_POLICY=round_robin \
  bash scripts/router.sh
  ```

- `ROUTER_POLICY` 可选：`random`、`round_robin`、`cache_aware`、`power_of_two`、
  `bucket`、`manual`、`consistent_hashing`、`prefix_hash`。
- 只打印命令不启动时可用 `ROUTER_DRY_RUN=1`。
- 日志：`scripts/logs/YYYYMMDD/router-pd-S1-S2-cache_aware_*.log`。

### 4. 发请求

```bash
curl http://<router-ip>:30016/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/model/GLM-5.3-Channel-INT4-w4a8","messages":[{"role":"user","content":"hello"}]}'
```

启动顺序：先 P、再 D，确认两端 health 正常后再启动 Router。三个进程均为前台运行，
日志同时输出到终端和 `logs/` 目录；停止时直接终止对应进程即可。

## 备注

- `sglang-das/` 是从 `InJuly7/sglang-das` 的 `feat/support_1M_context`
  分支导入的快照（源提交 `25720158d4beddc4586fcbf97f6f90f44b0b3974`），
  为自包含的单根提交，未携带源仓库的完整提交历史。
- `scripts/.gitkeep` 仅用于占位，无实际内容。
