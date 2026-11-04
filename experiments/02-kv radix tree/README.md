# KV radix tree 实验（T1–T6）

本目录包含服务启动脚本、观测钩子和六项 radix cache 实验。默认从仓库的 `sglang-das/python` 加载 SGLang 源码，使用 TP8、64 token 页、开启 radix cache 的单实例配置。

## 运行前准备

- 一台安装了 SGLang 项目依赖、可运行 GLM-5.2-w4a8 的 8 卡 HYGON/ROCm 机器。
- 将模型放在 `/model/GLM-5.2-w4a8`，或设置 `M1_MODEL_PATH`。
- 用环境变量 `M1_HOST` / `M1_IFNAME` 传入节点 IP 与网卡，或在 `lib/env.sh` 中自行添加节点别名。仓库只保留占位节点 `example`（`192.0.2.10`，RFC 5737 文档地址，不可路由），**运行前必须替换**。
- 客户端脚本默认连接 `127.0.0.1:31000`；服务不在本机时用 `--host` 指定。
- `lib/serve.sh` 默认将运行记录写入 `/tmp/m1-radix`，并为观测钩子尝试向 Python 的 site-packages 写入 `.pth` 文件；启动用户需要相应权限。实验结果会在运行时生成。

## 运行步骤

```bash
cd 'experiments/02-kv radix tree'

# 查看服务命令和源码路径，不启动服务
bash lib/serve.sh --node example --mode plan

# 终端一：启动服务
M1_MODEL_PATH=/path/to/GLM-5.2-w4a8 bash lib/serve.sh --node example

# 终端二：运行六项实验
bash run_m1.sh

# 停止服务
bash lib/stop.sh 31000
```

`serve.sh` 会把本次运行目录写入 `.state/last_run`，供 `run_m1.sh` 和各实验脚本读取。默认源码路径是 `../../sglang-das/python`；如需使用另一份源码，可设置 `M1_REPO=/path/to/sglang-das/python`。只运行指定实验时可用 `bash run_m1.sh --only t1_match_split,t3_page_boundary`。

## T1–T6 的作用

| 实验 | 脚本 | 检查内容 |
|---|---|---|
| T1 匹配与分裂 | `exp/t1_match_split.py` | 前缀命中长度、共享前缀分叉及树节点账本 |
| T2 共享引用与锁 | `exp/t2_share_lock.py` | 并发共享时的保护引用、请求完成后的锁释放 |
| T3 页边界 | `exp/t3_page_boundary.py` | 不同输入长度下的命中页数与页边界对齐 |
| T4 多轮与恢复 | `exp/t4_multiturn.py` | 多轮请求的缓存复用、未完成请求的增量命中和 flush |
| T5 miss 归因 | `exp/t5_miss_attr.py` | 冷启动、页尾、salt 不匹配和容量导致的未命中 |
| T6 ready、取消与释放 | `exp/t6_ready_cancel.py` | 请求取消、资源释放、缓存账本及重发后的缓存状态 |

## 常用参数

`lib/serve.sh` 支持 `--page`、`--kv bf16|fp8`、`--cap`、`--ctx`、`--policy`、`--radix on|off`、`--eagle 0|1`、`--obs 1|0`、`--port` 和 `--root`。例如容量压力测试可用 `bash lib/serve.sh --node example --cap 8192`。使用 `--mode plan` 可先检查完整启动命令。

结果目录默认是 `/tmp/m1-radix/runs/<时间>_<配置>_<pid>/`，其中包含实验的 JSON/CSV、radix 事件与快照、源码哈希和服务日志。
