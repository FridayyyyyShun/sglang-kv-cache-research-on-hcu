# T1/T2 HiCache 模型实验

淘汰策略的服务启动及 T3 命令见 [T3 淘汰策略测试](README.md)。

使用当前项目的真实模型服务，不复制之前手工填充 MHA KV 的小实验。
默认 MLA/sparse 模型继续使用 `page_first`；不要盲目改成 MHA 的 `layout_hcu`。

## 启动

```bash
cd '/workspace/kv-research/experiments/hicache-restore-eviction'

# 终端一：使用原启动脚本，保留 env.sh 的环境及 observer 初始化。
# T2 必须在启动时开启 KV 数据校验；不要使用 serve_no_hicache.sh。
M1_HICACHE_VERIFY=1 bash lib/serve.sh \
  --hicache on --hc-ratio 4 --hc-policy write_through \
  --hc-layout page_first --hc-io kernel \
  --radix on --kv bf16 --page 64 \
  --cap 16384 --ctx 32768 --chunk 8192 \
  --port 31000 --src repo --obs 1 --root /workspace/kv-runtime
# 只预览时在末尾加 --mode plan，核对后去掉再启动。
```

`--cap` 现在实际传给 `--max-total-tokens`。ctx 是单请求长度上限，
cap 是池容量，chunk 是单次 prefill 预算，三者不能互相替代。
示例容量需按模型实际运行情况调整；TP8 也不能把八卡容量直接相加。

必须打开观测（默认 `--obs 1`），更改观测脚本后需要重启服务。
上述命令对应示例服务器 IP `127.0.0.1`、模型
`/model/GLM-5.2-Channel-INT4-w4a8`；由服务器现有 `env.sh` 配置。
先同步新版 `exp/t2_hicache_load_back.py` 和 `lib/obs/m1_radix_observe.py`，
保留服务器自身的 `env.sh`。启动前先停止旧服务，等待 GPU 释放。
KV 校验会同步 GPU 并将目标数据复制到 CPU，仅用于正确性测试；
性能测试应去掉 `M1_HICACHE_VERIFY=1` 后重新启动服务。
先确认 server.log 显示 HybridCacheController/Host 池创建成功。
实验需要服务进程和客户端能访问同一份 RUN 目录。
T1/T2 不自动并入原 T1-T6，避免意外增加压力工作负载。

## 写穿

单独的启动、运行、结果解读及故障排查见 [T1 write-through 运行说明](README.md)。

```bash
# 终端二：服务就绪后执行，测试期间不要发送其他请求。
python3 exp/t1_hicache_write_through.py \
  --host 127.0.0.1 --port 31000 \
  --run-dir "$(cat .state/last_run)" \
  --page 64 --prompt-len 4096 --wait-seconds 60
```

通过条件：新 namespace 的冷请求、目标插入节点对应的备份提交与完成。
备份完成事件在生产 ack 处理之后记录；仅有 host_value 不等于搬运完成。
本实验验证 write_through，应使用该策略重启；write_back 应另设计触发条件。

## 回载

```bash
# 同一实验目录下运行；使用正式文件名，不是临时的 load_back1.py。
python3 exp/t2_hicache_load_back.py \
  --host 127.0.0.1 --port 31000 \
  --run-dir "$(cat .state/last_run)" \
  --page 64 --prompt-len 4096 \
  --pressure-len 8192 --max-rounds 6 \
  --expected-ranks 8 --wait-seconds 60
```

每轮发送递增数量的不共享压力输入，再探测目标。探测本身会使目标变热，
所以压力在探测之间递增。目标先在全部 8 个 rank 完成原始 KV 指纹采集和备份，
随后每个 rank 都必须观察到 GPU 零命中、足够的 Host 命中、
同一 PID/Host anchor 的成功 load_back、消费 ack，以及 KV 指纹一致才通过。
校验覆盖有效逻辑页的各层 KV 和 DSA index cache，不比较生成回答。
一条目标输入长 4096 token；每条压力输入长 8192 token，最多六轮分别发送
1、2、3、4、5、6 条压力输入，回载验证通过后提前停止。
单个压力输入加输出预算必须能放进 GPU 池，输入也不能超过 context 上限。
Host ratio 应留出余量，避免 GPU 驱逐之前/之后 Host 也丢失目标。
没有触发回载会报失败，不能解释为实现错误；查看 JSON/CSV 和事件区分
“未建立场景”“Host 数据已丢失”“观测缺失”“回载失败”。

所有结果写入现有 RUN 的 t1/t2 JSON/CSV，事件仍在 radix_events_<pid>.jsonl。
TP 多进程事件按 PID/node_id 关联：T1 证明至少一个 rank 的备份链路，
T2 要求全部预期 rank 的回载和指纹校验。T2 还保存 `.artifacts.json`
记录逐轮和完整事件。更多校验范围、限制见 [T2 KV 校验说明](README.md)。
本地仅执行静态检查和无 GPU 的观测模拟；实际模型/搬运验证需在云端运行。

## 启动参数

新增：`--hicache on|off`、`--hc-ratio`、`--hc-policy`、`--hc-layout`、`--hc-io`。
`--hicache off` 可作为无 L2 对照（T1/T2 预期无法通过）。axes.env 和 run tag
记录 HiCache 配置。脚本仍沿用现有 TP8、模型量化和 sparse 后端配置，
不是通用任意模型启动器。写入 .pth 的现有观测安装行为保持原样。
