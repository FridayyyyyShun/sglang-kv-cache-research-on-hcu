# T2/T3/T4 运行入口

## T1 写穿备份测试

同一入口现支持 `--only t1`。服务开启HiCache、write_through及obs，两个额外校验开关设为0。完整服务启动命令见 [T1说明](README.md) 及lib/serve.sh开头注释；按本机env.sh配置IP、模型和网卡。

```bash
# 服务就绪后，在实验目录执行：
bash run_hicache.sh --only t1 --host 127.0.0.1 --port 31000 \
  --page 64 --prompt-len 4096 --wait-seconds 60
```

T1发送1条目标请求，默认输入4096 token、生成1 token。通过条件为冷请求、目标节点的成功备份提交及完成、无observer错误。现有T1至少一条PID/节点链路即可通过，不强制8个rank，也不验证KV字节或回载。结果为当前RUN中的t1_hicache_write_through.json/.csv，入口退出码摘要为run_t1_summary.txt。

淘汰实验拆分为 T3 审计（exp/t3_eviction_policy.py）和 T4 性能对比（exp/t4_policy_benchmark.py）。历史回答一致性实验不运行。

服务仍须手动按模式启动。入口不会自动重启服务，也无法替你改变已运行服务的环境变量；终端提示是要求而非自动检查。

```bash
# T2：服务启动前加 M1_HICACHE_VERIFY=1 M1_EVICTION_VERIFY=0
bash run_hicache.sh --only t2 --host 127.0.0.1 --port 31000

# T4性能：服务启动前加 M1_HICACHE_VERIFY=0 M1_EVICTION_VERIFY=0
# 使用 --policy slru 启动服务后运行；其他策略分别重启对应配置。
bash run_hicache.sh --only t4 --host 127.0.0.1 \
  --policy slru --cycles 10 --repeats 3

# T3淘汰顺序审计：服务启动前加 M1_EVICTION_VERIFY=1 M1_HICACHE_VERIFY=0
bash run_hicache.sh --only t3 --host 127.0.0.1 --policy lru
```

三种模式的服务启动条件不同，因此要求显式选择--only，不默认连跑。
入口默认读取.state/last_run；可用--run-dir覆盖。默认page=64、prompt-len=4096、pressure-len=8192、max-rounds=6、expected-ranks=8、wait-seconds=60。
性能比较会清缓存，只用于空闲专用测试服务。

新结果仍写入当前服务RUN目录，但文件名前缀改为：

```text
t3_eviction_policy.json / .csv / .artifacts.json
t4_policy_benchmark_lru.json
t4_policy_benchmark_lfu.json
t4_policy_benchmark_slru.json
t4_policy_benchmark_priority.json
run_t2_summary.txt / run_t3_summary.txt / run_t4_summary.txt
```

服务RUN目录名由serve.sh的时间和配置决定，不含实验编号，无需修改。
历史output/t10_*.json不自动改名，保留来源；它们可由新版脚本--summarize读取。
审计编号为 T3，性能脚本的文件名、结果前缀、日志标签和命名空间为 T4；同步服务器时使用新文件名。
