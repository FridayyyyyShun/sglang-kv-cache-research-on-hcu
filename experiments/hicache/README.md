# HiCache · M1–M6 分层机制实验

这一 part 延续同级 `radix-tree` 的六项操作，只增加 L2/L3 状态、锁和生命周期证据。**不测 TTFT、吞吐或性能收益。**

- L1：设备 KV 池；L2：主机内存 KV 池；L3：外部 file storage。
- 现场仍使用 `UnifiedRadixCache`；registry 创建 Unified 后调用 `init_hicache()`，不是换成旧 HiRadixCache。
- 原 `radix-tree/` 只读，不调用其启动/停止/实验脚本，不更新其状态、handoff 或文件。

## 六项操作与原实验的对应

| 原实验 | 新实验 | 在 L2/L3 新观察什么 |
|---|---|---|
| T1 匹配/分裂 | `exp/m1_match_split.py` | A 退到 L2 后，B 在同一页内分叉；从 L3 恢复后再分叉，公共节点与原路径是否仍正确 |
| T2 共享引用/锁 | `exp/m2_share_lock.py` | 多个请求从 L2/L3 恢复共享前缀；设备锁、host锁、异步操作与驱逐保护 |
| T3 页边界 | `exp/m3_page_boundary.py` | L1、L2、L3 对同一组 P−1/P/P+1 输入，命中是否均遵守查询上限和整页约束 |
| T4 多轮/恢复 | `exp/m4_multiturn_restore.py` | 真实 output_ids 追加历史；L2 载回、flush后L3恢复、未完成请求增量发布的事件顺序 |
| T5 miss归因 | `exp/m5_miss_attribution.py` | 区分 cold、page/query exclusion、L1容量miss但L2命中、L1/L2清空但L3命中，以及salt契约边界 |
| T6 ready/取消/释放 | `exp/m6_cancel_release.py` | 真实abort、双层锁归零、异步队列排空、设备/主机/DSA辅助池恢复基线、重发与零命中冷参考一致 |

每项的操作步骤、预期状态和证据见 `docs/m1.md` 至 `docs/m6.md`。

## 运行：先 L2，再叠加 L3

环境：一台安装了 SGLang 依赖、可运行 GLM-5.2-w4a8 的 8 卡机器；以下命令均在本目录（`hicache/`）下执行。

```bash

# 默认只打印计划；不会安装全局.pth、写.state或启动服务
bash lib/serve.sh --tier host --mode plan
# 等8张卡有足够空闲显存时启动L1+L2
bash lib/serve.sh --tier host --mode run
```

第二个终端运行：

```bash
bash run_m1.sh
# 只跑指定机制
bash run_m1.sh --only m1,m3,m5
# 单项入口也可直接运行；指定远程服务及本次RUN
M1_RADIX_OBSERVE=0 PYTHONDONTWRITEBYTECODE=1 python3 -B exp/m1_match_split.py \
  --host 127.0.0.1 --port 32000
```

跑完停止该RUN，再以L3配置启动，重新运行同一套六项：

```bash
bash lib/stop.sh
bash lib/serve.sh --tier storage --storage file --mode plan
bash lib/serve.sh --tier storage --storage file --mode run
# 另一终端
bash run_m1.sh
```

默认 TP8、page64、device cap8192、HiCache ratio2、write_through、wait_complete、direct/layer_first。GLM DSA 的主KV走 MLA host pool，indexer走DSA辅助池，两者源码都支持 direct/layer_first；不要套用 MHA 专属 layout_hcu。M1–M6第一轮限制page64，待确认后端支持再扩页大小。

**本 part 要求 `--obs 1`。** 快照和事件是验收证据，不使用 `--obs 0` 的性能运行方式。各项开始前会等待自身独立服务排空并flush内存，M6只对自己创建的rid定向abort；不要把此套脚本连到生产服务或别人正在使用的端口。脚本核对RUN的host/port，服务启动也检查端口和8卡显存，不会停止既有服务。

2026-11-05检查时每张卡已占约127–139 GiB显存，因此本轮没有加载GLM或宣称六项实机通过。

## 分层观察和断言原则

1. 每次操作保留响应 `cached_tokens_details.device/host/storage/storage_backend` 与 request id。对齐断言来自普通生成请求的查询上限 Q=L−1：重复请求预期 `P×floor((L−1)/P)`；多轮新增历史可复用上轮已计算的全部输入，公式按各场景分别推导。
2. L2测试先确认目标页在**每个rank**均为 `device_slots=0, host_slots>0`，再回放。通过无关请求自然制造容量压力，不通过测试后门强改缓存/分配器。
3. L3测试等待备份和确认队列排空，再flush L1/L2；确认本地树空后回放，要求正的storage字段。flush不等于删除L3，不能把这次回放称为冷启动。
4. 快照由 scheduler.process_input_requests 读取唯一tag的mailbox，**不发送额外生成请求触发快照**，不调用cuda synchronize、不搬运KV张量。记录节点key/parent/children、Full和组件的device/host槽与引用、write/load pending状态，以及每个host/辅助pool的剩余量。
5. 锁/回收检查同时记录：ongoing_write_through、ongoing_load_back、ongoing_prefetch、ongoing_backup，ack队列、host页释放队列和prefetch占用量。只看running=0不等于ready。
6. 每个rank分别检查树内token守恒、`tokens=evictable+protected`、负锁、flush后空树、M6清理后物理池恢复基线。TP8同一请求出现8份事件，不能当作8个独立请求。
7. L3没有“树节点快照”。L3证据是提交/完成事件和实际恢复结果；不能拿`host_value`存在、host命中或L2树图代替外部存储命中。
8. `cache_salt`仅承诺进程内树/事件隔离。外部L3 keys按现场契约仍是token-only，M5对跨salt的L3命中只观察、不错误断言必须零命中。M6在任何缓存建立前先取得明确零命中的冷参考，避免依赖salt制造L3冷启动。

## 结果与未覆盖窗口

RUN位于 `/tmp/m1-hicache/runs/`，只更新 `hicache/.state/last_run`。主要产物：

- `m1_*.json` 至 `m6_*.json`：操作记录、PASS/FAIL/NOT_EXERCISED判据、逐rank检查点快照、事件；
- `run_m1_summary.txt`：六项汇总（PASS / NOT_EXERCISED / FAIL）；
- `hicache_events_PID.jsonl`、`hicache_snapshots_PID.jsonl`：原始观测；
- `axes.env`、`command.txt`、`source.sha256`、`server.log`：环境与源码；
- `storage/`：每RUN独立file目录，保留复查，不自动删除。

退出码：0=PASS，1=FAIL，3=NOT_EXERCISED。没有抓到host-only状态、共享并发锁、未完成请求窗口或L3预取中取消，会明确记为未覆盖。不能把“没有观测到出错”当作该并发交错已验证；部分覆盖也不会汇总成六项全通过。

M2额外报告是否自然碰到了“host驱逐与host锁竞争”；M4需要增量发布先于探测、源请求完成晚于探测；M6分别区分首token后的真实取消与**尚在L3预取中**取消，后者可能因窗口太短而未覆盖。已有T6输出一致性差异仍保留为失败，不根据账本正常推断KV内容一定正确。

```bash
# 只给该RUN记录且核对cmdline的PID发TERM，不按端口批量pkill
bash lib/stop.sh
# 不用GPU的fixture测试
M1_RADIX_OBSERVE=0 PYTHONDONTWRITEBYTECODE=1 python3 -B tests/test_mechanisms.py -v
```

