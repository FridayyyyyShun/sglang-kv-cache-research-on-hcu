# -*- coding: utf-8 -*-
"""
Session 并发压测脚本

数据集按 session_id 分组，每个并发 worker 负责处理一个 session：
  - 按 start_time 升序依次发送该 session 内的每条请求
  - 该 session 所有请求发送完毕后，从队列中取下一个 session 继续处理
  - 当没有剩余 session 可取，且当前活跃并发数 < 初始并发数 * stop_threshold 时，
    发出全局停止信号，取消/中断所有正在进行的请求

支持的数据格式（JSON 或 JSONL）：
  每条记录需要包含：
    - session_id  : session 标识（用于分组）
    - start_time  : 排序字段（字符串或数字均可）
    - messages    : 请求消息列表

用法示例：
# 多数据文件决定路径 + 多并发 + 跑完数据集所有数据为止
python3 session_concurrency_benchmark.py \
  --target_model kimi-k2.5 --base_url https://testapi.lkeap.cloud.tencent.com/v3 --api_key <KEY> \
  --data_files /users/a.json,/users/b.json \
  --concurrency_levels 10,20,50 

# 多数据文件 + 多并发 + 持续 10 分钟（数据不够时循环跑）
python3 session_concurrency_benchmark.py \
  --target_model kimi-k2.5 --base_url https://testapi.lkeap.cloud.tencent.com/v3 --api_key <KEY> \
  --data_files a.json,b.json \
  --concurrency_levels 10,20,50 \
  --duration 600 ...

# 扫描目录下所有数据集文件依次进行发送
python3 session_concurrency_benchmark.py \
  --target_model kimi-k2.5 --base_url https://testapi.lkeap.cloud.tencent.com/v3 --api_key <KEY> \
  --data_dir /path/to/data \
  --concurrency_levels 10,50 \
  --duration 600 ...

"""

import json
import os
import re
import sys
import time
import uuid
import copy
import queue
import threading
import unicodedata
from collections import defaultdict
from decimal import Decimal
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from threading import Event, Lock
import httpx
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
import argparse

from openai import OpenAI

try:
    import ijson  # 流式解析超大 JSON 数组，避免一次性 read+json.loads 导致 OOM
    _has_ijson = True
except ImportError:
    ijson = None
    _has_ijson = False


def _parse_time_to_ms(value) -> int:
    """将 start_time 转为毫秒时间戳，兼容纯数字字符串和 ISO 8601 字符串。"""
    s = str(value).strip()
    if s.isdigit():
        return int(s)
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except ValueError:
        raise ValueError(f"无法解析 time 值: {s!r}")

# ==================== 默认配置 ====================

# data_dir = os.path.join(os.path.dirname(__file__), 'data')
data_dir = os.path.join(os.path.dirname(__file__), '')

DEFAULT_DATA_FILE = os.path.join(data_dir, "glm5-avg70K-cache0.83_7h.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")

TARGET_MODEL = "your-model-name"
API_KEY = "sk-your-api-key"
BASE_URL = "https://api.example.com/v1"

CONCURRENCY = 10
CONCURRENCY_LEVELS = [CONCURRENCY]  # 支持多并发，main() 中解析覆盖
ROUNDS = 1                  # 每个 (数据集, 并发) 组合重复跑的轮数，main() 中解析覆盖
STOP_THRESHOLD = 0.95       # 活跃并发 < 初始并发 * 该比例时触发停止
ENABLE_STREAM = True
ENABLE_UUID_PREFIX = False
# 运行轮次盐值（per-run salt）。同一轮回放共用同一 salt（保留真实命中），
# 切换并发/循环复用时更换 salt（消除重复回放假命中），用于测真实缓存命中率。
# 样例：content = "[NONCE:a3f5c9d1e2b4] 帮我写一个快速排序"
ENABLE_RUN_SALT = False
ENABLE_THINKING = True      # true → extra_body 追加 {"thinking":{"type":"enabled"}}；false → "disabled" 
# 自定义 thinking.type：设置后直接覆盖 thinking.type 的值（如 minimax-m3 仅支持 "adaptive"/"disabled"）。
# 为空（None）时退回 ENABLE_THINKING 的 enabled/disabled 行为，保持向后兼容。
THINKING_TYPE = None
# reasoning_effort：设置后会在请求参数中追加 {"reasoning_effort": <值>}（如 "high"/"medium"/"low"/"minimal"）。
# 为空（None）时不追加该字段，保持向后兼容。
REASONING_EFFORT = None
DURATION = 0                # 压测持续时长（秒），0 表示跑完所有 session 为止
REQUEST_TIMEOUT = 120       # 单条请求超时时间（秒），超时直接报错
# 流式请求的 read 超时（秒）：两个 chunk 之间最长等待时间。
# stop_event 触发后线程最多再等这么长时间就会因超时退出，无需等待下一个 chunk 自然到来。
# 设为较小值（如 10s）可加快长尾中断；但若模型 thinking 阶段静默超过此值，正常请求也会超时报错。
STREAM_READ_TIMEOUT = 300

# max_tokens 上限：设置后会覆盖原始请求中的 max_tokens / max_completion_tokens 字段。
# None 或 0 表示不覆盖，沿用原始请求中的值（若原始请求也未设置则由服务端默认）。
MAX_TOKENS = None

# session_id 透传 header 名：设置后每条请求会在 HTTP header 中附带 {该 header: session_id}，
# 便于网关 / 推理框架按 session 做亲和性路由（如同一 session 命中同一实例的 KV cache）。
# 为空（None）表示不附带该 header。
SESSION_ID_HEADER = None

# ==================== 乱码检测与日志（异步）====================
# 当模型返回的内容既不是中文也不是英文（而是其它脚本/替换字符等乱码）时，
# 将输入 prompt 与乱码输出追加写入日志文件，便于事后排查模型解码异常。
#
# 为避免拖慢发压关键路径：请求线程只做一次极快的入队操作（put_nowait），
# 真正的「乱码检测 + 写日志」由独立的后台 worker 线程消费执行，
# 因此完全不占用请求线程的时间，也不会污染任何性能指标的测量。
GARBLED_LOG_FILE = None            # 乱码日志文件路径，main() 中初始化
_garbled_log_lock: Lock = Lock()   # 多线程写日志的互斥锁
GARBLED_RATIO_THRESHOLD = 0.3      # 可疑字符（非中文/英文/常见符号）占比阈值
GARBLED_MIN_LEN = 10               # 有效字符数小于该值不判定（样本太少易误判）


def _script_family(ch: str) -> str:
    """粗略推断字符所属的 Unicode 脚本"家族"（如 CYRILLIC / ARABIC / HANGUL / HIRAGANA /
    DEVANAGARI / THAI / HEBREW / LATIN 等），取自 unicodedata 字符名称的首个单词。

    用于判断模型输出中的"非中文/英文"字符究竟属于哪种语言文字，
    进而与输入 prompt 所使用的文字进行比对（是否为正常的其它语言问答）。
    取不到名称（如未分配码点）时返回空字符串。
    """
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return ""
    return name.split(" ", 1)[0]


def _script_family_present(text: str, family: str, min_count: int = 1) -> bool:
    """判断文本中是否显著出现某个脚本家族（如 CYRILLIC/ARABIC 等）的字符。

    用于判断输入 prompt 是否本就是用该语言书写的（若是，说明模型只是"用请求
    所用的语言正常回答"，而非真正的解码乱码）。
    """
    if not text or not family:
        return False
    count = 0
    for ch in text:
        if ch.isspace():
            continue
        if _script_family(ch) == family:
            count += 1
            if count >= min_count:
                return True
    return False


def _extract_text_from_messages(messages) -> str:
    """从请求的 messages 中提取全部文本内容并拼接为一个字符串。

    兼容 content 为纯字符串、或为多模态 parts 列表（取其中 text 字段）两种格式；
    用于与模型输出做"请求语言 vs 回复语言"的比对，解析失败时安全返回空字符串。
    """
    if not messages:
        return ""
    parts = []
    try:
        for msg in messages:
            content = msg.get("content") if isinstance(msg, dict) else None
            if content is None:
                continue
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str):
                            parts.append(text)
    except Exception:
        return ""
    return "\n".join(parts)


def _looks_garbled(text: str,
                   prompt_text: str = None,
                   threshold: float = None,
                   min_len: int = None) -> bool:
    """判断文本是否为"非中文/英文"的乱码内容。

    判定规则（逐字符统计，忽略空白）：
      - 以下字符视为"可接受"，不计入可疑：
          * ASCII（英文字母、数字、常见标点）
          * 中文及中日韩标点、全角字符
          * 通用标点 / 符号（含 emoji）/ 数字类别（P* / S* / N* / M*）
      - 以下字符计入可疑：
          * Unicode 替换字符 U+FFFD（编码残缺信号）
          * 非中文的字母类（西里尔、希腊、阿拉伯、谚文、假名等）
      - 当可疑字符占比 >= threshold 时才判为乱码。

    注意：U+FFFD 只计入占比、不再"见一个就判乱码"。因为流式(SSE)输出时，
    一个多字节 UTF-8 字符可能被拆到两个 chunk 边界、偶发解码出单个 U+FFFD，
    这属于正常传输现象；只有整体占比超阈值（大量替换字符）才是真正的乱码。

    有效字符数小于 min_len 时直接返回 False，避免短文本误判。

    豁免场景（不算乱码）：若可疑字符占比超阈值，但触发的字符主要来自某个具体的
    "其它语言"文字（如西里尔/阿拉伯/谚文等，而非解码残缺的替换字符），且传入的
    prompt_text（输入请求原文）中本就显著出现同一种文字，则说明这是用户用该语言
    提问、模型照实用该语言回答的正常多语言问答场景，不应判定为乱码。
    """
    if not text:
        return False
    thr = GARBLED_RATIO_THRESHOLD if threshold is None else threshold
    mlen = GARBLED_MIN_LEN if min_len is None else min_len

    suspicious = 0
    total = 0
    replacement_count = 0          # Unicode 替换字符计数（真实解码异常，不可豁免）
    script_counter: dict = {}      # 非替换字符的可疑字符，按脚本家族计数

    for ch in text:
        if ch.isspace():
            continue
        total += 1
        cp = ord(ch)
        # Unicode 替换字符：计入可疑参与占比（单个偶发不判乱码，大量才判）
        if ch == "\ufffd":
            suspicious += 1
            replacement_count += 1
            continue
        # ASCII：英文/数字/标点
        if cp < 128:
            continue
        # 中文及中日韩标点、全角 ASCII/标点
        if (
            0x4E00 <= cp <= 0x9FFF          # CJK 统一表意文字
            or 0x3400 <= cp <= 0x4DBF       # CJK 扩展 A
            or 0xF900 <= cp <= 0xFAFF       # CJK 兼容表意文字
            or 0x3000 <= cp <= 0x303F       # CJK 符号和标点
            or 0xFF00 <= cp <= 0xFFEF       # 全角 ASCII / 半角形式
        ):
            continue
        # 通用标点、符号（含 emoji）、数字：视为可接受
        cat = unicodedata.category(ch)
        if cat[0] in ("P", "S", "N", "M"):
            continue
        # 其余（非中文的字母类，如西里尔/希腊/阿拉伯/谚文/假名等）计入可疑
        suspicious += 1
        family = _script_family(ch)
        if family:
            script_counter[family] = script_counter.get(family, 0) + 1

    if total < mlen or suspicious == 0:
        return False
    if (suspicious / total) < thr:
        return False

    # ---- 触发比例阈值后，二次校验：排除"请求与回复用同一种其它语言"的正常场景 ----
    # 可疑字符主要是替换字符（真实解码残缺）时，不做语言豁免。
    if replacement_count / suspicious >= 0.5:
        return True

    if script_counter:
        # 回复中占比最高的"其它语言"脚本家族
        dominant_family, _ = max(script_counter.items(), key=lambda kv: kv[1])
        # 输入 prompt 本就显著使用同一脚本家族 => 属于正常的其它语言问答，不算乱码
        if prompt_text and _script_family_present(prompt_text, dominant_family,
                                                    min_count=max(mlen // 2, 1)):
            return False

    return True


def _log_garbled_content(session_id, req_id, messages, answer):
    """将检测到的乱码输出及其输入 prompt 线程安全地追加写入日志文件。"""
    if not GARBLED_LOG_FILE:
        return
    try:
        prompt_text = (
            json.dumps(messages, ensure_ascii=False, indent=2)
            if messages is not None else ""
        )
    except Exception:
        prompt_text = str(messages)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = (
        f"\n{'='*80}\n"
        f"[{ts}] 检测到疑似乱码输出  session_id={session_id}  req_id={req_id}\n"
        f"{'-'*80}\n"
        f"[输入 Prompt]\n{prompt_text}\n"
        f"{'-'*80}\n"
        f"[乱码输出]\n{answer}\n"
        f"{'='*80}\n"
    )
    with _garbled_log_lock:
        try:
            with open(GARBLED_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as ex:
            print(f"  ⚠️ 写入乱码日志失败: {ex}")


# ---- 异步处理：后台 worker 线程消费队列，执行检测 + 写日志 ----
_GARBLED_QUEUE_MAXSIZE = 10000     # 队列上限；满时丢弃新任务（只影响日志完整性，不阻塞发压）
_garbled_queue = None              # type: queue.Queue | None
_garbled_worker = None             # type: threading.Thread | None
_garbled_stop_sentinel = object()  # 停止哨兵：worker 取到它后退出
_garbled_dropped = [0]             # 队列满丢弃计数（列表包装以便闭包内修改）
_garbled_dropped_lock: Lock = Lock()


def _garbled_worker_loop():
    """后台 worker 主循环：从队列取任务，执行乱码检测并写日志。

    检测（CPU）与写盘（I/O）都在此线程内完成，请求线程完全不参与，
    因此不会增加请求耗时、也不会影响 TTFB/TTLB/TPM 等性能指标。
    """
    while True:
        item = _garbled_queue.get()
        try:
            if item is _garbled_stop_sentinel:
                break
            session_id, req_id, messages, answer = item
            try:
                prompt_text = _extract_text_from_messages(messages)
                if answer and _looks_garbled(answer, prompt_text=prompt_text):
                    _log_garbled_content(session_id, req_id, messages, answer)
            except Exception as ex:
                print(f"  ⚠️ 乱码异步处理异常: {ex}")
        finally:
            _garbled_queue.task_done()


def _start_garbled_worker():
    """启动乱码检测后台线程（幂等）。GARBLED_LOG_FILE 未设置时不启动。"""
    global _garbled_queue, _garbled_worker
    if not GARBLED_LOG_FILE:
        return
    if _garbled_worker is not None and _garbled_worker.is_alive():
        return
    _garbled_queue = queue.Queue(maxsize=_GARBLED_QUEUE_MAXSIZE)
    _garbled_worker = threading.Thread(
        target=_garbled_worker_loop, daemon=True, name="garbled-log-worker"
    )
    _garbled_worker.start()


def _enqueue_garbled_check(session_id, req_id, messages, answer):
    """把一次响应放入异步检测队列（非阻塞）。

    请求线程只调用它，耗时 = 一次 put_nowait（O(1)，微秒级）。
    队列满时直接丢弃并计数，绝不阻塞发压关键路径。
    """
    if not GARBLED_LOG_FILE or _garbled_queue is None or not answer:
        return
    try:
        _garbled_queue.put_nowait((session_id, req_id, messages, answer))
    except queue.Full:
        with _garbled_dropped_lock:
            _garbled_dropped[0] += 1


def _stop_garbled_worker(timeout: float = 60.0):
    """发送停止哨兵并等待后台线程把剩余队列处理完退出（压测全部结束后调用）。"""
    global _garbled_worker
    if _garbled_worker is None or _garbled_queue is None:
        return
    try:
        _garbled_queue.put(_garbled_stop_sentinel)
    except Exception:
        pass
    _garbled_worker.join(timeout=timeout)
    with _garbled_dropped_lock:
        dropped = _garbled_dropped[0]
    if dropped:
        print(f"  ⚠️ 乱码检测队列曾满，丢弃 {dropped} 条未检测"
              f"（如需完整可调大 _GARBLED_QUEUE_MAXSIZE）")
    _garbled_worker = None


# ==================== 重复输出（退化复读）检测与日志（异步）====================
# 部分模型在异常情况下会陷入"退化重复"：同一句话/同一行内容被反复输出，
# 直到达到 max_tokens 或超时才停止（俗称"复读机"现象）。这类输出虽然可能
# finish_reason 正常（如 "length"），但内容本身完全不可用，需要单独检测出来。
#
# 检测方式：先剥离文本中类似工具调用协议标签的样式（见下方 _strip_protocol_tags
# 说明），再按"行"、并对每行进一步按"句子结束符"分段（见下方 _split_into_segments
# 说明），得到一组自然语义单元（整行/整句）；统计每个分段【完全相同】的出现次数——
# 若某个分段重复出现次数达到阈值，说明这一整句/整行内容在文本中被反复输出，
# 再结合覆盖占比二次校验（见下方 REPEAT_MIN_COVERAGE_RATIO 说明）进一步排除
# 结构化重复导致的误判，即可判定为退化重复。
#
# 相比"固定大小字符滑动窗口"的做法，按行/按句分段后要求分段【整体完全一致】
# 才计数，天然能规避"多次工具调用参数只是共享较长公共前缀、但整行/整句内容
# 本身并不相同"（如仅文件名后缀不同的多个文件路径）导致的误判——因为只要
# 整行/整句有任何差异（如文件名不同），就不会被判定为同一分段，不会被计数。
#
# 与乱码检测同样采用「请求线程只做一次极快入队 + 后台线程异步检测写日志」的
# 设计，完全不占用请求关键路径、不影响任何性能指标的测量。
REPEAT_LOG_FILE = None             # 重复输出日志文件路径，main() 中初始化
_repeat_log_lock: Lock = Lock()    # 多线程写日志的互斥锁
REPEAT_MIN_COUNT = 5               # 同一分段(整行/整句)出现次数 >= 该值才进入候选（还需下方覆盖占比校验）
REPEAT_MIN_LEN = 40                # 输出长度小于该值不判定（短文本正常重复概率高，易误判）
REPEAT_SCAN_LIMIT = 50000          # 单次检测最多扫描的字符数（防止超长输出导致内存/耗时过大）
REPEAT_MIN_SEGMENT_LEN = 8         # 单个分段(整行/整句)长度小于该值不参与统计，避免"好的"/"是的"
                                    # 这类过短的正常应答或空行、短碎片被误判为重复
REPEAT_MIN_COVERAGE_RATIO = 0.3    # 重复分段覆盖字符数占全部有效分段总长度的最小比例，才判定为真正退化复读
                                    # 背景：模型进行多次相似的工具调用（如连续 Read 多个不同文件）时，
                                    # 各次调用天然会共享很长的协议标签模板 + 相同的目录路径前缀，
                                    # 仅看"某个分段出现 >=5 次"仍可能受个别巧合影响；而真正的退化
                                    # 复读会持续占据输出的绝大部分。因此要求命中分段的总覆盖字符数
                                    # （出现次数 × 分段长度）占全部有效分段总长度的比例也达到阈值，
                                    # 才认定为真实的模型退化重复。

# 类似工具调用协议标签的样式：形如 <invoke name="Read">、</parameter>、
# <｜DSML｜parameter name="file_path" string="true">、<｜DSML｜tool_calls> 等
# 尖括号包裹的标记（无论具体标签名/命名空间前缀是什么）。限制单个标签长度
# （不含换行、且不超过 80 字符）以避免误伤跨多行的正文内容或超长片段。
_PROTOCOL_TAG_PATTERN = re.compile(r'</?[^<>\n]{1,80}>')

# 句子结束符：用于在同一行内进一步切分出独立的句子分段（应对退化复读发生在
# 同一行、未换行分隔的情形）。中文句末标点（。！？；）不会出现在文件路径/域名/
# 版本号等标识符中，可直接在其后切分；但英文标点（.!?;）在文件路径（如
# "b.py"）、域名、版本号（如 "v1.2.3"）中大量正常出现，若同样"见到就切"会把
# 这类标识符错误拆散（如把 "D:/a/b.py" 切成 "D:/a/b." 和 "py" 两段），反而导致
# 完全相同的路径因为被拆散成不同分段而无法命中重复。因此英文标点只有在其后
# 紧跟空白字符或到达字符串末尾时（模拟正常英文书写习惯：句末标点后接空格或
# 结束），才视为真正的句子结束，否则视为标识符内部的普通字符，不切分。
_SENTENCE_SPLIT_PATTERN = re.compile(r'(?:(?<=[。！？；])\s*|(?<=[.!?;])(?=\s|$))')


def _strip_protocol_tags(text: str) -> str:
    """剥离文本中类似工具调用协议标签的样式，只保留标签之间/之外的实际内容
    （如具体的参数值、自然语言正文）用于重复检测。

    背景：模型进行多次相似的工具调用时，每次调用都会被同样的协议标签模板包裹
    （如 <invoke name="Read"><parameter name="file_path">...），这部分模板本身
    高度重复，但每次调用真正传入的参数值（如文件路径）往往并不相同——这类重复
    属于正常的结构化输出，不应被判定为模型退化复读。将标签模板剥离后再做分段
    统计，使检测只针对"实际内容是否重复"，不再受协议模板本身重复的干扰。
    """
    if not text:
        return text
    return _PROTOCOL_TAG_PATTERN.sub("", text)


def _split_into_segments(text: str) -> list:
    """将文本按"行"分段，再对每一行按"句子结束符"进一步切分，得到一组自然语义
    单元（整行 或 整句）的列表（已去除首尾空白，空分段已过滤）。

    这样得到的每个分段代表模型输出中一个完整的"行"或"句子"，用于后续按分段
    做精确匹配的重复统计——只有分段整体完全相同才算重复，不会像固定长度的
    字符滑动窗口那样，把"仅有细微差异（如文件名不同）但存在较长公共前缀"的
    不同内容误判为同一片段的重复。
    """
    segments = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for part in _SENTENCE_SPLIT_PATTERN.split(line):
            part = part.strip()
            if part:
                segments.append(part)
    return segments


def _looks_repetitive(text: str,
                      min_count: int = None,
                      min_len: int = None,
                      min_coverage_ratio: float = None,
                      min_segment_len: int = None) -> tuple:
    """检测文本中是否存在模型输出退化重复（"复读机"现象）。

    算法：先用 _strip_protocol_tags 剥离类似工具调用协议标签的样式（协议模板
    本身的重复不计入统计），再用 _split_into_segments 按"行/句子"分段，过滤掉
    长度小于 min_segment_len 的过短分段后，统计每个分段【完全相同】的出现
    次数；若某个分段的出现次数达到 min_count，先作为候选，再校验该分段的总
    覆盖字符数（出现次数 × 分段长度）占全部有效分段总长度的比例是否达到
    min_coverage_ratio，两者都满足才判定为退化重复。

    （要求分段整体完全一致才计数，天然规避"多次工具调用参数仅共享较长公共
    前缀、但整行/整句本身并不相同"导致的误判；覆盖占比校验进一步排除个别
    巧合重复——真正卡死的退化复读会持续占据输出的绝大部分。）

    剥离协议标签后文本长度小于 min_len 时直接判定为不重复，避免短文本正常
    重复被误判。超长文本仅扫描前 REPEAT_SCAN_LIMIT 个字符，避免内存/耗时过大
    （退化重复一般会在输出早期就已出现，扫描前缀足以检测）。

    Returns:
        (是否判定为重复, 重复片段, 出现次数)；未判定为重复时返回 (False, "", 0)。
    """
    if not text:
        return False, "", 0
    mcount = REPEAT_MIN_COUNT if min_count is None else min_count
    mlen = REPEAT_MIN_LEN if min_len is None else min_len
    mratio = REPEAT_MIN_COVERAGE_RATIO if min_coverage_ratio is None else min_coverage_ratio
    mseglen = REPEAT_MIN_SEGMENT_LEN if min_segment_len is None else min_segment_len

    stripped = _strip_protocol_tags(text)
    if len(stripped) < mlen:
        return False, "", 0

    scan_text = stripped[:REPEAT_SCAN_LIMIT]
    segments = [s for s in _split_into_segments(scan_text) if len(s) >= mseglen]
    if not segments:
        return False, "", 0

    counts: dict = {}
    for seg in segments:
        counts[seg] = counts.get(seg, 0) + 1
    best_segment, best_count = max(counts.items(), key=lambda kv: kv[1])

    if best_count < mcount:
        return False, "", 0

    total_len = sum(len(s) for s in segments)
    coverage = (best_count * len(best_segment)) / total_len if total_len else 0
    if coverage < mratio:
        return False, "", 0

    return True, best_segment, best_count


def _log_repeat_content(session_id, req_id, messages, answer, cot, source, fragment, count):
    """将检测到的退化重复输出及其输入 prompt 线程安全地追加写入日志文件。

    source: 命中重复检测的字段来源，取值 "content"（正文回答）或
            "reasoning_content"（思考过程），日志中会明确标注，便于区分是
            正文复读还是模型陷入循环思考。
    """
    if not REPEAT_LOG_FILE:
        return
    try:
        prompt_text = (
            json.dumps(messages, ensure_ascii=False, indent=2)
            if messages is not None else ""
        )
    except Exception:
        prompt_text = str(messages)

    source_label = "content（正文回答）" if source == "content" else "reasoning_content（思考过程）"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = (
        f"\n{'='*80}\n"
        f"[{ts}] 检测到疑似重复输出（退化复读）  session_id={session_id}  req_id={req_id}"
        f"  来源字段={source_label}\n"
        f"{'-'*80}\n"
        f"[输入 Prompt]\n{prompt_text}\n"
        f"{'-'*80}\n"
        f"[重复片段]（出现 {count} 次，分段长度 {len(fragment)} 字符）\n{fragment}\n"
        f"{'-'*80}\n"
        f"[完整 content]\n{answer}\n"
        f"{'-'*80}\n"
        f"[完整 reasoning_content]\n{cot}\n"
        f"{'='*80}\n"
    )
    with _repeat_log_lock:
        try:
            with open(REPEAT_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as ex:
            print(f"  ⚠️ 写入重复输出日志失败: {ex}")


# ---- 异步处理：后台 worker 线程消费队列，执行检测 + 写日志 ----
_REPEAT_QUEUE_MAXSIZE = 10000      # 队列上限；满时丢弃新任务（只影响日志完整性，不阻塞发压）
_repeat_queue = None               # type: queue.Queue | None
_repeat_worker = None              # type: threading.Thread | None
_repeat_stop_sentinel = object()   # 停止哨兵：worker 取到它后退出
_repeat_dropped = [0]              # 队列满丢弃计数（列表包装以便闭包内修改）
_repeat_dropped_lock: Lock = Lock()


def _repeat_worker_loop():
    """后台 worker 主循环：从队列取任务，执行重复输出检测并写日志。

    只检测模型的 content（正文回答）与 reasoning_content（思考过程）两个字段，
    二者分开独立判断（避免拼接处产生人为的假重复），命中任一即写日志并标注来源；
    tool_calls（工具调用的名称/参数）不在检测范围内 —— 工具调用参数天然可能包含
    结构相似甚至相同的片段（如连续多次调用同一函数），这不属于模型"退化复读"，
    不应被判定为重复输出。

    检测（CPU）与写盘（I/O）都在此线程内完成，请求线程完全不参与，
    因此不会增加请求耗时、也不会影响 TTFB/TTLB/TPM 等性能指标。
    """
    while True:
        item = _repeat_queue.get()
        try:
            if item is _repeat_stop_sentinel:
                break
            session_id, req_id, messages, answer, cot = item
            try:
                if answer:
                    is_repeat, fragment, count = _looks_repetitive(answer)
                    if is_repeat:
                        _log_repeat_content(session_id, req_id, messages, answer, cot,
                                            "content", fragment, count)
                if cot:
                    is_repeat, fragment, count = _looks_repetitive(cot)
                    if is_repeat:
                        _log_repeat_content(session_id, req_id, messages, answer, cot,
                                            "reasoning_content", fragment, count)
            except Exception as ex:
                print(f"  ⚠️ 重复输出异步检测异常: {ex}")
        finally:
            _repeat_queue.task_done()


def _start_repeat_worker():
    """启动重复输出检测后台线程（幂等）。REPEAT_LOG_FILE 未设置时不启动。"""
    global _repeat_queue, _repeat_worker
    if not REPEAT_LOG_FILE:
        return
    if _repeat_worker is not None and _repeat_worker.is_alive():
        return
    _repeat_queue = queue.Queue(maxsize=_REPEAT_QUEUE_MAXSIZE)
    _repeat_worker = threading.Thread(
        target=_repeat_worker_loop, daemon=True, name="repeat-log-worker"
    )
    _repeat_worker.start()


def _enqueue_repeat_check(session_id, req_id, messages, answer, cot=""):
    """把一次响应放入异步检测队列（非阻塞）。

    只传入 content（answer）与 reasoning_content（cot），不包含 tool_calls；
    请求线程只调用它，耗时 = 一次 put_nowait（O(1)，微秒级）。
    队列满时直接丢弃并计数，绝不阻塞发压关键路径。
    """
    if not REPEAT_LOG_FILE or _repeat_queue is None or not (answer or cot):
        return
    try:
        _repeat_queue.put_nowait((session_id, req_id, messages, answer, cot))
    except queue.Full:
        with _repeat_dropped_lock:
            _repeat_dropped[0] += 1


def _stop_repeat_worker(timeout: float = 60.0):
    """发送停止哨兵并等待后台线程把剩余队列处理完退出（压测全部结束后调用）。"""
    global _repeat_worker
    if _repeat_worker is None or _repeat_queue is None:
        return
    try:
        _repeat_queue.put(_repeat_stop_sentinel)
    except Exception:
        pass
    _repeat_worker.join(timeout=timeout)
    with _repeat_dropped_lock:
        dropped = _repeat_dropped[0]
    if dropped:
        print(f"  ⚠️ 重复输出检测队列曾满，丢弃 {dropped} 条未检测"
              f"（如需完整可调大 _REPEAT_QUEUE_MAXSIZE）")
    _repeat_worker = None


# ==================== 全局停止信号 ====================
stop_event: Event = Event()    # 全局中断：当前进行中的请求立即取消
                               # （duration>0 timer 到点 / duration=0 队列耗尽 + 活跃数 < 阈值）

# 活跃流集合：stop_event 触发后由 watcher 线程主动 close，避免等待下一个 chunk
_active_streams: set = set()
_streams_lock: Lock = Lock()


def _start_stream_watcher():
    """启动一个 watcher 线程：等待 stop_event，触发后持续 close 活跃流。

    每轮压测开始前调用一次，确保每轮都有有效的 watcher。

    持续监听设计：
      stop_event.set() 与工作线程 ``_active_streams.add(stream)`` 之间存在竞态
      窗口 —— ``client.chat.completions.create()`` 返回后才能拿到 stream 对象，
      在 add 之前如果 watcher 已经跑完一次 close 扫描，这个新流就会"漏关"，
      只能等 STREAM_READ_TIMEOUT 自然超时。
      因此 stop_event.set() 后进入多轮扫描，覆盖竞态期内陆续注册进来的流；
      当 _active_streams 连续若干轮为空时退出。
    """
    def _watch():
        stop_event.wait()
        # 已 close 过的流不再重复 close（避免无意义调用 / 异常日志）
        closed: set = set()
        # 连续空轮计数：达到阈值后退出（覆盖竞态窗口已足够）
        empty_rounds = 0
        max_empty_rounds = 20  # ≈ 2s（每轮 sleep 0.1s）
        # 硬上限：最多扫描 100 轮（10s），防止异常情况下 watcher 永不退出
        max_total_rounds = 100
        for _ in range(max_total_rounds):
            with _streams_lock:
                streams = [s for s in _active_streams if id(s) not in closed]
            if streams:
                empty_rounds = 0
                for stream in streams:
                    try:
                        stream.close()
                    except Exception:
                        pass
                    closed.add(id(stream))
            else:
                empty_rounds += 1
                if empty_rounds >= max_empty_rounds:
                    break
            time.sleep(0.1)
    t = threading.Thread(target=_watch, daemon=True)
    t.start()
    return t


def _drain_tail_futures(remaining, executor, label="长尾"):
    """统一的"等待长尾 worker 退出"逻辑（分两段超时 + 兜底 hard-cancel）。

    设计：
      第一段 ``close_window``：watcher close() 已生效时，正在阻塞在 stream
        ``__next__()`` 的 worker 应在秒级抛 ReadError/ClosedError 退出，绝大多数
        worker 在此期间结束。
      第二段 ``read_window``：兜底等待 SDK 内部尚未通过 socket read 走到 close
        感知点的请求，最长再等 STREAM_READ_TIMEOUT 秒（对应 httpx.Timeout.read）。
      仍未退出的 worker：调用 ``cancel()`` 把"未启动"的回收掉，然后
      ``shutdown(wait=False, cancel_futures=True)``（Py 3.9+，老版本退化为 wait=False），
      让 ThreadPoolExecutor 的 __exit__ 不再阻塞主流程。
      已 running 的 worker 仍会持续占用，但属于 daemon 线程，不影响进程退出；
      调用方应在档间重建 client 以释放它们持有的 httpx 连接池。

    Args:
        remaining: 需要等待的 future 列表（worker future）
        executor:  对应的 ThreadPoolExecutor，用于最终强制 shutdown
        label:     日志前缀，便于区分场景
    """
    if not remaining:
        return

    close_window = 30  # 第一段：等 watcher.close() 起效
    read_window = STREAM_READ_TIMEOUT  # 第二段：兜底等 ReadTimeout
    seen: set = set()  # 跨段去重，避免同一 future 被处理两次

    def _collect(timeout):
        try:
            for f in as_completed(remaining, timeout=timeout):
                if id(f) in seen:
                    continue
                seen.add(id(f))
                try:
                    f.result()
                except Exception as ex:
                    print(f"  ⚠️ Worker 异常: {ex}")
            return True
        except (TimeoutError, FuturesTimeoutError):
            # Python 3.10: concurrent.futures.TimeoutError 不是 builtin TimeoutError
            return False

    if _collect(close_window):
        return

    pending = sum(1 for f in remaining if not f.done())
    print(f"  ⏳ {label}：close() 后仍有 {pending} 个 Worker 未结束，"
          f"再兜底等待 {read_window}s（read timeout）...")

    if _collect(read_window):
        return

    unfinished = [f for f in remaining if not f.done()]
    print(f"  ⚠️ {label}等待超时（共 {close_window + read_window}s），"
          f"仍有 {len(unfinished)} 个 Worker 未完成，强制跳过；"
          f"对应工作线程将继续后台运行直到自身超时（建议档间重建 client）")

    for f in unfinished:
        try:
            f.cancel()
        except Exception:
            pass
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        executor.shutdown(wait=False)


# ==================== 参数解析 ====================

def parse_arguments():
    parser = argparse.ArgumentParser(description='Session 并发压测脚本')
    parser.add_argument('--data_files', type=str, default='',
                        help='测试数据文件列表，逗号分隔（支持绝对路径或相对 data_dir 的文件名）；'
                             '优先级高于 --data_dir')
    parser.add_argument('--data_dir', type=str, default='',
                        help='测试数据目录，自动加载目录内所有 *.json / *.jsonl 文件')
    parser.add_argument('--session_key', type=str, default='session_id',
                        help='用于区分 session 的字段名（默认: session_id）')
    parser.add_argument('--time_key', type=str, default='start_time',
                        help='session 内排序用的时间字段名（默认: start_time）')
    parser.add_argument('--session_order', type=str, default='file',
                        choices=['file', 'start_time', 'size', 'shuffle'],
                        help='session 之间的下发顺序：file=数据文件中首次出现顺序（默认，向后兼容）；'
                             'start_time=按 session 首条记录的时间升序；size=按 session 请求条数降序；'
                             'shuffle=按 --order_seed 固定随机打乱。四种模式都是确定性的，可复现。')
    parser.add_argument('--order_seed', type=int, default=0,
                        help='--session_order shuffle 的随机种子（默认: 0）。同一 seed + 同一数据文件 → 同一顺序。')
    parser.add_argument('--replay_id', type=str, default='',
                        help='本次回放标识，写入 --dump_plan 的输出与结果文件名，便于事后按 id 对照复现')
    parser.add_argument('--dump_plan', type=str, default='',
                        help='把本次实际下发计划（顺序 + 每条的时间戳 + prompt 指纹）落盘到该 JSON 文件，'
                             '两次运行 diff 即可确认"下发顺序是否完全一致"。空表示不落盘（默认: 空）')
    parser.add_argument('--session_id_header', type=str, default='',
                        help='将每条请求所属的 session_id 通过指定的 HTTP header 名透传给服务端，'
                             '便于网关/推理框架按 session 做亲和性路由（如 KV cache 命中），'
                             '例如 --session_id_header x-session-id。空表示不附带（默认: 空）')
    parser.add_argument('--target_model', type=str, default=TARGET_MODEL,
                        help=f'目标模型名称（默认: {TARGET_MODEL}）')
    parser.add_argument('--api_key', type=str, default=API_KEY,
                        help='API 密钥')
    parser.add_argument('--base_url', type=str, default=BASE_URL,
                        help=f'API 基础 URL（默认: {BASE_URL}）')
    parser.add_argument('--concurrency_levels', type=str, default=str(CONCURRENCY),
                        help=f'并发级别列表，逗号分隔，如 "10,20,50"（默认: {CONCURRENCY}）')
    parser.add_argument('--rounds', type=int, default=ROUNDS,
                        help=f'每个 (数据集, 并发) 组合重复跑的轮数，结果在汇总表里按轮次多行呈现。'
                             f'用于对同一配置多次测量观察稳定性（默认: {ROUNDS}）')
    parser.add_argument('--tail_threshold', type=float, default=STOP_THRESHOLD,
                        help=f'长尾中断阈值：活跃并发 < 初始并发 * 该比例时停止（默认: {STOP_THRESHOLD}）')
    parser.add_argument('--enable_stream', type=lambda x: x.lower() == 'true',
                        default=ENABLE_STREAM,
                        help='是否使用流式请求（默认: True）')
    parser.add_argument('--enable_uuid_prefix', type=lambda x: x.lower() == 'true',
                        default=ENABLE_UUID_PREFIX,
                        help='是否在 content 前添加随机 UUID 前缀以避免命中缓存（默认: False）。'
                             '注意：每请求换新 uuid 会破坏全部前缀缓存（含 session 内多轮）→ 命中率≈0，'
                             '测的是纯冷启动/最坏情况，而非真实命中率。')
    parser.add_argument('--enable_run_salt', type=lambda x: x.lower() == 'true',
                        default=ENABLE_RUN_SALT,
                        help='方案 B：是否启用"运行轮次盐值"（默认: False）。同一轮数据回放共用同一 salt，'
                             '保留 session 内多轮/跨 session 共享前缀的真实命中；切换并发级别或 duration '
                             '循环复用时更换 salt，消除"重复回放同一数据"的假命中，用于测真实缓存命中率。'
                             '与 --enable_uuid_prefix 互斥（uuid 前缀开启时本项不生效）。')
    parser.add_argument('--enable_thinking', type=lambda x: x.lower() == 'true',
                        default=ENABLE_THINKING,
                        help='thinking 模式：true → extra_body 追加 {"thinking":{"type":"enabled"}}，'
                             'false → 追加 {"thinking":{"type":"disabled"}}（默认: True）')
    parser.add_argument('--thinking_type', type=str, default=THINKING_TYPE,
                        help='直接指定 thinking.type 的值（如 enabled/disabled/adaptive）。'
                             '设置后覆盖 --enable_thinking；不设置时沿用 --enable_thinking 的 enabled/disabled。'
                             '例如 minimax-m3 仅支持 adaptive/disabled，需传 --thinking_type adaptive')
    parser.add_argument('--reasoning_effort', type=str, default=REASONING_EFFORT,
                        help='访问模型时传入的 reasoning_effort 参数（如 high/medium/low/minimal）。'
                             '设置后会在请求参数中追加 {"reasoning_effort": <值>}；不设置则不追加（默认: 不追加）')
    parser.add_argument('--duration', type=int, default=DURATION,
                        help='压测持续时长（秒），超时后自动停止；0 表示跑完所有 session 为止（默认: 0）。'
                             '开启后 session 队列耗尽时会自动从头循环，保持并发持续运行直到超时。')
    parser.add_argument('--request_timeout', type=int, default=REQUEST_TIMEOUT,
                        help=f'单条请求超时时间（秒），超时直接报错（默认: {REQUEST_TIMEOUT}）')
    parser.add_argument('--stream_read_timeout', type=int, default=STREAM_READ_TIMEOUT,
                        help=f'流式请求两个 chunk 之间的最长等待时间（秒），stop_event 触发后最多等待此时长退出（默认: {STREAM_READ_TIMEOUT}）')
    parser.add_argument('--max_tokens', type=int, default=None,
                        help='请求的 max_tokens 上限，设置后会覆盖原始请求中的 max_tokens / max_completion_tokens 字段；'
                             '不传或传 0 表示不覆盖，沿用原始请求中的值（默认: 不覆盖）')
    parser.add_argument('--pre_sample', type=int, default=0,
                        help='压测前按 start_time 顺序确定性地截取前 N 条记录作为输入数据集，'
                             '保证多次运行使用完全相同的请求集合（结果可重复），0 表示不截取（默认: 0）。')
    parser.add_argument('--max_samples_per_file', type=int, default=0,
                        help='【防 OOM】加载阶段每个数据文件最多读取的原始记录条数，0 表示不限。'
                             '对超大 JSON（GB 级）建议设置此值，使用 ijson 流式解析、读到该条数后立即停止；'
                             '与 --pre_sample 兼容：先在加载阶段截断，再做 pre_sample 精排（默认: 0 = 不限）。')
    parser.add_argument('--tpm_cache_filter', type=str, default='',
                        help='【TPM 趋势报告】按 cache_hit_ratio 筛选连续分钟窗口并在报告中聚合 TPM。'
                             '支持表达式：">=0.8" / ">0.8" / "<=0.5" / "<0.5" / "=0.9"，'
                             '以及闭区间 "0.7..0.9"。命中段会在 Excel 报告中新增 "Cache 区间汇总" sheet，'
                             '并在趋势图上叠加阴影标注。空字符串表示不做 cache 段筛选（默认: 空）。')
    parser.add_argument('--garbled_log', type=str, default='',
                        help='乱码日志文件路径：当模型输出为非中文/英文的乱码时，'
                             '将输入 prompt 与乱码输出追加写入该文件。'
                             '若输出的"非中文/英文"内容与输入 prompt 使用的语言一致'
                             '（如用户用俄语/韩语等提问，模型照实用该语言回答），则不算乱码。'
                             '空表示使用默认路径 results/<model>_garbled_<timestamp>.log（默认: 空）')
    parser.add_argument('--garbled_threshold', type=float, default=GARBLED_RATIO_THRESHOLD,
                        help='乱码判定阈值：可疑字符（非中文/英文/常见符号）占比 >= 该值判为乱码'
                             f'（默认: {GARBLED_RATIO_THRESHOLD}）')
    parser.add_argument('--repeat_log', type=str, default='',
                        help='重复输出（退化复读）日志文件路径：当模型输出中某个整行/整句内容'
                             '反复出现次数达到阈值时，将输入 prompt 与完整输出追加写入该文件。'
                             '空表示使用默认路径 results/<model>_repeat_<timestamp>.log（默认: 空）')
    parser.add_argument('--repeat_min_count', type=int, default=REPEAT_MIN_COUNT,
                        help='重复判定候选阈值：同一分段(整行/整句)在输出中出现的次数 >= 该值才进入候选'
                             f'（还需 --repeat_min_coverage_ratio 覆盖占比校验才最终判定，默认: {REPEAT_MIN_COUNT}）')
    parser.add_argument('--repeat_min_segment_len', type=int, default=REPEAT_MIN_SEGMENT_LEN,
                        help='参与重复统计的分段(整行/整句)最小长度（字符数），小于该值的分段'
                             '（如"好的"/"是的"等过短应答或空行）不参与统计，避免误判'
                             f'（默认: {REPEAT_MIN_SEGMENT_LEN}）')
    parser.add_argument('--repeat_min_coverage_ratio', type=float, default=REPEAT_MIN_COVERAGE_RATIO,
                        help='重复判定覆盖占比阈值：命中分段的总覆盖字符数（出现次数×分段长度）'
                             '占全部有效分段总长度的比例 >= 该值才判定为真正退化复读。用于进一步排除'
                             '个别巧合重复导致的误判'
                             f'（默认: {REPEAT_MIN_COVERAGE_RATIO}）')
    return parser.parse_args()


# ==================== 数据加载 ====================

def _normalize_decimals(obj):
    """递归把 ijson 产出的 Decimal 转成原生 int/float，避免后续 json.dumps 序列化失败。

    ijson 默认把 JSON 数字解析为 ``decimal.Decimal``，而标准库 ``json`` 无法序列化
    Decimal（报 ``Object of type Decimal is not JSON serializable``）。这里在产出
    记录前统一归一化为 int（无小数）或 float。
    """
    if isinstance(obj, Decimal):
        # 整数值用 int，否则 float
        return int(obj) if obj == obj.to_integral_value() else float(obj)
    if isinstance(obj, dict):
        return {k: _normalize_decimals(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize_decimals(v) for v in obj]
    return obj


def _iter_records_streaming(data_file: str, max_samples: int = 0):
    """流式逐条产出记录，避免对 GB 级文件一次性 read()+json.loads() 导致 OOM。

    自动嗅探格式：
      - 顶层 JSON 数组 ``[{...}, {...}]``：优先 ijson 流式解析。
      - JSON Lines（每行一个 object）：按行 ``json.loads``。
      - 顶层是单个 object：回退到 json.load 整体加载并把它当作一条记录。

    :param data_file: 数据文件路径
    :param max_samples: 最多产出多少条，0 表示不限
    :yield: (record_dict, mode_used_str)
    """
    # 嗅探首字符判断结构
    with open(data_file, "rb") as f:
        head = f.read(64).lstrip()
    is_array = head.startswith(b"[")
    is_object = head.startswith(b"{")
    ext = os.path.splitext(data_file)[1].lower()

    # 1) JSON 数组：ijson 流式
    if is_array:
        if not _has_ijson:
            raise RuntimeError(
                "检测到大型 JSON 数组，但未安装 ijson。请执行：pip install ijson"
            )
        count = 0
        with open(data_file, "rb") as f:
            for obj in ijson.items(f, "item"):
                yield _normalize_decimals(obj), "ijson(array)"
                count += 1
                if max_samples and count >= max_samples:
                    return
        return

    # 2) JSONL / NDJSON 或裸的非 [/{ 起始（罕见）
    if ext in (".jsonl", ".ndjson") or (not is_array and not is_object):
        count = 0
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line), "jsonl(line)"
                count += 1
                if max_samples and count >= max_samples:
                    return
        return

    # 3) 顶层是单个 object：无法流式，整体加载（小文件兜底）
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    yield data, "json.load(object)"


def load_data(data_file: str, session_key: str, time_key: str,
              max_samples: int = 0) -> dict:
    """
    加载数据文件，按 session_key 分组并按 time_key 排序。

    :param max_samples: 加载阶段最多读取的原始记录条数，0 表示不限。
                        用于防止 GB 级文件 OOM。

    Returns:
        {session_id: [record, ...]}  # 每个 list 已按 time_key 升序排列
    """
    records = []
    mode_used = None
    for rec, mode in _iter_records_streaming(data_file, max_samples=max_samples):
        records.append(rec)
        mode_used = mode

    truncated_tag = ""
    if max_samples and len(records) >= max_samples:
        truncated_tag = f"（已截断到前 {max_samples} 条）"
    print(f"  共加载 {len(records)} 条记录（方式={mode_used}）{truncated_tag}")

    # 按 session_key 分组
    sessions: dict = defaultdict(list)
    for rec in records:
        sid = rec.get(session_key)
        if sid is None:
            sid = '__no_session__'
        sessions[sid].append(rec)

    # 每个 session 内按 time_key 排序
    for sid in sessions:
        sessions[sid].sort(key=lambda r: _parse_time_to_ms(r.get(time_key, 0)))

    print(f"  共 {len(sessions)} 个 session，session_key='{session_key}'，time_key='{time_key}'")
    for sid, recs in list(sessions.items())[:3]:
        print(f"    session {sid}: {len(recs)} 条记录")

    return dict(sessions)


def pre_sample_sessions(sessions: dict, n: int, time_key: str) -> dict:
    """
    确定性地截取前 n 条记录，保证多次运行使用完全相同的请求集合。

    策略：将所有记录按 time_key 全局排序后取前 n 条，
    再按原 session 结构重新分组，保持 session 内顺序不变。
    """
    # 展平所有记录，带 session_id 标记
    all_records = []
    for sid, recs in sessions.items():
        for rec in recs:
            all_records.append((sid, rec))

    # 按 time_key 全局排序（与 load_data 内 session 内排序一致）
    # 兼容纯数字字符串和 ISO 8601 字符串（如 "2026-05-13T14:33:00+08:00"）
    all_records.sort(key=lambda x: _parse_time_to_ms(x[1].get(time_key, 0)))

    # 取前 n 条
    sampled = all_records[:n]

    # 重新按 session 分组，保持插入顺序
    new_sessions: dict = {}
    for sid, rec in sampled:
        if sid not in new_sessions:
            new_sessions[sid] = []
        new_sessions[sid].append(rec)

    total_records = sum(len(v) for v in new_sessions.values())
    print(f"  --pre_sample {n}：截取 {total_records} 条记录，{len(new_sessions)} 个 session")
    return new_sessions


def order_sessions(sessions: dict, mode: str, time_key: str, seed: int = 0) -> dict:
    """按 mode 决定 session 之间的下发顺序；四种模式在固定输入下都是确定性的。

    load_data() 返回的 dict 顺序 = session 在数据文件中首次出现的顺序，
    此处按需重排，不改变 session 内部的请求顺序（内部已在 load_data 里按 time_key 升序）。
    """
    items = list(sessions.items())

    if mode == 'file':
        ordered = items
    elif mode == 'start_time':
        def _first_ts(kv):
            recs = kv[1]
            return min(_parse_time_to_ms(r.get(time_key, 0)) for r in recs) if recs else 0
        ordered = sorted(items, key=_first_ts)
    elif mode == 'size':
        ordered = sorted(items, key=lambda kv: -len(kv[1]))
    elif mode == 'shuffle':
        import random as _random
        _random.Random(seed).shuffle(items)
        ordered = items
    else:
        raise ValueError(f'未知 --session_order: {mode}')

    print(f"  session 下发顺序: {mode}"
          + (f"（seed={seed}）" if mode == 'shuffle' else ""))
    for i, (sid, recs) in enumerate(ordered[:3]):
        print(f"    #{i + 1} session={sid}，{len(recs)} 条请求")
    return dict(ordered)


def dump_plan(plan_path: str, sessions: dict, time_key: str,
              replay_id: str, data_file_name: str) -> None:
    """把本次"预期下发计划"落盘：顺序 + 每条请求的时间戳与 prompt 指纹。

    注意：这是*计划*而非*实际发送记录*。若 --tail_threshold > 0 触发长尾中断，
    实际发出的条数会少于计划条数；要严格对齐就把 --tail_threshold 设为 0。
    """
    import hashlib

    rows = []
    seq = 0
    for sid, recs in sessions.items():
        for idx, rec in enumerate(recs):
            seq += 1
            text = json.dumps(rec.get('messages', ''), ensure_ascii=False, sort_keys=True)
            rows.append({
                'seq': seq,
                'session_id': str(sid),
                'req_idx': idx,
                'time_key': str(rec.get(time_key, '')),
                'prompt_sha1': hashlib.sha1(text.encode('utf-8')).hexdigest()[:16],
            })

    meta = {
        'replay_id': replay_id,
        'data_file': data_file_name,
        'session_count': len(sessions),
        'record_count': len(rows),
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'plan': rows,
    }
    os.makedirs(os.path.dirname(plan_path) or ".", exist_ok=True)
    with open(plan_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  📝 下发计划已写入: {plan_path}（{len(rows)} 条 / {len(sessions)} session）")


# ==================== 请求工具 ====================

def create_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def normalize_messages_to_openai(messages: list) -> list:
    """将 Anthropic 格式的 messages 转换为 OpenAI 格式。

    Anthropic → OpenAI 转换规则：
    ┌─────────────────────────────────────────────────────────────────┐
    │ content 为字符串                → 保持不变                      │
    │ content 为 None/空              → 跳过该消息                    │
    │ content 数组含 tool_use         → 拆出为独立 assistant 消息，   │
    │   (role=assistant)                tool_calls 字段；             │
    │                                   同组 text 并入       │
    │                                   该消息的 content              │
    │ content 数组含 tool_result      → 每个 tool_result 拆为独立的   │
    │   (role=user)                     role=tool 消息；              │
    │                                   同组 text 单独生成 user 消息  │
    │ content 数组仅含 text           → 提取 text 拼为字符串          │
    │ content 数组含 thinking         → thinking 块属于模型内部推理，丢弃  │
    └─────────────────────────────────────────────────────────────────┘
    """
    result = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")

        # ── 1. 纯字符串，直接保留 ──────────────────────────────────
        if isinstance(content, str):
            result.append(msg)
            continue

        # ── 2. None / 空 ──────────────────────────────────────────
        if not content:
            # 例外：assistant 携带 tool_calls 但 content 为空（缺失 / None / ""）时
            # 不能丢弃，否则其后紧跟的 role=tool 消息会失去配对的 tool_calls，
            # 触发 "Messages with role 'tool' must be a response to a preceding
            # message with 'tool_calls'"。补空串 content 后保留该消息。
            if role == "assistant" and msg.get("tool_calls"):
                new_msg = dict(msg)
                new_msg["content"] = ""
                result.append(new_msg)
            continue

        # ── 3. 数组格式，按 role 分别处理 ────────────────────────
        if not isinstance(content, list):
            result.append(msg)
            continue

        if role == "assistant":
            # 收集各类型（thinking 块丢弃，属于模型内部推理，不应发给目标模型）
            text_parts = []
            tool_uses = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                t = item.get("type")
                if t == "text" and item.get("text"):
                    text_parts.append(item["text"])
                elif t == "tool_use":
                    tool_uses.append(item)

            if tool_uses:
                # 构造带 tool_calls 的 assistant 消息
                new_msg = {
                    "role": "assistant",
                    "content": "\n".join(text_parts) if text_parts else None,
                    "tool_calls": [
                        {
                            "id": tu.get("id", f"call_{i}"),
                            "type": "function",
                            "function": {
                                "name": tu.get("name", ""),
                                "arguments": json.dumps(tu.get("input", {}), ensure_ascii=False),
                            },
                        }
                        for i, tu in enumerate(tool_uses)
                    ],
                }
                result.append(new_msg)
            elif text_parts:
                new_msg = dict(msg)
                new_msg["content"] = "\n".join(text_parts)
                result.append(new_msg)
            # 若两者均为空则跳过

        elif role == "user":
            tool_results = []
            text_parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                t = item.get("type")
                if t == "tool_result":
                    tool_results.append(item)
                elif t == "text" and item.get("text"):
                    text_parts.append(item["text"])

            # tool_result → role=tool 消息
            for tr in tool_results:
                tr_content = tr.get("content", "")
                if isinstance(tr_content, list):
                    tr_content = "\n".join(
                        c.get("text", "") for c in tr_content
                        if isinstance(c, dict) and c.get("type") == "text"
                    )
                result.append({
                    "role": "tool",
                    "tool_call_id": tr.get("tool_use_id", ""),
                    "content": str(tr_content),
                })

            # 同组 text → 普通 user 消息
            if text_parts:
                new_msg = dict(msg)
                new_msg["content"] = "\n".join(text_parts)
                result.append(new_msg)

        else:
            # 其他 role，仅提取 text
            text_parts = [
                item["text"]
                for item in content
                if isinstance(item, dict) and item.get("type") == "text" and item.get("text")
            ]
            if text_parts:
                new_msg = dict(msg)
                new_msg["content"] = "\n".join(text_parts)
                result.append(new_msg)

    return result


def ensure_assistant_tool_content(messages: list) -> list:
    """方案 B 兜底：确保每条携带 tool_calls 的 assistant 消息都有字符串 content。

    部分推理框架 / 网关要求 assistant 消息的 content 字段必须存在且为字符串，
    当 assistant 仅返回 tool_calls 而 content 缺失 / 为 None 时会丢弃该消息，
    导致其后的 role=tool 消息变成孤儿，报错：
    "Messages with role 'tool' must be a response to a preceding message with 'tool_calls'"。

    这里在发送前统一兜底：对带 tool_calls 且 content 为空的 assistant 消息补空串。
    补 "" 而非有意义文本，避免影响模型行为与 token 统计。
    """
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            if not isinstance(msg.get("content"), str):
                msg["content"] = ""
    return messages


def add_uuid_prefix_to_messages(messages):
    messages_copy = copy.deepcopy(messages)
    for msg in messages_copy:
        content = msg.get("content", "")
        if isinstance(content, str) and content:
            msg["content"] = f"[UUID:{uuid.uuid4().hex}] {content}"
            break
        elif isinstance(content, list) and content:
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                    item["text"] = f"[UUID:{uuid.uuid4().hex}] {item['text']}"
                    break
                elif isinstance(item, str) and item:
                    content[content.index(item)] = f"[UUID:{uuid.uuid4().hex}] {item}"
                    break
            break
    return messages_copy


def _new_run_salt() -> str:
    """生成一个新的"运行轮次盐值"。取 uuid 前 12 位，足够避免跨轮碰撞且很短。"""
    return uuid.uuid4().hex[:12]


def add_run_salt_to_messages(messages, salt):
    """在首条消息内容前注入统一的运行轮次盐值 salt（方案 B）。

    与 add_uuid_prefix_to_messages 的关键区别：
      - uuid 前缀：每条请求都换新 uuid → 破坏所有前缀缓存（含 session 内多轮）→ 命中率≈0，
        测的是纯冷启动/最坏情况，不是真实命中率。
      - run salt ：同一轮（同一次数据回放）内所有请求共用同一 salt → 保留 session 内多轮
        及跨 session 共享前缀的"真实命中"；仅在切换并发级别 / 循环复用时更换 salt，
        从而消除"重复回放同一份数据"带来的假命中，测出真实缓存命中率。

    注入位置与 uuid 前缀一致（首条含文本内容的消息最前面），因为前缀缓存从第一个 token 起匹配，
    改变开头即可让整段前缀相对上一轮失配。
    """
    if not salt:
        return messages
    messages_copy = copy.deepcopy(messages)
    for msg in messages_copy:
        content = msg.get("content", "")
        if isinstance(content, str) and content:
            msg["content"] = f"[NONCE:{salt}] {content}"
            break
        elif isinstance(content, list) and content:
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                    item["text"] = f"[NONCE:{salt}] {item['text']}"
                    break
                elif isinstance(item, str) and item:
                    content[content.index(item)] = f"[NONCE:{salt}] {item}"
                    break
            break
    return messages_copy


def normalize_tools_to_openai(tools: list) -> list:
    """将 Anthropic/Claude 格式的 tools（input_schema）转换为 OpenAI 格式（type=function + function.parameters）。

    Anthropic 格式:  {"name": "...", "description": "...", "input_schema": {...}}
    OpenAI   格式:  {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
    已是 OpenAI 格式的不做转换。
    """
    if not tools:
        return tools
    result = []
    for t in tools:
        if not isinstance(t, dict):
            result.append(t)
            continue
        # 已是 OpenAI 格式（有 type + function 包装）→ 不处理
        if t.get("type") == "function" and "function" in t:
            result.append(t)
        # Anthropic 格式（有 input_schema）→ 转换为 OpenAI 格式
        elif "input_schema" in t:
            result.append({
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t["input_schema"],
                }
            })
        else:
            result.append(t)
    return result


def build_request_params(original_req: dict, target_model: str, enable_uuid_prefix: bool, enable_thinking: bool = False, run_salt: str = None) -> dict:
    """构建请求参数，替换 model，保留其他合法字段。

    run_salt: 方案 B 的运行轮次盐值。非空且未开启 uuid 前缀时，在首条消息前注入该 salt。
              同一轮内为同一字符串，切换并发级别/循环复用时更换，用于消除重复回放的假命中。
    """
    req = copy.deepcopy(original_req)
    # 将 Anthropic 格式 content 数组转换为 OpenAI 纯文本格式
    req["messages"] = normalize_messages_to_openai(req["messages"])
    # 方案 B 兜底：确保带 tool_calls 的 assistant 都有字符串 content，
    # 防止其后的 role=tool 消息变孤儿
    req["messages"] = ensure_assistant_tool_content(req["messages"])
    # 将 Anthropic 格式 tools (input_schema) 转换为 OpenAI 格式 (type=function + parameters)
    if req.get("tools"):
        req["tools"] = normalize_tools_to_openai(req["tools"])
    if enable_uuid_prefix:
        # uuid 前缀：每请求换新 uuid，破坏全部前缀缓存（含 session 内多轮）→ 命中率≈0（纯冷启动）
        req["messages"] = add_uuid_prefix_to_messages(req["messages"])
    elif run_salt:
        # run salt：同一轮共用同一 salt，保留 session 内真实多轮命中，仅跨轮更换以消除假命中
        req["messages"] = add_run_salt_to_messages(req["messages"], run_salt)

    STANDARD_FIELDS = {
        "model", "messages", "max_tokens", "max_completion_tokens",
        "temperature", "top_p", "n", "stop", "presence_penalty",
        "frequency_penalty", "logit_bias", "logprobs", "top_logprobs",
        "response_format", "seed", "tools", "tool_choice",
        "parallel_tool_calls", "user", "function_call", "functions",
        "stream", "stream_options", "service_tier", "store", "metadata",
        "reasoning_effort",
    }

    params = {"model": target_model, "messages": req["messages"]}
    for field in STANDARD_FIELDS:
        if field in req and req[field] is not None:
            if field in ("model", "messages", "max_tokens", "max_completion_tokens"):
                continue
            params[field] = req[field]

    # max_tokens 处理：
    # 1) 命令行 --max_tokens 优先级最高，统一覆盖为该值
    # 2) 否则沿用原始请求中的 max_tokens / max_completion_tokens（若存在）
    if MAX_TOKENS:
        params["max_tokens"] = MAX_TOKENS
        params["max_completion_tokens"] = MAX_TOKENS
    else:
        for field in ("max_tokens", "max_completion_tokens"):
            if field in req and req[field] is not None:
                params[field] = req[field]

    extra_body = {k: v for k, v in req.items() if k not in STANDARD_FIELDS and v is not None}
    # THINKING_TYPE 优先：直接指定 thinking.type 字符串（兼容 minimax-m3 的 adaptive/disabled）；
    # 未设置时退回 enable_thinking 的 enabled/disabled。
    if THINKING_TYPE:
        extra_body["thinking"] = {"type": THINKING_TYPE}
    else:
        extra_body["thinking"] = {"type": "enabled" if enable_thinking else "disabled"}
    if extra_body:
        params["extra_body"] = extra_body

    # reasoning_effort：设置后作为顶层请求参数传入（如 high/medium/low/minimal）
    if REASONING_EFFORT:
        params["reasoning_effort"] = REASONING_EFFORT

    return params


def send_single_request(client: OpenAI, original_req: dict,
                        target_model: str, enable_stream: bool,
                        enable_uuid_prefix: bool,
                        enable_thinking: bool = False,
                        benchmark_start_time: float = None,
                        session_id=None,
                        run_salt: str = None) -> dict | None:
    """
    发送单条请求并收集性能指标。
    stop_event 已设置时跳过请求，返回 None。

    Args:
        benchmark_start_time: 压测起始的 wall-clock 时间戳（time.time()），
            用于计算 started_at / chunk.at 等相对时间，供 strict-window TPM 计算使用。
        session_id: 当前请求所属的 session 标识；当全局 SESSION_ID_HEADER 非空时，
            会通过该 header 名透传给服务端，用于按 session 做亲和性路由。
        run_salt: 方案 B 运行轮次盐值，非空时注入到首条消息前（见 build_request_params）。
    """
    if stop_event.is_set():
        return None

    params = build_request_params(original_req, target_model, enable_uuid_prefix, enable_thinking, run_salt=run_salt)

    # session_id 透传 header：SESSION_ID_HEADER 非空且 session_id 有效时附带
    extra_headers = None
    if SESSION_ID_HEADER and session_id is not None:
        extra_headers = {SESSION_ID_HEADER: str(session_id)}

    answer, cot, req_id = "", "", ""
    ttfb, ttlb, token_rates = 0.0, 0.0, 0.0
    prompt_tokens, completion_tokens, cached_tokens = 0, 0, 0
    reasoning_tokens = 0
    tool_calls = []
    error_msg = ""
    finish_reason = ""
    chunk_count = 0
    output_chunks = []   # 每个 chunk 的到达时间 + 文本长度权重，供 strict-window TPM 使用
    request_start_wall = time.time()
    started_at = round(request_start_wall - benchmark_start_time, 3) if benchmark_start_time else 0.0

    try:
        start_ts = time.perf_counter()

        if enable_stream:
            params["stream"] = True
            params["stream_options"] = {"include_usage": True}

            # create() 之前再 check 一次：stop_event 已 set 则直接放弃，避免新建无意义的连接
            if stop_event.is_set():
                return None

            # 用 httpx.Timeout 设置 read 超时：两个 chunk 之间最长等 STREAM_READ_TIMEOUT 秒。
            # stop_event 触发后线程最多再等这么长就会因 ReadTimeout 退出。
            stream_timeout = httpx.Timeout(connect=30.0, read=STREAM_READ_TIMEOUT, write=30.0, pool=30.0)
            chat_completion = client.chat.completions.create(**params, timeout=stream_timeout,
                                                             extra_headers=extra_headers)
            streaming_tool_calls: dict = {}

            # 注册到活跃流集合，stop_event 触发时 watcher 线程会主动 close，
            # 使正在阻塞的 __next__() 立刻抛异常，无需等待下一个 chunk 到来。
            # 竞态保护：若注册时 stop_event 已 set（watcher 可能已退出本轮扫描），
            # 这里立刻自行 close，避免漏关导致挂死等 ReadTimeout。
            with _streams_lock:
                _active_streams.add(chat_completion)
            if stop_event.is_set():
                try:
                    chat_completion.close()
                except Exception:
                    pass
                with _streams_lock:
                    _active_streams.discard(chat_completion)
                return None
            try:
                for chunk in chat_completion:
                    chunk_count += 1

                    if ttfb == 0 and chunk.choices:
                        ttfb = time.perf_counter() - start_ts

                    if not req_id and chunk.id:
                        req_id = chunk.id

                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        chunk_text_weight = 0
                        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                            cot += delta.reasoning_content
                            chunk_text_weight += len(delta.reasoning_content)
                        if hasattr(delta, 'content') and delta.content:
                            answer += delta.content
                            chunk_text_weight += len(delta.content)
                        if chunk_text_weight > 0 and benchmark_start_time:
                            output_chunks.append({
                                "at": round(time.time() - benchmark_start_time, 3),
                                "weight": chunk_text_weight,
                            })
                        if chunk.choices[0].finish_reason:
                            finish_reason = chunk.choices[0].finish_reason
                        if hasattr(delta, 'tool_calls') and delta.tool_calls:
                            for tc in delta.tool_calls:
                                idx = tc.index
                                if idx not in streaming_tool_calls:
                                    streaming_tool_calls[idx] = {
                                        "id": "", "type": "function",
                                        "function_name": "", "function_arguments": "",
                                    }
                                if tc.id:
                                    streaming_tool_calls[idx]["id"] = tc.id
                                if tc.type:
                                    streaming_tool_calls[idx]["type"] = tc.type
                                if tc.function:
                                    if tc.function.name:
                                        streaming_tool_calls[idx]["function_name"] = tc.function.name
                                    if tc.function.arguments:
                                        streaming_tool_calls[idx]["function_arguments"] += tc.function.arguments

                    if chunk.usage:
                        if chunk.usage.completion_tokens and chunk.usage.completion_tokens > completion_tokens:
                            completion_tokens = chunk.usage.completion_tokens
                        # reasoning_tokens：thinking / 推理阶段消耗的 token，
                        # 位于 completion_tokens_details.reasoning_tokens（是 completion_tokens 的子集）。
                        # 与 completion_tokens 一样逐 chunk 取最大值，避免末尾包重置为 0 覆盖。
                        completion_details = getattr(chunk.usage, 'completion_tokens_details', None)
                        if completion_details:
                            rt = getattr(completion_details, 'reasoning_tokens', 0) or 0
                            if rt > reasoning_tokens:
                                reasoning_tokens = rt
                        prompt_details = getattr(chunk.usage, 'prompt_tokens_details', None)

                        # cached_tokens：只要 prompt_tokens >= 当前值（含末尾纯 usage 包 prompt_tokens 不变的情况），
                        # 都尝试更新，确保服务端最终返回的非零 cached_tokens 被采集
                        if chunk.usage.prompt_tokens and chunk.usage.prompt_tokens >= prompt_tokens:
                            prompt_tokens = chunk.usage.prompt_tokens
                            if prompt_details:
                                ct = getattr(prompt_details, 'cached_tokens', 0) or 0
                                if ct > cached_tokens:
                                    cached_tokens = ct
            finally:
                with _streams_lock:
                    _active_streams.discard(chat_completion)

            ttlb = time.perf_counter() - start_ts
            if ttfb == 0 and chunk_count > 0:
                ttfb = ttlb
            # 仅当解码窗口足够大、且确实是多 chunk 流式输出时才计算 Rate。
            # 否则（如 tool_calls 一次性 flush、chunk_count<=2 等场景）分母接近 0
            # 会产生百万级离群值，污染 Avg/P90/P95/P99，这里直接置 0，
            # 由统计聚合层用 rate>0 过滤剔除。
            gen_time = ttlb - ttfb
            if gen_time > 0.5 and chunk_count > 2 and completion_tokens > 1:
                token_rates = completion_tokens / gen_time
            else:
                token_rates = 0.0

            if streaming_tool_calls:
                for idx in sorted(streaming_tool_calls.keys()):
                    tc_info = streaming_tool_calls[idx]
                    tool_calls.append({
                        "id": tc_info["id"], "type": tc_info["type"],
                        "function": {
                            "name": tc_info["function_name"],
                            "arguments": tc_info["function_arguments"],
                        }
                    })
        else:
            resp = client.chat.completions.create(**params, timeout=REQUEST_TIMEOUT,
                                                  extra_headers=extra_headers)
            answer = resp.choices[0].message.content or ""
            if hasattr(resp.choices[0].message, 'reasoning_content'):
                cot = resp.choices[0].message.reasoning_content or ""
            req_id = resp.id
            finish_reason = resp.choices[0].finish_reason or ""
            if resp.usage:
                prompt_tokens = resp.usage.prompt_tokens
                completion_tokens = resp.usage.completion_tokens
                prompt_details = getattr(resp.usage, 'prompt_tokens_details', None)
                if prompt_details:
                    cached_tokens = getattr(prompt_details, 'cached_tokens', 0) or 0
                completion_details = getattr(resp.usage, 'completion_tokens_details', None)
                if completion_details:
                    reasoning_tokens = getattr(completion_details, 'reasoning_tokens', 0) or 0
            ttlb = time.perf_counter() - start_ts
            ttfb = ttlb
            if hasattr(resp.choices[0].message, 'tool_calls') and resp.choices[0].message.tool_calls:
                for tc in resp.choices[0].message.tool_calls:
                    tool_calls.append({
                        "id": tc.id, "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                    })

    except Exception as e:
        # stop_event 已设置时，超时/连接关闭等异常均属于主动中断，不记录为错误
        if stop_event.is_set():
            return None
        error_msg = str(e)

    if stop_event.is_set() and ttlb == 0:
        return None

    if not error_msg and prompt_tokens == 0 and completion_tokens == 0 and not answer and not cot:
        error_msg = "empty_response"

    # 检测异常 finish_reason：服务端虽然返回 200/SSE，但中途因为 abort/cancelled/error
    # 等异常原因主动终止；这种 response 不应计入成功，否则 success_count 会虚高、
    # TTFT/TPOT 等延迟分布会被未完成请求污染。
    # 正常终止（'stop' / 'length' / 'tool_calls' / 'function_call' / 'content_filter'）
    # 一律放行；空字符串（流式但未收到 finish_reason）也不判失败，避免误杀。
    NORMAL_FINISH_REASONS = {
        "", "stop", "length", "tool_calls", "function_call", "content_filter",
    }
    if not error_msg and finish_reason and finish_reason not in NORMAL_FINISH_REASONS:
        error_msg = f"abnormal_finish_reason={finish_reason}"

    # TPOT（Time Per Output Token，与 evalscope 口径一致）：
    #   TPOT = (总耗时 - TTFT) / (输出token数 - 1)
    # 分母 -1 是因为第 1 个输出 token 的耗时已计入 TTFT，TPOT 只统计 decode 阶段
    # （第 2 个 token 起）的耗时，均摊到每个输出 token 上；completion_tokens<=1 时不参与统计。
    tpot = 0.0
    if completion_tokens > 1 and ttlb > ttfb:
        tpot = round((ttlb - ttfb) / (completion_tokens - 1) * 1000, 3)

    # ITL（Inter-Token/Chunk Latency，与 evalscope 口径一致）：基于真实的相邻 chunk
    # 到达时间差（output_chunks[i+1].at - output_chunks[i].at），而非用总耗时摊薄估算；
    # 一个请求会产生 len(output_chunks)-1 个原始样本，供聚合层跨请求汇总统计
    # （avg/percentile 对所有请求的全部原始间隔样本合并计算，而非先按请求求平均）。
    itl_samples = []
    if len(output_chunks) > 1:
        for _i in range(len(output_chunks) - 1):
            _gap = (output_chunks[_i + 1]["at"] - output_chunks[_i]["at"]) * 1000
            if _gap > 0:
                itl_samples.append(round(_gap, 3))
    itl = round(sum(itl_samples) / len(itl_samples), 3) if itl_samples else 0.0

    cache_hit_ratio = round(cached_tokens / prompt_tokens, 4) if prompt_tokens > 0 else 0.0

    # 乱码检测：模型输出既非中文也非英文（而是替换字符/其它脚本等乱码）时，
    # 记录 prompt + 输出到日志。此处仅做一次非阻塞入队（微秒级），
    # 真正的检测与写盘由后台 worker 线程异步完成，不占用请求关键路径。
    if not error_msg and answer:
        _enqueue_garbled_check(
            session_id=session_id,
            req_id=req_id,
            messages=params.get("messages"),
            answer=answer,
        )
    # 重复输出（退化复读）检测：只检测 content（answer）与 reasoning_content（cot）
    # 两个字段，不检测 tool_calls（工具调用参数天然可能有相似/相同片段，不属于
    # 模型退化复读）。同样只做一次非阻塞入队，检测与写日志异步完成。
    if not error_msg and (answer or cot):
        _enqueue_repeat_check(
            session_id=session_id,
            req_id=req_id,
            messages=params.get("messages"),
            answer=answer,
            cot=cot,
        )

    # 流式但未收集到 chunk 文本时，退化为整体计入（fallback 标记，供 strict-window 统计）
    if enable_stream and completion_tokens > 0 and not output_chunks and benchmark_start_time:
        output_chunks.append({
            "at": round(time.time() - benchmark_start_time, 3),
            "weight": completion_tokens,
            "fallback": True,
        })

    return {
        "req_id": req_id,
        "answer_len": len(answer),
        "cot_len": len(cot),
        "tool_calls_count": len(tool_calls),
        "ttfb": round(ttfb, 4),
        "ttlb": round(ttlb, 4),
        "token_rates": round(token_rates, 2),
        "itl": itl,
        "itl_samples": itl_samples,
        "tpot": tpot,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cache_hit_ratio": cache_hit_ratio,
        "finish_reason": finish_reason,
        "chunk_count": chunk_count,
        "started_at": started_at,
        "output_chunks": output_chunks,
        "error": error_msg,
    }


# ==================== 核心并发逻辑 ====================

def worker_process_sessions(
    worker_id: int,
    session_queue: list,
    all_sessions: list,     # 所有 session 的原始列表，duration 模式下循环复用
    queue_lock: Lock,
    active_counter: list,   # [int] 用列表包装以便跨线程修改
    active_lock: Lock,
    results: list,
    results_lock: Lock,
    client: OpenAI,
    target_model: str,
    enable_stream: bool,
    enable_uuid_prefix: bool,
    enable_thinking: bool,
    initial_concurrency: int,
    stop_threshold: float,
    print_lock: Lock,
    duration_mode: bool,    # True: 按时长运行（队列空时循环补充）；False: 跑完即止
    set_stop_fn=None,       # 触发全局中断的回调（duration>0 + duration=0 都用此通道）
    benchmark_start_time: float = 0.0,  # 压测起始时间戳，用于记录 completed_at
    run_salt_holder: list = None,  # 方案 B：[当前轮次 salt]，用列表包装以便循环复用时跨线程更换
):
    """
    单个 worker 线程主逻辑：
      1. 从 session_queue 取出一个 session（线程安全）
      2. 按 start_time 顺序依次发送该 session 的所有请求
      3. 当前 session 处理完后，检查 stop_event；未停止则取下一个 session
      4. duration_mode=True:  队列空时循环补充，直到 stop_event
         duration_mode=False: 队列空时检查 stop_threshold，达到阈值则触发 stop_event
                              （与 duration>0 同一个 stop_event 通道，统一由 watcher
                               + _drain_tail_futures 处理长尾，避免最后一批 session
                               中的慢请求把整轮压测拖到 N×REQUEST_TIMEOUT 才退出）
    """
    while not stop_event.is_set():
        # 取一个 session
        with queue_lock:
            if not session_queue:
                if duration_mode:
                    # duration 模式：队列空了重新填入所有 session，继续循环
                    if stop_event.is_set():
                        break
                    session_queue.extend(all_sessions)
                    # 方案 B：循环复用即进入新一轮回放 → 更换 salt，让这一轮相对上一轮"变冷"，
                    # 避免重复回放同一份数据命中上一轮残留的前缀缓存（假命中）。
                    if run_salt_holder is not None:
                        run_salt_holder[0] = _new_run_salt()
                    with print_lock:
                        print(f"  [Worker-{worker_id}] 队列已耗尽，循环复用 {len(all_sessions)} 个 session"
                              + (f"（新一轮 salt={run_salt_holder[0]}）" if run_salt_holder is not None else ""))
                    if not session_queue:
                        break
                else:
                    # duration=0 模式：队列空 → 检查阈值 → 触发 stop_event
                    # 与 duration>0 一致地使用 stop_event 通道：
                    #   - stop_event 触发后 watcher 会主动 close 流式连接
                    #   - 其他还在 send_single_request 的 worker 会因 ReadError 立刻退出
                    #   - 避免"最后一个 worker 卡在某个慢请求上拖几分钟"
                    with active_lock:
                        active_count = active_counter[0]
                    stop_count = int(initial_concurrency * stop_threshold)
                    if active_count < stop_count and set_stop_fn is not None:
                        set_stop_fn()
                        with print_lock:
                            print(f"  ⏱️ [Worker-{worker_id}] 活跃并发 {active_count} < "
                                  f"初始并发 {initial_concurrency} 的 "
                                  f"{stop_threshold:.0%}（{stop_count}），"
                                  f"发送全局停止信号中断长尾请求")
                    break
            session_id, session_records = session_queue.pop(0)
            # 方案 B：在取出 session 时快照当前轮次 salt，供该 session 内所有请求统一使用。
            # 快照（而非每请求实时读取）可保证：即便另一个 worker 触发了循环换 salt，
            # 本 session 的多轮请求仍共用同一 salt → session 内多轮的真实前缀命中不被边界打断。
            current_salt = run_salt_holder[0] if run_salt_holder is not None else None

        with print_lock:
            print(f"  [Worker-{worker_id}] 开始处理 session={session_id}，"
                  f"共 {len(session_records)} 条请求，"
                  f"队列剩余 {len(session_queue)} 个 session")

        # 按顺序发送该 session 内的每条请求
        for idx, record in enumerate(session_records):
            if stop_event.is_set():
                break

            result = send_single_request(
                client, record, target_model, enable_stream, enable_uuid_prefix, enable_thinking,
                benchmark_start_time=benchmark_start_time,
                session_id=session_id,
                run_salt=current_salt,
            )

            if stop_event.is_set():
                break

            if result is not None:
                with results_lock:
                    results.append({
                        **result,
                        "session_id": str(session_id),
                        "completed_at": round(time.time() - benchmark_start_time, 3),
                    })
                    seq = len(results)
                status = "✅" if not result["error"] else "❌"
                with print_lock:
                    print(f"  [{seq}] {status} session={session_id} req={idx+1}/{len(session_records)} "
                          f"ttfb={result['ttfb']}s ttlb={result['ttlb']}s "
                          f"rate={result['token_rates']}t/s "
                          f"prompt={result['prompt_tokens']} "
                          f"completion={result['completion_tokens']} "
                          f"reasoning={result['reasoning_tokens']} "
                          f"cached={result['cached_tokens']} "
                          f"finish={result['finish_reason']}"
                          f"{' err='+result['error'] if result['error'] else ''}")

    # Worker 退出，减少活跃计数
    with active_lock:
        active_counter[0] -= 1
        remaining_active = active_counter[0]

    with print_lock:
        print(f"  [Worker-{worker_id}] 退出，剩余活跃 worker={remaining_active}")

    # duration=0 模式：worker 退出后再次检查，队列空 + 活跃数低于阈值时触发 stop_event。
    # 这一路径覆盖"worker 是从 session 内部循环 break 出来"的情况（while 头不会再走到上面的取队列分支）。
    if not duration_mode and not stop_event.is_set():
        with queue_lock:
            queue_empty = len(session_queue) == 0
        if queue_empty:
            stop_count = int(initial_concurrency * stop_threshold)
            if remaining_active < stop_count and set_stop_fn is not None:
                set_stop_fn()
                with print_lock:
                    print(f"  ⏱️ [Worker-{worker_id}] 退出后触发全局停止：活跃并发 {remaining_active} "
                          f"< 初始并发 {initial_concurrency} 的 {stop_threshold:.0%}（{stop_count}）")


def run_session_benchmark(
    client: OpenAI,
    sessions: dict,
    concurrency: int,
    target_model: str,
    enable_stream: bool,
    enable_uuid_prefix: bool,
    enable_thinking: bool,
    stop_threshold: float,
    data_file_name: str,
    duration: int = 0,
    enable_run_salt: bool = False,
) -> list:
    """
    按 session 粒度并发压测。

    Args:
        duration: 压测持续时长（秒），0 表示跑完所有 session（含 pre_sample 截取后的集合）为止。
                  > 0 时：到时间自动触发 stop_event，session 队列耗尽时自动循环复用。

    Returns:
        results: 所有已完成请求的结果列表
    """
    stop_event.clear()
    _start_stream_watcher()  # 每轮压测重新启动 watcher，确保 stop_event 触发时能快速 close 流
    duration_mode = duration > 0

    # 将所有 session 放入队列（列表，线程安全通过锁保护）
    all_sessions = list(sessions.items())   # 原始副本，duration 模式下循环复用
    session_queue = list(all_sessions)      # 工作队列
    total_sessions = len(session_queue)
    total_records = sum(len(v) for v in sessions.values())

    # 并发数不能超过 session 数（duration=0 时多余的 worker 没有任务可做）
    if not duration_mode and concurrency > total_sessions:
        print(f"  ⚠️ 并发数 {concurrency} > session 数 {total_sessions}，自动调整为 {total_sessions}")
        concurrency = total_sessions

    print(f"\n{'='*70}")
    print(f"[{data_file_name}] Session 并发压测")
    print(f"  并发数: {concurrency}  Session 数: {total_sessions}  总请求数: {total_records}")
    if duration_mode:
        print(f"  运行模式: 固定时长 {duration}s（session 循环复用）")
    else:
        print(f"  运行模式: 跑完所有 session")
        print(f"  停止阈值: 活跃并发 < {concurrency} × {stop_threshold:.0%} = {int(concurrency * stop_threshold)}")
    print(f"{'='*70}")

    queue_lock = Lock()
    active_counter = [concurrency]   # 初始时所有 worker 都活跃
    active_lock = Lock()
    results: list = []
    results_lock = Lock()
    print_lock = Lock()

    # 方案 B：本次压测（本并发级别 × 本数据文件）的初始轮次 salt。
    # 用列表包装成 holder，duration 循环复用时由 worker 就地更换，供所有 worker 共享读取。
    run_salt_holder = [_new_run_salt()] if enable_run_salt else None
    if enable_run_salt:
        print(f"  🧂 已启用运行轮次盐值(方案 B)：初始 salt={run_salt_holder[0]}，"
              f"切换并发级别/循环复用时更换，用于消除重复回放的假命中")

    start_time = time.time()
    stop_time = None  # tpm 截止时间戳（stop_event 首次触发时记录）
    stop_time_lock = Lock()

    def _set_stop():
        """记录 stop_time + 触发全局中断（duration>0 timer / duration=0 队列耗尽 都用此）。

        stop_time 仅在第一次调用时记录，后续重复调用幂等。
        - duration>0：timer 到点
        - duration=0：worker 发现"队列空 + 活跃并发 < initial × stop_threshold"
        """
        nonlocal stop_time
        with stop_time_lock:
            if not stop_event.is_set():
                stop_time = time.time()
                stop_event.set()

    # duration 模式：启动 timer 线程，到时触发停止信号
    if duration_mode:
        def _duration_timer():
            time.sleep(duration)
            if not stop_event.is_set():
                _set_stop()
                print(f"\n  ⏰ 已达到压测时长 {duration}s，发送全局停止信号...")
        timer_thread = threading.Thread(target=_duration_timer, daemon=True)
        timer_thread.start()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for worker_id in range(concurrency):
            f = executor.submit(
                worker_process_sessions,
                worker_id,
                session_queue,
                all_sessions,
                queue_lock,
                active_counter,
                active_lock,
                results,
                results_lock,
                client,
                target_model,
                enable_stream,
                enable_uuid_prefix,
                enable_thinking,
                concurrency,
                stop_threshold,
                print_lock,
                duration_mode,
                _set_stop,
                start_time,
                run_salt_holder,
            )
            futures.append(f)

        # 等待所有 worker 完成（或被 stop_event 中断）
        # 分两段等待：先 close_window=30s 等 watcher.close() 起效，再兜底等 STREAM_READ_TIMEOUT；
        # 超时后调用 executor.shutdown(wait=False, cancel_futures=True) 强制退出 with 块，
        # 已 running 的 worker 仍是 daemon 线程，不影响进程退出，下一档建议重建 client。
        # duration>0：timer 到点 set stop_event；duration=0：队列耗尽 + 活跃数 < 阈值时
        # 由 worker 触发 set stop_event。两种模式都走同一长尾兜底路径。
        try:
            _drain_tail_futures(futures, executor, label="Worker 退出")
        except Exception as ex:
            print(f"  ⚠️ 等待 Worker 退出时异常: {ex}（已收集 {len(results)} 条结果，继续统计）")
            unfinished = [f for f in futures if not f.done()]
            for f in unfinished:
                try:
                    f.cancel()
                except Exception:
                    pass
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                executor.shutdown(wait=False)

    stop_offset = (stop_time if stop_time is not None else time.time()) - start_time
    total_time = time.time() - start_time
    print(f"\n  压测结束：stop_offset={stop_offset:.1f}s，总耗时={total_time:.1f}s，完成请求={len(results)}")
    return results, total_time, stop_offset


# ==================== 统计 ====================

def _tokens_per_minute(tokens, seconds):
    return round(tokens / seconds * 60, 2) if seconds and seconds > 0 else 0.0


def _tokens_per_second(tokens, seconds):
    return round(tokens / seconds, 2) if seconds and seconds > 0 else 0.0


def _sum_strict_window_output_tokens(results, window_start, window_end):
    """按 chunk 到达时间统计固定窗口内的输出 token。

    服务端只在最终 usage 里返回 completion_tokens，因此这里按 chunk 文本长度权重
    将最终 completion_tokens 分摊到各 chunk 的到达时间上。

    Returns:
        (window_tokens: float, fallback_requests: int)
    """
    window_tokens = 0.0
    fallback_requests = 0
    for r in results:
        completion_tokens = r.get("completion_tokens", 0) or 0
        if completion_tokens <= 0:
            continue

        chunks = r.get("output_chunks") or []
        if not chunks:
            # 非流式或缺少 chunk 记录时只能退化为 completed_at 落窗近似
            completed_at = r.get("completed_at", float("inf"))
            if window_start <= completed_at <= window_end:
                window_tokens += completion_tokens
                fallback_requests += 1
            continue

        total_weight = sum(max(c.get("weight", 0) or 0, 0) for c in chunks)
        if total_weight <= 0:
            completed_at = r.get("completed_at", float("inf"))
            if window_start <= completed_at <= window_end:
                window_tokens += completion_tokens
                fallback_requests += 1
            continue

        for c in chunks:
            chunk_at = c.get("at", float("inf"))
            if window_start <= chunk_at <= window_end:
                weight = max(c.get("weight", 0) or 0, 0)
                window_tokens += completion_tokens * (weight / total_weight)
            if c.get("fallback"):
                fallback_requests += 1

    return window_tokens, fallback_requests


def _sum_strict_window_input_tokens(results, window_start, window_end):
    """按"请求生命周期与窗口交集时间占比"分摊 prompt_tokens / uncached_tokens。

    设计动机：
      strict-window 起点 = P95 TTFB（用于剔除爬升），但 input tokens 之前按
      ``started_at`` 是否落入窗口"全有/全无"计入，会把"启动较早但 TTFB 较短"的
      请求整体抛弃 —— 这部分请求往往 prompt 偏大（cache 命中多），导致 input 分子
      缩水比窗口分母快，出现 strict_input_tpm < completed_input_tpm 的反直觉现象。

      方案 A：把每条请求的 prompt_tokens 视为"在该请求的整个生命周期
      ``[started_at, completed_at]`` 内均匀消耗"，按交集时间占比计入窗口。
      与 output 的"按 chunk 到达时间分摊"思路对齐，让 input 也是真正的"瞬时口径"。

    Args:
        results: 成功请求列表（含 started_at / completed_at / prompt_tokens / cached_tokens）
        window_start, window_end: strict 窗口的时间区间（秒，相对压测起点）

    Returns:
        dict 包含：
          - input_tokens:         按交集占比分摊后的 prompt_tokens 总和
          - uncached_input_tokens: 按交集占比分摊后的 max(prompt - cached, 0) 总和
          - request_count_eff:    按交集占比累加的"等效落窗请求数"（用于 RPS/RPM）
          - request_count_touched: 实际与窗口有交集的请求数（>=1 个 token 计入）
    """
    if window_end <= window_start:
        return {
            "input_tokens": 0.0,
            "uncached_input_tokens": 0.0,
            "request_count_eff": 0.0,
            "request_count_touched": 0,
        }

    input_tokens = 0.0
    uncached_tokens = 0.0
    req_eff = 0.0
    req_touched = 0
    for r in results:
        prompt = r.get("prompt_tokens", 0) or 0
        if prompt <= 0:
            continue
        started = r.get("started_at", None)
        completed = r.get("completed_at", None)
        if started is None or completed is None:
            continue
        # 请求生命周期 = [started, completed]；非法/零长度退化为按 started 全有/全无
        life_s = max(completed - started, 0.0)
        if life_s <= 0:
            if window_start <= started <= window_end:
                input_tokens += prompt
                uncached_tokens += max(prompt - (r.get("cached_tokens", 0) or 0), 0)
                req_eff += 1.0
                req_touched += 1
            continue
        # 交集
        ov_start = max(started, window_start)
        ov_end = min(completed, window_end)
        overlap = max(ov_end - ov_start, 0.0)
        if overlap <= 0:
            continue
        ratio = overlap / life_s
        input_tokens += prompt * ratio
        uncached_tokens += max(prompt - (r.get("cached_tokens", 0) or 0), 0) * ratio
        req_eff += ratio
        req_touched += 1
    return {
        "input_tokens": input_tokens,
        "uncached_input_tokens": uncached_tokens,
        "request_count_eff": req_eff,
        "request_count_touched": req_touched,
    }


def calculate_percentile(data, percentile):
    if not data:
        return 0
    return round(float(np.percentile(data, percentile)), 4)


def compute_statistics(results: list, duration_seconds: float = None,
                       tpm_cutoff_seconds: float = None) -> dict:
    success_results = [r for r in results if not r["error"]]
    failed_results = [r for r in results if r["error"]]

    total = len(results)
    success_count = len(success_results)
    fail_count = len(failed_results)
    success_rate = round(success_count / total * 100, 2) if total > 0 else 0

    stats = {
        "total_requests": total,
        "success_count": success_count,
        "fail_count": fail_count,
        "success_rate": f"{success_rate}%",
    }

    if success_count > 0:
        ttfb_list = [r["ttfb"] for r in success_results]
        ttlb_list = [r["ttlb"] for r in success_results]
        # Rate 只统计真实流式输出的请求（采集处已对 tool_calls 一次性 flush 等场景置 0）
        rate_list = [r["token_rates"] for r in success_results if r.get("token_rates", 0) > 0]
        # ITL 与 evalscope 口径一致：把所有成功请求的「原始相邻 chunk 间隔样本」直接
        # 汇总进一个大列表做全局 avg/percentile（而非先按请求求平均再统计），
        # 更能反映 decode 过程逐 token/chunk 的真实抖动分布。
        itl_pool = []
        for _r in success_results:
            itl_pool.extend(_r.get("itl_samples") or [])
        # TPOT 与 evalscope 口径一致：每个请求先算出 1 个 TPOT 值，再对请求求 avg/percentile。
        tpot_list = [r["tpot"] for r in success_results if r.get("tpot", 0) > 0]
        prompt_list = [r["prompt_tokens"] for r in success_results]
        completion_list = [r["completion_tokens"] for r in success_results]
        cached_list = [r.get("cached_tokens", 0) for r in success_results]
        reasoning_list = [r.get("reasoning_tokens", 0) for r in success_results]

        total_completion_tokens = sum(completion_list)
        total_prompt_tokens = sum(prompt_list)
        total_reasoning_tokens = sum(reasoning_list)
        cache_hit_ratio_list = [r.get("cache_hit_ratio", 0.0) for r in success_results]

        # ===== TPM/TPS 双口径计算 =====
        # 1) Strict-Window 口径：瞬时吞吐峰值视角，剔除爬升+长尾。
        #    窗口起点 = P95 TTFB（剔除爬升阶段），窗口终点 = tpm_cutoff（剔除长尾阶段）。
        #    completion 按各 chunk 文本长度权重分摊到 chunk.at；
        #    prompt 按"请求生命周期 [started_at, completed_at] 与窗口的交集时间占比"分摊
        #      —— 避免起点用 TTFB（而非 started_at）时，把"启动较早、TTFB 较短"的请求
        #      整体抛弃，导致 input 分子缩水比窗口分母快、出现 strict_input_tpm <
        #      completed_input_tpm 的反直觉现象。
        # 2) Completed-Request 口径：对齐 inference-perf，用所有成功请求的完整生命周期窗口
        #    [min(started_at), max(completed_at)] 计算，反映完整请求场景的平均吞吐。
        ramp_end = float(np.percentile(ttfb_list, 95)) if ttfb_list else 0.0

        # ---- Strict-Window ----
        strict_window_start = ramp_end
        strict_window_end = tpm_cutoff_seconds if (tpm_cutoff_seconds and tpm_cutoff_seconds > 0) else (duration_seconds or 0.0)
        strict_window_s = max(strict_window_end - strict_window_start, 0.0)
        strict_output_tokens, strict_fallback_requests = _sum_strict_window_output_tokens(
            success_results, strict_window_start, strict_window_end
        )
        # input 按生命周期交集占比分摊（方案 A）
        _si = _sum_strict_window_input_tokens(
            success_results, strict_window_start, strict_window_end
        )
        strict_input_tokens = _si["input_tokens"]
        strict_uncached_tokens = _si["uncached_input_tokens"]
        # 有效请求数：按交集占比累加（用于 RPS/RPM 的"瞬时"语义）
        strict_request_count_eff = _si["request_count_eff"]
        # 与窗口有交集的请求数（≥ eff，用于参考）
        strict_request_count = _si["request_count_touched"]

        strict_output_tpm = _tokens_per_minute(strict_output_tokens, strict_window_s)
        strict_output_tps = _tokens_per_second(strict_output_tokens, strict_window_s)
        strict_input_tpm = _tokens_per_minute(strict_input_tokens, strict_window_s)
        strict_input_tps = _tokens_per_second(strict_input_tokens, strict_window_s)
        strict_uncached_input_tpm = _tokens_per_minute(strict_uncached_tokens, strict_window_s)
        strict_total_tpm = _tokens_per_minute(strict_input_tokens + strict_output_tokens, strict_window_s)
        strict_total_tps = _tokens_per_second(strict_input_tokens + strict_output_tokens, strict_window_s)
        strict_rps = _tokens_per_second(strict_request_count_eff, strict_window_s)
        strict_rpm = _tokens_per_minute(strict_request_count_eff, strict_window_s)

        # ---- Completed-Lifecycle ----
        lifecycle_start = min((r.get("started_at", 0.0) for r in success_results), default=0.0)
        lifecycle_end = max((r.get("completed_at", 0.0) for r in success_results), default=0.0)
        lifecycle_s = max(lifecycle_end - lifecycle_start, 0.0)
        completed_uncached_tokens = sum(
            max(r.get("prompt_tokens", 0) - r.get("cached_tokens", 0), 0)
            for r in success_results
        )
        completed_output_tpm = _tokens_per_minute(total_completion_tokens, lifecycle_s)
        completed_output_tps = _tokens_per_second(total_completion_tokens, lifecycle_s)
        completed_input_tpm = _tokens_per_minute(total_prompt_tokens, lifecycle_s)
        completed_input_tps = _tokens_per_second(total_prompt_tokens, lifecycle_s)
        completed_uncached_input_tpm = _tokens_per_minute(completed_uncached_tokens, lifecycle_s)
        completed_total_tpm = _tokens_per_minute(total_prompt_tokens + total_completion_tokens, lifecycle_s)
        completed_total_tps = _tokens_per_second(total_prompt_tokens + total_completion_tokens, lifecycle_s)

        # 全局解码速率：总 completion_tokens / (整体执行耗时 - 平均TTFT)
        avg_ttft = float(np.mean(ttfb_list)) if ttfb_list else 0.0
        decode_window = (duration_seconds - avg_ttft
                         if duration_seconds and duration_seconds > avg_ttft else 0)
        global_token_rate = round(
            total_completion_tokens / decode_window, 2
        ) if decode_window > 0 else 0

        stats.update({
            "ttfb": {
                "avg": round(float(np.mean(ttfb_list)), 4),
                "p50": calculate_percentile(ttfb_list, 50),
                "p75": calculate_percentile(ttfb_list, 75),
                "p90": calculate_percentile(ttfb_list, 90),
                "p95": calculate_percentile(ttfb_list, 95),
                "p99": calculate_percentile(ttfb_list, 99),
            },
            "ttlb": {
                "avg": round(float(np.mean(ttlb_list)), 4),
                "p50": calculate_percentile(ttlb_list, 50),
                "p75": calculate_percentile(ttlb_list, 75),
                "p90": calculate_percentile(ttlb_list, 90),
                "p95": calculate_percentile(ttlb_list, 95),
                "p99": calculate_percentile(ttlb_list, 99),
            },
            "token_rates": {
                "avg": round(float(np.mean(rate_list)), 2) if rate_list else 0,
                "p50": round(float(np.percentile(rate_list, 50)), 2) if rate_list else 0,
                "p75": round(float(np.percentile(rate_list, 75)), 2) if rate_list else 0,
                "p90": round(float(np.percentile(rate_list, 90)), 2) if rate_list else 0,
                "p95": round(float(np.percentile(rate_list, 95)), 2) if rate_list else 0,
                "p99": round(float(np.percentile(rate_list, 99)), 2) if rate_list else 0,
            },
            "itl": {
                "avg": round(float(np.mean(itl_pool)), 3) if itl_pool else 0,
                "p50": round(float(np.percentile(itl_pool, 50)), 3) if itl_pool else 0,
                "p75": round(float(np.percentile(itl_pool, 75)), 3) if itl_pool else 0,
                "p90": round(float(np.percentile(itl_pool, 90)), 3) if itl_pool else 0,
                "p95": round(float(np.percentile(itl_pool, 95)), 3) if itl_pool else 0,
                "p99": round(float(np.percentile(itl_pool, 99)), 3) if itl_pool else 0,
            },
            "tpot": {
                "avg": round(float(np.mean(tpot_list)), 3) if tpot_list else 0,
                "p50": round(float(np.percentile(tpot_list, 50)), 3) if tpot_list else 0,
                "p75": round(float(np.percentile(tpot_list, 75)), 3) if tpot_list else 0,
                "p90": round(float(np.percentile(tpot_list, 90)), 3) if tpot_list else 0,
                "p95": round(float(np.percentile(tpot_list, 95)), 3) if tpot_list else 0,
                "p99": round(float(np.percentile(tpot_list, 99)), 3) if tpot_list else 0,
            },
            "avg_prompt_tokens": round(float(np.mean(prompt_list))),
            "avg_completion_tokens": round(float(np.mean(completion_list))),
            "avg_reasoning_tokens": round(float(np.mean(reasoning_list))),
            "avg_cached_tokens": round(float(np.mean(cached_list))),
            "avg_cache_hit_ratio": round(
                sum(cached_list) / sum(prompt_list) if sum(prompt_list) > 0 else 0.0, 4
            ),
            "cache_hit_ratio": {
                "avg": round(
                    sum(cached_list) / sum(prompt_list) if sum(prompt_list) > 0 else 0.0, 4
                ),
                "p50": round(float(np.percentile(cache_hit_ratio_list, 50)), 4),
                "p90": round(float(np.percentile(cache_hit_ratio_list, 90)), 4),
                "p99": round(float(np.percentile(cache_hit_ratio_list, 99)), 4),
            },
            "avg_total_time_ms": round(float(np.mean(ttlb_list)) * 1000, 2),
            "total_completion_tokens": total_completion_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "total_reasoning_tokens": total_reasoning_tokens,
            "global_token_rate": global_token_rate,

            # ===== Strict-Window 口径（瞬时吞吐峰值视角）=====
            # 窗口起点 = P95 TTFB（剔除爬升），窗口终点 = tpm_cutoff（剔除长尾）
            "strict_window_start_s": round(strict_window_start, 3),
            "strict_window_end_s": round(strict_window_end, 3),
            "strict_window_s": round(strict_window_s, 3),
            "strict_window_output_tokens": round(strict_output_tokens),
            "strict_window_input_tokens": round(strict_input_tokens, 2),
            "strict_window_uncached_input_tokens": round(strict_uncached_tokens, 2),
            "strict_window_request_count": strict_request_count,
            "strict_window_request_count_eff": round(strict_request_count_eff, 3),
            "strict_window_fallback_requests": strict_fallback_requests,
            "strict_window_output_tpm": strict_output_tpm,
            "strict_window_output_tps": strict_output_tps,
            "strict_window_input_tpm": strict_input_tpm,
            "strict_window_input_tps": strict_input_tps,
            "strict_window_uncached_input_tpm": strict_uncached_input_tpm,
            "strict_window_total_tpm": strict_total_tpm,
            "strict_window_total_tps": strict_total_tps,
            "strict_window_rps": strict_rps,
            "strict_window_rpm": strict_rpm,

            # ===== Completed-Lifecycle 口径（完整请求视角）=====
            "completed_lifecycle_start_s": round(lifecycle_start, 3),
            "completed_lifecycle_end_s": round(lifecycle_end, 3),
            "completed_lifecycle_s": round(lifecycle_s, 3),
            "completed_output_tpm": completed_output_tpm,
            "completed_output_tps": completed_output_tps,
            "completed_input_tpm": completed_input_tpm,
            "completed_input_tps": completed_input_tps,
            "completed_uncached_input_tpm": completed_uncached_input_tpm,
            "completed_total_tpm": completed_total_tpm,
            "completed_total_tps": completed_total_tps,

            # ===== 兼容旧字段：默认映射到 strict-window 口径 =====
            "output_tpm": strict_output_tpm,
            "output_tps": strict_output_tps,
            "input_tpm": strict_input_tpm,
            "input_tps": strict_input_tps,
            "tpm": strict_total_tpm,
            "tps": strict_total_tps,
            "tpm_ramp_end_s": round(ramp_end, 3) if tpm_cutoff_seconds else None,
            "tpm_window_s": round(strict_window_s, 3),
        })

    if fail_count > 0:
        error_types: dict = {}
        for r in failed_results:
            err = r["error"][:100]
            error_types[err] = error_types.get(err, 0) + 1
        stats["error_types"] = error_types

    return stats


def print_statistics(stats: dict, concurrency: int, data_file_name: str, time_elapsed: float):
    print(f"\n{'─'*70}")
    print(f"📊 统计结果: [{data_file_name}] 并发={concurrency}")
    print(f"{'─'*70}")
    print(f"  总请求数: {stats['total_requests']}")
    print(f"  成功数: {stats['success_count']}")
    print(f"  失败数: {stats['fail_count']}")
    print(f"  成功率: {stats['success_rate']}")
    print(f"  测试耗时: {time_elapsed:.1f} s")

    if stats['success_count'] > 0:
        print(f"\n  {'指标':<22} {'平均':>10} {'P50':>10} {'P75':>10} {'P90':>10} {'P99':>10}")
        print(f"  {'─'*68}")

        ttfb = stats["ttfb"]
        print(f"  {'首包耗时(s)':<18} {ttfb['avg']:>10.4f} {ttfb['p50']:>10.4f} "
              f"{ttfb['p75']:>10.4f} {ttfb['p90']:>10.4f} {ttfb['p99']:>10.4f}")

        ttlb = stats["ttlb"]
        print(f"  {'整包耗时(s)':<18} {ttlb['avg']:>10.4f} {ttlb['p50']:>10.4f} "
              f"{ttlb['p75']:>10.4f} {ttlb['p90']:>10.4f} {ttlb['p99']:>10.4f}")

        rate = stats["token_rates"]
        print(f"  {'输出速率(t/s)':<16} {rate['avg']:>10.2f} {rate['p50']:>10.2f} "
              f"{rate['p75']:>10.2f} {rate['p90']:>10.2f} {rate['p99']:>10.2f}")
        print(f"  {'  └全局(t/s)':<16} {stats.get('global_token_rate', 0):>10.2f}"
              f"{'':>44}  ← Σcompletion / (总时长 - 平均TTFT)")

        itl = stats.get("itl", {})
        if itl and itl.get("avg", 0) > 0:
            print(f"  {'ITL(ms/token)':<18} {itl['avg']:>10.3f} {itl['p50']:>10.3f} "
                  f"{itl['p75']:>10.3f} {itl['p90']:>10.3f} {itl['p99']:>10.3f}")

        tpot = stats.get("tpot", {})
        if tpot and tpot.get("avg", 0) > 0:
            print(f"  {'TPOT(ms/token)':<18} {tpot['avg']:>10.3f} {tpot['p50']:>10.3f} "
                  f"{tpot['p75']:>10.3f} {tpot['p90']:>10.3f} {tpot['p99']:>10.3f}")

        chr_ = stats.get("cache_hit_ratio", {})
        print(f"  {'Cache命中率':<18} {chr_.get('avg', 0):>10.2%} {chr_.get('p50', 0):>10.2%} "
              f"{chr_.get('p90', 0):>10.2%} {chr_.get('p99', 0):>10.2%}")

        print(f"\n  平均 prompt tokens:     {stats['avg_prompt_tokens']:.0f}")
        print(f"  平均 completion tokens: {stats['avg_completion_tokens']:.0f}")
        print(f"  平均 reasoning tokens:  {stats.get('avg_reasoning_tokens', 0):.0f}")
        print(f"  平均 cached tokens:     {stats.get('avg_cached_tokens', 0):.0f}")
        print(f"  全局 cache 命中率:      {stats.get('avg_cache_hit_ratio', 0):.2%}  "
              f"(cached_tokens 总量 / prompt_tokens 总量)")
        print(f"  总耗时平均值(ms):       {stats.get('avg_total_time_ms', '-')}")

        # ---- TPM 双口径 ----
        print(f"\n  TPM - Strict Window 口径（按 chunk 真实到达时间分摊，剔除爬升+长尾）")
        print(f"  统计窗口:                [{stats.get('strict_window_start_s', 0)}s, "
              f"{stats.get('strict_window_end_s', 0)}s]，{stats.get('strict_window_s', 0)}s")
        print(f"  输出 TPM（tokens/min）:  {stats.get('strict_window_output_tpm', '-')}")
        print(f"  输出 TPS（tokens/s）:    {stats.get('strict_window_output_tps', '-')}")
        print(f"  输入 TPM（tokens/min）:  {stats.get('strict_window_input_tpm', '-')}")
        print(f"  未缓存输入 TPM:          {stats.get('strict_window_uncached_input_tpm', '-')}")
        print(f"  总 TPM（Output+Input）:  {stats.get('strict_window_total_tpm', '-')}")
        print(f"  总 TPS（Output+Input）:  {stats.get('strict_window_total_tps', '-')}")
        if stats.get("strict_window_fallback_requests", 0):
            print(f"  近似计入请求数:          {stats.get('strict_window_fallback_requests', 0)} "
                  f"（流式失败、退化为整体计入的请求）")

        print(f"\n  TPM - Completed Request 口径（完整成功请求生命周期）")
        print(f"  生命周期窗口:            [{stats.get('completed_lifecycle_start_s', 0)}s, "
              f"{stats.get('completed_lifecycle_end_s', 0)}s]，{stats.get('completed_lifecycle_s', 0)}s")
        print(f"  输出 TPM（tokens/min）:  {stats.get('completed_output_tpm', '-')}")
        print(f"  输出 TPS（tokens/s）:    {stats.get('completed_output_tps', '-')}")
        print(f"  输入 TPM（tokens/min）:  {stats.get('completed_input_tpm', '-')}")
        print(f"  未缓存输入 TPM:          {stats.get('completed_uncached_input_tpm', '-')}")
        print(f"  总 TPM（Output+Input）:  {stats.get('completed_total_tpm', '-')}")
        print(f"  总 TPS（Output+Input）:  {stats.get('completed_total_tps', '-')}")

    if stats.get("error_types"):
        print(f"\n  错误类型分布:")
        for err, cnt in stats["error_types"].items():
            print(f"    [{cnt}次] {err}")

    print(f"{'─'*70}")


# ==================== Excel 输出 ====================

def save_to_excel(all_benchmark_results: dict, output_dir: str, timestamp: str, target_model: str):
    """
    将所有 (数据集 × 并发数) 组合的统计数据保存到单个 Excel 文件中。

    Args:
        all_benchmark_results: {data_file_name: {concurrency_N: benchmark_result}}
        output_dir: 输出目录
        timestamp: 时间戳字符串
        target_model: 目标模型名
    """
    wb = Workbook()

    header_font = Font(bold=True, color="FFFFFF", size=11)
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell_alignment = Alignment(horizontal="center", vertical="center")
    thin_border = Border(
        left=Side(style="thin"), right=Side(style="thin"),
        top=Side(style="thin"), bottom=Side(style="thin"),
    )
    success_fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
    fail_fill = PatternFill(start_color="FCE4EC", end_color="FCE4EC", fill_type="solid")

    def style_header_row(ws, row, max_col):
        for col in range(1, max_col + 1):
            cell = ws.cell(row=row, column=col)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_alignment
            cell.border = thin_border

    def style_data_cell(ws, row, col, value=None):
        cell = ws.cell(row=row, column=col)
        if value is not None:
            cell.value = value
        cell.alignment = cell_alignment
        cell.border = thin_border
        return cell

    # 展平所有结果为 list，按 (数据集, 并发) 排序
    flat_results = []
    for data_file_name, file_results in all_benchmark_results.items():
        for _, br in file_results.items():
            flat_results.append(br)
    flat_results.sort(key=lambda br: (br.get("data_file_name", ""), br.get("concurrency", 0)))

    # ===== Sheet 1: 汇总（每行一个组合）=====
    ws_summary = wb.active
    ws_summary.title = "汇总"
    summary_headers = [
        "数据集", "并发数", "轮次",
        # —— 常用核心指标（前置便于快速对比）——
        "★Avg TTFT(s)", "★Avg ITL(ms)", "★Avg TPOT(ms)", "★Avg Cache Hit Ratio",
        "★Strict Total TPM", "★Strict Output TPM",
        "压测耗时(s)", "执行耗时(s)", "TPM计算窗口(s)", "Session数", "总请求数", "成功数", "失败数", "成功率",
        "Avg TTFT(s)", "P50 TTFT(s)", "P75 TTFT(s)", "P90 TTFT(s)", "P95 TTFT(s)", "P99 TTFT(s)",
        "Avg TTLT(s)", "P50 TTLT(s)", "P75 TTLT(s)", "P90 TTLT(s)", "P95 TTLT(s)", "P99 TTLT(s)",
        "Avg Rate(t/s)", "P50 Rate(t/s)", "P75 Rate(t/s)", "P90 Rate(t/s)", "P95 Rate(t/s)", "P99 Rate(t/s)",
        "Global Token Rate(t/s)",
        "Avg ITL(ms)", "P50 ITL(ms)", "P75 ITL(ms)", "P90 ITL(ms)", "P95 ITL(ms)", "P99 ITL(ms)",
        "Avg TPOT(ms)", "P50 TPOT(ms)", "P75 TPOT(ms)", "P90 TPOT(ms)", "P95 TPOT(ms)", "P99 TPOT(ms)",
        "Avg Prompt Tokens", "Avg Completion Tokens", "Avg Reasoning Tokens", "Avg Cached Tokens",
        "Avg Cache Hit Ratio", "P50 Cache Hit Ratio", "P90 Cache Hit Ratio", "P99 Cache Hit Ratio",
        "Avg Total Time(ms)",
        # Strict-Window 口径（瞬时吞吐峰值视角）
        "Strict Window(s)", "Strict Output TPM", "Strict Output TPS",
        "Strict Input TPM", "Strict Uncached Input TPM", "Strict Total TPM", "Strict Total TPS",
        # Completed-Lifecycle 口径（完整请求视角）
        "Completed Lifecycle(s)", "Completed Output TPM", "Completed Output TPS",
        "Completed Input TPM", "Completed Uncached Input TPM", "Completed Total TPM", "Completed Total TPS",
    ]
    for col_idx, header in enumerate(summary_headers, 1):
        ws_summary.cell(row=1, column=col_idx, value=header)
    style_header_row(ws_summary, 1, len(summary_headers))

    for row_idx, benchmark_result in enumerate(flat_results, start=2):
        s = benchmark_result["statistics"]
        c = benchmark_result["concurrency"]
        ttfb = s.get("ttfb", {})
        ttlb = s.get("ttlb", {})
        rate = s.get("token_rates", {})
        itl = s.get("itl", {})
        tpot = s.get("tpot", {})
        chr_ = s.get("cache_hit_ratio", {})
        values = [
            benchmark_result.get("data_file_name", "-"),
            c,
            benchmark_result.get("run_round", 1),
            # —— 常用核心指标（前置）——
            ttfb.get("avg", "-"),
            itl.get("avg", "-"),
            tpot.get("avg", "-"),
            chr_.get("avg", "-"),
            s.get("strict_window_total_tpm", "-"),
            s.get("strict_window_output_tpm", "-"),
            benchmark_result.get("benchmark_duration", "-"),
            benchmark_result.get("exec_time", "-"),
            s.get("tpm_window_s", "-"),
            benchmark_result.get("session_count", "-"),
            s.get("total_requests", 0),
            s.get("success_count", 0),
            s.get("fail_count", 0),
            s.get("success_rate", "0%"),
            ttfb.get("avg", "-"), ttfb.get("p50", "-"), ttfb.get("p75", "-"), ttfb.get("p90", "-"),
            ttfb.get("p95", "-"), ttfb.get("p99", "-"),
            ttlb.get("avg", "-"), ttlb.get("p50", "-"), ttlb.get("p75", "-"), ttlb.get("p90", "-"),
            ttlb.get("p95", "-"), ttlb.get("p99", "-"),
            rate.get("avg", "-"), rate.get("p50", "-"), rate.get("p75", "-"), rate.get("p90", "-"),
            rate.get("p95", "-"), rate.get("p99", "-"),
            s.get("global_token_rate", "-"),
            itl.get("avg", "-"), itl.get("p50", "-"), itl.get("p75", "-"), itl.get("p90", "-"),
            itl.get("p95", "-"), itl.get("p99", "-"),
            tpot.get("avg", "-"), tpot.get("p50", "-"), tpot.get("p75", "-"), tpot.get("p90", "-"),
            tpot.get("p95", "-"), tpot.get("p99", "-"),
            s.get("avg_prompt_tokens", "-"), s.get("avg_completion_tokens", "-"),
            s.get("avg_reasoning_tokens", "-"),
            s.get("avg_cached_tokens", "-"),
            chr_.get("avg", "-"), chr_.get("p50", "-"), chr_.get("p90", "-"), chr_.get("p99", "-"),
            s.get("avg_total_time_ms", "-"),
            # Strict-Window 列
            s.get("strict_window_s", "-"),
            s.get("strict_window_output_tpm", "-"), s.get("strict_window_output_tps", "-"),
            s.get("strict_window_input_tpm", "-"), s.get("strict_window_uncached_input_tpm", "-"),
            s.get("strict_window_total_tpm", "-"), s.get("strict_window_total_tps", "-"),
            # Completed-Lifecycle 列
            s.get("completed_lifecycle_s", "-"),
            s.get("completed_output_tpm", "-"), s.get("completed_output_tps", "-"),
            s.get("completed_input_tpm", "-"), s.get("completed_uncached_input_tpm", "-"),
            s.get("completed_total_tpm", "-"), s.get("completed_total_tps", "-"),
        ]
        for col_idx, val in enumerate(values, 1):
            cell = style_data_cell(ws_summary, row_idx, col_idx, val)
            # 成功数/失败数列因常用指标前置而后移 6 列：原 8/9 → 现 14/15
            # 加「轮次」列后再后移 1 列：现 15/16
            if col_idx == 15:
                cell.fill = success_fill
            elif col_idx == 16 and isinstance(val, (int, float)) and val > 0:
                cell.fill = fail_fill

    for col_idx in range(1, len(summary_headers) + 1):
        ws_summary.column_dimensions[get_column_letter(col_idx)].width = min(
            len(str(summary_headers[col_idx - 1])) + 4, 30
        )
    ws_summary.freeze_panes = "A2"

    # ===== Sheet 2: 错误统计（跨组合聚合）=====
    ws_errors = wb.create_sheet("错误统计")
    error_headers = ["数据集", "并发数", "轮次", "错误类型", "次数"]
    for col_idx, header in enumerate(error_headers, 1):
        ws_errors.cell(row=1, column=col_idx, value=header)
    style_header_row(ws_errors, 1, len(error_headers))
    err_row = 2
    for benchmark_result in flat_results:
        s = benchmark_result["statistics"]
        c = benchmark_result["concurrency"]
        for err_msg, count in s.get("error_types", {}).items():
            style_data_cell(ws_errors, err_row, 1, benchmark_result.get("data_file_name", "-"))
            style_data_cell(ws_errors, err_row, 2, c)
            style_data_cell(ws_errors, err_row, 3, benchmark_result.get("run_round", 1))
            style_data_cell(ws_errors, err_row, 4, err_msg)
            cell = style_data_cell(ws_errors, err_row, 5, count)
            cell.fill = fail_fill
            err_row += 1
    for col_idx in range(1, len(error_headers) + 1):
        ws_errors.column_dimensions[get_column_letter(col_idx)].width = (
            40 if col_idx == 4 else 25 if col_idx == 1 else 12
        )
    ws_errors.freeze_panes = "A2"

    # ===== Sheet 3+: 请求明细（每个组合一个 sheet）=====
    detail_headers = [
        "Session ID", "序号", "Request ID",
        "TTFB(s)", "TTLB(s)", "Rate(t/s)", "ITL(ms)", "TPOT(ms)",
        "Prompt Tokens", "Completion Tokens", "Reasoning Tokens", "Cached Tokens", "Cache Hit Ratio",
        "Finish Reason", "Chunk Count",
        "Answer长度", "CoT长度", "Tool Calls数", "错误信息",
    ]
    used_sheet_names = set()
    for benchmark_result in flat_results:
        data_name = benchmark_result.get("data_file_name", "data")
        c = benchmark_result.get("concurrency", 0)
        # Excel sheet 名称限制 31 字符，需清洗特殊字符
        run_round = benchmark_result.get("run_round", 1)
        data_stem = os.path.splitext(data_name)[0]
        sheet_name = f"{data_stem}_c{c}_r{run_round}"
        # 替换非法字符 [/ \ ? * : [ ]]
        for ch in r'/\?*:[]':
            sheet_name = sheet_name.replace(ch, '_')
        # 截断到 31 字符
        sheet_name = sheet_name[:31]
        # 去重
        base_name = sheet_name
        suffix = 1
        while sheet_name in used_sheet_names:
            suffix_str = f"_{suffix}"
            sheet_name = (base_name[:31 - len(suffix_str)]) + suffix_str
            suffix += 1
        used_sheet_names.add(sheet_name)

        ws_detail = wb.create_sheet(sheet_name)
        for col_idx, header in enumerate(detail_headers, 1):
            ws_detail.cell(row=1, column=col_idx, value=header)
        style_header_row(ws_detail, 1, len(detail_headers))

        detail_row = 2
        for seq, r in enumerate(benchmark_result.get("raw_results", []), 1):
            row_vals = [
                r.get("session_id", ""),
                seq,
                r.get("req_id", ""),
                r.get("ttfb", 0), r.get("ttlb", 0),
                r.get("token_rates", 0), r.get("itl", 0), r.get("tpot", 0),
                r.get("prompt_tokens", 0), r.get("completion_tokens", 0),
                r.get("reasoning_tokens", 0),
                r.get("cached_tokens", 0), r.get("cache_hit_ratio", 0),
                r.get("finish_reason", ""), r.get("chunk_count", 0),
                r.get("answer_len", 0), r.get("cot_len", 0),
                r.get("tool_calls_count", 0), r.get("error", ""),
            ]
            for col_idx, val in enumerate(row_vals, 1):
                cell = style_data_cell(ws_detail, detail_row, col_idx, val)
                if r.get("error") and col_idx == len(detail_headers):
                    cell.fill = fail_fill
            detail_row += 1

        for col_idx in range(1, len(detail_headers) + 1):
            ws_detail.column_dimensions[get_column_letter(col_idx)].width = (
                20 if col_idx in (1, 3, 19) else 16
            )
        ws_detail.freeze_panes = "A2"

    os.makedirs(output_dir, exist_ok=True)
    safe_model = target_model.replace("/", "_")
    excel_file = os.path.join(
        output_dir,
        f"{safe_model}_session_benchmark_{timestamp}.xlsx"
    )
    os.makedirs(os.path.dirname(excel_file) or ".", exist_ok=True)
    wb.save(excel_file)
    print(f"\n📊 Excel 统计表已保存到: {excel_file}")
    return excel_file


def incremental_save(all_benchmark_results: dict, output_dir: str, timestamp: str,
                     target_model: str, benchmark_start_time: float, data_files: list,
                     concurrency_levels: list, stop_threshold: float, duration: int,
                     pre_sample: int, session_key: str, time_key: str,
                     enable_stream: bool, enable_uuid_prefix: bool,
                     tpm_cache_filter: str = ""):
    """每完成一个 (数据集, 并发) 组合后增量保存当前所有结果到 JSONL 和 Excel。

    JSONL 文件结构对齐 `http_model_concurrency_benchmark.py` 的输出，便于复用
    分析工具：
      {
        "total_elapsed_seconds": ...,
        "total_elapsed_display": ...,
        "config": {...},                        # session 脚本特有的运行配置
        "datasets": {
          "<data_file_name>": {
            "concurrency_<N>": {
              ...benchmark_result_meta...,
              "statistics": {...},
              "raw_results": [ {req级明细}, ... ]   # ★ 单条请求原始结果
            }
          }
        }
      }
    """
    elapsed = round(time.time() - benchmark_start_time, 1)
    final_output = {
        "total_elapsed_seconds": elapsed,
        "total_elapsed_display": f"{int(elapsed)//60}分{int(elapsed)%60}秒",
        "config": {
            "target_model": target_model,
            "base_url": BASE_URL,
            "concurrency_levels": concurrency_levels,
            "rounds": ROUNDS,
            "stop_threshold": stop_threshold,
            "duration": duration,
            "pre_sample": pre_sample,
            "data_files": [os.path.basename(p) for p in data_files],
            "session_key": session_key,
            "time_key": time_key,
            "enable_stream": enable_stream,
            "enable_uuid_prefix": enable_uuid_prefix,
            "enable_run_salt": ENABLE_RUN_SALT,
        },
        # 嵌套结构 + 完整 raw_results，与 http_model_concurrency_benchmark 输出对齐
        "datasets": all_benchmark_results,
    }
    safe_model = target_model.replace("/", "_")
    output_jsonl = os.path.join(output_dir, f"{safe_model}_session_benchmark_{timestamp}.jsonl")
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    print(f"  💾 增量保存 JSONL: {output_jsonl}")

    excel_file = save_to_excel(all_benchmark_results, output_dir, timestamp, target_model)

    # 在 Excel 统计表中追加 TPM 趋势分析（数据 + 图 + Cache 段汇总，可选）
    # 任何失败都只 warning，不影响发压主流程。
    try:
        # 同目录脚本，直接 import 即可
        from compute_tpm_per_minute import enrich_xlsx_with_tpm_trend
        enrich_xlsx_with_tpm_trend(
            jsonl_path=output_jsonl,
            xlsx_path=excel_file,
            cache_filter=tpm_cache_filter,  # CLI --tpm_cache_filter；空字符串表示不做 cache 段筛选
            verbose=True,
        )
    except Exception as _e:
        print(f"  ⚠️ TPM 趋势嵌入失败（跳过，不影响主流程）: {_e}")




def main():
    args = parse_arguments()

    # ── 解析并发级别 ──────────────────────────────────────────────────────────
    global TARGET_MODEL, API_KEY, BASE_URL, CONCURRENCY, CONCURRENCY_LEVELS, ROUNDS, \
           STOP_THRESHOLD, ENABLE_STREAM, ENABLE_UUID_PREFIX, ENABLE_RUN_SALT, ENABLE_THINKING, THINKING_TYPE, DURATION, REQUEST_TIMEOUT, \
           MAX_TOKENS, SESSION_ID_HEADER, GARBLED_LOG_FILE, GARBLED_RATIO_THRESHOLD, REASONING_EFFORT, \
           REPEAT_LOG_FILE, REPEAT_MIN_COUNT, REPEAT_MIN_SEGMENT_LEN, REPEAT_MIN_COVERAGE_RATIO
    TARGET_MODEL = args.target_model
    API_KEY = args.api_key
    BASE_URL = args.base_url
    STOP_THRESHOLD = args.tail_threshold
    ENABLE_STREAM = args.enable_stream
    ENABLE_UUID_PREFIX = args.enable_uuid_prefix
    ENABLE_RUN_SALT = args.enable_run_salt
    ENABLE_THINKING = args.enable_thinking
    THINKING_TYPE = args.thinking_type if args.thinking_type else None
    REASONING_EFFORT = args.reasoning_effort if args.reasoning_effort else None
    DURATION = args.duration
    REQUEST_TIMEOUT = args.request_timeout
    globals()['STREAM_READ_TIMEOUT'] = args.stream_read_timeout  # patch: 同步到模块全局，供 worker 的 httpx 读超时使用
    MAX_TOKENS = args.max_tokens if args.max_tokens and args.max_tokens > 0 else None
    SESSION_ID_HEADER = args.session_id_header.strip() if args.session_id_header else None
    GARBLED_RATIO_THRESHOLD = args.garbled_threshold
    REPEAT_MIN_COUNT = args.repeat_min_count
    REPEAT_MIN_SEGMENT_LEN = args.repeat_min_segment_len
    REPEAT_MIN_COVERAGE_RATIO = args.repeat_min_coverage_ratio

    if args.concurrency_levels:
        CONCURRENCY_LEVELS = [int(x) for x in args.concurrency_levels.split(',') if x.strip()]
    CONCURRENCY = CONCURRENCY_LEVELS[0]
    ROUNDS = args.rounds if args.rounds and args.rounds > 0 else 1

    # ── 解析数据文件 ──────────────────────────────────────────────────────────
    if args.data_files:
        candidates = [f.strip() for f in args.data_files.split(',') if f.strip()]
        data_files = []
        for f in candidates:
            path = f if os.path.isabs(f) else os.path.join(data_dir, f)
            if os.path.isfile(path):
                data_files.append(path)
            else:
                print(f"⚠️ 警告: 数据文件不存在，跳过: {path}")
        if not data_files:
            print("❌ 错误: 没有有效的数据文件")
            sys.exit(1)
    elif args.data_dir:
        scan_dir = args.data_dir
        if not os.path.isdir(scan_dir):
            print(f"❌ 错误: data_dir 不存在: {scan_dir}")
            sys.exit(1)
        data_files = sorted(
            os.path.join(scan_dir, fn)
            for fn in os.listdir(scan_dir)
            if fn.endswith(('.json', '.jsonl')) and os.path.isfile(os.path.join(scan_dir, fn))
        )
        if not data_files:
            print(f"❌ 错误: 目录 {scan_dir} 内无 .json/.jsonl 文件")
            sys.exit(1)
        print(f"  从目录加载 {len(data_files)} 个文件: {scan_dir}")
    else:
        data_files = [DEFAULT_DATA_FILE]

    # ── 打印配置 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("📋 Session 并发压测配置")
    print(f"{'='*70}")
    print(f"  目标模型:             {TARGET_MODEL}")
    print(f"  API 地址:             {BASE_URL}")
    print(f"  并发级别:             {CONCURRENCY_LEVELS}")
    print(f"  重复轮数:             {ROUNDS}" + (f"（每个 数据集×并发 组合跑 {ROUNDS} 轮，汇总表每轮一行）" if ROUNDS > 1 else ""))
    if DURATION > 0:
        print(f"  压测时长:             {DURATION}s（固定时长模式，session 循环复用）")
    else:
        print(f"  停止阈值:             {STOP_THRESHOLD}")
    print(f"  流式请求:             {'启用' if ENABLE_STREAM else '禁用'}")
    print(f"  UUID 前缀:            {'启用' if ENABLE_UUID_PREFIX else '禁用'}")
    print(f"  运行轮次盐值(方案B):  {'启用' if ENABLE_RUN_SALT else '禁用'}"
          + ("（uuid 前缀已启用，本项不生效）" if (ENABLE_RUN_SALT and ENABLE_UUID_PREFIX) else ""))
    print(f"  Think Enable:      {'启用' if ENABLE_THINKING else '禁用'}")
    print(f"  thinking.type:        {THINKING_TYPE if THINKING_TYPE else ('enabled' if ENABLE_THINKING else 'disabled')}")
    print(f"  reasoning_effort:     {REASONING_EFFORT if REASONING_EFFORT else '不传（沿用原始请求）'}")
    print(f"  流式read超时:         {STREAM_READ_TIMEOUT}s")
    print(f"  max_tokens 覆盖:      {MAX_TOKENS if MAX_TOKENS else '不覆盖（沿用原始请求）'}")
    _msf = args.max_samples_per_file if args.max_samples_per_file and args.max_samples_per_file > 0 else 0
    print(f"  每文件最大样本数:     {_msf if _msf else '不限（加载全部）'}")
    print(f"  数据文件:             {len(data_files)} 个")
    for f in data_files:
        print(f"    - {f}")
    if args.pre_sample > 0:
        print(f"  预截取记录:           前 {args.pre_sample} 条（按 start_time 顺序，结果可重复）")
    print(f"  Session 字段:         {args.session_key}")
    print(f"  时间字段:             {args.time_key}")
    print(f"  Session 下发顺序:     {args.session_order}"
          + (f"（seed={args.order_seed}）" if args.session_order == 'shuffle' else ""))
    print(f"  回放标识:             {args.replay_id if args.replay_id else '未设置'}")
    if args.dump_plan:
        print(f"  下发计划落盘:         {args.dump_plan}")
    print(f"  Session ID Header:    {SESSION_ID_HEADER if SESSION_ID_HEADER else '不透传'}")
    print(f"{'='*70}\n")

    # ── 预加载所有数据文件 ────────────────────────────────────────────────────
    loaded_data = {}
    max_samples = args.max_samples_per_file if args.max_samples_per_file and args.max_samples_per_file > 0 else 0
    for data_file in data_files:
        data_file_name = os.path.basename(data_file)
        print(f"📂 加载数据文件: {data_file}")
        sessions = load_data(data_file, args.session_key, args.time_key,
                             max_samples=max_samples)
        if args.pre_sample > 0:
            sessions = pre_sample_sessions(sessions, args.pre_sample, args.time_key)
        # session 之间的下发顺序：显式决定，避免"文件首次出现顺序"依赖数据文件排布
        sessions = order_sessions(sessions, args.session_order, args.time_key,
                                  args.order_seed)
        if args.dump_plan:
            plan_path = args.dump_plan
            if len(data_files) > 1:
                # 多文件时按数据文件名派生，避免互相覆盖
                root, ext = os.path.splitext(args.dump_plan)
                plan_path = f"{root}.{data_file_name}{ext or '.json'}"
            dump_plan(plan_path, sessions, args.time_key, args.replay_id, data_file_name)

        loaded_data[data_file_name] = sessions

    client = create_client(API_KEY, BASE_URL)
    benchmark_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 初始化乱码日志文件路径：优先使用 --garbled_log 指定路径，否则用默认命名
    if args.garbled_log:
        GARBLED_LOG_FILE = args.garbled_log
    else:
        _safe_model = TARGET_MODEL.replace("/", "_")
        GARBLED_LOG_FILE = os.path.join(OUTPUT_DIR, f"{_safe_model}_garbled_{timestamp}.log")
    os.makedirs(os.path.dirname(GARBLED_LOG_FILE) or ".", exist_ok=True)
    print(f"  乱码日志文件:         {GARBLED_LOG_FILE}（阈值 {GARBLED_RATIO_THRESHOLD}，异步处理）")
    _start_garbled_worker()  # 启动后台线程：异步执行乱码检测 + 写日志

    # 初始化重复输出（退化复读）日志文件路径：优先使用 --repeat_log 指定路径，否则用默认命名
    if args.repeat_log:
        REPEAT_LOG_FILE = args.repeat_log
    else:
        _safe_model = TARGET_MODEL.replace("/", "_")
        REPEAT_LOG_FILE = os.path.join(OUTPUT_DIR, f"{_safe_model}_repeat_{timestamp}.log")
    os.makedirs(os.path.dirname(REPEAT_LOG_FILE) or ".", exist_ok=True)
    print(f"  重复输出日志文件:     {REPEAT_LOG_FILE}（分段最小长度 {REPEAT_MIN_SEGMENT_LEN} 字符/次数阈值 {REPEAT_MIN_COUNT} 次/覆盖占比阈值 {REPEAT_MIN_COVERAGE_RATIO}，异步处理）")
    _start_repeat_worker()  # 启动后台线程：异步执行重复输出检测 + 写日志

    all_benchmark_results = {}  # data_file_name -> {concurrency_N -> benchmark_result}

    # ── 双层循环：并发数 × 数据文件 ──────────────────────────────────────────
    for concurrency in CONCURRENCY_LEVELS:
        print(f"\n{'='*70}")
        print(f"🔧 并发数: {concurrency}")

        for data_file in data_files:
            data_file_name = os.path.basename(data_file)
            sessions = loaded_data.get(data_file_name)
            if not sessions:
                continue

            for round_idx in range(1, ROUNDS + 1):
                print(f"\n{'─'*70}")
                print(f"  数据文件: {data_file_name}  Session 数: {len(sessions)}"
                      + (f"  第 {round_idx}/{ROUNDS} 轮" if ROUNDS > 1 else ""))

                if data_file_name not in all_benchmark_results:
                    all_benchmark_results[data_file_name] = {}

                try:
                    results, actual_duration, stop_offset = run_session_benchmark(
                        client=client,
                        sessions=sessions,
                        concurrency=concurrency,
                        target_model=TARGET_MODEL,
                        enable_stream=ENABLE_STREAM,
                        enable_uuid_prefix=ENABLE_UUID_PREFIX,
                        enable_thinking=ENABLE_THINKING,
                        stop_threshold=STOP_THRESHOLD,
                        data_file_name=data_file_name,
                        enable_run_salt=ENABLE_RUN_SALT,
                        duration=DURATION,
                    )

                    stats = compute_statistics(
                        results,
                        duration_seconds=actual_duration,
                        # 两种模式都使用 stop_offset 作为 tpm 截止时间：
                        # duration>0：stop_offset = timer 触发 stop_event 的时刻（= DURATION）
                        # duration=0：stop_offset = 活跃并发首次降到阈值的时刻
                        # 均排除全局中断后仍在运行的长尾请求对 tpm 的影响
                        tpm_cutoff_seconds=stop_offset,
                    )
                    print_statistics(stats, concurrency, data_file_name, actual_duration)

                    all_benchmark_results[data_file_name][f"concurrency_{concurrency}_run{round_idx}"] = {
                        "data_file_name": data_file_name,
                        "concurrency": concurrency,
                        "run_round": round_idx,
                        "stop_threshold": STOP_THRESHOLD,
                        "session_count": len(sessions),
                        "benchmark_duration": round(stop_offset, 1),
                        "exec_time": round(actual_duration, 1),
                        "duration_seconds": round(actual_duration, 1),
                        "tpm_cutoff_seconds": round(stop_offset, 1),
                        "statistics": stats,
                        "raw_results": results,
                    }

                except Exception as e:
                    print(f"  ❌ 压测异常: {e}")
                    import traceback
                    traceback.print_exc()

                finally:
                    # 无论成功还是异常，每完成一个组合就增量保存
                    incremental_save(
                        all_benchmark_results, OUTPUT_DIR, timestamp,
                        target_model=TARGET_MODEL,
                        benchmark_start_time=benchmark_start,
                        data_files=data_files,
                        concurrency_levels=CONCURRENCY_LEVELS,
                        stop_threshold=STOP_THRESHOLD,
                        duration=DURATION,
                        pre_sample=args.pre_sample,
                        session_key=args.session_key,
                        time_key=args.time_key,
                        enable_stream=ENABLE_STREAM,
                        enable_uuid_prefix=ENABLE_UUID_PREFIX,
                        tpm_cache_filter=args.tpm_cache_filter,
                    )

                    # 档间重建 client：上一档可能残留"无法被 stop_event 唤醒"的悬挂 worker，
                    # 它们仍持有当前 client 的 httpx 连接池资源。重建 client 后，旧连接池
                    # 会随悬挂线程结束（或 GC）自然释放，新档使用全新连接池，避免连接复用
                    # 命中半死链接 / 排队阻塞等问题。
                    try:
                        old_client = client
                        client = create_client(API_KEY, BASE_URL)
                        try:
                            old_client.close()
                        except Exception:
                            pass
                    except Exception as _e:
                        print(f"  ⚠️ 重建 client 失败（沿用旧 client）：{_e}")

    # ── 最终保存（与增量保存文件相同，确保最终状态完整）─────────────────────
    total_elapsed = time.time() - benchmark_start
    incremental_save(
        all_benchmark_results, OUTPUT_DIR, timestamp,
        target_model=TARGET_MODEL,
        benchmark_start_time=benchmark_start,
        data_files=data_files,
        concurrency_levels=CONCURRENCY_LEVELS,
        stop_threshold=STOP_THRESHOLD,
        duration=DURATION,
        pre_sample=args.pre_sample,
        session_key=args.session_key,
        time_key=args.time_key,
        enable_stream=ENABLE_STREAM,
        enable_uuid_prefix=ENABLE_UUID_PREFIX,
        tpm_cache_filter=args.tpm_cache_filter,
    )
    print(f"\n✅ 最终结果已保存")

    # 压测全部结束：等待后台 worker 处理完队列中剩余的乱码/重复输出检测任务后退出
    _stop_garbled_worker()
    _stop_repeat_worker()

    total_minutes, total_seconds = divmod(int(total_elapsed), 60)
    print(f"\n⏱️  整体压测耗时: {total_minutes}分{total_seconds}秒 ({total_elapsed:.1f}s)")


if __name__ == "__main__":
    main()
