#!/bin/bash
set -euo pipefail

source /opt/dtk-26.04-blas-nightly-1019/env.sh
export LD_LIBRARY_PATH=/opt/dtk-26.04-blas-nightly-1019/lib/rocblas/:$LD_LIBRARY_PATH
# Usage:
#   bash decode.sh        # S3 (default)
#   bash decode.sh s1     # S1 10.41.103.12  enp10s0f4u2c2
#   bash decode.sh s2     # S2 10.41.103.10  enp10s0f4u2
#   bash decode.sh s3     # S3 10.41.101.164 eth0
#   bash decode.sh s4     # S4 10.41.103.4   enp10s0f4u2
NODE="${1:-${DECODE_NODE:-s2}}"
NODE="$(echo "${NODE}" | tr '[:upper:]' '[:lower:]')"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${NODE}" in
s1|1)
        HOST=10.41.103.12
        SOCKET_IFNAME=enp10s0f4u2c2
        NODE=s1
        ;;
    s2|2)
        HOST=10.41.103.10
        SOCKET_IFNAME=enp10s0f4u2
        NODE=s2
        ;;
    s3|3)
        HOST=10.41.101.164
        SOCKET_IFNAME=eth0
        NODE=s3
        ;;
    s4|4)
        HOST=10.41.103.4
        SOCKET_IFNAME=enp10s0f4u2
        NODE=s4
        ;;
    s7|7)
        HOST=10.41.103.8
        SOCKET_IFNAME=enp10s0f4u2
        NODE=s7
        ;;
    *)
        echo "Usage: $0 [s1|s2|s3|s4]   # s2=default 10.41.103.10" >&2
        exit 1
        ;;
esac

MODEL_PATH=/model/GLM-5.3-Channel-INT4-w4a8
IB_DEVICES=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7

export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
DIST_INIT_HOST="${HOST}"
NODE_RANK=0
export PYTHONPATH=/workspace/kv-research/glm5.3/sglang-das_songge/python:${PYTHONPATH:-}
#SGLANG_SRC="/workspace/glm5.3_int4/sglang-das_glm5.3-w4a8/python"
#python3 - <<PY
#import os, sglang
#want = os.path.realpath("${SGLANG_SRC}")
#got = os.path.realpath(sglang.__file__)
#print("using sglang:", got)
#if not got.startswith(want + os.sep) and got != os.path.join(want, "sglang", "__init__.py"):
#    raise SystemExit(f"ERROR: sglang not from PYTHONPATH:\n  {got}")
#PY

# ===================== RCCL / NCCL =====================
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
#export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=${IB_DEVICES}
export NCCL_SOCKET_IFNAME=${SOCKET_IFNAME}
export GLOO_SOCKET_IFNAME=${SOCKET_IFNAME}

#unset NCCL_IB_GID_INDEX

#export NCCL_NET_PLUGIN=shca
#export NCCL_PLUGIN_P2P=ib

# 关闭 debug 刷屏
unset NCCL_DEBUG
unset NCCL_DEBUG_SUBSYS
unset NCCL_DEBUG_FILE
unset RCCL_DEBUG
unset RCCL_DEBUG_SUBSYS
unset TORCH_NCCL_DEBUG
export TORCH_CPP_LOG_LEVEL=ERROR
export PYTHONWARNINGS="ignore"


# ===================== SGLang / 通用性能参数 =====================
export SGLANG_ENABLE_SPEC_V2=1
#export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export HSA_FORCE_FINE_GRAIN_PCIE=1
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=0
export W8A8_SUPPORT_METHODS=3
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_USE_MODELSCOPE=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export HSA_USE_SVM=0


# ===================== DEEPEP =====================
#export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
#export ROCSHMEM_MAX_NUM_CONTEXTS=60
export ROCSHMEM_ALLOWED_IBV_DEVICES=${IB_DEVICES}
#export ROCSHMEM_HEAP_SIZE=3737418240 # 419430400 # 400 MB
export ROCSHMEM_IB_GID_INDEX=0
export ROCSHMEM_GDR_DISABLE_XDP=0
#export ROCSHMEM_DISABLE_HDP_FLUSH=1
# export ROCSHMEM_TOPO_FILE_FORCE=${TOPO_FILE}
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
export DEEPEP_ENABLE_LL_LAYERED_OPT=1 # deepep分层
#export DEEPEP_ENABLE_MORI_EP_LL_LAYERED=1 #mori-ep分层

# ===================== Mooncake / Disaggregation =====================
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_GID_INDEX=0
export MC_TE_FILTERS=${IB_DEVICES}
export MC_ALLOWED_IBV_DEVICES=${IB_DEVICES}
# unset MC_GID_INDEX

export SGLANG_DEBUG_RADIX_CACHE_TIME=0
# export SGLANG_NSA_HCU_INT8_INDEX_K_CACHE=1
export SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT=120
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1

# export SGLANG_SIMULATE_ACC_LEN=3.3
# export SGLANG_SIMULATE_ACC_METHOD=match-expected

# export SGLANG_NSA_HCU_REUSE_SORTED_TOPK=1
export SGLANG_NSA_FUSE_TOPK=1
export SGLANG_OPT_USE_TOPK_V2=0

export LIGHTOP_MQA_LOGITS_BETA_FP8=1

export SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=0 
export SGLANG_USE_W4A8_MASKED_HIPC=1


# export ROCBLAS_LAYER=3
# export FLASH_MLA_DSA_DEBUG=1
# export DEEPGEMM_PRINT_OP_SIZE=1 
# export DEEPGEMM_PRINT_OP_SIZE_STATS=1

# export SGLANG_DEBUG_DECODE_RADIX_CACHE=1

LOG_DIR="${SCRIPT_DIR}/logs/$(date +%Y%m%d)"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/decode-dp8ep8_$(date +%Y%m%d_%H%M%S).log"
echo "decode log: ${LOG_FILE}"
cd "${SCRIPT_DIR}"
export PYTHONUNBUFFERED=1

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256

# ===================== 启动 SGLang Decode =====================
sglang serve \
  --model-path ${MODEL_PATH} \
  --trust-remote-code \
  --host ${HOST} \
  --port 30023 \
  --dist-init-addr ${DIST_INIT_HOST}:6000 \
  --nnodes 1 \
  --node-rank ${NODE_RANK} \
  --tp-size 8 \
  --moe-dense-tp-size 1 \
  --dp-size 8 \
  --ep-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --enable-dp-lm-head \
  --deepep-mode low_latency \
  --page-size 64 \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --quantization slimquant_w4a8_marlin \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.82 \
  --chunked-prefill-size -1 \
  --cuda-graph-max-bs 16 \
  --max-running-requests 128 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --speculative-algorithm EAGLE \
  --json-model-override-args '{"index_share_for_mtp_iteration": false}' \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --speculative-draft-lm-head-vp-size 8 \
  --tokenizer-worker-num 8 \
  --enable-cache-report \
  --enable-metrics \
  --enable-request-time-stats-logging \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device "${IB_DEVICES}" \
  --disaggregation-bootstrap-port 8299 \
  --load-balance-method total_tokens \
  --disaggregation-decode-enable-radix-cache \
   2>&1 | tee "${LOG_FILE}"
#   --disable-radix-cache \
#   --speculative-num-steps 5 \
#   --speculative-eagle-topk 1 \
#   --speculative-num-draft-tokens 6 \
#   --speculative-draft-lm-head-vp-size 8 \



#    --json-model-override-args '{"index_share_for_mtp_iteration": false}' \
# --ep-dispatch-algorithm static \
#   --ep-static-dispatch-policy locality_fair \
#   --ep-num-redundant-experts 64 \
#   --disable-radix-cache \
#--enable-request-time-stats-logging \
#    --enable-metrics \
#    --expert-balancedness-report-mode both \

