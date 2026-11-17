#!/usr/bin/env bash
set -euo pipefail

source /opt/dtk-26.04-blas-nightly-1019/env.sh
export LD_LIBRARY_PATH=/opt/dtk-26.04-blas-nightly-1019/lib/rocblas/:$LD_LIBRARY_PATH
#export ROCBLAS_TENSILE_LIBPATH=./library_gpu6

NODE="${1:-${PREFILL_NODE:-s1}}"
NODE="$(echo "${NODE}" | tr '[:upper:]' '[:lower:]')"
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

export PYTHONPATH=/workspace/kv-research/glm5.3/sglang-das_songge/python:${PYTHONPATH:-}

# python3 - <<PY
# import os, sglang
# want = os.path.realpath("${SGLANG_SRC}")
# got = os.path.realpath(sglang.__file__)
# print("using sglang:", got)
# if not got.startswith(want + os.sep) and got != os.path.join(want, "sglang", "__init__.py"):
#     raise SystemExit(f"ERROR: sglang not from PYTHONPATH:\n  {got}")
# PY

export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=5
export SGLANG_USE_MODELSCOPE=1

export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0

# H2D / D2H
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512


# hip launch
export HSA_KERNARG_POOL_SIZE=8388608
export HSA_FORCE_FINE_GRAIN_PCIE=1
export ROC_AQL_QUEUE_SIZE=131072

# nccl / rccl
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_IB_HCA=${IB_DEVICES}
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${SOCKET_IFNAME}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${SOCKET_IFNAME}}"

# unset NCCL_IB_GID_INDEX

# export NCCL_NET_PLUGIN=shca
# export NCCL_PLUGIN_P2P=ib


# mooncake
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_TE_FILTERS="${IB_DEVICES}"
export MC_ALLOWED_IBV_DEVICES="${IB_DEVICES}"
# allreduce overlap
export ALLREDUCE_STREAM_WITH_COMPUTE=1

# 融合算子
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=0
export W8A8_SUPPORT_METHODS=3

# graph / tuning
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0


####dbg
export SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q=1
export SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB=8
export SGLANG_NSA_FUSE_TOPK=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
#export SGLANG_NSA_HCU_INT8_INDEX_K_CACHE=1
### dbg
export SGLANG_DEBUG_RADIX_CACHE_TIME=0
export SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT=120
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1

#export SGLANG_W4A8_TPMOE_BACKEND=aiter
export SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=0
export SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1

export SGLANG_USE_DEEPGEMM_MOE=1

export ROCSHMEM_ALLOWED_IBV_DEVICES="${IB_DEVICES}"
export MC_ALLOWED_IBV_DEVICES="${IB_DEVICES}"
#export ROCSHMEM_MAX_NUM_CONTEXTS=60
#export ROCSHMEM_DISABLE_HDP_FLUSH=1
#export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_GDR_DISABLE_XDP=0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export ROCSHMEM_IB_GID_INDEX=0
#export DEEPEP_ENABLE_LL_LAYERED_OPT=1

export MC_GID_INDEX=0
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_NSA_HCU_REUSE_SORTED_TOPK=1

export SGLANG_OPT_USE_TOPK_V2=0

# export ROCBLAS_LAYER=3
# export FLASH_MLA_DSA_DEBUG=1
export LIGHTOP_MQA_LOGITS_BETA_FP8=1

export SGLANG_HCU_COMPACT_INDEX_BROADCAST=1

PORT="30000"
MASTER_IP="${HOST}"
DIST_PORT="6000"
DIST_INIT_ADDR="${MASTER_IP}:${DIST_PORT}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs/$(date +%Y%m%d)"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/prefill-cp8_$(date +%Y%m%d_%H%M%S).log"
echo "prefill log: ${LOG_FILE}"
cd "${SCRIPT_DIR}"
export PYTHONUNBUFFERED=1


export LIGHTOP_MQA_LOGITS_BETA_FP8=1

# export DEEPGEMM_PRINT_OP_SIZE=1
# export DEEPGEMM_PRINT_OP_SIZE_STATS=1
# Optimized MegaMoE C014 W4A8 runtime (gfx938).
export SGLANG_HCU_MEGA_MOE_RUNTIME=megamoe
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=32768
export SGLANG_HCU_W4A8_SHARED_OVERLAP=1
export MEGAMOE_W4A8_LDS_SWIZZLE=1
export MEGAMOE_W4A8_LOCAL_SCATTER=1
export MEGAMOE_W4A8_HOST_SYNC_ROWS=1
export K3_USE_ASM_TAIL_REDUCE=0

export SGLANG_HCU_COMPACT_INDEX_BROADCAST=1

# export SGLANG_USE_LIGHTOP_ROPE=1
# export FLASH_MLA_DSA_DEBUG=1

# SGLANG_SRC="/workspace/glm5.3_int4/sglang-das_glm5.3-w4a8/python"
# python3 - <<PY
# import os, sglang
# want = os.path.realpath("${SGLANG_SRC}")
# got = os.path.realpath(sglang.__file__)
# print("using sglang:", got)
# if not got.startswith(want + os.sep) and got != os.path.join(want, "sglang", "__init__.py"):
#     raise SystemExit(f"ERROR: sglang not from PYTHONPATH:\n  {got}")
# PY


export KV_EXPERIMENT_ID="${KV_EXPERIMENT_ID:-b1-c64-mem080}"
export SGLANG_KV_REUSE_OBSERVER_PATH=/workspace/kv-research/glm5.3/scripts/baseline/logs/${KV_EXPERIMENT_ID}.kv.jsonl
export SGLANG_KV_REUSE_OBSERVER_INTERVAL=1

sglang serve \
    --model-path "${MODEL_PATH}" \
    --trust-remote-code \
    --host "${HOST}" \
    --port "${PORT}" \
    --dist-init-addr "${DIST_INIT_ADDR}" \
    --nnodes 1 \
    --node-rank 0 \
    --tp-size 8 \
    --ep-size 8 \
    --moe-dense-tp-size 1 \
    --moe-a2a-backend megamoe \
    --deepep-mode auto \
    --attn-cp-size 8 \
    --enable-nsa-prefill-context-parallel \
    --nsa-prefill-cp-mode round-robin-split \
    --context-length 262144 \
    --kv-cache-dtype fp8_e4m3 \
    --dtype bfloat16 \
    --mem-fraction-static 0.8 \
    --chunked-prefill-size 32768 \
    --max-prefill-tokens 32768 \
    --page-size 64 \
    --nsa-prefill-backend flashmla_sparse \
    --nsa-decode-backend flashmla_kv \
    --quantization slimquant_w4a8_marlin \
    --enable-cache-report \
    --enable-metrics \
    --enable-request-time-stats-logging \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6 \
    --json-model-override-args '{"index_share_for_mtp_iteration": false}' \
    --disable-cuda-graph \
    --tokenizer-worker-num 8 \
    --enable-dsa-cache-layer-split \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port 8299 \
    --disaggregation-ib-device "${IB_DEVICES}" \
    2>&1 | tee "${LOG_FILE}"
    # --enable-hierarchical-cache \
    # --hicache-ratio 1.2 \
    # --hicache-write-policy write_through \
    # --hicache-mem-layout page_first \
    # --hicache-io-backend kernel \
    
    
    
    
    # --moe-a2a-backend megamoe \
    # --deepep-mode auto \

    #    --json-model-override-args '{"index_share_for_mtp_iteration": false}' \
    #     --init-expert-location ./glm5.2-w4a8-ep8-cp8-prefill.pt \
    # --ep-dispatch-algorithm static \
    # --ep-num-redundant-experts 64 \
    #     --deepep-config ./ep_config.json \


#--enable-request-time-stats-logging \
#    --enable-metrics \
#    --expert-balancedness-report-mode both \

