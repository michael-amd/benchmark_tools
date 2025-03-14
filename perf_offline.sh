# 1. build docker rocm docker.rocm

# git clone https://github.com/sgl-project/sglang.git
# cd docker 
# docker build --build-arg SGL_BRANCH=v0.4.3.post4 -t v0.4.3.post4-rocm630-$date -f Dockerfile.rocm .

# or pull lasted docker  
# docker pull rocm/sgl-dev:20250310rc

# 2. enter docker 
docker start sgl-dev_20250310rc
docker attach sgl-dev_20250310rc

cd /mnt/raid/michael/HaiShaw/sglang/

# run offline 
# CK_MOE=1 USE_INT4_WEIGHT=1 python -m sglang.bench_one_batch --batch-size 32 --input 1024 --output 128 --model /mnt/raid/models/amd--grok-1-W4A8KV8/ --tokenizer-path Xenova/grok-1-tokenizer --tp 8 --quantization fp8 --trust-remote-code

CK_MOE=1 USE_INT4_WEIGHT=1 MOE_PADDING=0 python3 -m sglang.bench_one_batch --model /mnt/raid/models/amd--grok-1-W4A8KV8/ --tokenizer-path Xenova/grok-1-tokenizer --tp 8 --batch-size 1 --input 1024 --output 128 --attention-backend aiter --sampling-backend  pytorch --quantization fp8 --trust-remote-code --cuda-graph-max-bs 1024

# template
# CK_MOE=1 USE_INT4_WEIGHT=1 MOE_PADDING=0 python3 -m sglang.bench_one_batch --model ${model} --tokenizer-path Xenova/grok-1-tokenizer --tp ${tp} --batch-size $bs --input $ilen --output $olen --attention-backend aiter --sampling-backend  pytorch --quantization fp8 --trust-remote-code --cuda-graph-max-bs 1024

# result: median throughput: 1776.12 token/s

# Prefill. latency: 4.75348 s, throughput:    215.42 token/s
# Decode.  latency: 1.39892 s, throughput:      0.71 token/s
# Decode.  latency: 0.01613 s, throughput:     61.98 token/s
# Decode.  latency: 0.01586 s, throughput:     63.06 token/s
# Decode.  latency: 0.01585 s, throughput:     63.10 token/s
# Decode.  latency: 0.01583 s, throughput:     63.19 token/s
# Decode.  median latency: 0.01585 s, median throughput:     63.10 token/s
# Total. latency:  6.248 s, throughput:    165.18 token/s
# Benchmark ...
# Prefill. latency: 0.32363 s, throughput:   3164.09 token/s
# Decode.  latency: 0.01589 s, throughput:     62.93 token/s
# Decode.  latency: 0.01590 s, throughput:     62.88 token/s
# Decode.  latency: 0.01581 s, throughput:     63.26 token/s
# Decode.  latency: 0.01584 s, throughput:     63.14 token/s
# Decode.  latency: 0.01580 s, throughput:     63.29 token/s
# Decode.  median latency: 0.01577 s, median throughput:     63.41 token/s
# Total. latency:  2.327 s, throughput:    495.07 token/s