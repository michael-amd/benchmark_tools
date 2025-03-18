#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# perf_offline_csv.sh
#
# This script runs the offline benchmark command for TP=8 and multiple batch
# sizes. For each run, it:
#   1. Creates a folder named {date}_GROK1_CK-MOE-I4F8-AITER-DECODE-ATTN_pref_offline.
#   2. Runs the benchmark and captures output.
#   3. Parses the following metrics from the final result block (after "Benchmark ..."):
#        - Prefill latency (s)
#        - Prefill throughput (token/s)
#        - Median decode latency (s)
#        - Median decode throughput (token/s)
#        - Total (E2E) latency (s)
#        - E2E throughput (token/s)
#   4. Appends a CSV row with columns in the following order:
#         TP, batch_size, IL, OL, Prefill_latency(s), Median_decode_latency(s),
#         E2E_Latency(s), Prefill_Throughput(token/s), Median_Decode_Throughput(token/s),
#         E2E_Throughput(token/s)
#   5. If a result.jsonl file is produced by the benchmark, it moves (renames) it
#      to {date}_GROK1_CK-MOE-I4F8-AITER-DECODE-ATTN_pref_offline.jsonl (overwriting
#      any previous file) so that only the current run’s data is saved.
#
# ------------------------------------------------------------------------------
 
# Model and tokenizer paths
MODEL="/mnt/raid/models/amd--grok-1-W4A8KV8/"
TOKENIZER="Xenova/grok-1-tokenizer"

# Input/Output lengths
ILEN=1024
OLEN=128

# Only use TP=8
TP_VALUES=(8)
BATCH_SIZES=(1 2 4 8 16 32 64 128 256)

# Get current date for naming (e.g., 20250315)
current_date=$(date +%Y%m%d)
# Build folder name
folder="${current_date}_GROK1_CK-MOE-I4F8-AITER-DECODE-ATTN_pref_offline"
mkdir -p "$folder"

# CSV file inside the folder
OUTPUT_CSV="${folder}/${current_date}_GROK1_CK-MOE-I4F8-AITER-DECODE-ATTN_pref_offline.csv"

# Write CSV header with new ordering
echo "TP,batch_size,IL,OL,Prefill_latency(s),Median_decode_latency(s),E2E_Latency(s),Prefill_Throughput(token/s),Median_Decode_Throughput(token/s),E2E_Throughput(token/s)" > "$OUTPUT_CSV"

# Loop over TP and batch sizes (TP is fixed to 8)
for tp in "${TP_VALUES[@]}"; do
  for bs in "${BATCH_SIZES[@]}"; do
    echo "Running TP=${tp}, batch_size=${bs} ..."
    
    # Run the benchmark command and capture output.
    out=$(
      CK_MOE=1 USE_INT4_WEIGHT=1 MOE_PADDING=0 \
      python3 -m sglang.bench_one_batch \
        --model "${MODEL}" \
        --tokenizer-path "${TOKENIZER}" \
        --tp "${tp}" \
        --batch-size "${bs}" \
        --input "${ILEN}" \
        --output "${OLEN}" \
        --attention-backend aiter \
        --sampling-backend pytorch \
        --quantization fp8 \
        --trust-remote-code \
        --cuda-graph-max-bs 1024 2>&1
    )

    # Isolate the section after "Benchmark ..." (the final block of results)
    last_section=$(echo "$out" | awk '/Benchmark/ {flag=1; next} flag')
    
    # Parse the metrics from the final block:
    prefill_latency=$(echo "$last_section" | grep -oP 'Prefill\. latency:\s*\K[\d.]+' | tail -n 1)
    prefill_throughput=$(echo "$last_section" | grep -oP 'Prefill\. latency:.*throughput:\s*\K[\d.]+' | tail -n 1)
    
    decode_median_latency=$(echo "$last_section" | grep -oP 'Decode\.\s+median latency:\s*\K[\d.]+' | tail -n 1)
    decode_median_throughput=$(echo "$last_section" | grep -oP 'Decode\.\s+median latency:.*median throughput:\s*\K[\d.]+' | tail -n 1)
    
    total_latency=$(echo "$last_section" | grep -oP 'Total\. latency:\s*\K[\d.]+' | tail -n 1)
    e2e_throughput=$(echo "$last_section" | grep -oP 'Total\. latency:.*throughput:\s*\K[\d.]+' | tail -n 1)
    
    # Append a CSV row in the new order:
    # TP, batch_size, IL, OL, Prefill_latency(s), Median_decode_latency(s),
    # E2E_Latency(s), Prefill_Throughput(token/s), Median_Decode_Throughput(token/s),
    # E2E_Throughput(token/s)
    echo "${tp},${bs},${ILEN},${OLEN},${prefill_latency},${decode_median_latency},${total_latency},${prefill_throughput},${decode_median_throughput},${e2e_throughput}" >> "$OUTPUT_CSV"
    
    # If a result file (result.jsonl) is produced by the benchmark, move it.
    # Always rename it to the same file so that only the current run’s JSON result is kept.
    if [ -f result.jsonl ]; then
      dest_json="${folder}/${current_date}_GROK1_CK-MOE-I4F8-AITER-DECODE-ATTN_pref_offline.jsonl"
      mv result.jsonl "$dest_json"
      echo "Saved JSON result to ${dest_json}"
    fi

  done
done

echo "All done! Results saved to ${OUTPUT_CSV} and the current run's JSON result stored as ${folder}/${current_date}_GROK1_CK-MOE-I4F8-AITER-DECODE-ATTN_pref_offline.jsonl"
