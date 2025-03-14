#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# perf_offline_csv.sh
#
# This script runs sglang.bench_one_batch for TP=8 and multiple batch sizes.
# It extracts the final metrics (from the section after "Benchmark ...") for:
#   - Prefill latency (s)
#   - Decode median latency (s)
#   - Total latency (s)
#   - Throughput (token/s)
#
# The results are saved into a CSV file with no spaces in the header.
# ------------------------------------------------------------------------------
 
# Model and tokenizer paths
MODEL="/mnt/raid/models/amd--grok-1-W4A8KV8/"
TOKENIZER="Xenova/grok-1-tokenizer"

# Input/Output lengths
ILEN=1024
OLEN=128

# Use only TP=8 and define batch sizes
TP_VALUES=(8)
BATCH_SIZES=(1 2 4 8 16 32 64 128 256)

# Get the current date (e.g. 20250313)
current_date=$(date +%Y%m%d)
# Set header name with no spaces
header_name="GROK1_CK-MOE-I4F8-AITER-DECODE-ATTN_pref_offline"
OUTPUT_CSV="${current_date}_${header_name}.csv"

# Write header lines to CSV (first line is title, second line is column headers)
echo "${header_name}" > "$OUTPUT_CSV"
echo "TP,batch_size,IL,OL,Prefill_latency(s),median_decode_latency(s),E2E_Latency(s),E2E_Throughput(token/s)" >> "$OUTPUT_CSV"

# Loop over each batch size (TP is fixed to 8)
for tp in "${TP_VALUES[@]}"; do
  for bs in "${BATCH_SIZES[@]}"; do
    echo "Running TP=${tp}, batch_size=${bs} ..."

    # Run the benchmark command and capture output
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

    # Capture the section after the "Benchmark ..." line
    last_section=$(echo "$out" | awk '/Benchmark/ {flag=1; next} flag')

    # Extract the four metrics:
    prefill_latency=$(echo "$last_section" | grep -oP 'Prefill\. latency:\s*\K[\d.]+' | tail -n 1)
    decode_median_latency=$(echo "$last_section" | grep -oP 'Decode\.\s+median latency:\s*\K[\d.]+' | tail -n 1)
    total_latency=$(echo "$last_section" | grep -oP 'Total\. latency:\s*\K[\d.]+' | tail -n 1)
    throughput=$(echo "$last_section" | grep -oP 'throughput:\s*\K[\d.]+' | tail -n 1)

    # Append the metrics to the CSV row (fields remain empty if not found)
    echo "${tp},${bs},${ILEN},${OLEN},${prefill_latency},${decode_median_latency},${total_latency},${throughput}" >> "$OUTPUT_CSV"
  done
done

echo "All done! Results saved to ${OUTPUT_CSV}"
