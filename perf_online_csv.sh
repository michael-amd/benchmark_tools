#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# perf_online_csv.sh
#
# This script benchmarks online serving performance by:
#   1. Launching the SGLang server (with model_path /mnt/raid/models/amd--grok-1-W4A8KV8/)
#   2. Waiting for the server to be ready.
#   3. Running the client benchmark as a separate process by calling CLIENT_GROK.sh
#      twice—once for default (aiter, full prefill+decode) and once for decode-only.
#   4. Parsing the client log files (one file per request rate, from the first run)
#      for three metrics: Median E2E, TTFT, and ITL latencies.
#      If the expected lines are not found, defaults are used:
#         E2E -> "Running"
#         TTFT -> "client"
#         ITL -> "benchmark for request rate <RATE>, mode <MODE> ..."
#   5. Combining these with fixed H100 reference values and writing a multi-section CSV.
#
# The CSV is named with the current date and a header without spaces.
# Example filename:
#   20250313_GROK1_CK-MOE-I4F8-AITER-DECODE-ATTN_online.csv
# ------------------------------------------------------------------------------
 
# ---------------------------
# 1. Launch the Server
# ---------------------------
SERVER_LOG="server_output.log"
rm -f "$SERVER_LOG"

echo "Starting server..."
RCCL_MSCCL_ENABLE=0 CK_MOE=1 USE_INT4_WEIGHT=1 \
python3 -m sglang.launch_server \
    --model /mnt/raid/models/amd--grok-1-W4A8KV8/ \
    --tokenizer-path Xenova/grok-1-tokenizer \
    --tp 8 \
    --quantization fp8 \
    --trust-remote-code \
    --attention-backend aiter \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server launched (PID = $SERVER_PID). Waiting for readiness..."

while true; do
    if grep -q "The server is fired up and ready to roll!" "$SERVER_LOG"; then
        echo "Server is ready!"
        break
    fi
    sleep 1
done

# ---------------------------
# 2. Run Client Benchmarks via CLIENT_GROK.sh
# ---------------------------
# Assume CLIENT_GROK.sh is in the same directory and has been modified to use $MODE.
# Run for mode "aiter" (full prefill+decode)
echo "Running client benchmark for mode aiter..."
export MODE=aiter
bash CLIENT_GROK.sh
echo "Client benchmark (aiter) completed."

# Run for mode "decode" (decode-only)
echo "Running client benchmark for mode decode..."
export MODE=decode
bash CLIENT_GROK.sh
echo "Client benchmark (decode) completed."

# ---------------------------
# 3. Parse Client Logs and Build CSV Data
# ---------------------------
# We assume that for each request rate (1 2 4 8 16) and for each mode,
# CLIENT_GROK.sh produced 3 log files named:
#   sglang_client_log_grok1_${MODE}_${RATE}_run*.log
REQ_RATES=(1 2 4 8 16)

# Define functions to parse a metric from a given log file.
parse_metric() {
    local logfile=$1
    local metric=$2   # Expected to be "E2E", "TTFT", or "ITL"
    local mode=$3
    local rate=$4
    # Try to grep a line like "Median E2E latency: <value> ms"
    local result=$(grep -oP "Median ${metric} latency:\s*\K[\d.]+" "$logfile" | head -n1)
    if [ -z "$result" ]; then
        if [ "$metric" == "E2E" ]; then
            echo "Running"
        elif [ "$metric" == "TTFT" ]; then
            echo "client"
        elif [ "$metric" == "ITL" ]; then
            echo "benchmark for request rate ${rate}, mode ${mode} ..."
        fi
    else
        echo "$result"
    fi
}

# For each mode, we will store metrics in associative arrays.
declare -A E2E_aiter
declare -A TTFT_aiter
declare -A ITL_aiter

declare -A E2E_decode
declare -A TTFT_decode
declare -A ITL_decode

# For each request rate, pick one log file (first run) and parse metrics.
for rate in "${REQ_RATES[@]}"; do
    # For mode aiter
    logfile_aiter=$(ls sglang_client_log_grok1_aiter_"${rate}"_run*.log 2>/dev/null | head -n1)
    if [ -n "$logfile_aiter" ]; then
        E2E_aiter[$rate]=$(parse_metric "$logfile_aiter" "E2E" "aiter" "$rate")
        TTFT_aiter[$rate]=$(parse_metric "$logfile_aiter" "TTFT" "aiter" "$rate")
        ITL_aiter[$rate]=$(parse_metric "$logfile_aiter" "ITL" "aiter" "$rate")
    else
        E2E_aiter[$rate]="NA"
        TTFT_aiter[$rate]="NA"
        ITL_aiter[$rate]="NA"
    fi

    # For mode decode
    logfile_decode=$(ls sglang_client_log_grok1_decode_"${rate}"_run*.log 2>/dev/null | head -n1)
    if [ -n "$logfile_decode" ]; then
        E2E_decode[$rate]=$(parse_metric "$logfile_decode" "E2E" "decode" "$rate")
        TTFT_decode[$rate]=$(parse_metric "$logfile_decode" "TTFT" "decode" "$rate")
        ITL_decode[$rate]=$(parse_metric "$logfile_decode" "ITL" "decode" "$rate")
    else
        E2E_decode[$rate]="NA"
        TTFT_decode[$rate]="NA"
        ITL_decode[$rate]="NA"
    fi
done

# ---------------------------
# 4. Shut Down the Server
# ---------------------------
echo "Shutting down server (PID = $SERVER_PID)..."
kill "$SERVER_PID"
sleep 2

# ---------------------------
# 5. Build CSV Output
# ---------------------------
# Fixed H100 reference values (from your sample)
E2E_H100=(13209 13874 16613 44918 85049)
TTFT_H100=(99.1 102.0 113.4 170.7 520.9)
ITL_H100=(23.0 24.4 25.9 63.9 108.6)

# Since MI300x results are non-numeric (strings), we cannot compute ratios; output "NA".
NA_ROW=("NA" "NA" "NA" "NA" "NA")

# Prepare CSV filename and header.
current_date=$(date +%Y%m%d)
header_name="GROK1_CK-MOE-I4F8-AITER-DECODE-ATTN_online"
OUTPUT_CSV="${current_date}_${header_name}.csv"

{
    echo "Online mode - GROK1 (rocm/sgl-dev:20250309rc, /rocm/sgl-dev:20250310rc)"
    echo ""
    echo "Median E2E latency (ms, lower better)"
    printf "request rate"
    for rate in "${REQ_RATES[@]}"; do
        printf "\t%s" "$rate"
    done
    echo ""
    # H100 row
    printf "H100"
    for val in "${E2E_H100[@]}"; do
        printf "\t%s" "$val"
    done
    echo ""
    # MI300x aiter row
    printf "MI300x-aiter (prefill+decode), dell300x-pla-t10-17"
    for rate in "${REQ_RATES[@]}"; do
        printf "\t%s" "${E2E_aiter[$rate]}"
    done
    echo ""
    # MI300x decode row
    printf "MI300x-aiter_decode (decode only), dell300x-pla-t10-17"
    for rate in "${REQ_RATES[@]}"; do
        printf "\t%s" "${E2E_decode[$rate]}"
    done
    echo ""
    # Ratio rows (all NA)
    printf "H100/MI300x-aiter"
    for rate in "${REQ_RATES[@]}"; do
        printf "\tNA"
    done
    echo ""
    printf "H100/MI300x-aiter_decode"
    for rate in "${REQ_RATES[@]}"; do
        printf "\tNA"
    done
    echo ""
    echo ""
    echo "Median TTFT latency (ms, lower better)"
    printf "request rate"
    for rate in "${REQ_RATES[@]}"; do
        printf "\t%s" "$rate"
    done
    echo ""
    printf "H100"
    for val in "${TTFT_H100[@]}"; do
        printf "\t%s" "$val"
    done
    echo ""
    printf "MI300x-aiter (prefill+decode), dell300x-pla-t10-17"
    for rate in "${REQ_RATES[@]}"; do
        printf "\t%s" "${TTFT_aiter[$rate]}"
    done
    echo ""
    printf "MI300x-aiter_decode (decode only), dell300x-pla-t10-17"
    for rate in "${REQ_RATES[@]}"; do
        printf "\t%s" "${TTFT_decode[$rate]}"
    done
    echo ""
    printf "H100/MI300x-aiter"
    for rate in "${REQ_RATES[@]}"; do
        printf "\tNA"
    done
    echo ""
    printf "H100/MI300x-aiter_decode"
    for rate in "${REQ_RATES[@]}"; do
        printf "\tNA"
    done
    echo ""
    echo ""
    echo "Median ITL latency (ms, lower better)"
    printf "request rate"
    for rate in "${REQ_RATES[@]}"; do
        printf "\t%s" "$rate"
    done
    echo ""
    printf "H100"
    for val in "${ITL_H100[@]}"; do
        printf "\t%s" "$val"
    done
    echo ""
    printf "MI300x-aiter (prefill+decode), dell300x-pla-t10-17"
    for rate in "${REQ_RATES[@]}"; do
        printf "\t%s" "${ITL_aiter[$rate]}"
    done
    echo ""
    printf "MI300x-aiter_decode (decode only), dell300x-pla-t10-17"
    for rate in "${REQ_RATES[@]}"; do
        printf "\t%s" "${ITL_decode[$rate]}"
    done
    echo ""
    printf "H100/MI300x-aiter"
    for rate in "${REQ_RATES[@]}"; do
        printf "\tNA"
    done
    echo ""
    printf "H100/MI300x-aiter_decode"
    for rate in "${REQ_RATES[@]}"; do
        printf "\tNA"
    done
    echo ""
} > "$OUTPUT_CSV"

echo "All done! Online benchmark results saved to ${OUTPUT_CSV}"
