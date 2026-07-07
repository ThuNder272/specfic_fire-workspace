#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/nvidia/miniforge3/envs/robomaster/bin/python"
APP="/home/nvidia/Desktop/specific_fire/aim_scheduler.py"

CMD=(
  "$PYTHON_BIN"
  "$APP"
  --gun-offset-y 42
  --no-show-tx
  --target-color red
  --port /dev/ttyTHS1
  --baud 115200
  --bullet-speed 50
  --system-latency-ms 255
  --exclude-class-ids 11
  --rate-fast-alpha 0.50
  --rate-slow-alpha 0.08
  --ec-t0-ms 35
  --max-yaw-rate 400
  --max-pitch-rate 200
)

echo "[service-bootstrap] starting aim_scheduler"
exec "${CMD[@]}"
