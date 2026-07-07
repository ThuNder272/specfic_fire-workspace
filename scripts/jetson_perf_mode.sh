#!/usr/bin/env bash
set -euo pipefail

echo "[jetson] switching to MAXN + jetson_clocks"
sudo nvpmodel -m 0
sudo jetson_clocks

echo "[jetson] current status"
sudo nvpmodel -q
sudo jetson_clocks --show
