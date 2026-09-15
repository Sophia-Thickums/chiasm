#!/usr/bin/env bash
# Build both probes. Requires: gcc, libvulkan, Vulkan headers.
set -euo pipefail
cd "$(dirname "$0")"

INC="vendor/Vulkan-Headers/include"
if [ ! -d "$INC" ]; then
  echo "Vulkan headers not found at $INC"
  echo "Fetch them with: ./fetch-headers.sh"
  exit 1
fi

mkdir -p build
gcc -O2 -Wall -Wextra -o build/probe     src/dma_buf_bridge_probe.c     -I"$INC" -lvulkan
gcc -O2 -Wall -Wextra -o build/bandwidth src/dma_buf_bridge_bandwidth.c -I"$INC" -lvulkan
echo "Built: build/probe  build/bandwidth"
echo
echo "Run:  ./build/probe"
echo "      ./build/bandwidth 64 30"
