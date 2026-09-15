#!/usr/bin/env bash
# Build all three tools. Requires: gcc, libvulkan, Vulkan headers.
set -euo pipefail
cd "$(dirname "$0")"

INC="vendor/Vulkan-Headers/include"
if [ ! -d "$INC" ]; then
  echo "Vulkan headers not found at $INC"
  echo "Fetch them with: ./fetch-headers.sh"
  exit 1
fi

mkdir -p build
gcc -O2 -Wall -Wextra -o build/probe     src/chiasm_probe.c     -I"$INC" -lvulkan
gcc -O2 -Wall -Wextra -o build/bandwidth src/chiasm_bandwidth.c -I"$INC" -lvulkan
gcc -O2 -Wall -Wextra -o build/classic   src/classic_hop.c     -I"$INC" -lvulkan
echo "Built: build/probe  build/bandwidth  build/classic"
echo
echo "Run:  ./build/probe            # correctness: handoff + byte-exact verification"
echo "      ./build/bandwidth 64 30  # chiasm throughput, both directions"
echo "      ./build/classic          # the host-staged baseline for comparison"
