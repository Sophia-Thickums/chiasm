#!/usr/bin/env bash
# Fetch the Khronos Vulkan-Headers (the only build dependency not assumed present).
set -euo pipefail
cd "$(dirname "$0")"
V="vulkan-sdk-1.4.321.0"
mkdir -p vendor && cd vendor
if [ ! -d "Vulkan-Headers-$V" ]; then
  curl -sSL -o vh.tar.gz "https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/$V.tar.gz"
  tar xzf vh.tar.gz && rm -f vh.tar.gz
fi
mkdir -p Vulkan-Headers
ln -sfn "Vulkan-Headers-$V/include" Vulkan-Headers/include
echo "Headers ready: vendor/Vulkan-Headers/include"
