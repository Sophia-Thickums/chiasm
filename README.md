# dma-buf-bridge

**Cross-device GPU data handoff on Linux, via Vulkan external memory. No host copy.**

Two GPUs. One buffer. Zero bytes through system RAM.

This is a small, self-contained proof-of-concept plus a measured bandwidth characterisation,
for the mechanism that lets a buffer living in one GPU's VRAM be handed to a *different* GPU
and read there as if it were local.

```
GPU A VRAM ──[ dma_buf fd ]──> GPU B
             no memcpy
             no host staging
             microseconds to hand over
```

---

## The idea

A **terminal stage** takes a finished artifact and returns a finished artifact. Nothing about
such a stage requires it to run on the device that produced the input — so it can be moved.

That reframing already exists in other ecosystems: NVIDIA's ReShade-based *Neural Coprocessor*
project ("MGPU Bridge") runs a game's neural post-processing on a second GPU by passing a
completed frame across a D3D12 cross-adapter shared heap. That project is Windows/D3D12 and
RTX-only.

**The same idea has a native Linux carrier, and it is standard Vulkan:**

| | Windows / D3D12 | Linux / Vulkan |
|---|---|---|
| Sharing primitive | cross-adapter shared heap | `dma_buf` file descriptor |
| Extensions | D3D12 shared resources | `VK_KHR_external_memory_fd`, `VK_EXT_external_memory_dma_buf` |
| Copy through host? | no | no |

This repo demonstrates the Linux carrier, verifies the data survives, and measures what it costs.

## What it is not

- **Not unified memory.** Importing a buffer does not add VRAM to anything. `16 GiB + 16 GiB` is
  not `32 GiB`. This will not make a model larger than one card fit across two.
- **Not tensor parallelism**, not NCCL/RCCL collective communication, not a sharding solution.
- **Not a merge of mixed-architecture devices.** Two GPUs of different ISA generations still
  cannot share a compute kernel. This moves data between them; it does not unify them.
- **Not frame generation, DLSS, or a graphics mod.** No NVIDIA code is used or required.

What it *does* is remove the host round-trip from any pipeline that already hands data between
devices — the common case being `device A -> system RAM -> device B`.

---

## Requirements

- Linux with Mesa/RADV (or any Vulkan driver exposing the external-memory extensions)
- **Two** Vulkan physical devices (an integrated GPU counts)
- `libvulkan`, `gcc`, Vulkan headers
- No ROCm required

## Build

On a fresh clone, fetch the Vulkan headers first (the only dependency not assumed present):

```bash
./fetch-headers.sh
./build.sh
# or manually:
inc=vendor/Vulkan-Headers/include
gcc -O2 -Wall -o build/probe     src/dma_buf_bridge_probe.c     -I"$inc" -lvulkan
gcc -O2 -Wall -o build/bandwidth src/dma_buf_bridge_bandwidth.c -I"$inc" -lvulkan
```

## Run

```bash
./build/probe              # correctness: handoff + byte-for-byte verification
./build/bandwidth 64 30    # throughput: 64 MiB buffer, 30 iterations per direction
```

---

## Results

### Correctness — `probe`

16 MiB buffer, discrete GPU allocates and fills, second device imports and reads back.

```
A dma_buf EXPORTABLE: YES (features=0x6)
B dma_buf IMPORTABLE: YES (features=0x6)

A: 16 MiB pattern written into VRAM
exported dma_buf fd=3 in 0.004 ms
B: fd memory type bits = 0x104 -> bound to memory type 2 in 0.015 ms

bytes compared : 16777216
mismatched     : 0
tail sentinel  : intact

RESULT: BRIDGE-WORKS
```

**Handshake cost: 0.019 ms.** One allocation, two devices, byte-identical on the far side.

The probe is built to be hard to fool:

- **Tail sentinel** — the final 4 KiB is poisoned with `0xAB` before transfer. A handoff that
  silently does nothing reports `LOST` rather than passing.
- **Full byte-exact comparison** of all 16 MiB, not a checksum of a sample.
- **Capability query before allocation** (`vkGetPhysicalDeviceExternalBufferProperties`), so a
  failure names the layer that failed.
- Per-device names are printed, so there is no ambiguity about which device did what.

### Throughput — `bandwidth`

64 MiB buffer, 30 iterations per direction, discrete GPU ↔ integrated GPU:

| Direction | combined | per-direction effective | 1080p RGBA frame |
|---|---|---|---|
| discrete → integrated | 5.65 GB/s | 2.83 GB/s | **2.86 ms** |
| integrated → discrete | 6.07 GB/s | 3.04 GB/s | **2.67 ms** |

**Treat these as a floor.** Timing is CPU-side around `submit + wait` on *both* devices, so the
measurement includes a full serialization stall — the devices take turns rather than overlapping.
Overlapped operation requires external semaphores / timeline fences and will beat these numbers.

Frame-budget read: ~2.7 ms to move a 1080p frame is roughly **17% of a 16.7 ms (60 fps) budget**.
Usable at frame scale or reduced resolution; not free.

---

## How it works

1. Device A allocates a buffer in `DEVICE_LOCAL` memory with `VkExportMemoryAllocateInfo`
   requesting a `dma_buf` handle.
2. `vkGetMemoryFdKHR` returns a plain file descriptor. That fd *is* the buffer.
3. Device B calls `vkGetMemoryFdPropertiesKHR` to learn which of its own memory types can accept
   it, then allocates with `VkImportMemoryFdInfoKHR` carrying the same fd.
4. Device B binds its buffer to that memory and reads it normally. No copy occurred.

The `dma_buf` is the kernel's own buffer-sharing primitive — the same one used by DRM/KMS,
Wayland compositors, and camera pipelines. Vulkan simply exposes it.

## Where this is useful

- Moving a frame or tensor between a discrete GPU and an idle integrated GPU so the iGPU can do
  work (analysis, filtering, inference) without a CPU round-trip.
- Any pipeline currently shaped `device A -> CPU -> device B`.
- Overlapping compute across two devices where the handoff, not the compute, was the bottleneck.

## Next steps (not implemented here)

1. **Overlapped synchronization** — external semaphores + timeline fences, so both devices run
   concurrently. This is what converts the throughput floor above into a real number.
2. **Direct producer taps** — capturing frames as `dma_buf` handles (KMS/DRM) rather than CPU pixels,
   which is what makes a capture→analyze→return loop possible without touching system RAM.
3. **Multi-arch validation** — testing across GCN/RDNA generations on RADV.

## Provenance & prior art

The terminal-stage framing and the second-GPU-as-neural-coprocessor demonstration come from
[`maohgad-web/Neural-coprocessor`](https://github.com/maohgad-web/Neural-coprocessor) (MIT),
which publishes measurements on dual RTX 5060 Ti under Windows/D3D12. That developer's reported
result — offloading a neural stage to a second card restoring **86%** of available upscaling
performance vs **39%** when run on the render card — is their measurement, not independently
reproduced here. Their numbers motivated this port.

No NVIDIA code, binaries, or licensed DLLs are used or required. The Linux mechanism described
here is standard Vulkan and predates the project that inspired the framing.

## License

MIT. See `LICENSE`.
