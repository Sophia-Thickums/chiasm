# chiasm

**Cross-device GPU data handoff on Linux via Vulkan external memory. One buffer, two devices, no host copy.**

Named for the optic chiasm — the structure where two separate signal paths cross without merging.
That is exactly what this mechanism does, and exactly what it does not do:

- it lets a buffer in one GPU's VRAM be **addressed by a different GPU**;
- it does **not** merge memory. `16 GiB + 16 GiB` is not `32 GiB`. Each device still has its own.

If you remember one thing from this repository, remember that distinction.

---

## Read this before you build anything on it

**On a discrete + integrated pair, the classic host-staged path is faster than chiasm.**
Measured, same 64 MiB payload, same machine:

| Path | Time for 64 MiB | Effective one-way |
|---|---|---|
| Classic (D2H → host memcpy → H2D) | **5.80 ms** | 10.78 GB/s |
| chiasm (dma_buf handoff) | **~10.7 ms** | ~5.7 GB/s combined |

This repository contains the benchmark that beats its own thesis (`build/classic`), because a
mechanism with an unmeasured alternative is a pitch, not a result.

**Why the classic path wins here:** an integrated GPU shares system RAM. The "host leg" of the
round-trip is therefore nearly free, while a peer read *across PCIe* by the weaker device is the
slow part. Remove the host and you keep the expensive leg.

**When chiasm is the right tool anyway** — and these are real, they are just not throughput:

1. **The host leg is expensive.** On a discrete-to-discrete pair there is no cheap host path;
   everything crosses PCIe regardless. That is the configuration this was designed for, and the
   first thing to re-measure when a second discrete card is available.
2. **The CPU is the actual bottleneck.** No staging allocation, no `memcpy`, no CPU ever touching
   the payload. In a pipeline where the CPU is already saturated doing analysis, "the transfer
   costs no CPU" matters more than "the transfer is 5 ms faster."
3. **Zero copy is a correctness property, not a performance one.** One allocation, one owner, no
   second copy that can drift out of sync with the first.

`build/classic` is included so you can run the comparison on *your* hardware pair instead of
trusting mine. Your result may differ — integrated memory arrangements vary and that is the
dominant term.

---

## What it is not

- **Not unified memory.** Importing a buffer does not add VRAM to anything. This will not make a
  model that exceeds one card fit across two.
- **Not tensor parallelism**, not NCCL/RCCL, not a sharding solution.
- **Not a merge of mixed-architecture devices.** Two GPUs of different ISA generations still cannot
  share a compute kernel. This moves data between them; it does not unify them.
- **Not frame generation, DLSS, or a graphics mod.** No NVIDIA code is used or required.

---

## Requirements

- Linux with Mesa/RADV, or any Vulkan driver exposing the external-memory extensions
- **Two** Vulkan physical devices (an integrated GPU counts)
- `libvulkan`, `gcc`, Vulkan headers (fetched by script)
- No ROCm required

## Build

```bash
./fetch-headers.sh     # Khronos Vulkan-Headers — the only non-assumed dependency
./build.sh
```

## Run

```bash
./build/probe            # correctness: handoff + byte-for-byte verification
./build/bandwidth 64 30  # throughput, both directions
./build/classic          # the host-staged baseline, for comparison
```

---

## Results

### Correctness — `probe` (verified on the reference machine)

16 MiB buffer. Discrete GPU allocates and fills in VRAM; integrated GPU imports the fd and reads it back.

```
== chiasm probe: cross-device handoff (Linux/Vulkan)
== A (export side) : AMD Radeon RX 9070 XT  (RADV GFX1201)   heap 15.9 GiB
== B (import side) : AMD Ryzen 7 9800X3D    (RADV RAPHAEL)   heap 10.5 GiB
== A dma_buf EXPORTABLE: YES (features=0x6)
== B dma_buf IMPORTABLE: YES (features=0x6)
== exported dma_buf fd=3 in 0.003 ms
== B: imported fd bound to memory type 2 in 0.016 ms

   bytes compared : 16777216
   mismatched     : 0
   tail sentinel  : intact
   round-trip     : export 0.003 ms + import 0.016 ms

RESULT: BRIDGE-WORKS
```

**Handshake: 0.019 ms.** One allocation, two devices, byte-identical on the far side.

The probe is built to be hard to fool:

- **Tail sentinel** — the final 4 KiB is poisoned with `0xAB` before transfer, so a handoff that
  silently does nothing reports `LOST` rather than passing.
- **Full byte-exact comparison** of all 16 MiB, not a checksum of a sample.
- **Capability query before allocation** (`vkGetPhysicalDeviceExternalBufferProperties`), so a
  failure names the layer that failed.
- Device names printed, so there is no ambiguity about which device did what.

### Throughput — `bandwidth`

64 MiB buffer, 30 iterations per direction, discrete ↔ integrated:

| Direction | combined | per-direction effective | 1080p RGBA frame |
|---|---|---|---|
| discrete → integrated | 5.69 GB/s | 2.85 GB/s | **2.84 ms** |
| integrated → discrete | 5.79 GB/s | 2.89 GB/s | **2.80 ms** |

**Treat these as a floor, not a ceiling.** Timing is CPU-side around `submit + wait` on *both*
devices, so the measurement includes a full serialization stall — the devices take turns. Overlapped
operation requires external semaphores / timeline fences and will beat these numbers.

Frame-budget read: ~2.8 ms to move a 1080p frame is about **17% of a 16.7 ms (60 fps) budget**.
Usable at frame scale; not free.

---

## How it works

1. Device A allocates a buffer in `DEVICE_LOCAL` memory with `VkExportMemoryAllocateInfo` requesting
   a `dma_buf` handle.
2. `vkGetMemoryFdKHR` returns a plain file descriptor. That fd *is* the buffer.
3. Device B calls `vkGetMemoryFdPropertiesKHR` to learn which of its own memory types can accept the
   fd, then allocates with `VkImportMemoryFdInfoKHR` carrying the same fd.
4. Device B binds its buffer to that memory and reads it normally. No copy occurred.

`dma_buf` is the kernel's own buffer-sharing primitive — the same one behind DRM/KMS, Wayland
compositors, and camera pipelines. Vulkan simply exposes it.

## Where this is useful

- Any pipeline shaped `device A → CPU → device B` where you would rather it were `device A → device B`.
- Handing a completed frame or tensor to an otherwise idle device for analysis, filtering, or
  inference, without a CPU round-trip.
- Cases where the CPU cost of staging is the bottleneck rather than the transfer time.

## Next steps (not implemented here)

1. **Overlapped synchronization** — external semaphores + timeline fences so both devices run
   concurrently. This is what converts the throughput floor above into a real number.
2. **Producer taps** — capturing frames as `dma_buf` handles (KMS/DRM) instead of CPU pixels, which
   is what makes a capture → analyze → return loop possible with no system-RAM leg at all.
3. **Discrete-to-discrete validation** — the configuration where the argument for this mechanism is
   strongest, and therefore the one that most needs re-measuring.
4. **Multi-architecture validation** across GCN/RDNA generations on RADV.

## Provenance & prior art

The *terminal stage* framing — a stage that takes a finished artifact and returns a finished
artifact, and can therefore be moved to another device — and the second-GPU-as-neural-coprocessor
demonstration come from [`maohgad-web/Neural-coprocessor`](https://github.com/maohgad-web/Neural-coprocessor)
(MIT), which publishes measurements on dual RTX 5060 Ti under Windows/D3D12. That developer's
reported result — offloading a neural stage to a second card restoring **86%** of available
upscaling performance versus **39%** on the render card — is their measurement, not independently
reproduced here. Their numbers motivated this port.

The transport mechanism itself is standard Vulkan and predates that project. What this repository
adds is the Linux carrier, byte-exact verification, a measured cost, and an honest baseline.

No NVIDIA code, binaries, or licensed DLLs are used or required.

## License

MIT. See `LICENSE`.

## Related

- **gpu-attrib** — <https://github.com/Sophia-Thickums/gpu-attrib> — which GPU is *actually*
  doing the work, by kernel engine-time counters.
- **cf-probe** — <https://github.com/Sophia-Thickums/cf-probe> — can two cards be made to
  cooperate at the presentation layer at all? One Vulkan call, yes/no, ~2 seconds.
- **chiasm** — <https://github.com/Sophia-Thickums/chiasm> — cross-GPU `dma_buf` handoff with
  no host copy, and the benchmark that beat its own thesis.

**Long-form notes:** *Which GPU is actually rendering?* — the method, the two wrong answers,
and one recorded belief that did not survive measurement:
<https://github.com/Sophia-Thickums/gpu-attrib/blob/main/docs/WHICH_GPU_IS_RENDERING.md>
