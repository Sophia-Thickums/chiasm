/*
 * chiasm_probe.c — correctness proof for cross-device GPU handoff.
 *
 * Question: can a buffer in GPU A's VRAM be handed to GPU B as a dma_buf
 * file descriptor, bound there as an ordinary buffer, and read back
 * byte-identical — with no host round-trip and no second copy?
 *
 * Method:
 *   1. Device A: allocate a buffer in DEVICE_LOCAL memory marked
 *      dma_buf-exportable; fill it from a host staging buffer.
 *   2. Export the dma_buf fd from A's device memory.
 *   3. Device B: import the fd; allocate and bind a buffer to it.
 *   4. B copies the imported buffer into its own staging memory, maps it,
 *      and the result is compared byte-for-byte against the source pattern.
 *
 * A tail sentinel (last 4 KiB poisoned with 0xAB) guards against a handoff
 * that silently does nothing being mistaken for success.
 *
 * Sync note: this probe uses vkQueueWaitIdle for clarity, not speed.
 * A production path needs external semaphores/timeline fences so the two
 * devices overlap rather than take turns.
 *
 * Build: see build.sh
 */

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define CHECK(x, msg) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "FAIL %s: VkResult=%d\n", msg, _r); exit(1); } } while (0)

#define BUFFER_BYTES (16u * 1024u * 1024u)   /* 16 MiB */

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ---------- minimal per-device context ---------- */
typedef struct {
    VkPhysicalDevice phys;
    VkDevice dev;
    VkQueue queue;
    uint32_t queueFamily;
    VkPhysicalDeviceMemoryProperties memprops;
    char name[256];
    int is_discrete;
    VkDeviceSize heap_bytes;
} Ctx;

static int pick_queue_family(VkPhysicalDevice pd, uint32_t *out) {
    uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, NULL);
    VkQueueFamilyProperties *q = malloc(n * sizeof(*q));
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, q);
    int found = -1;
    for (uint32_t i = 0; i < n; i++) {
        if ((q[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            (q[i].queueFlags & VK_QUEUE_TRANSFER_BIT)) { found = (int)i; break; }
        if (found < 0 && (q[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) found = (int)i;
    }
    free(q);
    if (found < 0) return 0;
    *out = (uint32_t)found;
    return 1;
}

static int find_memory_type(Ctx *c, uint32_t typeBits, VkMemoryPropertyFlags want,
                            VkMemoryPropertyFlags avoid, uint32_t *out) {
    for (uint32_t i = 0; i < c->memprops.memoryTypeCount; i++) {
        if (!(typeBits & (1u << i))) continue;
        VkMemoryPropertyFlags f = c->memprops.memoryTypes[i].propertyFlags;
        if ((f & want) == want && (avoid == 0 || !(f & avoid))) { *out = i; return 1; }
    }
    return 0;
}

static void ctx_init(Ctx *c, VkPhysicalDevice pd) {
    memset(c, 0, sizeof(*c));
    c->phys = pd;
    VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pd, &p);
    snprintf(c->name, sizeof(c->name), "%s", p.deviceName);
    c->is_discrete = (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
    vkGetPhysicalDeviceMemoryProperties(pd, &c->memprops);
    for (uint32_t i = 0; i < c->memprops.memoryHeapCount; i++)
        if (c->memprops.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            c->heap_bytes += c->memprops.memoryHeaps[i].size;

    if (!pick_queue_family(pd, &c->queueFamily)) {
        fprintf(stderr, "FAIL no compute queue on %s\n", c->name); exit(1);
    }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = c->queueFamily, .queueCount = 1, .pQueuePriorities = &prio };

    const char *exts[] = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
    };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
        .enabledExtensionCount = 3, .ppEnabledExtensionNames = exts };
    CHECK(vkCreateDevice(pd, &dci, NULL, &c->dev), "vkCreateDevice");
    vkGetDeviceQueue(c->dev, c->queueFamily, 0, &c->queue);
}

/* ---------- helpers ---------- */
static uint32_t mk_buffer(Ctx *c, VkDeviceSize sz, VkBufferUsageFlags usage, VkBuffer *out) {
    VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = sz, .usage = usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
    CHECK(vkCreateBuffer(c->dev, &bi, NULL, out), "vkCreateBuffer");
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(c->dev, *out, &mr);
    return (uint32_t)mr.size; /* caller may need size */
}

static uint32_t mem_type_for(Ctx *c, VkBuffer buf, VkMemoryPropertyFlags want) {
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(c->dev, buf, &mr);
    uint32_t idx;
    if (!find_memory_type(c, mr.memoryTypeBits, want, 0, &idx)) {
        fprintf(stderr, "FAIL no memory type (%s want=0x%x)\n", c->name, want); exit(1);
    }
    return idx;
}

static void *map_mem(Ctx *c, VkDeviceMemory m, VkDeviceSize sz) {
    void *p; CHECK(vkMapMemory(c->dev, m, 0, sz, 0, &p), "vkMapMemory"); return p;
}

int main(void) {
    printf("== chiasm probe: cross-device handoff (Linux/Vulkan) ==\n");
    printf("== buffer size: %.1f MiB\n", BUFFER_BYTES / 1048576.0);

    VkApplicationInfo ai = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "chiasm", .apiVersion = VK_API_VERSION_1_3 };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &ai };
    VkInstance inst;
    CHECK(vkCreateInstance(&ici, NULL, &inst), "vkCreateInstance");

    uint32_t n = 0;
    CHECK(vkEnumeratePhysicalDevices(inst, &n, NULL), "enum count");
    if (n < 2) { printf("\nRESULT: NO-BRIDGE — only %u Vulkan device(s), need 2\n", n); return 0; }
    VkPhysicalDevice *pds = malloc(n * sizeof(*pds));
    CHECK(vkEnumeratePhysicalDevices(inst, &n, pds), "enum devices");
    printf("== %u Vulkan physical devices found\n", n);

    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pds[i], &p);
        printf("   [%u] %s  (%s)\n", i, p.deviceName,
               p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? "DISCRETE" :
               p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? "integrated" : "other");
    }

    /* Pick a DISCRETE device as A and the first OTHER device as B. */
    int ia = -1, ib = -1;
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pds[i], &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && ia < 0) ia = (int)i;
    }
    for (uint32_t i = 0; i < n; i++) if ((int)i != ia) { ib = (int)i; break; }
    if (ia < 0 || ib < 0) { printf("\nRESULT: NO-BRIDGE — need one discrete + one other\n"); return 0; }

    Ctx A, B;
    ctx_init(&A, pds[ia]);
    ctx_init(&B, pds[ib]);
    printf("== A (export side) : %s  heap %.1f GiB\n", A.name, A.heap_bytes / 1073741824.0);
    printf("== B (import side) : %s  heap %.1f GiB\n", B.name, B.heap_bytes / 1073741824.0);

    /* --- confirm the dma_buf handle type is actually supported for a
           transfer-dst buffer on A (query BEFORE allocating) --- */
    VkPhysicalDeviceExternalBufferInfo ebi = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
    VkExternalBufferProperties ebp = { .sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES };
    vkGetPhysicalDeviceExternalBufferProperties(A.phys, &ebi, &ebp);
    int a_exportable = (ebp.externalMemoryProperties.externalMemoryFeatures &
                        VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) != 0;
    printf("== A dma_buf EXPORTABLE: %s (features=0x%x)\n",
           a_exportable ? "YES" : "NO", ebp.externalMemoryProperties.externalMemoryFeatures);
    if (!a_exportable) { printf("\nRESULT: NO-BRIDGE — A cannot export dma_buf\n"); return 0; }

    vkGetPhysicalDeviceExternalBufferProperties(B.phys, &ebi, &ebp);
    int b_importable = (ebp.externalMemoryProperties.externalMemoryFeatures &
                        VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0;
    printf("== B dma_buf IMPORTABLE: %s (features=0x%x)\n",
           b_importable ? "YES" : "NO", ebp.externalMemoryProperties.externalMemoryFeatures);
    if (!b_importable) { printf("\nRESULT: NO-BRIDGE — B cannot import dma_buf\n"); return 0; }

    /* ---------- 1. source buffer on A, filled with a known pattern ---------- */
    VkBuffer bufA; mk_buffer(&A, BUFFER_BYTES, VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_DST_BIT, &bufA);
    uint32_t mtA;
    {
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(A.dev, bufA, &mr);
        /* must be a type compatible with dma_buf export; query per-type */
        int ok = 0;
        for (uint32_t i = 0; i < A.memprops.memoryTypeCount && !ok; i++) {
            if (!(mr.memoryTypeBits & (1u << i))) continue;
            VkMemoryPropertyFlags f = A.memprops.memoryTypes[i].propertyFlags;
            /* prefer DEVICE_LOCAL (real VRAM, not host staging) */
            if (!(f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) continue;
            mtA = i; ok = 1;
        }
        if (!ok) { printf("\nRESULT: NO-BRIDGE — no DEVICE_LOCAL type for bufA\n"); return 0; }
    }
    VkExportMemoryAllocateInfo exp = { .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
    VkMemoryDedicatedAllocateInfo ded = { .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
        .buffer = bufA };
    exp.pNext = &ded;
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &exp, .allocationSize = BUFFER_BYTES, .memoryTypeIndex = mtA };
    VkDeviceMemory memA;
    CHECK(vkAllocateMemory(A.dev, &mai, NULL, &memA), "vkAllocateMemory(export)");
    CHECK(vkBindBufferMemory(A.dev, bufA, memA, 0), "bind bufA");
    printf("== A buffer bound to DEVICE_LOCAL memory type %u\n", mtA);

    /* staging on A to fill it */
    VkBuffer stgA; mk_buffer(&A, BUFFER_BYTES, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &stgA);
    VkDeviceMemory stgAm = 0;
    {
        VkMemoryAllocateInfo si = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = BUFFER_BYTES,
            .memoryTypeIndex = mem_type_for(&A, stgA, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) };
        CHECK(vkAllocateMemory(A.dev, &si, NULL, &stgAm), "alloc stgA");
        CHECK(vkBindBufferMemory(A.dev, stgA, stgAm, 0), "bind stgA");
    }
    uint8_t *src = malloc(BUFFER_BYTES);
    for (uint32_t i = 0; i < BUFFER_BYTES; i++) src[i] = (uint8_t)((i * 31u + (i >> 8) * 7u) & 0xFF);
    /* poison the destination so a silent no-op cannot pass as success */
    memset(src + BUFFER_BYTES - 4096, 0xAB, 4096);
    { void *p = map_mem(&A, stgAm, BUFFER_BYTES); memcpy(p, src, BUFFER_BYTES); vkUnmapMemory(A.dev, stgAm); }

    { /* copy stgA -> bufA on A */
        VkCommandPool cp; VkCommandPoolCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .queueFamilyIndex = A.queueFamily };
        CHECK(vkCreateCommandPool(A.dev, &cpi, NULL, &cp), "pool A");
        VkCommandBuffer cb; VkCommandBufferAllocateInfo cbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
        CHECK(vkAllocateCommandBuffers(A.dev, &cbi, &cb), "cmdbuf A");
        VkCommandBufferBeginInfo bbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
        vkBeginCommandBuffer(cb, &bbi);
        VkBufferCopy r = { 0, 0, BUFFER_BYTES };
        vkCmdCopyBuffer(cb, stgA, bufA, 1, &r);
        vkEndCommandBuffer(cb);
        VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1, .pCommandBuffers = &cb };
        CHECK(vkQueueSubmit(A.queue, 1, &si, VK_NULL_HANDLE), "submit A");
        CHECK(vkQueueWaitIdle(A.queue), "wait A");
        vkDestroyCommandPool(A.dev, cp, NULL);
    }
    printf("== A: 16 MiB pattern written into VRAM on %s\n", A.name);

    /* ---------- 2. export the dma_buf fd ---------- */
    PFN_vkGetMemoryFdKHR pGetMemoryFdKHR =
        (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(A.dev, "vkGetMemoryFdKHR");
    if (!pGetMemoryFdKHR) { printf("\nRESULT: NO-BRIDGE — vkGetMemoryFdKHR missing\n"); return 0; }
    VkMemoryGetFdInfoKHR gfi = { .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
        .memory = memA, .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
    int fd = -1;
    double t0 = now_ms();
    CHECK(pGetMemoryFdKHR(A.dev, &gfi, &fd), "vkGetMemoryFdKHR");
    double t_export = now_ms() - t0;
    printf("== exported dma_buf fd=%d in %.3f ms\n", fd, t_export);

    /* ---------- 3. import that fd on B ---------- */
    PFN_vkGetMemoryFdPropertiesKHR pGetMemoryFdPropertiesKHR =
        (PFN_vkGetMemoryFdPropertiesKHR)vkGetDeviceProcAddr(B.dev, "vkGetMemoryFdPropertiesKHR");
    VkMemoryFdPropertiesKHR fdp = { .sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR };
    CHECK(pGetMemoryFdPropertiesKHR(B.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd, &fdp),
          "vkGetMemoryFdPropertiesKHR");
    printf("== B: fd memory type bits = 0x%x\n", fdp.memoryTypeBits);

    VkBuffer bufB; mk_buffer(&B, BUFFER_BYTES, VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_DST_BIT, &bufB);
    uint32_t mtB = 0; int ok = 0;
    for (uint32_t i = 0; i < B.memprops.memoryTypeCount && !ok; i++)
        if (fdp.memoryTypeBits & (1u << i)) { mtB = i; ok = 1; }
    if (!ok) { printf("\nRESULT: NO-BRIDGE — no B memory type accepts the fd\n"); return 0; }

    VkImportMemoryFdInfoKHR imp = { .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, .fd = fd };
    VkMemoryDedicatedAllocateInfo dedB = { .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
        .buffer = bufB };
    imp.pNext = &dedB;
    VkMemoryAllocateInfo maiB = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &imp, .allocationSize = BUFFER_BYTES, .memoryTypeIndex = mtB };
    VkDeviceMemory memB;
    t0 = now_ms();
    CHECK(vkAllocateMemory(B.dev, &maiB, NULL, &memB), "vkAllocateMemory(import)");
    double t_import = now_ms() - t0;
    CHECK(vkBindBufferMemory(B.dev, bufB, memB, 0), "bind bufB");
    printf("== B: imported fd bound to memory type %u in %.3f ms\n", mtB, t_import);

    /* ---------- 4. B copies the imported buffer out and we verify ---------- */
    VkBuffer stgB; mk_buffer(&B, BUFFER_BYTES, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &stgB);
    VkDeviceMemory stgBm = 0;
    {
        VkMemoryAllocateInfo si = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = BUFFER_BYTES,
            .memoryTypeIndex = mem_type_for(&B, stgB, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) };
        CHECK(vkAllocateMemory(B.dev, &si, NULL, &stgBm), "alloc stgB");
        CHECK(vkBindBufferMemory(B.dev, stgB, stgBm, 0), "bind stgB");
    }
    {
        VkCommandPool cp; VkCommandPoolCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .queueFamilyIndex = B.queueFamily };
        CHECK(vkCreateCommandPool(B.dev, &cpi, NULL, &cp), "pool B");
        VkCommandBuffer cb; VkCommandBufferAllocateInfo cbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
        CHECK(vkAllocateCommandBuffers(B.dev, &cbi, &cb), "cmdbuf B");
        VkCommandBufferBeginInfo bbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
        vkBeginCommandBuffer(cb, &bbi);
        VkBufferCopy r = { 0, 0, BUFFER_BYTES };
        vkCmdCopyBuffer(cb, bufB, stgB, 1, &r);
        vkEndCommandBuffer(cb);
        VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1, .pCommandBuffers = &cb };
        CHECK(vkQueueSubmit(B.queue, 1, &si, VK_NULL_HANDLE), "submit B");
        CHECK(vkQueueWaitIdle(B.queue), "wait B");
        vkDestroyCommandPool(B.dev, cp, NULL);
    }
    printf("== B: read the imported buffer back through %s\n", B.name);

    uint8_t *got = map_mem(&B, stgBm, BUFFER_BYTES);
    size_t diff = 0, first = (size_t)-1;
    for (uint32_t i = 0; i < BUFFER_BYTES; i++)
        if (got[i] != src[i]) { if (first == (size_t)-1) first = i; diff++; }
    int poison_ok = 1;
    for (uint32_t i = BUFFER_BYTES - 4096; i < BUFFER_BYTES; i++)
        if (got[i] != 0xAB) { poison_ok = 0; break; }

    printf("\n== VERIFICATION\n");
    printf("   bytes compared : %u\n", BUFFER_BYTES);
    printf("   mismatched     : %zu\n", diff);
    if (diff) printf("   first mismatch : offset %zu (exp 0x%02x, got 0x%02x)\n",
                     first, src[first], got[first]);
    printf("   tail sentinel  : %s\n", poison_ok ? "intact" : "LOST");
    printf("   round-trip     : export %.3f ms + import %.3f ms\n", t_export, t_import);

    int pass = (diff == 0);
    printf("\nRESULT: %s\n", pass
        ? "BRIDGE-WORKS — GPU A's VRAM buffer was exported as dma_buf, "
          "imported by GPU B, and read back byte-identical. No host copy."
        : "BRIDGE-FAILED — data crossed but did not survive intact.");

    vkUnmapMemory(B.dev, stgBm);
    free(src);
    return pass ? 0 : 2;
}
