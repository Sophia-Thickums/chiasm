/*
 * dma_buf_bridge_bandwidth.c — cost characterisation for the dma_buf bridge.
 *
 * Follow-up to dma_buf_bridge_probe.c. That proves the handoff works.
 * This measures whether it is useful: sustained throughput across the
 * bridge, both directions, at frame scale and larger.
 *
 * Method: one shared buffer, exported once and imported once. Then N
 * iterations of (A writes -> B reads out) and (B writes -> A reads out).
 * Timing is CPU-side around submit+wait on both devices, so it includes a
 * full serialization stall. THIS IS A FLOOR, NOT A CEILING: overlapping the
 * two devices needs external semaphores and will beat these numbers.
 *
 * Frame-scale reference: 1920x1080 RGBA = 8.29 MiB.
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

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

typedef struct {
    VkPhysicalDevice phys; VkDevice dev; VkQueue queue; uint32_t qf;
    VkPhysicalDeviceMemoryProperties memprops; char name[256];
    VkCommandPool pool; VkCommandBuffer cb;
} Ctx;

static void ctx_init(Ctx *c, VkPhysicalDevice pd) {
    memset(c, 0, sizeof(*c)); c->phys = pd;
    VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pd, &p);
    snprintf(c->name, sizeof(c->name), "%s", p.deviceName);
    vkGetPhysicalDeviceMemoryProperties(pd, &c->memprops);
    uint32_t n = 0, pick = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, NULL);
    VkQueueFamilyProperties *q = malloc(n * sizeof(*q));
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, q);
    for (uint32_t i = 0; i < n; i++)
        if ((q[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && (q[i].queueFlags & VK_QUEUE_TRANSFER_BIT)) { pick = i; break; }
    free(q);
    c->qf = pick;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = pick, .queueCount = 1, .pQueuePriorities = &prio };
    const char *exts[] = { VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
        .enabledExtensionCount = 3, .ppEnabledExtensionNames = exts };
    CHECK(vkCreateDevice(pd, &dci, NULL, &c->dev), "vkCreateDevice");
    vkGetDeviceQueue(c->dev, pick, 0, &c->queue);
    VkCommandPoolCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = pick, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT };
    CHECK(vkCreateCommandPool(c->dev, &cpi, NULL, &c->pool), "pool");
    VkCommandBufferAllocateInfo cbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = c->pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
    CHECK(vkAllocateCommandBuffers(c->dev, &cbi, &c->cb), "cmdbuf");
}

static VkBuffer mkbuf(Ctx *c, VkDeviceSize sz, VkBufferUsageFlags u) {
    VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = sz, .usage = u, .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
    VkBuffer b; CHECK(vkCreateBuffer(c->dev, &bi, NULL, &b), "vkCreateBuffer"); return b;
}

static uint32_t memtype(Ctx *c, VkBuffer b, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) {
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(c->dev, b, &mr);
    for (uint32_t i = 0; i < c->memprops.memoryTypeCount; i++) {
        if (!(mr.memoryTypeBits & (1u << i))) continue;
        VkMemoryPropertyFlags f = c->memprops.memoryTypes[i].propertyFlags;
        if ((f & want) == want && (avoid == 0 || !(f & avoid))) return i;
    }
    fprintf(stderr, "FAIL no memtype want=0x%x avoid=0x%x\n", want, avoid); exit(1);
}

static void submit_copy(Ctx *c, VkBuffer src, VkBuffer dst, VkDeviceSize sz) {
    VkCommandBufferBeginInfo bbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    CHECK(vkBeginCommandBuffer(c->cb, &bbi), "begin");
    VkBufferCopy r = { 0, 0, sz };
    vkCmdCopyBuffer(c->cb, src, dst, 1, &r);
    CHECK(vkEndCommandBuffer(c->cb), "end");
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &c->cb };
    CHECK(vkQueueSubmit(c->queue, 1, &si, VK_NULL_HANDLE), "submit");
    CHECK(vkQueueWaitIdle(c->queue), "wait");
}

int main(int argc, char **argv) {
    VkDeviceSize SZ = (argc > 1 ? (VkDeviceSize)atol(argv[1]) : 64) * 1024 * 1024;
    int ITER = argc > 2 ? atoi(argv[2]) : 30;

    printf("== dma-buf-bridge bandwidth probe (Linux/Vulkan)\n");
    printf("== buffer %.0f MiB, %d iterations per direction\n", SZ / 1048576.0, ITER);
    printf("== frame-scale reference: 1080p RGBA = 8.29 MiB\n\n");

    VkApplicationInfo ai = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "dma-buf-bridge", .apiVersion = VK_API_VERSION_1_3 };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai };
    VkInstance inst; CHECK(vkCreateInstance(&ici, NULL, &inst), "instance");

    uint32_t n = 0; CHECK(vkEnumeratePhysicalDevices(inst, &n, NULL), "count");
    VkPhysicalDevice *pds = malloc(n * sizeof(*pds));
    CHECK(vkEnumeratePhysicalDevices(inst, &n, pds), "devices");
    int ia = -1, ib = -1;
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pds[i], &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && ia < 0) ia = (int)i;
    }
    for (uint32_t i = 0; i < n; i++) if ((int)i != ia) { ib = (int)i; break; }
    if (ia < 0 || ib < 0) { printf("NO-BRIDGE: need 2 devices\n"); return 0; }

    Ctx A, B; ctx_init(&A, pds[ia]); ctx_init(&B, pds[ib]);
    printf("A (export): %s\nB (import): %s\n\n", A.name, B.name);

    /* shared buffer on A, exported */
    VkBuffer shA = mkbuf(&A, SZ, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    uint32_t mtA = memtype(&A, shA, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
    VkExportMemoryAllocateInfo ex = { .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
    VkMemoryDedicatedAllocateInfo dd = { .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .buffer = shA };
    ex.pNext = &dd;
    VkMemoryAllocateInfo ma = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &ex, .allocationSize = SZ, .memoryTypeIndex = mtA };
    VkDeviceMemory memA; CHECK(vkAllocateMemory(A.dev, &ma, NULL, &memA), "alloc A");
    CHECK(vkBindBufferMemory(A.dev, shA, memA, 0), "bind A");

    PFN_vkGetMemoryFdKHR getFd = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(A.dev, "vkGetMemoryFdKHR");
    VkMemoryGetFdInfoKHR gfi = { .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
        .memory = memA, .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
    int fd = -1; CHECK(getFd(A.dev, &gfi, &fd), "getFd");

    PFN_vkGetMemoryFdPropertiesKHR getFdp =
        (PFN_vkGetMemoryFdPropertiesKHR)vkGetDeviceProcAddr(B.dev, "vkGetMemoryFdPropertiesKHR");
    VkMemoryFdPropertiesKHR fdp = { .sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR };
    CHECK(getFdp(B.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd, &fdp), "fdp");

    VkBuffer shB = mkbuf(&B, SZ, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    uint32_t mtB = 0; for (uint32_t i = 0; i < B.memprops.memoryTypeCount; i++)
        if (fdp.memoryTypeBits & (1u << i)) { mtB = i; break; }
    VkImportMemoryFdInfoKHR im = { .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, .fd = fd };
    VkMemoryDedicatedAllocateInfo ddB = { .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .buffer = shB };
    im.pNext = &ddB;
    VkMemoryAllocateInfo mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &im, .allocationSize = SZ, .memoryTypeIndex = mtB };
    VkDeviceMemory memB; CHECK(vkAllocateMemory(B.dev, &mb, NULL, &memB), "alloc B");
    CHECK(vkBindBufferMemory(B.dev, shB, memB, 0), "bind B");
    printf("shared buffer: A memtype %u <-> B memtype %u (one allocation, two devices)\n\n", mtA, mtB);

    /* staging on each side */
    VkBuffer stA = mkbuf(&A, SZ, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    VkMemoryAllocateInfo sa = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = SZ,
        .memoryTypeIndex = memtype(&A, stA, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0) };
    VkDeviceMemory stAm; CHECK(vkAllocateMemory(A.dev, &sa, NULL, &stAm), "stA");
    CHECK(vkBindBufferMemory(A.dev, stA, stAm, 0), "bind stA");

    VkBuffer stB = mkbuf(&B, SZ, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    VkMemoryAllocateInfo sb = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = SZ,
        .memoryTypeIndex = memtype(&B, stB, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0) };
    VkDeviceMemory stBm; CHECK(vkAllocateMemory(B.dev, &sb, NULL, &stBm), "stB");
    CHECK(vkBindBufferMemory(B.dev, stB, stBm, 0), "bind stB");

    /* ---- Direction 1: A renders/writes into VRAM, B reads it out ---- */
    submit_copy(&A, stA, shA, SZ);           /* warm */
    double best = 1e18, tot = 0;
    for (int i = 0; i < ITER; i++) {
        double t = now_ms();
        submit_copy(&A, stA, shA, SZ);       /* A writes its VRAM */
        submit_copy(&B, shB, stB, SZ);       /* B reads the same memory out */
        double dt = now_ms() - t;
        tot += dt; if (dt < best) best = dt;
    }
    double avg = tot / ITER, bw = (SZ / 1073741824.0) / (avg / 1000.0);
    printf("DIRECTION A->B  (A writes VRAM, B reads it out)\n");
    printf("   avg %7.3f ms/frame   best %7.3f ms   %.2f GB/s (write+read combined)\n",
           avg, best, bw);
    printf("   per-direction effective      %.2f GB/s  | 1080p frame would take ~%.3f ms\n",
           bw / 2.0, (8.29 / 1024.0) / (bw / 2.0) * 1000.0);

    /* ---- Direction 2: B writes into the same memory, A reads it back ---- */
    submit_copy(&B, stB, shB, SZ);           /* warm */
    best = 1e18; tot = 0;
    for (int i = 0; i < ITER; i++) {
        double t = now_ms();
        submit_copy(&B, stB, shB, SZ);       /* B writes into shared, via dma_buf */
        submit_copy(&A, shA, stA, SZ);       /* A reads it out */
        double dt = now_ms() - t;
        tot += dt; if (dt < best) best = dt;
    }
    avg = tot / ITER; bw = (SZ / 1073741824.0) / (avg / 1000.0);
    printf("\nDIRECTION B->A  (B writes via the bridge, A reads it out)\n");
    printf("   avg %7.3f ms/frame   best %7.3f ms   %.2f GB/s (write+read combined)\n",
           avg, best, bw);
    printf("   per-direction effective      %.2f GB/s  | 1080p frame would take ~%.3f ms\n",
           bw / 2.0, (8.29 / 1024.0) / (bw / 2.0) * 1000.0);

    printf("\nNOTE: timings are CPU-side around submit+wait on both devices, so they\n");
    printf("include a full serialization stall. Overlapped operation needs external\n");
    printf("semaphores and will beat these numbers. Treated as a floor.\n");

    return 0;
}
