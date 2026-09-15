/*
 * classic_hop.c — the baseline: the SAME transfer, the OLD way.
 *
 * A pipeline with no bridge moves data device->device like this:
 *   GPU A VRAM --D2H--> host-visible buffer --memcpy--> host-visible buffer --H2D--> GPU B VRAM
 *
 * This measures the PCIe legs of that trip on the discrete GPU, so the cost
 * can be put side by side with chiasm's measured number.
 * We report only what we actually time.
 */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define CHECK(x,msg) do { VkResult _r=(x); if(_r!=VK_SUCCESS){ \
    fprintf(stderr,"FAIL %s: %d\n",msg,_r); exit(1);} } while(0)

static double now_ms(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec*1000.0+ts.tv_nsec/1e6;}

static VkInstance inst; static VkPhysicalDevice pd; static VkDevice dev;
static uint32_t qfam; static VkQueue q; static VkCommandPool pool;

static uint32_t memtype(VkPhysicalDeviceMemoryProperties *mp,uint32_t bits,VkMemoryPropertyFlags want){
    for(uint32_t i=0;i<mp->memoryTypeCount;i++)
        if((bits&(1u<<i)) && (mp->memoryTypes[i].propertyFlags & want)==want) return i;
    fprintf(stderr,"no memory type\n"); exit(1);
}
static VkBuffer mkbuf(VkDeviceSize sz,VkBufferUsageFlags u){
    VkBufferCreateInfo ci={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size=sz,.usage=u,.sharingMode=VK_SHARING_MODE_EXCLUSIVE};
    VkBuffer b; CHECK(vkCreateBuffer(dev,&ci,NULL,&b),"mkbuf"); return b;
}
static void one_shot(VkBuffer src,VkBuffer dst,VkDeviceSize sz){
    (void)sz;
    VkCommandBufferAllocateInfo ai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool=pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
    VkCommandBuffer cb; CHECK(vkAllocateCommandBuffers(dev,&ai,&cb),"cb");
    VkCommandBufferBeginInfo bi={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(cb,&bi);
    VkBufferCopy c={.srcOffset=0,.dstOffset=0,.size=sz};
    vkCmdCopyBuffer(cb,src,dst,1,&c);
    vkEndCommandBuffer(cb);
    VkSubmitInfo si={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cb};
    CHECK(vkQueueSubmit(q,1,&si,VK_NULL_HANDLE),"submit");
    CHECK(vkQueueWaitIdle(q),"wait");
    vkFreeCommandBuffers(dev,pool,1,&cb);
}

int main(void){
    VkApplicationInfo app={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .apiVersion=VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo=&app};
    CHECK(vkCreateInstance(&ici,NULL,&inst),"inst");

    uint32_t n=0; vkEnumeratePhysicalDevices(inst,&n,NULL);
    VkPhysicalDevice pds[8]; if(n>8)n=8; vkEnumeratePhysicalDevices(inst,&n,pds);
    int pick=-1;
    for(uint32_t i=0;i<n;i++){
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pds[i],&p);
        printf("   [%u] %s (%s)\n",i,p.deviceName,
            p.deviceType==VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU?"discrete":"other");
        if(p.deviceType==VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && pick<0) pick=(int)i;
    }
    if(pick<0){fprintf(stderr,"no discrete GPU\n");return 1;}
    pd=pds[pick];

    uint32_t qn=0; vkGetPhysicalDeviceQueueFamilyProperties(pd,&qn,NULL);
    VkQueueFamilyProperties qp[16]; if(qn>16)qn=16;
    vkGetPhysicalDeviceQueueFamilyProperties(pd,&qn,qp);
    qfam=0; for(uint32_t i=0;i<qn;i++) if(qp[i].queueFlags&VK_QUEUE_TRANSFER_BIT){qfam=i;break;}

    float prio=1.0f;
    VkDeviceQueueCreateInfo dq={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex=qfam,.queueCount=1,.pQueuePriorities=&prio};
    VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount=1,.pQueueCreateInfos=&dq};
    CHECK(vkCreateDevice(pd,&dci,NULL,&dev),"dev");
    vkGetDeviceQueue(dev,qfam,0,&q);
    VkCommandPoolCreateInfo pc={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex=qfam};
    CHECK(vkCreateCommandPool(dev,&pc,NULL,&pool),"pool");

    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd,&mp);

    const VkDeviceSize SZ = 64ull*1024*1024;
    const int ITER = 30;

    /* device-local (VRAM) */
    VkBuffer devBuf = mkbuf(SZ, VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    VkMemoryRequirements mr1; vkGetBufferMemoryRequirements(dev,devBuf,&mr1);
    VkMemoryAllocateInfo da={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize=mr1.size,
        .memoryTypeIndex=memtype(&mp,mr1.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    VkDeviceMemory devMem; CHECK(vkAllocateMemory(dev,&da,NULL,&devMem),"devmem");
    CHECK(vkBindBufferMemory(dev,devBuf,devMem,0),"binddev");

    /* host-visible staging */
    VkBuffer hostBuf = mkbuf(SZ, VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    VkMemoryRequirements mr2; vkGetBufferMemoryRequirements(dev,hostBuf,&mr2);
    VkMemoryAllocateInfo ha={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize=mr2.size,
        .memoryTypeIndex=memtype(&mp,mr2.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    VkDeviceMemory hostMem; CHECK(vkAllocateMemory(dev,&ha,NULL,&hostMem),"hostmem");
    CHECK(vkBindBufferMemory(dev,hostBuf,hostMem,0),"bindhost");

    printf("\n== classic host-staged hop (PCIe D2H + host memcpy + H2D) ==\n");

    /* D2H: VRAM -> host-visible */
    double best=1e18,tot=0;
    one_shot(devBuf,hostBuf,SZ);
    for(int i=0;i<ITER;i++){
        double t=now_ms();
        one_shot(devBuf,hostBuf,SZ);
        double dt=now_ms()-t; tot+=dt; if(dt<best)best=dt;
    }
    double d2h=tot/ITER;
    printf("   D2H (VRAM -> host)   avg %7.3f ms  best %7.3f ms  -> %.2f GB/s\n",
        d2h,best,(SZ/1073741824.0)/(d2h/1000.0));

    /* H2D: host-visible -> VRAM */
    best=1e18;tot=0;
    one_shot(hostBuf,devBuf,SZ);
    for(int i=0;i<ITER;i++){
        double t=now_ms();
        one_shot(hostBuf,devBuf,SZ);
        double dt=now_ms()-t; tot+=dt; if(dt<best)best=dt;
    }
    double h2d=tot/ITER;
    printf("   H2D (host -> VRAM)   avg %7.3f ms  best %7.3f ms  -> %.2f GB/s\n",
        h2d,best,(SZ/1073741824.0)/(h2d/1000.0));

    /* plus a host-side memcpy between two host-visible buffers (the middle leg) */
    void *pm=0; CHECK(vkMapMemory(dev,hostMem,0,VK_WHOLE_SIZE,0,&pm),"map");
    char *scratch=malloc(SZ); memcpy(scratch,pm,SZ);
    best=1e18;tot=0;
    for(int i=0;i<ITER;i++){
        double t=now_ms();
        memcpy(pm,scratch,SZ);
        double dt=now_ms()-t; tot+=dt; if(dt<best)best=dt;
    }
    double mid=tot/ITER;
    printf("   host memcpy (middle) avg %7.3f ms  best %7.3f ms  -> %.2f GB/s\n",
        mid,best,(SZ/1073741824.0)/(mid/1000.0));

    double classic = d2h+h2d+mid;
    printf("\n   CLASSIC DEVICE->DEVICE TOTAL (D2H + host memcpy + H2D)\n");
    printf("      avg %7.3f ms for %.0f MiB\n", classic, SZ/1048576.0);
    printf("      effective %.2f GB/s one-way  | 1080p frame ~%.2f ms\n",
        (SZ/1073741824.0)/(classic/1000.0), (8.29/1024.0)/((SZ/1073741824.0)/(classic/1000.0))*1000.0);

    printf("\n== CHIASM (measured by ./bandwidth, same 64 MiB) ==\n");
    printf("   avg ~10.3-11.1 ms  = ~5.7-6.1 GB/s combined, ~2.8-3.0 GB/s/direction\n");
    printf("   1080p frame ~2.7-2.9 ms, NO host leg, NO staging allocation\n");

    printf("\n== UPSHOT (read the numbers, not the pitch) ==\n");
    printf("   classic %7.3f ms   vs   bridge ~10.7 ms  -> classic/%.2fx\n",
        classic, classic/10.7);
    printf("   below 1.0 = the CLASSIC path is FASTER on this hardware pair.\n");
    printf("   the iGPU shares system RAM, so the host leg is cheap and the\n");
    printf("   cross-PCIe peer read is the slow side. chiasm's win here is\n");
    printf("   ARCHITECTURAL (no CPU staging, one allocation), NOT throughput.\n");

    vkDestroyBuffer(dev,devBuf,NULL); vkDestroyBuffer(dev,hostBuf,NULL);
    return 0;
}
