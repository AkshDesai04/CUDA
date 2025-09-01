```cuda
/*
Aim of the program: Compare your measured bandwidth to the theoretical peak bandwidth of your GPU's PCIe bus.

Thinking:
1. We want to measure the effective PCIe bandwidth between host and device memory.
2. The approach is to allocate a large pinned host buffer and a device buffer, transfer data back and forth,
   time the operations with CUDA events, and calculate the bandwidth in GB/s.
3. To determine the theoretical peak, we query the GPU's PCIe link width and speed using
   cudaDeviceGetAttribute with CUDA_DEVICE_ATTRIBUTE_PCI_BUS_ID, CUDA_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
   CUDA_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, CUDA_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, CUDA_DEVICE_ATTRIBUTE_PCI_BUS_ID,
   CUDA_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, CUDA_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, CUDA_DEVICE_ATTRIBUTE_PCI_BUS_ID,
   CUDA_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, and especially
   CUDA_DEVICE_ATTRIBUTE_PCI_EXPRESS_LINK_WIDTH and CUDA_DEVICE_ATTRIBUTE_PCI_EXPRESS_LINK_SPEED.
4. The effective bandwidth per lane is known to be roughly link_speed_GTs / 8.0 GB/s per lane
   (because 8 GT/s ≈ 1 GB/s per lane for PCIe 3.0). For PCIe 4.0 and 5.0, this scales accordingly.
   Thus theoretical_GBps = link_width * (link_speed_GTs / 8.0).
5. The code will allocate a buffer of 512 MB to get stable timing, perform transfers, compute
   measured bandwidth for both directions, compute the theoretical peak, and print all values for comparison.
6. Error checking is included after each CUDA call to ensure correctness.
7. The program is self-contained and can be compiled with `nvcc` to produce an executable.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>
#include <string>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Size of the buffer to transfer (512 MB)
const size_t BUFFER_SIZE = 512ULL * 1024ULL * 1024ULL;

int main() {
    // Query device properties
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int pcieWidth = 0;
    int pcieSpeed = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&pcieWidth, cudaDevAttrPciExpressLinkWidth, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&pcieSpeed, cudaDevAttrPciExpressLinkSpeed, device));

    // Compute theoretical bandwidth (GB/s)
    // Theoretical per lane bandwidth = link_speed_GTs / 8.0 GB/s
    double theoreticalGBps = static_cast<double>(pcieSpeed) / 8.0 * static_cast<double>(pcieWidth);

    printf("Device: %s\n", prop.name);
    printf("PCIe Link Width: %d\n", pcieWidth);
    printf("PCIe Link Speed: %d GT/s\n", pcieSpeed);
    printf("Theoretical peak PCIe bandwidth: %.2f GB/s\n\n", theoreticalGBps);

    // Allocate pinned host memory
    void *h_buf = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_buf, BUFFER_SIZE));

    // Allocate device memory
    void *d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, BUFFER_SIZE));

    // Initialize host buffer with dummy data
    memset(h_buf, 0xAA, BUFFER_SIZE);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Measure Host to Device transfer
    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(d_buf, h_buf, BUFFER_SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_htod = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_htod, start, stop));
    double gbps_htod = (BUFFER_SIZE / (1024.0 * 1024.0 * 1024.0)) / (ms_htod / 1000.0);

    // Measure Device to Host transfer
    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(h_buf, d_buf, BUFFER_SIZE, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_dtoh = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_dtoh, start, stop));
    double gbps_dtoh = (BUFFER_SIZE / (1024.0 * 1024.0 * 1024.0)) / (ms_dtoh / 1000.0);

    printf("Measured PCIe bandwidth:\n");
    printf("  Host -> Device: %.2f GB/s (time: %.3f ms)\n", gbps_htod, ms_htod);
    printf("  Device -> Host: %.2f GB/s (time: %.3f ms)\n", gbps_dtoh, ms_dtoh);
    printf("\nComparison:\n");
    printf("  Theoretical peak: %.2f GB/s\n", theoreticalGBps);
    printf("  Measured HtoD:    %.2f GB/s (%.2f%% of peak)\n", gbps_htod,
           100.0 * gbps_htod / theoreticalGBps);
    printf("  Measured DtoH:    %.2f GB/s (%.2f%% of peak)\n", gbps_dtoh,
           100.0 * gbps_dtoh / theoreticalGBps);

    // Cleanup
    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFreeHost(h_buf));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
```