```cpp
/*
Aim of the program:
Compare your measured bandwidth to the theoretical peak bandwidth of your GPU's PCIe bus.

Thinking process:
1.  To measure PCIe bandwidth we need to copy a large amount of data between host and device.
    The copy direction (host→device or device→host) is irrelevant for the peak transfer rate,
    but we will measure both and take the average to reduce noise.

2.  Use pinned host memory (cudaMallocHost) so that the copy uses the fastest DMA path
    rather than going through pageable memory.  This gives a more accurate reflection of
    the PCIe link performance.

3.  Timing is performed with CUDA events (cudaEventRecord / cudaEventSynchronize) because
    they provide nanosecond resolution and are GPU-aware, avoiding CPU overhead.

4.  The theoretical peak bandwidth depends on PCIe generation and link width.
    CUDA provides the attributes MaxPciLinkSpeed (in GT/s) and MaxPciLinkWidth.
    For PCIe 3.0 the raw speed is 8 GT/s, but 128b/130b encoding reduces the effective
    data rate to 7.875 GT/s, which corresponds to 0.984375 GB/s per lane.
    Thus the effective bandwidth per lane = MaxPciLinkSpeed * 0.984375 GB/s.
    Total theoretical bandwidth = effective per lane * link width.

5.  The program queries these attributes, computes the theoretical value, then
    reports the measured transfer rates for host→device and device→host, the
    average measured rate, and a comparison ratio.

6.  Error checking is performed after each CUDA call to catch any issues early.

Note: The program allocates 512 MB of data; this is large enough to average out
    small timing fluctuations but small enough to fit on most GPUs.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>

// Helper macro for error checking
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess) {                                                 \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl;                    \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

int main() {
    // Choose device 0 for simplicity
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    // Get PCIe attributes
    int pcie_speed_gts = 0; // max PCIe speed in GT/s
    int pcie_width = 0;     // link width (number of lanes)
    CUDA_CHECK(cudaDeviceGetAttribute(&pcie_speed_gts, cudaDevAttrMaxPciLinkSpeed, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&pcie_width, cudaDevAttrMaxPciLinkWidth, device));

    // Compute theoretical bandwidth (GB/s)
    // Effective data rate per lane: speed_gts * 0.984375 GB/s
    const double per_lane_gbps = pcie_speed_gts * 0.984375; // GB/s per lane
    const double theoretical_gbps = per_lane_gbps * static_cast<double>(pcie_width);

    // Allocate pinned host memory
    const size_t bytes = 512 * 1024 * 1024; // 512 MB
    void* host_mem = nullptr;
    CUDA_CHECK(cudaMallocHost(&host_mem, bytes));

    // Allocate device memory
    void* device_mem = nullptr;
    CUDA_CHECK(cudaMalloc(&device_mem, bytes));

    // Initialize host memory with some data
    unsigned char* h_ptr = static_cast<unsigned char*>(host_mem);
    for (size_t i = 0; i < bytes; ++i) {
        h_ptr[i] = static_cast<unsigned char>(i & 0xFF);
    }

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Measure host-to-device transfer
    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(device_mem, host_mem, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_hd = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_hd, start, stop));

    // Measure device-to-host transfer
    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpy(host_mem, device_mem, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_dh = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_dh, start, stop));

    // Compute bandwidths
    double bw_hd = (bytes * 1e-9) / (ms_hd / 1000.0); // GB/s
    double bw_dh = (bytes * 1e-9) / (ms_dh / 1000.0); // GB/s
    double bw_avg = (bw_hd + bw_dh) / 2.0;

    // Output results
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "PCIe Link: " << pcie_width << "x" << pcie_speed_gts << " GT/s" << std::endl;
    std::cout << "Theoretical peak bandwidth: " << theoretical_gbps << " GB/s" << std::endl;
    std::cout << "Measured host->device bandwidth: " << bw_hd << " GB/s" << std::endl;
    std::cout << "Measured device->host bandwidth: " << bw_dh << " GB/s" << std::endl;
    std::cout << "Average measured bandwidth: " << bw_avg << " GB/s" << std::endl;
    std::cout << "Measured / Theoretical ratio: " << bw_avg / theoretical_gbps << std::endl;

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(device_mem));
    CUDA_CHECK(cudaFreeHost(host_mem));

    return 0;
}
```