```cuda
/*********************************************************************
 * Aim of the program (as requested): 
 * "Plot the measured bandwidth for different transfer sizes (e.g., 1KB, 1MB, 16MB, 128MB). 
 * Notice that bandwidth is lower for small transfers due to latency overhead."
 * 
 * My thinking while writing this program:
 * 1. We want to measure the effective data transfer bandwidth between host and device
 *    for several transfer sizes: 1KB, 1MB, 16MB, 128MB.
 * 2. CUDA provides cudaEvent_t objects that give high‑resolution timing on the GPU.
 *    We will use cudaEventRecord before and after each copy and then use 
 *    cudaEventElapsedTime to get the elapsed time in milliseconds.
 * 3. For each size we will:
 *    - Allocate host memory with cudaHostAlloc to enable page‑locked (pinned) memory,
 *      which is optimal for DMA transfers.
 *    - Allocate device memory with cudaMalloc.
 *    - Initialize host memory with some data (not strictly necessary for transfer
 *      measurement but useful to avoid page‑faults).
 *    - Perform a warm‑up copy (host → device) to avoid first‑time overhead.
 *    - Record the time for a single copy and compute bandwidth = size / time.
 *    - Print the size in bytes and the bandwidth in GB/s.
 * 4. The program will output a simple table that can be plotted externally.
 * 5. We will also include a second measurement of device → host copy to demonstrate
 *    that bandwidth is similar in both directions, but the goal is just to showcase
 *    the bandwidth versus size effect.
 * 6. Error checking is done via a helper macro that prints the error string
 *    and exits on failure.
 * 7. The program is self‑contained in a single .cu file and can be compiled with:
 *    nvcc -O2 -arch=sm_70 bandwidth_measure.cu -o bandwidth_measure
 *********************************************************************/

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(err)                                                     \
    do {                                                                    \
        cudaError_t err__ = (err);                                          \
        if (err__ != cudaSuccess) {                                         \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__    \
                      << " - " << cudaGetErrorString(err__) << std::endl;  \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

int main() {
    // Define transfer sizes in bytes
    const size_t sizes[] = {
        1 * 1024,           // 1 KB
        1 * 1024 * 1024,    // 1 MB
        16 * 1024 * 1024,   // 16 MB
        128 * 1024 * 1024   // 128 MB
    };
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Transfer Size (bytes)\tBandwidth (GB/s)\tDirection\n";
    std::cout << "-----------------------------------------------\n";

    for (int i = 0; i < num_sizes; ++i) {
        size_t size = sizes[i];

        // Allocate pinned host memory
        float *h_data = nullptr;
        CUDA_CHECK(cudaHostAlloc((void**)&h_data, size, cudaHostAllocDefault));

        // Allocate device memory
        float *d_data = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&d_data, size));

        // Initialize host memory (not strictly necessary)
        for (size_t j = 0; j < size / sizeof(float); ++j) {
            h_data[j] = static_cast<float>(j);
        }

        // Warm‑up copy: host -> device
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

        // Create CUDA events for timing
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Measure host -> device copy
        CUDA_CHECK(cudaEventRecord(start, 0));
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms_time;
        CUDA_CHECK(cudaEventElapsedTime(&ms_time, start, stop));
        float gbps = (size / (1024.0f * 1024.0f * 1024.0f)) / (ms_time / 1000.0f);

        std::cout << size << "\t\t" << gbps << "\t\tH->D\n";

        // Measure device -> host copy
        CUDA_CHECK(cudaEventRecord(start, 0));
        CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(&ms_time, start, stop));
        gbps = (size / (1024.0f * 1024.0f * 1024.0f)) / (ms_time / 1000.0f);

        std::cout << size << "\t\t" << gbps << "\t\tD->H\n";

        // Clean up events
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        // Free memory
        CUDA_CHECK(cudaFree(d_data));
        CUDA_CHECK(cudaFreeHost(h_data));
    }

    return 0;
}
```