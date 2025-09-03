```cu
/*
Aim: Compare your measured bandwidth to the theoretical peak bandwidth of your GPU's PCIe bus.

Thinking process:
1. The goal is to quantify how well memory transfers between host and device perform
   compared to the absolute maximum bandwidth the PCIe interface can provide.
2. PCIe bandwidth depends on link width and generation. Many modern GPUs use PCIe 3.0
   x16, which theoretically offers about 15.75 GB/s per direction (this value is
   often quoted in GPU documentation). For PCIe 3.0 the raw per‑lane bandwidth is
   8 GT/s, but due to 128b/130b encoding the effective bandwidth is roughly
   7.88 GB/s per lane. For a 16‑lane link that yields ≈126 GB/s; however, GPU
   manufacturers typically report a more conservative figure (~15.75 GB/s) to
   account for overhead and to be comparable across generations.
3. To perform a fair measurement:
   - Allocate a large buffer (e.g., 100 MiB) on host and device.
   - Use cudaMemcpy for H→D and D→H transfers, timing each with CUDA events.
   - Repeat several times to average out noise.
4. Compute bandwidth in GB/s as (bytes transferred / seconds).
5. Print both the measured bandwidth and the theoretical peak for comparison.
6. The code is kept simple: no command line parsing, just a hard‑coded buffer size.
7. Compile with `nvcc` and run on a system with a CUDA‑capable GPU.

The program below implements this plan. It uses CUDA runtime API functions only
and outputs the measured bandwidths along with the assumed theoretical peak
value (15.75 GB/s for PCIe 3.0 x16). Feel free to adjust the theoretical value
if your hardware uses a different PCIe configuration.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main(void) {
    const size_t MB = 1024 * 1024;
    const size_t bufSize = 100 * MB;          // 100 MiB buffer
    const int repetitions = 10;

    // Allocate host memory (pinned for faster transfer)
    void *h_buf = NULL;
    CHECK_CUDA(cudaMallocHost(&h_buf, bufSize));

    // Allocate device memory
    void *d_buf = NULL;
    CHECK_CUDA(cudaMalloc(&d_buf, bufSize));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float totalTimeHtoD = 0.0f;
    float totalTimeDtoH = 0.0f;

    // Warm-up copy
    CHECK_CUDA(cudaMemcpy(d_buf, h_buf, bufSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(h_buf, d_buf, bufSize, cudaMemcpyDeviceToHost));

    for (int i = 0; i < repetitions; ++i) {
        // Host to Device
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cudaMemcpy(d_buf, h_buf, bufSize, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        totalTimeHtoD += ms;

        // Device to Host
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cudaMemcpy(h_buf, d_buf, bufSize, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        totalTimeDtoH += ms;
    }

    // Compute average times in seconds
    float avgTimeHtoD = (totalTimeHtoD / repetitions) / 1000.0f;
    float avgTimeDtoH = (totalTimeDtoH / repetitions) / 1000.0f;

    // Bandwidth calculation (GB/s)
    double bwHtoD = (double)bufSize / (1024.0 * 1024.0 * 1024.0) / avgTimeHtoD;
    double bwDtoH = (double)bufSize / (1024.0 * 1024.0 * 1024.0) / avgTimeDtoH;

    // Theoretical peak bandwidth for PCIe 3.0 x16 (approx.)
    const double theoreticalPeakGBps = 15.75;

    printf("Measured Host->Device bandwidth: %.2f GB/s\n", bwHtoD);
    printf("Measured Device->Host bandwidth: %.2f GB/s\n", bwDtoH);
    printf("Theoretical PCIe 3.0 x16 peak bandwidth: %.2f GB/s\n", theoreticalPeakGBps);

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_buf));
    CHECK_CUDA(cudaFreeHost(h_buf));

    return 0;
}
```