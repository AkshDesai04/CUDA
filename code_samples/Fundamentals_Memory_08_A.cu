```cuda
/* 
Aim of the program: Perform the same timing for a device-to-host copy. Is it typically faster, slower, or the same?

Thinking process:
1. I need to create a CUDA program that measures the time taken for a device-to-host (d2h) memory copy.
2. I will allocate an array on the host (H) and an array on the device (D). 
3. For a meaningful copy I will first copy data from host to device to initialize the device memory.
4. Then I will perform multiple device-to-host copies (e.g., 1000 iterations) and time them using CUDA events to get a reliable measurement.
5. To provide context I will also time a host-to-device (h2d) copy, so the user can see the relative speeds.
6. I will calculate average times and print them out.
7. The output will indicate that d2h copies are typically slower due to PCIe bandwidth limitations, but the code itself will just show the measurement.
8. The code will be self‑contained in a single .cu file, with all necessary includes, error checking, and cleanup.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N (1 << 24)            // 16 million elements (~64 MB)
#define ITERATIONS 1000

// Helper macro for CUDA error checking
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main(void) {
    // Allocate host memory
    float *h_vec = (float*)malloc(N * sizeof(float));
    if (!h_vec) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    // Initialize host data
    for (size_t i = 0; i < N; ++i) h_vec[i] = (float)i;

    // Allocate device memory
    float *d_vec;
    CHECK_CUDA(cudaMalloc((void**)&d_vec, N * sizeof(float)));

    // Copy from host to device once (initialization)
    CHECK_CUDA(cudaMemcpy(d_vec, h_vec, N * sizeof(float), cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Timing device-to-host copy
    float elapsedTimeD2H = 0.0f;
    for (int i = 0; i < ITERATIONS; ++i) {
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cudaMemcpy(h_vec, d_vec, N * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        elapsedTimeD2H += ms;
    }
    elapsedTimeD2H /= ITERATIONS;

    // Timing host-to-device copy
    float elapsedTimeH2D = 0.0f;
    for (int i = 0; i < ITERATIONS; ++i) {
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cudaMemcpy(d_vec, h_vec, N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        elapsedTimeH2D += ms;
    }
    elapsedTimeH2D /= ITERATIONS;

    // Output results
    printf("Average time over %d iterations:\n", ITERATIONS);
    printf("Device-to-Host (d2h):   %f ms\n", elapsedTimeD2H);
    printf("Host-to-Device (h2d):   %f ms\n", elapsedTimeH2D);
    if (elapsedTimeD2H > elapsedTimeH2D)
        printf("Typically, device-to-host copy is slower than host-to-device.\n");
    else if (elapsedTimeD2H < elapsedTimeH2D)
        printf("Typically, device-to-host copy is faster than host-to-device.\n");
    else
        printf("Device-to-host and host-to-device copies have similar performance.\n");

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_vec));
    free(h_vec);

    return 0;
}
```