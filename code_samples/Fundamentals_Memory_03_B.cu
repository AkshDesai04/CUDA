```c
/*
Time the `cudaMemcpyDeviceToDevice` operation. Compare this timing to a DtoH followed by an HtoD copy. The DtoD copy should be much faster.

Thought Process:
1. Define a large data buffer (e.g., 10 million floats) to ensure measurable copy times.
2. Allocate two device buffers `d_src` and `d_dst` of the same size.
3. Allocate a host buffer `h_buf` for the intermediate copy in the DtoH/HtoD scenario.
4. Use CUDA events to time operations with high resolution:
   - Create events for start and stop.
   - Record start before the copy, stop after, and then compute elapsed time.
5. Perform the following in a loop (e.g., 100 iterations) to average out jitter:
   a) DtoD: copy from `d_src` to `d_dst`.
   b) DtoH: copy from `d_src` to `h_buf`.
   c) HtoD: copy from `h_buf` back to `d_dst`.
   Measure the combined DtoH+HtoD time separately.
6. Compute average times for both methods.
7. Print results in milliseconds, highlighting that DtoD is faster.
8. Clean up all allocated memory and destroy events.
9. Add basic error checking after CUDA API calls to make the program robust.

The final program prints the average DtoD time and the average DtoH+HtoD time, demonstrating the expected speed difference.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 10000000          // Number of floats (≈40 MB)
#define ITER 100            // Number of repetitions for averaging

// Utility macro for checking CUDA errors
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main(void) {
    size_t bytes = N * sizeof(float);
    float *h_buf = (float*)malloc(bytes);
    if (!h_buf) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    float *d_src = NULL;
    float *d_dst = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_src, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_dst, bytes));

    // Initialize source buffer on device (e.g., all ones)
    CHECK_CUDA(cudaMemset(d_src, 0x00, bytes));  // zeroing for simplicity

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float timeDtoD = 0.0f;
    float timeDtoH_HtoD = 0.0f;

    for (int i = 0; i < ITER; ++i) {
        // ---- Device to Device ----
        CHECK_CUDA(cudaEventRecord(start, 0));
        CHECK_CUDA(cudaMemcpy(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        timeDtoD += ms;

        // ---- Device to Host then Host to Device ----
        CHECK_CUDA(cudaEventRecord(start, 0));
        CHECK_CUDA(cudaMemcpy(h_buf, d_src, bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(d_dst, h_buf, bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        timeDtoH_HtoD += ms;
    }

    // Compute average times
    timeDtoD /= ITER;
    timeDtoH_HtoD /= ITER;

    printf("Average time over %d iterations:\n", ITER);
    printf("  Device-to-Device (DtoD)    : %6.3f ms\n", timeDtoD);
    printf("  Device-to-Host + Host-to-Device (DtoH + HtoD): %6.3f ms\n",
           timeDtoH_HtoD);
    printf("  DtoD is faster by a factor of %.2f\n",
           timeDtoH_HtoD / timeDtoD);

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    free(h_buf);

    return EXIT_SUCCESS;
}
```