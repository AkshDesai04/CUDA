/*
Aim of the program: Run for `STRIDE=32`. The performance should be very poor.

Thinking:
- To illustrate poor performance, we deliberately use a memory access pattern that is highly uncoalesced.
- We define a kernel where each thread writes to a global array element that is spaced apart by `STRIDE` elements.
- Since `STRIDE` is 32, threads within a warp will access memory locations far apart (32 elements apart), causing each thread to generate a separate memory transaction.
- This pattern prevents memory coalescing, increases memory traffic, and will degrade performance noticeably compared to a contiguous access pattern.
- The program allocates an array large enough to demonstrate the effect, copies data to the device, runs the kernel, copies results back, and measures execution time using CUDA events.
- Error checking macros are included to ensure proper CUDA API usage.
*/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N (1 << 24)            // Size of the array (16M elements)
#define STRIDE 32              // Stride between elements accessed by consecutive threads
#define THREADS_PER_BLOCK 256  // Common block size

// Macro for checking CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Kernel: Each thread writes its index to a location that is STRIDE elements apart
__global__ void write_with_stride_kernel(int *d_arr, int stride, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int offset = idx * stride;
        if (offset < total_elements) {
            d_arr[offset] = idx; // Simple write
        }
    }
}

int main() {
    // Allocate host memory
    int *h_arr = (int*)malloc(N * sizeof(int));
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host array (not strictly necessary for this test)
    for (int i = 0; i < N; ++i) {
        h_arr[i] = i;
    }

    // Allocate device memory
    int *d_arr = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_arr, N * sizeof(int)));

    // Copy data to device (though not needed, we copy for completeness)
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    write_with_stride_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_arr, STRIDE, N);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time (uncoalesced stride %d): %f ms\n", STRIDE, milliseconds);

    // Copy back to host to ensure the kernel finished (optional)
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify a few elements
    int errors = 0;
    for (int i = 0; i < 10; ++i) {
        int idx = i * STRIDE;
        if (idx < N) {
            if (h_arr[idx] != i) {
                errors++;
                printf("Verification error at idx %d: expected %d, got %d\n", idx, i, h_arr[idx]);
            }
        }
    }
    if (errors == 0) {
        printf("Verification passed for first 10 elements.\n");
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_arr));
    free(h_arr);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
