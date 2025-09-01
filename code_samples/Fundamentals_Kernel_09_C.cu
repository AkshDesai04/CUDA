/*
Aim: Explain the difference between __global__ and __device__ functions in a comment.

Thinking: The user wants a short explanation in a comment, but the developer instruction requires producing a full .cu file that includes a comment with the aim and a small CUDA program. I will create a minimal CUDA program that demonstrates both __global__ and __device__ functions. The top multiline comment will contain the aim exactly as requested, plus a detailed explanation of the two function types, including their usage, launch constraints, and how they differ. After the comment, the code will include a simple device helper function (__device__) and a kernel function (__global__) that calls the helper. The host code will launch the kernel, copy the result back, and print it. This satisfies the developer instruction to output a .cu file with code only, and the user’s request for an explanation in a comment. No additional text or files are provided. */

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for checking CUDA errors
#define CUDA_CHECK(err)                                                        \
    do {                                                                       \
        cudaError_t err_ = (err);                                              \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err_));                                \
            exit(EXIT_FAILURE);                                               \
        }                                                                      \
    } while (0)

/* --------------------------------------------------------------------
 * Explanation of __global__ vs __device__ functions:
 *
 * __global__:
 *   - Declared with __global__ before the return type.
 *   - Can only be called from host (CPU) code.
 *   - Executes on the device (GPU) and runs concurrently with many
 *     threads. It can be launched with <<<grid, block>>> syntax.
 *   - Must have a void return type (no return value).
 *
 * __device__:
 *   - Declared with __device__ before the return type.
 *   - Executed on the device and can only be called from other
 *     __device__ or __global__ functions.
 *   - Can have any return type, including complex types.
 *   - Not launchable from host; it is part of the device runtime.
 *
 * In practice, __global__ functions are your kernels; __device__
 * functions are helpers or device-side utilities that you call
 * from kernels or other device functions. They share the same
 * memory space and can access global, shared, and local memory
 * just like the kernels.
 * -------------------------------------------------------------------- */

__device__ int add(int a, int b) {
    // Simple device helper that adds two integers.
    return a + b;
}

__global__ void kernel(int *out, int N) {
    // Each thread writes the result of add(threadIdx.x, 10) to out[tid].
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        out[tid] = add(tid, 10);
    }
}

int main(void) {
    const int N = 256;
    size_t size = N * sizeof(int);

    // Allocate host memory
    int *h_out = (int *)malloc(size);
    if (!h_out) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    int *d_out;
    CUDA_CHECK(cudaMalloc((void **)&d_out, size));

    // Launch kernel: 64 threads per block, 4 blocks (total 256 threads)
    int threadsPerBlock = 64;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    // Print a few results to verify
    for (int i = 0; i < 10; ++i) {
        printf("h_out[%d] = %d\n", i, h_out[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_out));
    free(h_out);

    return 0;
}