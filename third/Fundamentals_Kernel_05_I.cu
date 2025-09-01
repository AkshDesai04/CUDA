/*
 * Aim: Calculate the global ID starting from 1 instead of 0.
 *
 * Thinking: In CUDA, the global ID of a thread in a 1D grid can be computed as:
 *     global_id0 = blockIdx.x * blockDim.x + threadIdx.x
 * We want IDs starting from 1, so we add 1 to the result.
 * The kernel will write these IDs into an array for verification.
 * We choose a 1D grid and 1D blocks for simplicity. The host will allocate an array
 * on device, launch the kernel, copy back results, and print them.
 *
 * Additionally, we will add bounds checking: only threads where global_id0 < N
 * will write to the array.
 *
 * The program demonstrates basic CUDA memory allocation, kernel launch,
 * and data transfer. The kernel uses simple arithmetic; no dynamic parallelism
 * or recursion is required.
 *
 * The code is written in standard C++ style for CUDA (.cu file).
 *
 * Build: nvcc -o global_id global_id.cu
 *
 * Run: ./global_id
 *
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",       \
                    __func__, __FILE__, __LINE__,                   \
                    cudaGetErrorString(err));                       \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Kernel that writes global ID + 1 into output array
__global__ void computeGlobalID(int *out, int N)
{
    // Compute the 0‑based global index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Write only if within bounds
    if (idx < N)
    {
        // Global ID starting from 1
        out[idx] = idx + 1;
    }
}

int main(void)
{
    const int N = 16;                 // Number of elements / threads
    const int threadsPerBlock = 4;    // You can change this value

    // Compute grid size
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate host memory
    int *h_out = (int *)malloc(N * sizeof(int));
    if (!h_out) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    int *d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_out, N * sizeof(int)));

    // Launch kernel
    computeGlobalID<<<blocks, threadsPerBlock>>>(d_out, N);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print the results
    printf("Global IDs starting from 1:\n");
    for (int i = 0; i < N; ++i) {
        printf("Thread %d -> ID %d\n", i, h_out[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_out));
    free(h_out);

    return 0;
}
