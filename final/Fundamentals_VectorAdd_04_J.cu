/*
 * Aim: Verify the results of subtraction on the CPU.
 * 
 * This program demonstrates how to perform element-wise subtraction of two arrays
 * on the GPU using CUDA and then verify the correctness of the results by
 * recomputing the subtraction on the CPU. The process is:
 * 1. Allocate two input arrays (a and b) and an output array (c_gpu) on the host.
 * 2. Initialize a and b with sample data.
 * 3. Allocate corresponding arrays on the device.
 * 4. Copy a and b to device memory.
 * 5. Launch a CUDA kernel that computes c_gpu[i] = a[i] - b[i] for each element.
 * 6. Copy the result back to the host (c_gpu).
 * 7. Perform the same subtraction on the host (c_cpu).
 * 8. Compare c_gpu and c_cpu element-wise to verify that the GPU computation
 *    produced the expected results.
 * 9. Report success or failure.
 *
 * The program also includes simple error handling for CUDA API calls
 * and demonstrates basic CUDA memory management and kernel launch syntax.
 *
 * Thought process:
 * - Decide array size and data types (int for simplicity).
 * - Write a straightforward kernel that uses thread index to index arrays.
 * - Use a block size (e.g., 256 threads) and grid size to cover all elements.
 * - After GPU computation, use a loop to compare each element with CPU result.
 * - Print a message indicating whether all elements match.
 *
 * This example can be expanded to use floating point data, larger arrays,
 * or more complex verification logic as needed.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* CUDA error checking macro */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/* Kernel for element-wise subtraction */
__global__ void subtractKernel(const int *a, const int *b, int *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] - b[idx];
    }
}

int main(void)
{
    const int N = 1 << 20;          // Number of elements (about 1 million)
    const int bytes = N * sizeof(int);

    /* Host allocations */
    int *h_a = (int *)malloc(bytes);
    int *h_b = (int *)malloc(bytes);
    int *h_c_gpu = (int *)malloc(bytes);   // Result from GPU
    int *h_c_cpu = (int *)malloc(bytes);   // Result from CPU

    if (!h_a || !h_b || !h_c_gpu || !h_c_cpu) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize input data */
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = i % 3;   // Simple pattern for easy verification
    }

    /* Device allocations */
    int *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void **)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_c, bytes));

    /* Copy inputs to device */
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    /* Launch kernel */
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    subtractKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost));

    /* CPU subtraction for verification */
    for (int i = 0; i < N; ++i) {
        h_c_cpu[i] = h_a[i] - h_b[i];
    }

    /* Verify results */
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (h_c_gpu[i] != h_c_cpu[i]) {
            if (errors < 10) {   // Print first few mismatches
                fprintf(stderr, "Mismatch at index %d: GPU=%d, CPU=%d\n",
                        i, h_c_gpu[i], h_c_cpu[i]);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("Verification PASSED: All %d results match.\n", N);
    } else {
        printf("Verification FAILED: %d mismatches found.\n", errors);
    }

    /* Clean up */
    free(h_a);
    free(h_b);
    free(h_c_gpu);
    free(h_c_cpu);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}
