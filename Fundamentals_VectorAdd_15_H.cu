```cuda
/*
Aim of the program: Handle large vectors with boundary checks in your fused kernel.

Thought process:
- The requirement is to create a CUDA program that processes large vectors and ensures boundary checks in a fused kernel.
- A "fused kernel" typically performs multiple operations in a single launch to reduce memory traffic and kernel launch overhead. 
- For this example, I chose a common fused operation: element‑wise addition of two input vectors A and B, followed by scaling by a constant alpha, producing output vector C:  
  C[i] = alpha * (A[i] + B[i]).
- To handle "large vectors", the code will accept a vector size N that can be much larger than a single block's capacity. It will launch enough blocks to cover the entire array.
- Boundary checks: Each thread computes a global index `idx`. It must check `idx < N` before accessing array elements to avoid out‑of‑bounds memory accesses when the number of threads exceeds N.
- Host side: Allocate vectors A, B, C on the host, initialize A and B with some values, allocate corresponding device memory, copy data to device, launch the fused kernel, copy back result, verify correctness, and free memory.
- Performance: Use CUDA events to time the kernel execution.
- Error checking: A simple macro `CUDA_CHECK` is used to wrap CUDA API calls and report any errors.
- The code is self‑contained, uses only the CUDA runtime API, and compiles with `nvcc -o fused_vector fused_vector.cu`.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a CUDA API call
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",          \
                    __FUNCTION__, __FILE__, __LINE__,                 \
                    cudaGetErrorString(err));                         \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// Fused kernel: C[i] = alpha * (A[i] + B[i])
__global__ void fusedAddScaleKernel(const float *A, const float *B, float *C, float alpha, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = alpha * (A[idx] + B[idx]);
    }
}

int main(int argc, char *argv[])
{
    // Parse vector size from command line or use default
    size_t N = 1 << 24; // ~16 million elements (~64MB per float array)
    if (argc > 1) {
        N = strtoul(argv[1], NULL, 10);
    }
    printf("Vector size: %zu elements\n", N);

    // Host allocations
    float *h_A = (float *)malloc(N * sizeof(float));
    float *h_B = (float *)malloc(N * sizeof(float));
    float *h_C = (float *)malloc(N * sizeof(float));

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = 1.0f;                 // example data
        h_B[i] = 2.0f;
    }

    // Device allocations
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_C, N * sizeof(float)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (int)((N + threadsPerBlock - 1) / threadsPerBlock);

    float alpha = 0.5f; // scaling factor

    // Timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Launch kernel
    fusedAddScaleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, alpha, N);

    // Record stop
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Compute elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %f ms\n", milliseconds);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify a few elements
    int errors = 0;
    for (size_t i = 0; i < N && errors < 10; ++i) {
        float expected = alpha * (h_A[i] + h_B[i]);
        if (fabs(h_C[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %zu: host=%f, device=%f, expected=%f\n",
                    i, h_C[i], expected, h_A[i], h_B[i]);
            errors++;
        }
    }
    if (errors == 0) {
        printf("Verification passed.\n");
    } else {
        printf("Verification failed with %d errors.\n", errors);
    }

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```