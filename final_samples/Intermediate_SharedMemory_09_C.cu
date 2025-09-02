/*
Aim of the program:
Write a kernel where `output[i] = shared_mem[threadIdx.x]`. This should be conflict-free.

Thinking process:

1. To satisfy the requirement, we create a CUDA kernel that writes each element of an output array
   by reading from a shared memory location indexed by `threadIdx.x`.  
2. We need to ensure that each thread reads a unique element of shared memory so that there is no
   write or read contention. By allocating the shared memory array to have at least
   `blockDim.x` elements and using `threadIdx.x` as the index, each thread accesses its own
   distinct location.  
3. To demonstrate the kernel, we also copy data from a global input array into shared memory
   before reading it back. This shows how the shared memory is initialized and used in a
   conflict‑free manner.  
4. The host code allocates memory for input and output arrays, initializes the input,
   launches the kernel with a suitable number of blocks and threads, copies the output back
   to the host, verifies the result, and finally cleans up.  
5. Bank conflicts in shared memory are avoided because every thread accesses a distinct
   word, and with a linear layout the access pattern maps each thread to a separate bank
   (or at least does not create a conflict pattern).  
6. The kernel and host use standard CUDA API calls with error checking for robustness.

This program can be compiled with `nvcc` and executed on any CUDA-capable GPU.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Kernel: each thread writes output[i] = shared_mem[threadIdx.x]
__global__ void conflict_free_kernel(const int *input, int *output, int n)
{
    // Shared memory allocation (size equal to blockDim.x)
    extern __shared__ int shared_mem[];

    // Compute global index
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    // Load data from global memory into shared memory
    shared_mem[threadIdx.x] = input[gid];

    // Optional: __syncthreads(); // not strictly needed here since each thread writes/read its own element

    // Write output from shared memory
    output[gid] = shared_mem[threadIdx.x];
}

int main()
{
    const int N = 1 << 20; // 1M elements
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    size_t size_bytes = N * sizeof(int);

    // Allocate host memory
    int *h_input = (int *)malloc(size_bytes);
    int *h_output = (int *)malloc(size_bytes);
    if (!h_input || !h_output) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_input[i] = i;
    }

    // Allocate device memory
    int *d_input = nullptr;
    int *d_output = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_input, size_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_output, size_bytes));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size_bytes, cudaMemcpyHostToDevice));

    // Launch kernel with shared memory size equal to threadsPerBlock * sizeof(int)
    conflict_free_kernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(
        d_input, d_output, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size_bytes, cudaMemcpyDeviceToHost));

    // Verify results
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (h_output[i] != h_input[i]) {
            fprintf(stderr, "Mismatch at index %d: expected %d, got %d\n",
                    i, h_input[i], h_output[i]);
            success = false;
            break;
        }
    }
    if (success) {
        printf("Kernel executed successfully. All values match.\n");
    } else {
        printf("Kernel produced incorrect results.\n");
    }

    // Clean up
    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
