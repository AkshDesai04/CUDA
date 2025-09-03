```cuda
/*
Aim of the program:
Launch your grid-stride kernel for a vector of size 1M, but only launch a grid of 1024 threads total. Verify the correctness.

Thinking:
- We need a 1 million element vector. 1M fits in a single CUDA device easily.
- The requirement is to launch a grid that uses exactly 1024 threads total. That can be achieved by launching a single block with 1024 threads (blockDim.x = 1024, gridDim.x = 1). The grid stride for the kernel will thus be 1024.
- We will write a simple grid‑stride kernel that increments each element by 1.0f. Each thread will loop over the array indices it is responsible for: starting at its global index and stepping by the stride.
- After the kernel finishes, we will copy the data back to host memory and verify that each element has been incremented correctly. For simplicity, we will initialize the array to zeros on the host; after the kernel, each element should be 1.0f.
- Error checking is included for CUDA API calls and kernel launches.
- The program will print a success message if all elements are correct, otherwise it will report an error.
- We keep the code simple and self‑contained, no external dependencies beyond the CUDA runtime.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1000000          // Vector size: 1M
#define THREADS_PER_BLOCK 1024

// CUDA error checking macro
#define CHECK_CUDA(call)                                    \
    do {                                                    \
        cudaError_t err = call;                             \
        if (err != cudaSuccess) {                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    } while (0)

// Grid-stride kernel that adds 1.0f to each element
__global__ void add_one(float *d_data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        d_data[i] += 1.0f;
    }
}

int main(void)
{
    float *h_data = (float*)malloc(N * sizeof(float));
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host data to 0.0f
    for (int i = 0; i < N; ++i) {
        h_data[i] = 0.0f;
    }

    // Allocate device memory
    float *d_data = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_data, N * sizeof(float)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel with 1 block of 1024 threads
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim(1);  // Only one block, total threads = 1024
    add_one<<<gridDim, blockDim>>>(d_data, N);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify correctness
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (h_data[i] != 1.0f) {
            if (errors < 10) { // Print first few errors
                fprintf(stderr, "Error at index %d: expected 1.0f, got %f\n", i, h_data[i]);
            }
            ++errors;
        }
    }

    if (errors == 0) {
        printf("All %d elements correctly incremented to 1.0f.\n", N);
    } else {
        printf("Found %d errors in output.\n", errors);
    }

    // Clean up
    free(h_data);
    CHECK_CUDA(cudaFree(d_data));

    return 0;
}
```