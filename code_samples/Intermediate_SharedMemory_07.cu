```cpp
/*
Write a kernel to reverse an array within a single block. Each thread `i` loads `input[i]` into `shared_mem[i]`. After a `__syncthreads()`, thread `i` reads from `shared_mem[BLOCK_SIZE - 1 - i]` and writes to `output[i]`. This is much faster than global memory reversal.

My thinking:
1. The goal is to implement a simple CUDA kernel that performs an in‑block reversal of an array.
2. Each thread loads an element from global memory into shared memory. Since shared memory is fast, the reverse read/write will be cheaper.
3. The kernel will use `__syncthreads()` to ensure all loads are complete before any thread reads from shared memory.
4. We will launch the kernel with exactly one block; the block size will be set to a compile‑time constant `BLOCK_SIZE`.  
   If the input array is larger than `BLOCK_SIZE`, we will only reverse the first `BLOCK_SIZE` elements for demonstration.
5. The host code will:
   - Allocate a test array of integers of size `N` (<= BLOCK_SIZE).
   - Allocate device memory for input and output.
   - Copy the input array to the device.
   - Launch the kernel with one block and `N` threads.
   - Copy the output back to host and print both original and reversed arrays.
6. Error checking will be performed after each CUDA API call.
7. The code will be self‑contained and compile with `nvcc`.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

__global__ void reverse_in_block(const int* input, int* output, int n) {
    // Allocate shared memory dynamically sized to BLOCK_SIZE
    extern __shared__ int shared_mem[];

    int i = threadIdx.x;

    // Guard against out‑of‑bounds threads
    if (i >= n) return;

    // Each thread loads its element into shared memory
    shared_mem[i] = input[i];

    // Synchronize to ensure all loads are complete
    __syncthreads();

    // Each thread writes the reversed element to output
    output[i] = shared_mem[n - 1 - i];
}

int main(void) {
    // Example array size (must be <= BLOCK_SIZE)
    const int n = 20;
    size_t size = n * sizeof(int);

    // Host allocation
    int* h_input = (int*)malloc(size);
    int* h_output = (int*)malloc(size);

    // Initialize input array
    printf("Original array:\n");
    for (int i = 0; i < n; ++i) {
        h_input[i] = i + 1;
        printf("%d ", h_input[i]);
    }
    printf("\n");

    // Device allocation
    int *d_input = NULL;
    int *d_output = NULL;
    cudaError_t err = cudaMalloc((void**)&d_input, size);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_input error: %s\n", cudaGetErrorString(err)); return EXIT_FAILURE; }
    err = cudaMalloc((void**)&d_output, size);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_output error: %s\n", cudaGetErrorString(err)); return EXIT_FAILURE; }

    // Copy input to device
    err = cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy H2D error: %s\n", cudaGetErrorString(err)); return EXIT_FAILURE; }

    // Launch kernel with one block and n threads
    dim3 blockDim(n);
    dim3 gridDim(1);
    size_t sharedMemSize = n * sizeof(int);
    reverse_in_block<<<gridDim, blockDim, sharedMemSize>>>(d_input, d_output, n);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err)); return EXIT_FAILURE; }

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize error: %s\n", cudaGetErrorString(err)); return EXIT_FAILURE; }

    // Copy result back to host
    err = cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy D2H error: %s\n", cudaGetErrorString(err)); return EXIT_FAILURE; }

    // Print reversed array
    printf("Reversed array:\n");
    for (int i = 0; i < n; ++i) {
        printf("%d ", h_output[i]);
    }
    printf("\n");

    // Clean up
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return EXIT_SUCCESS;
}
```