/*
Aim: Define `dim3` variables for the grid and block sizes and use them in the kernel launch configuration.

Thinking process:
1. We need a complete CUDA program (.cu) that demonstrates how to declare and use `dim3` for grid and block dimensions.
2. A simple vector addition kernel (`addKernel`) is ideal because it clearly shows thread indexing and the need for a grid of blocks.
3. To make the example practical, we will:
   - Allocate host and device arrays.
   - Initialize host data.
   - Copy data to device.
   - Launch the kernel using the defined `dim3` grid and block sizes.
   - Copy the result back to host.
   - Verify correctness.
4. For the grid and block configuration:
   - Block size is chosen as 256 threads per block (common choice for memory-bound kernels).
   - Grid size is computed as the ceiling of (number of elements / block size).
   - These are stored in `dim3` variables (`block` and `grid`) and used directly in the kernel launch.
5. Additional details:
   - Include basic error checking for CUDA API calls.
   - Use `cudaMalloc`, `cudaMemcpy`, `cudaFree`.
   - Use `printf` for output.
6. The final code will be self‑contained and ready to compile with `nvcc`.

Below is the complete CUDA C code (.cu file) implementing the above plan.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void addKernel(const int *a, const int *b, int *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main()
{
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(int);

    // Allocate host memory
    int *h_a = (int*)malloc(size);
    int *h_b = (int*)malloc(size);
    int *h_c = (int*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = N - i;
    }

    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // Copy data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define block and grid sizes using dim3
    dim3 block(256); // 256 threads per block
    dim3 grid((N + block.x - 1) / block.x); // enough blocks to cover all elements

    // Launch kernel
    addKernel<<<grid, block>>>(d_a, d_b, d_c, N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify result
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            if (errors < 10) { // print first few errors
                printf("Mismatch at %d: %d + %d != %d\n", i, h_a[i], h_b[i], h_c[i]);
            }
            ++errors;
        }
    }
    if (errors == 0) {
        printf("All results correct.\n");
    } else {
        printf("Total mismatches: %d\n", errors);
    }

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
