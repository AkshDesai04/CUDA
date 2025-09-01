```cpp
/*
 * Aim of the program: Remember to use `cudaFreeHost` to free memory allocated with `cudaMallocHost`.
 *
 * My thinking while writing this program:
 * 1. The prompt specifically asks to demonstrate the use of `cudaFreeHost` for memory
 *    allocated via `cudaMallocHost`. I therefore need a small CUDA application that:
 *    - Allocates pinned (page-locked) host memory using `cudaMallocHost`.
 *    - Allocates corresponding device memory with `cudaMalloc`.
 *    - Copies data from host to device and back.
 *    - Launches a simple kernel to show that the device memory is usable.
 *    - Properly frees the pinned host memory with `cudaFreeHost` after the kernel
 *      execution and data copy back.
 *
 * 2. I will keep the program minimal, yet complete:
 *    - Include necessary headers.
 *    - Define a small kernel that increments each element.
 *    - Use a simple `int` array of size 10 for clarity.
 *    - Print results to confirm correctness.
 *
 * 3. I must ensure that the program compiles with nvcc and runs without errors.
 *    So I will check for errors after each CUDA call using a helper macro.
 *
 * 4. The focus is on showing the pair `cudaMallocHost` + `cudaFreeHost`, so I will
 *    comment those parts clearly and make sure the rest of the code is straightforward.
 *
 * 5. Finally, I will output the entire code file as a .cu source file content,
 *    following the instruction to output only the .cu file, with no extra commentary.
 */

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CHECK_CUDA(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",      \
                    #call, __FILE__, __LINE__,                    \
                    cudaGetErrorString(err));                     \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

// Simple kernel that increments each element by 1
__global__ void increment_kernel(int *d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] += 1;
    }
}

int main(void) {
    const int N = 10;
    const size_t size = N * sizeof(int);

    // Allocate pinned (page-locked) host memory
    int *h_arr;
    CHECK_CUDA(cudaMallocHost(&h_arr, size));

    // Initialize host array
    for (int i = 0; i < N; ++i) {
        h_arr[i] = i;
    }

    // Allocate device memory
    int *d_arr;
    CHECK_CUDA(cudaMalloc((void **)&d_arr, size));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    increment_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost));

    // Print results
    printf("Result after kernel execution:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Free device memory
    CHECK_CUDA(cudaFree(d_arr));

    // Free pinned host memory using cudaFreeHost
    CHECK_CUDA(cudaFreeHost(h_arr));

    return 0;
}
```