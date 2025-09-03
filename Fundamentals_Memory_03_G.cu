```c
/*
Aim of the program: 
After a DtoD copy, modify `d_a`. Does `d_b` change? (No, it's a deep copy).

Thinking process:
1. Allocate two device arrays `d_a` and `d_b` of the same size.
2. Initialize `d_a` with known values on the host and copy it to the device.
3. Perform a device-to-device copy from `d_a` to `d_b` using `cudaMemcpy` with
   the `cudaMemcpyDeviceToDevice` flag. This should create a deep copy, so
   `d_b` will have its own independent memory.
4. Launch a simple kernel that modifies each element of `d_a` (e.g., increment
   by 10) to demonstrate that changes to `d_a` do not affect `d_b`.
5. Copy both arrays back to the host and print them to verify that `d_b`
   remains unchanged while `d_a` reflects the modifications.
6. Clean up device memory and exit.

Edge cases:
- Ensure error checking for CUDA API calls.
- Use a small array size for simplicity and clarity.
- Synchronize after kernel launch before copying back to host.

The program will output two arrays: the original `d_a` values and the modified
`d_a` after the kernel, plus the untouched `d_b` values to confirm the deep
copy behavior.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple error checking macro
#define CHECK_CUDA(call) do {                                 \
    cudaError_t err = call;                                  \
    if (err != cudaSuccess) {                                \
        fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",    \
                #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                  \
    }                                                        \
} while (0)

// Kernel to increment each element of the array by 10
__global__ void increment_kernel(int *d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] += 10;
    }
}

int main() {
    const int N = 10;
    const size_t bytes = N * sizeof(int);
    int h_a[N];
    int h_b[N];

    // Initialize host array h_a
    for (int i = 0; i < N; ++i) {
        h_a[i] = i * 2;  // e.g., 0, 2, 4, ...
    }

    // Device pointers
    int *d_a = NULL;
    int *d_b = NULL;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void **)&d_a, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_b, bytes));

    // Copy data from host to device d_a
    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    // Device-to-device copy from d_a to d_b
    CHECK_CUDA(cudaMemcpy(d_b, d_a, bytes, cudaMemcpyDeviceToDevice));

    // Launch kernel to modify d_a
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    increment_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy back both arrays to host
    CHECK_CUDA(cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost));

    // Print results
    printf("After incrementing d_a by 10:\n");
    printf("h_a (modified): ");
    for (int i = 0; i < N; ++i) printf("%d ", h_a[i]);
    printf("\n");

    printf("h_b (should remain unchanged): ");
    for (int i = 0; i < N; ++i) printf("%d ", h_b[i]);
    printf("\n");

    // Clean up
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));

    return 0;
}
```