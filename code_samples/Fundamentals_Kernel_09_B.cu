```c
/*
Create a `__device__` function `void increment(int* x)` that increments the value pointed to by x. The kernel should call this on an element of an array in device memory.

Thoughts:
- We need a device function that takes a pointer to int and increments the dereferenced value.
- The kernel will launch many threads; each thread will determine its global index.
- For each valid index, the kernel will call `increment(&d_arr[idx])`.
- We'll allocate an array on the host, copy it to device, launch the kernel, copy back, and print the results.
- Basic CUDA error checking will be included for robustness.
- The code will be a complete .cu file ready to compile with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Device function that increments the value pointed to by x
__device__ void increment(int* x) {
    (*x)++;
}

// Kernel that calls increment on each element of the array
__global__ void inc_kernel(int* d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        increment(&d_arr[idx]);
    }
}

int main() {
    const int N = 10;
    int h_arr[N];

    // Initialize host array
    for (int i = 0; i < N; ++i) {
        h_arr[i] = i;
    }

    // Allocate device memory
    int* d_arr = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_arr, N * sizeof(int)));

    // Copy host array to device
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    inc_kernel<<<blocks, threadsPerBlock>>>(d_arr, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    printf("Array after increment:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Clean up
    CUDA_CHECK(cudaFree(d_arr));

    return 0;
}
```