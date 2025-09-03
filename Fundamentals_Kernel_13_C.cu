```cpp
/*
Aim: Write a kernel that performs the operation in-place. It takes a single pointer `a` and computes `a[i] = a[i] + 10;`.

Thinking:
The requirement is to create a CUDA kernel that modifies an array in place by adding 10 to each element.
We need to pass the pointer `a` and the number of elements `n` so that the kernel knows how many indices to process.
A common pattern is to compute the global thread index as
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
and then guard against out-of-range indices:
    if (idx < n) { a[idx] += 10; }
The kernel will be launched with an appropriate grid and block size; a typical block size is 256 threads.
On the host side, we allocate an array, initialize it, copy to device memory, launch the kernel, copy the results back, and print them.
We include error checking after CUDA API calls for robustness.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",       \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Kernel that adds 10 to each element of the array in-place
__global__ void add10_kernel(int *a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] += 10;
    }
}

int main() {
    const int N = 1024;                // Number of elements
    const int bytes = N * sizeof(int);

    // Allocate host memory and initialize
    int *h_a = (int *)malloc(bytes);
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;  // Example data
    }

    // Allocate device memory
    int *d_a;
    CUDA_CHECK(cudaMalloc((void **)&d_a, bytes));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    // Launch kernel: 256 threads per block
    const int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    add10_kernel<<<blocks, threadsPerBlock>>>(d_a, N);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost));

    // Verify results (print first 10 elements)
    printf("First 10 elements after adding 10:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_a[i]);  // Should be i + 10
    }
    printf("\n");

    // Clean up
    CUDA_CHECK(cudaFree(d_a));
    free(h_a);

    return 0;
}
```