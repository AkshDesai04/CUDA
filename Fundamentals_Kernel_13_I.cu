/*
Aim of the program: 
Write a kernel that swaps the values between two arrays: `int temp = a[i]; a[i] = b[i]; b[i] = temp;`.

Thinking:
1. Need to write a CUDA kernel that takes two integer pointers a and b and swaps elements at the same index i.  
2. Use a simple element-wise loop: `int temp = a[i]; a[i] = b[i]; b[i] = temp;`.  
3. Launch the kernel with enough threads to cover all array elements; guard against out-of-bounds indices.  
4. In the host code, allocate two host arrays, initialize them with sample values, copy them to device memory, launch the kernel, copy the results back, and print them to verify the swap.  
5. Include basic CUDA error checking for good practice.  
6. Wrap everything in a single .cu file that can be compiled with `nvcc`.  
7. Ensure the code is self‑contained and uses only the C/C++ standard library and CUDA runtime API.  
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n",            \
                    __FILE__, __LINE__, err, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Kernel that swaps values between two arrays element-wise
__global__ void swapKernel(int *a, int *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int temp = a[i];
        a[i] = b[i];
        b[i] = temp;
    }
}

int main(void) {
    const int N = 10;                      // Number of elements in each array
    size_t size = N * sizeof(int);

    // Allocate host memory
    int *h_a = (int *)malloc(size);
    int *h_b = (int *)malloc(size);
    if (!h_a || !h_b) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = i + 1;          // 1, 2, 3, ...
        h_b[i] = (i + 1) * 10;   // 10, 20, 30, ...
    }

    // Allocate device memory
    int *d_a, *d_b;
    CHECK_CUDA(cudaMalloc((void **)&d_a, size));
    CHECK_CUDA(cudaMalloc((void **)&d_b, size));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Define block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    swapKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost));

    // Print results
    printf("After swapping:\n");
    for (int i = 0; i < N; ++i) {
        printf("Index %2d: a = %3d, b = %3d\n", i, h_a[i], h_b[i]);
    }

    // Clean up
    free(h_a);
    free(h_b);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
