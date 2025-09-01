```cuda
/*
Aim of the program: Create a header file (my_device_functions.cuh) that contains a __device__ function, include this header in the main .cu file, and demonstrate its usage in a simple CUDA kernel that adds two arrays element-wise.

Thinking:
1. Define a device function `add` in the header that simply returns the sum of two integers.
2. Protect the header with include guards.
3. In the main .cu file, include this header, launch a kernel that uses `add`, and perform a basic array addition.
4. Allocate host and device memory, copy data, launch the kernel, copy results back, and print them for verification.
5. Ensure all necessary headers are included and use standard CUDA conventions.

Header file content (to be placed in a separate file named `my_device_functions.cuh`):
#ifndef MY_DEVICE_FUNCTIONS_CUH
#define MY_DEVICE_FUNCTIONS_CUH

__device__ int add(int x, int y) {
    return x + y;
}

#endif // MY_DEVICE_FUNCTIONS_CUH
*/

#include "my_device_functions.cuh"
#include <stdio.h>
#include <stdlib.h>

#define N 10
#define BLOCK_SIZE 256

__global__ void addKernel(const int *a, const int *b, int *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = add(a[idx], b[idx]);  // use the __device__ function from the header
    }
}

int main(void) {
    int h_a[N], h_b[N], h_c[N];
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = N - i;
    }

    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_c, N * sizeof(int));

    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    addKernel<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result of element-wise addition:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
```