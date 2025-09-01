```cuda
/*
Aim of the program:
Create a __device__ function add_two(int a, int b) that returns a+b. Call it from your kernel.

Thought process:
1. The requirement is straightforward: define a device function that adds two integers.
2. The kernel should call this function to compute results.
3. To demonstrate the device function, the kernel will add corresponding elements of two integer arrays.
4. The host code will allocate input arrays, initialize them, allocate output array, launch the kernel,
   copy the result back to host, and print the output.
5. We keep the example minimal: 10 elements, one thread per element.
6. Error checking is included for CUDA API calls for robustness.
7. The final output is a complete .cu file that compiles with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Device function that adds two integers
__device__ int add_two(int a, int b) {
    return a + b;
}

// Kernel that uses the device function to add elements of two arrays
__global__ void addKernel(const int* A, const int* B, int* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = add_two(A[idx], B[idx]);
    }
}

// Utility macro for checking CUDA errors
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main() {
    const int N = 10;
    const size_t size = N * sizeof(int);

    // Allocate host memory
    int h_A[N], h_B[N], h_C[N];
    for (int i = 0; i < N; ++i) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // Allocate device memory
    int *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel: one thread per element
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());  // Check for kernel launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Print results
    printf("Result of adding two arrays:\n");
    for (int i = 0; i < N; ++i) {
        printf("h_A[%d] + h_B[%d] = %d\n", i, i, h_C[i]);
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
```