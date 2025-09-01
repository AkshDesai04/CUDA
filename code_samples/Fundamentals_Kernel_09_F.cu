/*
Place your `__device__` function definition after the `__global__` kernel that calls it. Does it compile? (It might require a forward declaration).

Thinking:
- CUDA requires that a __device__ function be declared before it is used in a __global__ kernel.
- If the definition appears after the kernel, the compiler will not know about it unless a forward declaration is provided.
- Therefore, I will provide a forward declaration of the device function before the kernel.
- The kernel will call this device function.
- After the kernel, I will define the device function.
- The program will perform a simple element-wise addition of two arrays to demonstrate that the device function is called correctly.
- The code will allocate host and device memory, copy data, launch the kernel, copy back results, and print them.
- If the forward declaration is omitted, the code would fail to compile.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 10

// Forward declaration of the device function
__device__ int add(int a, int b);

// Kernel that calls the device function
__global__ void vectorAdd(const int *A, const int *B, int *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = add(A[idx], B[idx]); // call to device function
    }
}

// Definition of the device function (after the kernel)
__device__ int add(int a, int b) {
    return a + b;
}

int main() {
    int hostA[N], hostB[N], hostC[N];
    int *devA, *devB, *devC;

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        hostA[i] = i;
        hostB[i] = 2 * i;
    }

    // Allocate device memory
    cudaMalloc((void**)&devA, N * sizeof(int));
    cudaMalloc((void**)&devB, N * sizeof(int));
    cudaMalloc((void**)&devC, N * sizeof(int));

    // Copy data to device
    cudaMemcpy(devA, hostA, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, hostB, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(devA, devB, devC, N);

    // Copy result back to host
    cudaMemcpy(hostC, devC, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    printf("Result of A + B:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d + %d = %d\n", hostA[i], hostB[i], hostC[i]);
    }

    // Free device memory
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    return 0;
}
