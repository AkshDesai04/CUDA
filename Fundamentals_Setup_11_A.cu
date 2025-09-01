```c
/* 
Aim of the program: Modify the program to print 'Hello from host!' before the kernel launch and 'Kernel finished!' after the kernel launch.

Thinking:
1. We need a simple CUDA program that launches a kernel. 
2. Before launching the kernel, we will print "Hello from host!" using printf.
3. We will launch a trivial kernel (e.g., it writes values to an array or does nothing) just to demonstrate kernel execution.
4. After launching the kernel, we will synchronize the device with cudaDeviceSynchronize() to ensure the kernel has finished executing before proceeding.
5. Once synchronization completes, we will print "Kernel finished!" to indicate that the kernel has completed.
6. We'll include basic error checking after the kernel launch and synchronization to catch any potential issues.
7. The code will be self-contained in a single .cu file and use standard CUDA C constructs.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that writes a value into an array
__global__ void dummyKernel(int *d_arr, int val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] = val;
    }
}

// Helper macro for error checking
#define CUDA_CHECK(err) do { \
    cudaError_t err__ = (err); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s (errnum=%d) at %s:%d\n", \
                cudaGetErrorString(err__), err__, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

int main(void) {
    const int N = 10;
    int h_arr[N];
    int *d_arr = NULL;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_arr, N * sizeof(int)));

    // Print message before kernel launch
    printf("Hello from host!\n");

    // Launch kernel with 1 block of N threads
    dummyKernel<<<1, N>>>(d_arr, 42, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Synchronize to ensure kernel has finished
    CUDA_CHECK(cudaDeviceSynchronize());

    // Print message after kernel has finished
    printf("Kernel finished!\n");

    // Copy result back to host (optional)
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print the array to verify
    printf("Device array contents: ");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Clean up
    CUDA_CHECK(cudaFree(d_arr));

    return 0;
}
```