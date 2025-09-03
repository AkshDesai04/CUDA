/*
Write a kernel that swaps the values between two arrays: `int temp = a[i]; a[i] = b[i]; b[i] = temp;`.
The goal of this program is to demonstrate a simple element‑wise swap operation between two arrays on the GPU.
To accomplish this, we write a CUDA kernel that each thread handles one index i.  
We use a typical 1‑D grid and block layout, ensuring that the kernel covers all n elements by launching enough blocks.  
The host code allocates and initializes two integer arrays on the CPU, copies them to the GPU, calls the kernel, and copies the results back.  
After the swap we print both arrays to verify that the operation was successful.  
The program includes basic error checking for CUDA API calls and kernel launch errors.  
All code is self‑contained in a single .cu file and can be compiled with nvcc (e.g. `nvcc -o swap_arrays swap_arrays.cu`).
*/

#include <cstdio>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a CUDA API call
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Kernel that swaps the elements of two integer arrays
__global__ void swapKernel(int* a, int* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int temp = a[i];
        a[i] = b[i];
        b[i] = temp;
    }
}

int main() {
    const int N = 10;                       // Number of elements
    const size_t size = N * sizeof(int);    // Size in bytes

    // Allocate host arrays
    int h_a[N], h_b[N];
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;           // 0, 1, 2, ...
        h_b[i] = i + 10;      // 10, 11, 12, ...
    }

    // Device pointers
    int *d_a, *d_b;
    CHECK_CUDA(cudaMalloc((void**)&d_a, size));
    CHECK_CUDA(cudaMalloc((void**)&d_b, size));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    swapKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, N);
    CHECK_CUDA(cudaGetLastError());          // Check for kernel launch errors
    CHECK_CUDA(cudaDeviceSynchronize());     // Ensure kernel finished

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));

    // Print results
    printf("Array a after swap: ");
    for (int i = 0; i < N; ++i) printf("%d ", h_a[i]);
    printf("\n");

    printf("Array b after swap: ");
    for (int i = 0; i < N; ++i) printf("%d ", h_b[i]);
    printf("\n");

    return 0;
}
