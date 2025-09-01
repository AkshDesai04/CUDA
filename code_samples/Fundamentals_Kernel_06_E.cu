/*
Write a kernel that takes two pointers, `int* in` and `int* out`. It should read the value from `in`, multiply it by 2, and write the result to `out`.
[Thinking]
The kernel should be written in CUDA C, so it will be compiled with nvcc.  
It takes two device pointers `int* in` and `int* out`.  For each thread we simply load the element from `in`, multiply it by two, and store it to the corresponding location in `out`.  The thread index is derived from `blockIdx.x`, `blockDim.x`, and `threadIdx.x` so that the kernel can work on arbitrary array sizes.  The host code demonstrates usage: allocate host arrays, copy them to device, launch the kernel, copy the result back, and print it.  Basic error checking is included via a helper macro `CUDA_CHECK`.  The program uses `cudaMalloc`, `cudaMemcpy`, `cudaFree`, and launches the kernel with a single block if the array is small, or multiple blocks otherwise.  This keeps the example compact while still showing a realistic CUDA workflow. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple macro to check CUDA errors
#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                          \
    } while (0)

// Kernel that doubles each element from 'in' and writes to 'out'
__global__ void doubleKernel(const int* in, int* out, size_t n)
{
    // Compute global thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        out[idx] = in[idx] * 2;
    }
}

int main()
{
    const size_t N = 10;                      // Number of elements
    const size_t bytes = N * sizeof(int);     // Bytes needed for each array

    // Host memory allocation
    int h_in[N], h_out[N];
    for (size_t i = 0; i < N; ++i)
        h_in[i] = (int)i;                     // Initialize input

    // Device memory allocation
    int *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_in, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_out, bytes));

    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Determine block and grid sizes
    const int threadsPerBlock = 256;
    int blocks = (int)((N + threadsPerBlock - 1) / threadsPerBlock);

    // Launch kernel
    doubleKernel<<<blocks, threadsPerBlock>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());   // Check for kernel launch errors
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel to finish

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // Print results
    printf("Input:  ");
    for (size_t i = 0; i < N; ++i)
        printf("%d ", h_in[i]);
    printf("\nOutput: ");
    for (size_t i = 0; i < N; ++i)
        printf("%d ", h_out[i]);
    printf("\n");

    // Free device memory
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
