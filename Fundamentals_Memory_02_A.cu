/*
Aim: Write a loop that iterates through the arrays and prints an error message if a mismatch is found, then breaks.

Thinking:
The assignment requests a CUDA program that demonstrates checking two arrays for equality and printing an error message when a mismatch occurs, then terminating the loop immediately.  
To satisfy this requirement we need:
1. Two host arrays (`h_a` and `h_b`) of a fixed size `N`.  
2. Allocate corresponding device arrays (`d_a` and `d_b`).  
3. Copy data from host to device.  
4. Launch a simple CUDA kernel that, for illustration, copies the contents of `d_a` into `d_b`.  
   (This step is optional for the mismatch test but demonstrates data movement.)  
5. Copy the result back to the host.  
6. Iterate over the arrays on the host, compare element-wise, print an error if a mismatch is found, and break the loop.  
7. Clean up resources and exit.  

The loop that does the comparison is purely on the host side because printing to `stdout` is a host operation.  
The CUDA kernel is trivial – a copy kernel – but is included to show typical CUDA data flow.  
We also include basic error checking for CUDA API calls.  
This program can be compiled with `nvcc -o compare_arrays compare_arrays.cu` and executed to see the mismatch detection in action.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

// Simple kernel that copies data from src to dst
__global__ void copyKernel(const int *src, int *dst, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        dst[idx] = src[idx];
    }
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                              \
    do                                                               \
    {                                                                \
        cudaError_t err = call;                                      \
        if (err != cudaSuccess)                                      \
        {                                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

int main(void)
{
    int h_a[N], h_b[N];
    int *d_a = NULL, *d_b = NULL;

    // Initialize host arrays
    for (int i = 0; i < N; ++i)
    {
        h_a[i] = i;            // all elements are 0..N-1
        h_b[i] = i;            // same as h_a initially
    }

    // Introduce a deliberate mismatch for demonstration
    h_b[100] = -1;              // mismatch at index 100

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_a, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_b, N * sizeof(int)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice));

    // Launch copy kernel to overwrite d_b with d_a
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    copyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_b, d_b, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Host loop to compare arrays and report mismatch
    for (int i = 0; i < N; ++i)
    {
        if (h_a[i] != h_b[i])
        {
            fprintf(stderr, "Mismatch found at index %d: h_a[%d] = %d, h_b[%d] = %d\n",
                    i, i, h_a[i], i, h_b[i]);
            break; // exit loop immediately after first mismatch
        }
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));

    printf("Array comparison completed.\n");
    return 0;
}
