/*
Aim: Can tools like Valgrind detect CUDA device memory leaks? (Not directly, but `cuda-memcheck --leak-check` can).
Thinking: This simple CUDA program demonstrates that Valgrind cannot detect device memory leaks because Valgrind only tracks host memory. The program allocates memory on the GPU using `cudaMalloc`, launches a trivial kernel, and deliberately does NOT free the device memory. Running this program under `cuda-memcheck --leak-check=all` will report the leak, whereas running it under Valgrind will not report anything about GPU memory. The code includes basic error checking macros to make sure any CUDA API errors are caught. The goal is to illustrate the difference in leak detection capabilities between Valgrind and cuda-memcheck.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",        \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Simple kernel that writes to an array
__global__ void hello_kernel(int *d_arr, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] = idx;
    }
}

int main(void)
{
    const int N = 1024;
    int *d_arr = NULL;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_arr, N * sizeof(int)));

    // Launch kernel
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    hello_kernel<<<gridSize, blockSize>>>(d_arr, N);
    CUDA_CHECK(cudaGetLastError());

    // Intentionally NOT freeing d_arr to create a device memory leak
    // Comment the following line if you want to see the program run without leak
    // CUDA_CHECK(cudaFree(d_arr));

    // Optional: Synchronize to ensure kernel completion before exit
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Program finished. If run under cuda-memcheck, a device memory leak should be reported.\n");

    return 0;
}
