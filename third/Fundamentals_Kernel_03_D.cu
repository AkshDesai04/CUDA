/*
Aim of the program:
Modify the kernel so that only threads with an even index print their ID.

Thinking:
The task is to create a simple CUDA program that demonstrates how to filter threads based on
their index before performing an operation—in this case, printing their ID.  
CUDA allows device-side printing via `printf`, but it must be enabled at compile time with the
`--compiler-options` flag or by using the appropriate compiler options in the Makefile or
NVCC command. Since this is a minimal example, we will rely on the default compilation settings
which enable device printf.

Key points considered:
1. Use a single block with a reasonable number of threads (e.g., 256) to keep output manageable.
2. In the kernel, use `threadIdx.x` as the thread identifier and check if it is even using
   the modulus operator.
3. Only threads that satisfy the condition will call `printf`.
4. After launching the kernel, the host program calls `cudaDeviceSynchronize()` to wait for all
   device work to finish and ensure that all printed messages appear before the program exits.
5. Basic error checking macro is provided for clarity, although for a simple demo it may be
   optional.

The program compiles to a .cu file and can be compiled with:
nvcc -arch=sm_30 -o even_threads even_threads.cu
or with any other architecture that supports device printf.

This file contains the entire code from start to finish, ready to be compiled and executed.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Simple macro for checking CUDA errors */
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",       \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

/* Kernel that prints the thread ID only if the index is even */
__global__ void printEvenThreadIDs()
{
    unsigned int tid = threadIdx.x;
    if ((tid % 2) == 0) {
        printf("Thread ID: %u\n", tid);
    }
}

int main(void)
{
    const int threadsPerBlock = 256;
    const int blocks = 1;

    /* Launch the kernel */
    printEvenThreadIDs<<<blocks, threadsPerBlock>>>();
    CHECK_CUDA(cudaGetLastError());

    /* Wait for the kernel to finish */
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("Kernel execution completed.\n");
    return 0;
}
