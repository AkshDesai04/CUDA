/*
Write a program that allocates a large amount of memory on the device (e.g., 512MB) and then immediately frees it using `cudaFree`. Use robust error checking for all API calls to ensure the allocation and deallocation were successful.

Thinking:
- Use CUDA Runtime API (cudaMalloc and cudaFree) for simplicity.
- Define a helper function `checkCudaError` to wrap API calls and print descriptive error messages, then exit on failure.
- Allocate 512 * 1024 * 1024 bytes (~512MB) of device memory.
- Immediately free the memory after checking that the allocation succeeded.
- Include minimal includes: <cuda_runtime.h> for CUDA API and <stdio.h> for printing.
- Ensure the program compiles with nvcc and runs on any GPU with at least 512MB free device memory.
- Provide a main function that returns 0 on success.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/* Helper function to check CUDA API return status */
static void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error [%s]: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    size_t bytes = 512ULL * 1024ULL * 1024ULL; /* 512 MB */
    void *devPtr = NULL;

    /* Allocate device memory */
    cudaError_t err = cudaMalloc(&devPtr, bytes);
    checkCudaError(err, "cudaMalloc");

    printf("Successfully allocated %zu bytes on device.\n", bytes);

    /* Free device memory */
    err = cudaFree(devPtr);
    checkCudaError(err, "cudaFree");

    printf("Successfully freed device memory.\n");

    return 0;
}
