/*
Aim: Call `cudaGetLastError` after a failed malloc to retrieve the error.
Thinking:
1. The purpose of this program is to demonstrate how to catch a CUDA runtime error that occurs during a failed memory allocation (cudaMalloc) by querying the last error with cudaGetLastError().
2. We intentionally request an extremely large block of device memory (e.g., 4 petabytes) which is guaranteed to fail on any realistic GPU.
3. Immediately after the failed cudaMalloc, we call cudaGetLastError() to fetch the error status and print a human‑readable message using cudaGetErrorString().
4. To illustrate that cudaGetLastError clears the error flag, we perform a subsequent, valid allocation and check that no error remains.
5. The program uses standard CUDA runtime API functions and includes minimal error handling for clarity.
6. No external libraries or complex logic are required; the program focuses solely on error retrieval after a failed malloc.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

int main(void)
{
    cudaError_t err;

    // Intentionally allocate an enormous amount of memory to trigger failure
    size_t hugeSize = (size_t)4 * 1024 * 1024 * 1024 * 1024ULL; // 4 TB
    void *devPtr = NULL;
    err = cudaMalloc(&devPtr, hugeSize);
    if (err != cudaSuccess)
    {
        // cudaMalloc failed; retrieve the error using cudaGetLastError()
        err = cudaGetLastError();
        printf("cudaMalloc failed as expected. cudaGetLastError returned: %s\n",
               cudaGetErrorString(err));
    }
    else
    {
        // This block should never be executed
        printf("Unexpected success allocating %zu bytes.\n", hugeSize);
        cudaFree(devPtr);
    }

    // Verify that cudaGetLastError cleared the error state
    err = cudaMalloc(&devPtr, 1024); // Small allocation that should succeed
    if (err != cudaSuccess)
    {
        printf("Second cudaMalloc unexpectedly failed: %s\n",
               cudaGetErrorString(err));
    }
    else
    {
        // Successful allocation; now check the error state
        err = cudaGetLastError();
        if (err == cudaSuccess)
        {
            printf("After successful cudaMalloc, cudaGetLastError reports no error as expected.\n");
        }
        else
        {
            printf("After successful cudaMalloc, cudaGetLastError returned: %s\n",
                   cudaGetErrorString(err));
        }
        cudaFree(devPtr);
    }

    // Reset the device to clean up
    cudaDeviceReset();
    return 0;
}
