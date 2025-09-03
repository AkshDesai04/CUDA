/*
Use your macro on a call that is guaranteed to succeed, like `cudaGetDeviceCount`, to ensure it doesn't trigger incorrectly.
The aim of this program is to demonstrate a macro that checks CUDA API calls for errors and reports them. We want to confirm that the macro does not falsely trigger when the CUDA function succeeds. 
We will create a macro `CUDA_CHECK` that executes a CUDA call, checks the return status, and if it's not `cudaSuccess`, prints an error message with file and line information and exits the program. 
In `main`, we will call `cudaGetDeviceCount`, which is a function that is very likely to succeed on any CUDA-capable system. By wrapping this call with `CUDA_CHECK`, we can verify that the macro does not erroneously report an error when the call succeeds. 
After retrieving the device count, we will print it out and exit successfully. This minimal program therefore serves as a test harness for the error-checking macro.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro to check CUDA errors */
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

int main(void)
{
    int deviceCount = 0;

    /* This call is expected to succeed; macro should not trigger */
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    printf("Number of CUDA devices available: %d\n", deviceCount);

    return 0;
}
