/*
Use your macro on a call that is guaranteed to succeed, like `cudaGetDeviceCount`, to ensure it doesn't trigger incorrectly.

Thoughts:
1. We need a macro that checks the return status of a CUDA API call.
2. The macro should capture the return value, compare it to cudaSuccess, and if not successful, print an error message and exit.
3. We then use this macro on a call that we know will succeed, namely cudaGetDeviceCount, to confirm the macro does not falsely trigger.
4. For completeness, the program will print the number of CUDA devices found.
5. Since this is a .cu file, we include the necessary headers and use nvcc compatible code.
6. The macro will be named CUDA_CHECK for clarity.
7. We'll use a simple main function that demonstrates the macro.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Macro to check CUDA API calls
#define CUDA_CHECK(call)                                                   \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                            \
        }                                                                   \
    } while (0)

int main(void)
{
    int deviceCount = 0;

    // Use the macro on a guaranteed successful call
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    printf("Number of CUDA devices available: %d\n", deviceCount);

    // Optionally, perform a minimal kernel launch to show program flow
    // (not necessary for the macro test, but demonstrates a typical usage)
    const int N = 256;
    int *d_array;
    CUDA_CHECK(cudaMalloc((void**)&d_array, N * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_array, 0, N * sizeof(int)));
    CUDA_CHECK(cudaFree(d_array));

    return 0;
}
