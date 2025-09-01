```cpp
/*
Aim of the program:
What happens if you have two bad kernel launches in a row, but only check for an error after the second one? The error from the first launch may be reported.

Thinking:
The CUDA runtime reports errors asynchronously. When a kernel launch fails (for example due to an illegal memory access), the device goes into an error state. Subsequent launches will not succeed until the error is cleared. If you perform two erroneous launches back‑to‑back and only query cudaGetLastError() after the second launch, the error you receive may actually be the one from the first launch – because the second launch might not produce a new error (the device was already in an error state) or it might be overridden. This program demonstrates that behaviour by:

1. Launching a kernel that writes out of bounds (bad kernel #1).
2. Launching a second kernel that also writes out of bounds (bad kernel #2).
3. Calling cudaGetLastError() only after the second launch.
4. Printing the error string. The output may show the error from the first launch.

The code intentionally does not correct the error after the first launch to keep the situation simple. It also includes helper functions for printing CUDA errors and for allocating device memory safely.

This example is for educational purposes and shows how CUDA error handling works.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Helper macro to check CUDA API calls
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",     \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// A kernel that performs an illegal memory access (writes out of bounds)
__global__ void badKernelOutOfBounds(int *data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Intentionally go out of bounds when idx >= size
    data[idx] = idx; // May write beyond the allocated array
}

// A kernel that performs division by zero (illegal instruction)
__global__ void badKernelDivByZero()
{
    int a = 1 / 0; // Division by zero causes illegal instruction
    (void)a; // Suppress unused variable warning
}

int main()
{
    const int arraySize = 10;
    int *d_array = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_array, arraySize * sizeof(int)));

    // First bad kernel launch (out-of-bounds write)
    int threadsPerBlock = 32;
    int blocks = 2; // 64 threads, but arraySize is only 10
    badKernelOutOfBounds<<<blocks, threadsPerBlock>>>(d_array, arraySize);
    // We deliberately do NOT call cudaGetLastError() here

    // Second bad kernel launch (division by zero)
    badKernelDivByZero<<<1, 1>>>();
    // Again, we do NOT call cudaGetLastError() here

    // Now check for errors AFTER the second launch
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        // Print the error message. It may report the error from the first launch
        printf("Error after second launch: %s\n", cudaGetErrorString(err));
    } else {
        printf("No errors reported after second launch.\n");
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_array));

    return 0;
}
```