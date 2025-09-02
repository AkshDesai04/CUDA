```cpp
/*
Demonstrate its use by launching a kernel with an invalid configuration (e.g., too many threads) and catching the error with `CHECK(cudaPeekAtLastError())`.

The idea of this program is to illustrate how CUDA error handling works when an invalid kernel launch configuration is used. CUDA allows the programmer to query device properties, such as the maximum number of threads per block that a particular GPU architecture supports. By intentionally exceeding this limit, we create a situation where the kernel launch will fail. After the launch attempt, we immediately call `cudaPeekAtLastError()` to fetch the error state set by the runtime. The `CHECK` macro is a simple helper that evaluates a CUDA runtime call and prints a meaningful error message if the call fails. By using `CHECK(cudaPeekAtLastError())`, we demonstrate how to capture and report launch-time errors without requiring a separate call to `cudaGetLastError()`. The program prints the error string returned by the CUDA runtime, confirming that the launch was indeed invalid.

The program follows these steps:
1. Define a `CHECK` macro that handles CUDA error codes.
2. Create a trivial kernel (`dummyKernel`) that performs no work.
3. In `main`, query the device for `maxThreadsPerBlock`.
4. Set an intentionally invalid block size (`maxThreadsPerBlock + 1`).
5. Launch `dummyKernel` with the invalid block size.
6. Immediately call `CHECK(cudaPeekAtLastError())` to catch the launch error.
7. Print the error string to verify the failure.
8. Exit the program.

This demonstrates how to catch kernel launch errors early and respond appropriately.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Macro to check CUDA runtime calls for errors
#define CHECK(call)                                                    \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// A trivial kernel that does nothing
__global__ void dummyKernel()
{
    // No operation
}

int main()
{
    // Query device properties to determine maximum threads per block
    cudaDeviceProp devProp;
    CHECK(cudaGetDeviceProperties(&devProp, 0));

    printf("Device: %s\n", devProp.name);
    printf("Maximum threads per block: %d\n", devProp.maxThreadsPerBlock);

    // Intentionally set an invalid block size: exceed the maximum
    int invalidBlockSize = devProp.maxThreadsPerBlock + 1;

    printf("Launching kernel with %d threads per block (invalid)...\n", invalidBlockSize);

    // Launch the kernel with the invalid configuration
    dummyKernel<<<1, invalidBlockSize>>>();

    // Immediately check for launch errors
    CHECK(cudaPeekAtLastError());

    // If the program reaches this point, the error was not caught
    printf("Kernel launch succeeded (unexpected).\n");

    return 0;
}
```