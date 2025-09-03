```cpp
/*
Write code that triggers a synchronous error and use the CHECK macro to catch it immediately.

Thinking:
In CUDA, a synchronous error occurs during the kernel launch if the launch configuration
violates device limits (e.g., too many threads per block). Such errors are reported
synchronously by the launch API and can be detected immediately using
cudaGetLastError().

To implement this:
1. Define a simple kernel (doesn't need to do anything).
2. Define a CHECK macro that checks a cudaError_t and prints the file, line, and error message
   if the error is not cudaSuccess.
3. In main(), launch the kernel with a block size that exceeds the device's maximum
   threads per block (e.g., 1024 on a device that supports only 512). This triggers a
   synchronous error of type cudaErrorInvalidConfiguration.
4. Immediately after the launch, call CHECK(cudaGetLastError()); to catch and handle the
   error.

The program will terminate with an error message indicating the synchronous error.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Macro to check CUDA API calls and report errors immediately
#define CHECK(call)                                      \
    do {                                                 \
        cudaError_t err = call;                          \
        if (err != cudaSuccess) {                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                          \
        }                                                \
    } while (0)

// Simple kernel that does nothing
__global__ void dummyKernel()
{
    // No operation
}

int main(void)
{
    // Get device properties to know max threads per block
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("Device: %s\n", prop.name);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);

    // Intentionally launch with too many threads per block to trigger a synchronous error
    int threadsPerBlock = prop.maxThreadsPerBlock + 256; // exceed limit
    dummyKernel<<<1, threadsPerBlock>>>();

    // Immediately check for errors from the launch
    CHECK(cudaGetLastError());

    // If no error (which shouldn't happen), we print a success message
    printf("Kernel launched successfully (unexpected).\n");

    return 0;
}
```