/*
Aim: Write code that triggers a synchronous error and use the `CHECK` macro to catch it immediately.

Thinking:
To trigger a synchronous error we need to call a CUDA runtime API that returns an error code immediately upon failure. One common example is attempting to copy more data than was allocated to a device pointer. This will cause `cudaMemcpy` to return an error (`cudaErrorInvalidValue`) synchronously. We wrap the CUDA API calls with a `CHECK` macro that verifies the return status and prints an error message before exiting. This ensures the error is caught and handled right away.

The `CHECK` macro and helper function `gpuAssert` are defined to print the error string, file, and line number, and then terminate the program if any CUDA call fails.

In `main`, we:
1. Allocate a host buffer of 512 bytes.
2. Allocate a device buffer of 512 bytes.
3. Intentionally request a copy of 1024 bytes from host to device, exceeding the allocated size.
4. Wrap this `cudaMemcpy` call with `CHECK` so the error is caught immediately.

The program will terminate with a clear error message pointing to the offending line.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

int main()
{
    const size_t hostSize = 512;
    const size_t devSize = 512;
    const size_t copySize = 1024; // Intentional error: copy more than allocated

    // Allocate host memory
    unsigned char *h_data = (unsigned char*)malloc(hostSize);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    unsigned char *d_data;
    CHECK(cudaMalloc((void**)&d_data, devSize));

    // Intentionally trigger a synchronous error
    CHECK(cudaMemcpy(d_data, h_data, copySize, cudaMemcpyHostToDevice));

    // This line will not be reached due to the error above
    printf("Memcpy succeeded unexpectedly.\n");

    // Clean up (unreachable in this example)
    cudaFree(d_data);
    free(h_data);

    return 0;
}
