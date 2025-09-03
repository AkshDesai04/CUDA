/*
Aim: Call `cudaGetLastError` after a failed malloc to retrieve the error.

Thinking:
The task is to demonstrate the use of cudaGetLastError() after a failed
cudaMalloc() call. To force a failure, we can attempt to allocate an
extremely large amount of device memory (e.g. 1 TB). This allocation will
almost certainly fail on any normal GPU. After the call we then invoke
cudaGetLastError() to retrieve the error code that was set by the failed
allocation. We convert this error code to a human‑readable string using
cudaGetErrorString() and print it out. This example does not use the
return value of cudaMalloc() directly, but rather relies on the global
error state that cudaGetLastError() reports. The program is a complete
CUDA C++ source file (.cu) that can be compiled with nvcc and run on a
device with the CUDA runtime library.
*/

#include <cstdio>
#include <cuda_runtime.h>

int main()
{
    int *d_ptr = nullptr;
    size_t hugeSize = static_cast<size_t>(1ULL << 40); // 1 TB

    // Intentionally cause a failure by requesting an unrealistically large allocation
    cudaMalloc((void**)&d_ptr, hugeSize);

    // Retrieve the last error that occurred on the device
    cudaError_t err = cudaGetLastError();

    // Print the error code and its description
    printf("cudaMalloc failed with error code %d: %s\n", err, cudaGetErrorString(err));

    // Clean up: reset device (optional)
    cudaDeviceReset();

    return 0;
}
