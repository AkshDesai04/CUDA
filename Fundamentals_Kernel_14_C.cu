```cpp
/*
Aim of the program:
Use `cudaPeekAtLastError()` immediately after the kernel launch to get the error code without waiting for a synchronization.

Thinking:
The goal is to demonstrate how `cudaPeekAtLastError()` can be used to capture asynchronous kernel launch errors
right after the launch, without forcing a device synchronization. To trigger a launch error we deliberately launch
the kernel with a block dimension that exceeds the maximum supported threads per block on the device
(typically 1024). This causes the launch to fail and sets the last error code to `cudaErrorInvalidValue`.
`cudaPeekAtLastError()` retrieves this error code without resetting it, allowing us to inspect it immediately
after the launch. We also call `cudaGetLastError()` afterwards to show that it clears the error status.
Finally, we perform a `cudaDeviceSynchronize()` to ensure that any deferred errors (none in this case)
are caught, and we print the resulting error status for completeness. This pattern is useful when you
want to detect launch failures early in the code flow without incurring the overhead of synchronizing
the device.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// A simple kernel that writes to a device array.
__global__ void dummyKernel(int *data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx;
    }
}

int main()
{
    const int N = 256;
    int *d_data = NULL;
    cudaError_t err;

    // Allocate device memory.
    err = cudaMalloc((void**)&d_data, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Intentionally launch with an invalid block size (exceeds max threads per block).
    dim3 blockSize(2048); // Typically > 1024 -> launch error
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    // Launch the kernel.
    dummyKernel<<<gridSize, blockSize>>>(d_data, N);

    // Immediately check for launch error without synchronization.
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("cudaPeekAtLastError after launch: %s\n", cudaGetErrorString(err));
    } else {
        printf("cudaPeekAtLastError after launch: No error\n");
    }

    // Retrieve and clear the error status.
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cudaGetLastError after launch: %s\n", cudaGetErrorString(err));
    } else {
        printf("cudaGetLastError after launch: No error\n");
    }

    // Synchronize to catch any deferred errors (none expected here).
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("cudaDeviceSynchronize error: %s\n", cudaGetErrorString(err));
    } else {
        printf("cudaDeviceSynchronize: No error\n");
    }

    // Clean up.
    cudaFree(d_data);

    return 0;
}
```