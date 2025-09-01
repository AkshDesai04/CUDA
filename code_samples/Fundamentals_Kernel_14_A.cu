/*
Trigger `cudaErrorInvalidConfiguration` by requesting more shared memory per block than is available on the device.

This program demonstrates how to force a CUDA runtime error of type cudaErrorInvalidConfiguration by asking for more per‑block shared memory than the GPU can provide. The steps are:

1. Query the device for its maximum shared memory per block (deviceProp.sharedMemPerBlock).
2. Compute a requested shared memory size that is intentionally larger (e.g., add 1KB to the maximum).
3. Launch a simple kernel with the over‑requested shared memory size using the <<<grid, block, sharedMemSize>>> syntax.
4. After the launch, immediately check the error with cudaGetLastError().
   Because the requested shared memory exceeds the limit, the runtime will return cudaErrorInvalidConfiguration.
5. The program prints the error string to demonstrate the failure.

The kernel does not need to do any work; its purpose is solely to trigger the configuration error. The program is fully self‑contained and can be compiled with nvcc:

    nvcc -o trigger_shared_mem_error trigger_shared_mem_error.cu

Running the executable will print the error message:

    Error launching kernel: invalid configuration argument

This satisfies the requirement of demonstrating the error condition while keeping the code minimal.
*/

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void dummyKernel(void) {
    // Intentionally empty kernel
}

int main(void) {
    cudaDeviceProp deviceProp;
    int device = 0;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceProp, device);

    size_t maxSharedMem = deviceProp.sharedMemPerBlock;
    size_t requestedSharedMem = maxSharedMem + 1024; // 1KB over the limit

    printf("Device %d max shared memory per block: %zu bytes\n", device, maxSharedMem);
    printf("Requesting %zu bytes of shared memory per block.\n", requestedSharedMem);

    // Launch kernel with over‑requested shared memory
    dummyKernel<<<1, 1, requestedSharedMem>>>();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error launching kernel: %s\n", cudaGetErrorString(err));
    } else {
        // If somehow it didn't error, wait for completion
        cudaDeviceSynchronize();
        printf("Kernel launched successfully (unexpected).\n");
    }

    return 0;
}
