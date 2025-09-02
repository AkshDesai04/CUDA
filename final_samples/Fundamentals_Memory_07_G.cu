```c
/*
 * Aim: Write a __global__ function that takes an int* and dereferences it. Call it with a host pointer. Observe the error at runtime.
 *
 * Thinking:
 * 1. In CUDA, device kernels can only access device memory. If a host pointer is passed to a kernel,
 *    the pointer refers to host memory, which is not visible to the device. The device will attempt
 *    to read/write through that address, leading to an invalid memory access.
 * 2. We will write a simple kernel that dereferences the passed pointer and writes a value.
 * 3. In host code, we will allocate an int on the host, obtain its address, and pass that address
 *    to the kernel. This is illegal because the kernel expects a device pointer.
 * 4. When the kernel runs, it will crash or produce a runtime error (e.g., CUDA_ERROR_INVALID_DEVICE_FUNCTION
 *    or CUDA_ERROR_INVALID_VALUE). We will check for errors after kernel launch and after cudaDeviceSynchronize().
 * 5. This example demonstrates that passing host memory to device code is disallowed and will cause a runtime error.
 */

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that dereferences the passed pointer
__global__ void derefKernel(int *ptr) {
    // Attempt to write to the memory location
    *ptr = 42;
}

int main() {
    // Host integer
    int hostInt = 0;
    int *hostPtr = &hostInt;

    // Launch kernel with host pointer (incorrect)
    derefKernel<<<1, 1>>>(hostPtr);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    // Synchronize device to catch runtime errors from kernel execution
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
    } else {
        printf("Kernel executed successfully. hostInt = %d\n", hostInt);
    }

    return 0;
}
```