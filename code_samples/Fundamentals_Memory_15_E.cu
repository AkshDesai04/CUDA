/*
Use `cudaMemcpyToSymbol` and `cudaMemcpyFromSymbol` to initialize and read the `__device__` variable from the host.

Thinking:
1. The program demonstrates how to copy data from host memory to a device variable using cudaMemcpyToSymbol and how to copy it back using cudaMemcpyFromSymbol.
2. A simple `__device__` integer variable `devVar` is declared and initialized on the device to zero.
3. On the host, an integer value is set and copied to `devVar` with `cudaMemcpyToSymbol`. The size argument is `sizeof(int)` and the offset is zero.
4. A kernel `incrementDevVar` is launched that adds a constant to `devVar`. This shows that the device variable can be accessed from the device side.
5. After the kernel, the host copies the updated value back from `devVar` using `cudaMemcpyFromSymbol` into a host variable.
6. The program prints the original value, the value after the kernel, and the final value read back to confirm that the copy operations succeeded.
7. Basic error checking is included to catch CUDA API failures.
8. The program compiles with `nvcc` and runs on any CUDA-capable device.
*/

#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

// Device variable to be accessed via cudaMemcpyToSymbol / FromSymbol
__device__ int devVar = 0;

// Simple kernel that increments the device variable
__global__ void incrementDevVar(int increment) {
    // Using atomicAdd for safety if multiple threads are used
    atomicAdd(&devVar, increment);
}

int main(void) {
    // Host variable to initialize the device variable
    int hostValToSet = 42;
    printf("Host: Setting device variable to %d\n", hostValToSet);

    // Copy value from host to device variable
    CHECK_CUDA(cudaMemcpyToSymbol(devVar, &hostValToSet, sizeof(int), 0, cudaMemcpyHostToDevice));

    // Launch kernel to increment devVar by 10
    int inc = 10;
    incrementDevVar<<<1, 1>>>(inc);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Read back the updated value from device
    int hostValFromDevice;
    CHECK_CUDA(cudaMemcpyFromSymbol(&hostValFromDevice, devVar, sizeof(int), 0, cudaMemcpyDeviceToHost));

    printf("Host: Value of device variable after kernel = %d\n", hostValFromDevice);

    // Clean up
    CHECK_CUDA(cudaDeviceReset());
    return 0;
}
