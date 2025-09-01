/*
Aim of the program:
Use `sizeof()` on the host and device to print the size of the struct and verify they are the same.

Thinking:
To demonstrate that the size of a struct is identical on both host and device, we:
1. Define a simple struct with a mix of data types.
2. Use `sizeof()` on the host side to get the size.
3. Allocate a device variable to hold the size from the device.
4. Launch a tiny kernel that writes `sizeof(MyStruct)` into that device variable.
5. Copy the device size back to the host.
6. Print both host and device sizes and check that they match.
Additionally, we use `printf` in the kernel to illustrate printing from device, but the main verification is done by copying the size back to host and comparing.
*/

#include <stdio.h>
#include <cuda_runtime.h>

struct MyStruct {
    int    a;
    double b;
    char   c[3];
};

__device__ size_t devSize;

__global__ void computeSizeKernel() {
    // Compute the size of MyStruct on the device
    devSize = sizeof(MyStruct);
    // Optional: print from device
    printf("Device size of MyStruct: %zu bytes\n", devSize);
}

int main() {
    // Host size
    size_t hostSize = sizeof(MyStruct);
    printf("Host size of MyStruct: %zu bytes\n", hostSize);

    // Launch kernel to compute device size
    computeSizeKernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    // Copy device size back to host
    size_t deviceSize;
    cudaMemcpy(&deviceSize, &devSize, sizeof(size_t), cudaMemcpyDeviceToHost);

    // Verify that host and device sizes match
    if (hostSize == deviceSize) {
        printf("Verification succeeded: host and device sizes match.\n");
    } else {
        printf("Verification failed: host size = %zu, device size = %zu\n", hostSize, deviceSize);
    }

    // Clean up
    cudaDeviceReset();
    return 0;
}
