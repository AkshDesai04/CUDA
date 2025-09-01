```cuda
/*
Aim of the program:
Use `cudaMemcpyToSymbol` and `cudaMemcpyFromSymbol` to initialize and read the `__device__` variable from the host.

Thinking:
- The goal is to demonstrate how to transfer data to and from a device-side variable that is declared with `__device__` using the symbol-based copy functions.
- We declare a global device variable `deviceVar`.
- On the host we have a local integer `hostVar` that we set to a known value.
- `cudaMemcpyToSymbol` copies this value into `deviceVar`.
- We launch a simple kernel (even if it does nothing) to ensure any device-side side effects are completed before we copy back.
- After the kernel, `cudaMemcpyFromSymbol` copies the value from `deviceVar` back into `hostVar`.
- Finally we print the host value to verify that the round‑trip worked.
- Proper error checking is included for all CUDA API calls to aid debugging.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Device variable
__device__ int deviceVar;

// Simple kernel (does nothing but ensures synchronization)
__global__ void dummyKernel() {
    // Could perform operations on deviceVar if desired
}

int main() {
    // Host variable to initialize deviceVar
    int hostVar = 42;
    printf("Host before copy to device: %d\n", hostVar);

    // Copy hostVar to device variable deviceVar
    cudaError_t err = cudaMemcpyToSymbol(deviceVar, &hostVar, sizeof(int), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Launch dummy kernel to enforce ordering (optional)
    dummyKernel<<<1, 1>>>();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy back from deviceVar to hostVar
    err = cudaMemcpyFromSymbol(&hostVar, deviceVar, sizeof(int), 0, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyFromSymbol failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Host after copy from device: %d\n", hostVar);

    return 0;
}
```