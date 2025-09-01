```cuda
/* 
Aim: Can a kernel launch fail with `cudaErrorInvalidDevice`? (Unlikely, as the context is already established).

Thinking:
- The CUDA API reports `cudaErrorInvalidDevice` when a device ID is invalid or when the context is not available.
- A kernel launch fails with this error if the current device context has been destroyed (e.g., by `cudaDeviceReset`) or if the device ID is out of range.
- To demonstrate this, we:
  1. Set the device to a valid ID (0) and create a simple context.
  2. Allocate device memory and copy data from host to device.
  3. Reset the device, which destroys the context.
  4. Attempt to launch a trivial kernel. Since the context no longer exists, the launch should fail with `cudaErrorInvalidDevice`.
- The program checks and prints the error code after the kernel launch, confirming the expected failure.
- This illustrates that while a context is usually valid, explicit resets can invalidate the device, causing kernel launches to fail with `cudaErrorInvalidDevice`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void dummyKernel(int *d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d_arr[idx] = d_arr[idx] * 2;
}

int main() {
    int device = 0;
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        printf("Failed to set device %d: %s\n", device, cudaGetErrorString(err));
        return -1;
    }

    const int N = 10;
    int h_arr[N];
    for (int i = 0; i < N; ++i) h_arr[i] = i;

    int *d_arr = nullptr;
    err = cudaMalloc((void**)&d_arr, N * sizeof(int));
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return -1;
    }

    // Destroy the current context by resetting the device
    err = cudaDeviceReset();
    if (err != cudaSuccess) {
        printf("cudaDeviceReset failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return -1;
    }

    // Attempt to launch kernel after context reset
    dim3 block(4);
    dim3 grid((N + block.x - 1) / block.x);
    dummyKernel<<<grid, block>>>(d_arr, N);

    err = cudaGetLastError();  // Check kernel launch status
    if (err != cudaSuccess) {
        printf("Kernel launch failed with error: %s (%d)\n", cudaGetErrorString(err), err);
    } else {
        printf("Kernel launched successfully (unexpected).\n");
    }

    // Clean up (in practice, the context is gone, but freeing pointers is safe)
    cudaFree(d_arr);
    return 0;
}
```