```cuda
/*
Create a `__host__ __device__` function (a function that can be called from both host and device). Call it from both `main` and your kernel.

Thinking process:
- The goal is to demonstrate a function that is callable from both the host (CPU) and the device (GPU). 
- In CUDA, such a function must be annotated with `__host__ __device__`.
- To illustrate usage, the function will perform a simple arithmetic operation (e.g., summing two integers).
- From the host side (`main`), we will call the function directly and print the result.
- On the device side, we will launch a kernel that also calls the same function to compute results for each thread.
- The kernel will write the results to a device array, which is then copied back to the host for verification.
- The code includes basic error checking for CUDA API calls.
- The program uses only standard CUDA runtime API, making it straightforward to compile and run with `nvcc`.
- The code is self‑contained and compiles as a single `.cu` file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// __host__ __device__ function that can be called from both host and device.
// It simply adds two integers and returns the sum.
__host__ __device__
int add(int a, int b) {
    return a + b;
}

// CUDA kernel that calls the __host__ __device__ function `add`.
__global__
void addKernel(int *out, int a, int b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread computes the sum of a and b and stores it in the output array.
    out[idx] = add(a, b);
}

int main() {
    const int N = 10;                     // Number of elements
    const int a = 5, b = 7;               // Test values

    // Allocate host memory
    int h_out[N];
    // Allocate device memory
    int *d_out;
    cudaError_t err = cudaMalloc((void**)&d_out, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Call the __host__ __device__ function from host
    int hostResult = add(a, b);
    printf("Host called add(%d, %d) = %d\n", a, b, hostResult);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, a, b);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        return 1;
    }

    // Copy results back to host
    err = cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        return 1;
    }

    // Print results from device
    printf("Device results:\n");
    for (int i = 0; i < N; ++i) {
        printf("Thread %d: %d\n", i, h_out[i]);
    }

    // Clean up
    cudaFree(d_out);
    return 0;
}
```