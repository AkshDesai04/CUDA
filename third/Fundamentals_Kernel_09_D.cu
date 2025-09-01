```cuda
/*
Aim: Can you call a __global__ function from a __device__ function? (No, unless using Dynamic Parallelism). Can you call a __device__ function from a __global__ function? (Yes). Explain this in a comment.

Thinking:
1. In CUDA, kernel functions are marked with __global__ and are invoked from the host with a launch configuration. They execute on the device but the call syntax is special.
2. Device functions (marked __device__) run on the device and can be called from other device or global functions. They are regular C++ functions with no launch configuration.
3. It is illegal to invoke a __global__ kernel from a __device__ function because a kernel launch requires the host to specify grid and block dimensions, which a device function cannot provide.
4. The exception is Dynamic Parallelism: a kernel can launch another kernel from device code if the device supports it, but this is a special feature and requires __launch_bounds__ or device runtime.
5. Therefore, the correct relationship is:
   - __global__ can call __device__ (straightforward).
   - __device__ cannot call __global__ unless using Dynamic Parallelism, which is optional and not part of standard CUDA programming.
The code below demonstrates a simple __device__ function used by a __global__ kernel. It also includes a comment illustrating the restriction.
*/

#include <stdio.h>

// A simple __device__ function that adds two integers
__device__ int addDevice(int a, int b) {
    return a + b;
}

// Kernel that calls the __device__ function
__global__ void kernelAdd(int *out, int a, int b) {
    // Only one thread performs the addition for demonstration
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int result = addDevice(a, b);   // __global__ -> __device__ call
        *out = result;
    }
}

int main() {
    int h_out = 0;
    int *d_out;
    size_t size = sizeof(int);

    // Allocate device memory
    cudaMalloc((void**)&d_out, size);

    // Launch kernel with a single thread
    kernelAdd<<<1, 1>>>(d_out, 3, 4);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(&h_out, d_out, size, cudaMemcpyDeviceToHost);

    printf("Result of 3 + 4 = %d\n", h_out);

    // Clean up
    cudaFree(d_out);
    return 0;
}
```