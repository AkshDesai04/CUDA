/*
Aim of program:
Can you call a `__global__` function from a `__device__` function? (No, unless using Dynamic Parallelism). Can you call a `__device__` function from a `__global__` function? (Yes). Explain this in a comment.

Thinking:
- In CUDA, `__global__` functions are kernels that must be launched from host code. They cannot be called directly from device code because a device does not have a launch context for kernels unless dynamic parallelism is enabled (which is a feature that allows a kernel to launch other kernels).
- Therefore, attempting to call a `__global__` function from a `__device__` function without dynamic parallelism will result in a compile error.
- On the other hand, `__device__` functions are ordinary functions that run on the GPU and can be called from any other device or global function. A kernel (`__global__`) can freely call a `__device__` function.
- This program demonstrates these concepts: it defines a `__device__` function and a kernel that calls it. It also contains commented-out code that would try to call a kernel from a device function, illustrating that this is not allowed (unless dynamic parallelism is used). The program prints output from the kernel to show that the device function was invoked correctly.
*/

#include <cstdio>

// A __device__ function that can be called from device or global functions
__device__ int deviceAdd(int a, int b) {
    return a + b;
}

// A __global__ function (kernel) that calls the __device__ function
__global__ void kernelCallDevice(int *out, int a, int b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) { // only one thread writes to output for demonstration
        int result = deviceAdd(a, b);
        *out = result;
        printf("Kernel called deviceAdd: %d + %d = %d\n", a, b, result);
    }
}

// Uncommenting the following code would attempt to call a __global__ kernel from a __device__ function,
// which is illegal without dynamic parallelism and would result in a compilation error.
// __device__ void deviceCallKernel() {
//     // Attempt to launch a kernel from device (invalid without dynamic parallelism)
//     // kernelCallDevice<<<1,1>>>(...);
// }

// A __device__ function that attempts to call a kernel (invalid without dynamic parallelism)
// __device__ void invalidDeviceCall() {
//     // kernelCallDevice<<<1,1>>>(...); // This is not allowed
// }

int main() {
    int *d_out;
    int h_out;
    size_t size = sizeof(int);

    cudaMalloc(&d_out, size);

    // Launch kernel to compute 5 + 10
    kernelCallDevice<<<1, 1>>>(d_out, 5, 10);

    cudaDeviceSynchronize();

    cudaMemcpy(&h_out, d_out, size, cudaMemcpyDeviceToHost);

    printf("Result from device memory: %d\n", h_out);

    cudaFree(d_out);
    return 0;
}
