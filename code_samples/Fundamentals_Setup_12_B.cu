/*
Aim: Create an error by calling a regular C++ function (that is not marked __device__) from within your kernel.

Thinking:
We want to demonstrate that a device kernel cannot call a host function. To do that, we write a simple C++ function add that is not annotated with __device__ or __host__ __device__. Then we call this function inside a __global__ kernel. When compiling with nvcc, the compiler will emit an error like "error: calling a host function from a device function". This will confirm that such calls are disallowed. We also include a minimal main to launch the kernel. The code is purposely incorrect to produce the error.
*/

#include <stdio.h>

// Regular C++ function (host only)
int add(int a, int b) {
    return a + b;
}

// Kernel attempting to call the host function
__global__ void kernelCallHostFunc(int *d_out) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // This line will cause a compilation error
    d_out[idx] = add(idx, idx);
}

int main() {
    const int N = 10;
    int h_out[N];
    int *d_out;

    cudaMalloc((void**)&d_out, N * sizeof(int));

    // Launch kernel
    kernelCallHostFunc<<<1, N>>>(d_out);

    // Copy result back
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < N; ++i) {
        printf("h_out[%d] = %d\n", i, h_out[i]);
    }

    cudaFree(d_out);
    return 0;
}
