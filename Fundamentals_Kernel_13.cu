/*
Aim of the program:
Write a kernel that accepts two integer pointers, `a` and `b`. The kernel (with a single thread) should read the value from `a`, add 10 to it, and write the result to `b`. The host must allocate, initialize `a`, and verify the result in `b`.

Thinking:
The requirement is very simple: a kernel that reads a value from one integer pointer, adds 10, and writes it to another integer pointer. We need only a single thread, so we launch the kernel with one block and one thread.  
The host will allocate two integer variables, set the first one to some known value, copy that value to device memory, launch the kernel, copy the result back, and then compare it with the expected value (`a + 10`).  
We’ll include basic CUDA error checking for robustness. The program will print the input and output values and a success message if the result matches the expectation.

The code below follows the CUDA C++ conventions and can be compiled with `nvcc`. It contains:
- `add10Kernel`: the device kernel.
- `checkCudaError`: helper for error checking.
- `main`: host code that sets up data, launches the kernel, and verifies the result.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Simple CUDA error checking helper
inline void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Kernel that reads from *a, adds 10, and writes to *b
__global__ void add10Kernel(const int* a, int* b) {
    // Since there's only one thread, we don't need threadIdx or blockIdx
    *b = *a + 10;
}

int main() {
    // Host variables
    int h_a = 7;          // example input
    int h_b = 0;          // will hold the result
    int device_a, device_b;

    // Allocate device memory
    checkCudaError(cudaMalloc((void**)&device_a, sizeof(int)), "cudaMalloc device_a");
    checkCudaError(cudaMalloc((void**)&device_b, sizeof(int)), "cudaMalloc device_b");

    // Copy input from host to device
    checkCudaError(cudaMemcpy(device_a, &h_a, sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy H2D device_a");

    // Launch kernel with 1 block and 1 thread
    add10Kernel<<<1, 1>>>(device_a, device_b);

    // Check for kernel launch errors
    checkCudaError(cudaGetLastError(), "Kernel launch");

    // Copy result back to host
    checkCudaError(cudaMemcpy(&h_b, device_b, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy D2H device_b");

    // Verify the result
    if (h_b == h_a + 10) {
        printf("Success: %d + 10 = %d\n", h_a, h_b);
    } else {
        printf("Error: expected %d but got %d\n", h_a + 10, h_b);
    }

    // Clean up
    cudaFree(device_a);
    cudaFree(device_b);

    return 0;
}
