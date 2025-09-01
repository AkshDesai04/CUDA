```cpp
/*
Create a C++ helper function `void checkKernelLaunch()` that calls `cudaDeviceSynchronize()` and checks for errors. Call this function after every kernel launch in your programs to ensure the kernel executed without an asynchronous error.

Thought process:
- The goal is to provide a simple helper that synchronizes the device after a kernel launch and checks for any runtime errors that may have occurred asynchronously during kernel execution.
- The helper will call `cudaDeviceSynchronize()` to force the CPU to wait until all previously issued GPU work is finished. This ensures any errors produced by the kernel are caught.
- After synchronization, we query the returned error code. If it is not `cudaSuccess`, we print a helpful message and exit the program to avoid silent failures.
- To demonstrate usage, I’ll write a small vector addition kernel and call `checkKernelLaunch()` immediately after launching it.
- The example includes memory allocation, kernel launch, error checking, memory copy back, and cleanup. It serves as a template that users can copy and adapt for their own kernels.
- The file is a valid `.cu` source that can be compiled with `nvcc`. No external dependencies beyond the CUDA runtime are needed.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

/* Helper that synchronizes the device and checks for kernel launch errors */
void checkKernelLaunch()
{
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after kernel launch: "
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

/* Simple vector addition kernel */
__global__ void addKernel(const int *a, const int *b, int *c, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main()
{
    const size_t N = 1 << 20;           // 1M elements
    const size_t SIZE = N * sizeof(int);

    // Allocate host memory
    int *h_a = (int*)malloc(SIZE);
    int *h_b = (int*)malloc(SIZE);
    int *h_c = (int*)malloc(SIZE);

    // Initialize input data
    for (size_t i = 0; i < N; ++i) {
        h_a[i] = static_cast<int>(i);
        h_b[i] = static_cast<int>(2 * i);
    }

    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, SIZE);
    cudaMalloc((void**)&d_b, SIZE);
    cudaMalloc((void**)&d_c, SIZE);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, SIZE, cudaMemcpyHostToDevice);

    // Launch kernel
    const size_t THREADS_PER_BLOCK = 256;
    const size_t BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    addKernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);

    // Check for launch errors
    checkKernelLaunch();

    // Copy result back to host
    cudaMemcpy(h_c, d_c, SIZE, cudaMemcpyDeviceToHost);

    // Verify result (optional)
    bool ok = true;
    for (size_t i = 0; i < N; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            ok = false;
            std::cerr << "Verification failed at index " << i
                      << ": " << h_c[i] << " != " << h_a[i] + h_b[i] << std::endl;
            break;
        }
    }
    if (ok) {
        std::cout << "Vector addition successful!" << std::endl;
    }

    // Clean up
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
```