/*
Generalize the kernel to work on arrays. It should take `int* a`, `int* b`, and `int N`. Each thread `i` should compute `b[i] = a[i] + 10;`.

Thinking:
1. Create a kernel `addTenKernel` that receives device pointers `a` and `b`, and the array length `N`.  
2. Each thread computes its global index `i = blockIdx.x * blockDim.x + threadIdx.x`.  
3. Guard against out-of-bounds: `if (i < N) b[i] = a[i] + 10;`.  
4. In `main`, allocate host arrays `h_a` and `h_b`, initialize `h_a` with sample data, and compute `N`.  
5. Allocate device memory `d_a` and `d_b`.  
6. Copy `h_a` to `d_a`.  
7. Launch kernel with a reasonable block size (e.g., 256) and compute grid size to cover `N`.  
8. Copy `d_b` back to `h_b`.  
9. Print a few results to verify.  
10. Clean up memory and check for CUDA errors using a macro.  
*/

#include <iostream>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA error in " << __FILE__ << ":"            \
                      << __LINE__ << " : " << cudaGetErrorString(err)  \
                      << std::endl;                                    \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// Kernel that adds 10 to each element of array a and stores result in b
__global__ void addTenKernel(const int* a, int* b, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        b[i] = a[i] + 10;
    }
}

int main()
{
    const int N = 1024;                    // Size of arrays
    const int bytes = N * sizeof(int);

    // Allocate host memory
    int* h_a = new int[N];
    int* h_b = new int[N];

    // Initialize host input array
    for (int i = 0; i < N; ++i)
        h_a[i] = i;

    // Allocate device memory
    int *d_a = nullptr;
    int *d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));

    // Copy input from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addTenKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost));

    // Verify a few results
    std::cout << "First 10 results:" << std::endl;
    for (int i = 0; i < 10; ++i)
    {
        std::cout << "b[" << i << "] = " << h_b[i] << " (expected " << h_a[i] + 10 << ")" << std::endl;
    }

    // Clean up
    delete[] h_a;
    delete[] h_b;
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));

    // Reset device
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
