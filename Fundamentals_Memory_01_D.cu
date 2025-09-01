```cuda
/* 
Aim of the program: Instead of `malloc`, use `new int[256]` for the host allocation.

Thinking:
- The goal is to demonstrate how to replace C-style memory allocation (`malloc`) with C++ style (`new[]`) in a CUDA program.
- I decided to implement a simple vector addition kernel that operates on arrays of 256 integers. 
- Host arrays `h_A` and `h_B` will be allocated using `new int[256]` and initialized with sample values.
- Device arrays `d_A`, `d_B`, and `d_C` will be allocated with `cudaMalloc`.
- The kernel `vectorAdd` will add corresponding elements from `d_A` and `d_B`, storing the result in `d_C`.
- Results are copied back to the host array `h_C` using `cudaMemcpy`.
- After computation, I clean up by deleting the host arrays with `delete[]` and freeing device memory with `cudaFree`.
- Basic error checking is included to ensure CUDA API calls succeed.
- The code is fully self-contained and can be compiled with `nvcc` to produce a runnable executable.
*/

#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel for vector addition
__global__ void vectorAdd(const int *A, const int *B, int *C, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    const int N = 256;
    const size_t bytes = N * sizeof(int);

    // Allocate host memory using C++ new[]
    int *h_A = new int[N];
    int *h_B = new int[N];
    int *h_C = new int[N];

    // Initialize host arrays
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    // Allocate device memory
    int *d_A = nullptr;
    int *d_B = nullptr;
    int *d_C = nullptr;
    cudaError_t err;

    err = cudaMalloc((void**)&d_A, bytes);
    if (err != cudaSuccess) { std::cerr << "cudaMalloc d_A failed: " << cudaGetErrorString(err) << std::endl; return 1; }

    err = cudaMalloc((void**)&d_B, bytes);
    if (err != cudaSuccess) { std::cerr << "cudaMalloc d_B failed: " << cudaGetErrorString(err) << std::endl; return 1; }

    err = cudaMalloc((void**)&d_C, bytes);
    if (err != cudaSuccess) { std::cerr << "cudaMalloc d_C failed: " << cudaGetErrorString(err) << std::endl; return 1; }

    // Copy data from host to device
    err = cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { std::cerr << "cudaMemcpy to d_A failed: " << cudaGetErrorString(err) << std::endl; return 1; }

    err = cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { std::cerr << "cudaMemcpy to d_B failed: " << cudaGetErrorString(err) << std::endl; return 1; }

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) { std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl; return 1; }

    // Copy result back to host
    err = cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { std::cerr << "cudaMemcpy from d_C failed: " << cudaGetErrorString(err) << std::endl; return 1; }

    // Verify results
    bool success = true;
    for (int i = 0; i < N; ++i)
    {
        if (h_C[i] != h_A[i] + h_B[i])
        {
            std::cerr << "Verification failed at index " << i << ": " << h_C[i] << " != " << h_A[i] + h_B[i] << std::endl;
            success = false;
            break;
        }
    }
    if (success)
        std::cout << "Vector addition succeeded!" << std::endl;

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
```