```cpp
/* 
 * Aim of the program:
 * Implement D[i] = (A[i] > B[i]) ? A[i] * s1 : B[i] * s2, where s1 and s2 are scalars.
 *
 * Thinking:
 * 1. We need a CUDA kernel that iterates over array indices and applies the ternary
 *    expression element‑wise.
 * 2. For simplicity, we use float arrays. The kernel will read from input arrays A and B,
 *    write to output array D, and use two scalar parameters s1 and s2.
 * 3. On the host we allocate device memory, copy input data, launch the kernel with
 *    a sufficient number of blocks/threads to cover all elements, then copy the result
 *    back to host memory for verification.
 * 4. Error checking is added after each CUDA API call to catch failures early.
 * 5. A small example with N = 10 is provided; the host initializes A and B with
 *    deterministic values so that the result can be inspected.
 * 6. The program prints both input arrays and the resulting D array for clarity.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// CUDA kernel to compute D[i] = (A[i] > B[i]) ? A[i] * s1 : B[i] * s2
__global__ void computeKernel(const float* A, const float* B, float* D,
                              float s1, float s2, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        float a = A[idx];
        float b = B[idx];
        D[idx] = (a > b) ? (a * s1) : (b * s2);
    }
}

// Utility function to check CUDA errors
void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << "Error: " << msg << " (" << cudaGetErrorString(err) << ")\n";
        exit(EXIT_FAILURE);
    }
}

int main()
{
    const int N = 10;               // Number of elements
    const float s1 = 2.0f;          // Scalar multiplier for A when A > B
    const float s2 = 0.5f;          // Scalar multiplier for B when B >= A

    // Host vectors
    std::vector<float> h_A(N);
    std::vector<float> h_B(N);
    std::vector<float> h_D(N, 0.0f);

    // Initialize host input data
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = static_cast<float>(i);        // 0,1,2,...
        h_B[i] = static_cast<float>(N - i);    // 10,9,8,...
    }

    // Device pointers
    float *d_A = nullptr, *d_B = nullptr, *d_D = nullptr;

    size_t size = N * sizeof(float);

    // Allocate device memory
    checkCudaError(cudaMalloc(&d_A, size), "Allocating d_A");
    checkCudaError(cudaMalloc(&d_B, size), "Allocating d_B");
    checkCudaError(cudaMalloc(&d_D, size), "Allocating d_D");

    // Copy data from host to device
    checkCudaError(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice), "Memcpy A to device");
    checkCudaError(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice), "Memcpy B to device");

    // Determine execution configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    computeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_D, s1, s2, N);
    checkCudaError(cudaGetLastError(), "Kernel launch");

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_D.data(), d_D, size, cudaMemcpyDeviceToHost), "Memcpy D to host");

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D);

    // Print results
    std::cout << "A: ";
    for (float val : h_A) std::cout << val << ' ';
    std::cout << "\nB: ";
    for (float val : h_B) std::cout << val << ' ';
    std::cout << "\nD: ";
    for (float val : h_D) std::cout << val << ' ';
    std::cout << std::endl;

    return 0;
}
```