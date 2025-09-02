```cpp
/* 
Aim of the program: 
In your helper function, if an error is found, print the error string using `cudaGetErrorString`.

Thinking:
- We need a CUDA program written in C++ (.cu file) that demonstrates the usage of a helper function for error handling.
- The helper function will accept a cudaError_t and a message string, check if the error is cudaSuccess, and if not, print the error message along with the error string from cudaGetErrorString, then exit the program.
- The program will allocate memory on the device, launch a simple kernel (e.g., element-wise addition), copy results back, and free resources. After each CUDA API call and kernel launch, we will use the helper function to validate the result.
- This ensures that any CUDA error is reported with a human-readable string, satisfying the requirement.
- The code will be self-contained, compile with nvcc, and run a basic example to illustrate the helper.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// Helper function for CUDA error checking
void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << msg << " failed: " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Simple CUDA kernel: element-wise addition of two arrays
__global__ void addKernel(const float* a, const float* b, float* c, size_t N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        c[idx] = a[idx] + b[idx];
    }
}

int main()
{
    const size_t N = 1 << 20;            // 1M elements
    const size_t bytes = N * sizeof(float);

    // Host memory allocation
    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_c = new float[N];

    // Initialize host arrays
    for (size_t i = 0; i < N; ++i)
    {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    // Device memory allocation
    float *d_a, *d_b, *d_c;
    checkCudaError(cudaMalloc((void**)&d_a, bytes), "cudaMalloc d_a");
    checkCudaError(cudaMalloc((void**)&d_b, bytes), "cudaMalloc d_b");
    checkCudaError(cudaMalloc((void**)&d_c, bytes), "cudaMalloc d_c");

    // Copy data from host to device
    checkCudaError(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice), "cudaMemcpy h_a -> d_a");
    checkCudaError(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice), "cudaMemcpy h_b -> d_b");

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    checkCudaError(cudaGetLastError(), "Kernel launch");

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy d_c -> h_c");

    // Simple validation
    bool success = true;
    for (size_t i = 0; i < N; ++i)
    {
        if (h_c[i] != h_a[i] + h_b[i])
        {
            std::cerr << "Mismatch at index " << i << ": " << h_c[i] << " != " << h_a[i] + h_b[i] << std::endl;
            success = false;
            break;
        }
    }
    if (success)
        std::cout << "All results are correct." << std::endl;

    // Clean up
    checkCudaError(cudaFree(d_a), "cudaFree d_a");
    checkCudaError(cudaFree(d_b), "cudaFree d_b");
    checkCudaError(cudaFree(d_c), "cudaFree d_c");

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}
```