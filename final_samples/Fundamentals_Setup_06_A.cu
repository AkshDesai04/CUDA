```cuda
/*
Aim of the program:
From the maximum dimensions [X, Y, Z], verify that X * Y * Z is not necessarily equal to
maxThreadsPerBlock (since X, Y, and Z are individual dimension limits).

Thinking:
The CUDA device properties expose the maximum number of threads that can be launched
in each dimension (maxThreadsDim[0], maxThreadsDim[1], maxThreadsDim[2]) as well
as the maximum total number of threads per block (maxThreadsPerBlock).
While each dimension is limited individually, the product of the three limits
can exceed maxThreadsPerBlock.  This program queries those properties,
prints the values, and then attempts to launch a kernel using the maximum
per‑dimension values as the block size.  If the product exceeds maxThreadsPerBlock,
the launch will fail with cudaErrorInvalidConfiguration or a similar error.
The program checks for this and reports the outcome, demonstrating that the
product is not necessarily equal to maxThreadsPerBlock.
*/

#include <cuda_runtime.h>
#include <iostream>

// Simple kernel that writes its linear thread index into global memory
__global__ void testKernel(int* out)
{
    // Compute the linear index within the block
    int idx = threadIdx.x
            + threadIdx.y * blockDim.x
            + threadIdx.z * blockDim.x * blockDim.y;

    // Safety check (though idx should always be valid here)
    if (idx < blockDim.x * blockDim.y * blockDim.z)
        out[idx] = idx;
}

int main()
{
    // Use device 0 for this demonstration
    cudaSetDevice(0);

    // Retrieve device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Maximum threads per dimension
    int maxDimX = prop.maxThreadsDim[0];
    int maxDimY = prop.maxThreadsDim[1];
    int maxDimZ = prop.maxThreadsDim[2];

    // Maximum threads per block
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;

    // Compute product of maximum dimensions
    long long productDims = static_cast<long long>(maxDimX) *
                            static_cast<long long>(maxDimY) *
                            static_cast<long long>(maxDimZ);

    std::cout << "Device properties for device 0:\n";
    std::cout << "  maxThreadsDim[0] = " << maxDimX << "\n";
    std::cout << "  maxThreadsDim[1] = " << maxDimY << "\n";
    std::cout << "  maxThreadsDim[2] = " << maxDimZ << "\n";
    std::cout << "  maxThreadsPerBlock = " << maxThreadsPerBlock << "\n";
    std::cout << "  Product of maxThreadsDim = " << productDims << "\n";

    // Allocate memory for kernel output (size equal to product of dimensions)
    int* d_out = nullptr;
    size_t size = static_cast<size_t>(productDims) * sizeof(int);
    cudaError_t err = cudaMalloc(&d_out, size);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    // Prepare block dimensions using the maximum per-dimension values
    dim3 blockDim(maxDimX, maxDimY, maxDimZ);

    // Attempt to launch the kernel
    testKernel<<<1, blockDim>>>(d_out);

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Kernel launch failed as expected due to configuration limits.\n";
        std::cout << "  cudaError: " << cudaGetErrorString(err) << "\n";
    } else {
        std::cout << "Kernel launched successfully.\n";
        // In the unlikely event that launch succeeded, we can synchronize and check data
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(err) << "\n";
        } else {
            std::cout << "Kernel execution completed.\n";
        }
    }

    // Clean up
    cudaFree(d_out);

    return 0;
}
```