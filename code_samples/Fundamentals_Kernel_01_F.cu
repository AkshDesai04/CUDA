/*
Aim of the program: Create a second empty kernel with a different name and launch both kernels from the host.

Thinking process:
1. The requirement is straightforward: we need two distinct kernel functions that are empty (do nothing) and then launch both from the host.
2. To keep the program minimal yet functional, we'll use the CUDA runtime API. 
3. We'll define:
   - __global__ void firstKernel() { }
   - __global__ void secondKernel() { }
4. In main, we will:
   - Launch firstKernel with a simple <<<1,1>>> configuration.
   - Launch secondKernel with the same launch configuration.
   - Use cudaDeviceSynchronize() to wait for both kernels to finish.
   - Check for any launch errors using cudaGetLastError() and cudaDeviceSynchronize() error codes.
5. We'll include <cuda_runtime.h> for CUDA runtime functions and <iostream> for basic console output (to confirm the program ran without error).
6. The program will be self‑contained in a single .cu file and can be compiled with nvcc. No external dependencies or data are required.

The final code below implements these steps.
*/

#include <cuda_runtime.h>
#include <iostream>

// First empty kernel
__global__ void firstKernel()
{
    // No operations performed
}

// Second empty kernel with a different name
__global__ void secondKernel()
{
    // No operations performed
}

int main()
{
    // Launch the first kernel
    firstKernel<<<1, 1>>>();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Error launching firstKernel: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    // Launch the second kernel
    secondKernel<<<1, 1>>>();
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Error launching secondKernel: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    // Synchronize to ensure both kernels have finished execution
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "Error during cudaDeviceSynchronize: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Both kernels executed successfully." << std::endl;
    return EXIT_SUCCESS;
}
