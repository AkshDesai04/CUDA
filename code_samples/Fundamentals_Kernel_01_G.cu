/*
Aim: Write a helper function `void launch_empty_kernel()` that contains the kernel launch call.

Thinking:
1. The goal is to provide a small helper that encapsulates the launch of a trivial CUDA kernel.  
2. A kernel that does nothing is sufficient, so we define `__global__ void empty_kernel()` with an empty body.  
3. The helper `launch_empty_kernel()` will launch this kernel with a simple 1×1 grid configuration.  
4. For robustness we include error checking: after the launch we call `cudaGetLastError()` and print any error message.  
5. A `cudaDeviceSynchronize()` ensures the host waits for the kernel to finish.  
6. To demonstrate usage, a minimal `main()` calls the helper and prints a success message.  
7. All necessary headers (`stdio.h`) are included. This code compiles as a standard `.cu` file with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void empty_kernel()
{
    // No operation performed
}

void launch_empty_kernel()
{
    // Launch the kernel with 1 block and 1 thread
    empty_kernel<<<1, 1>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    // Synchronize to ensure completion
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
    }
}

int main()
{
    launch_empty_kernel();
    printf("Kernel launched and completed successfully.\n");
    return 0;
}
