/*
Pass a float literal (e.g., 3.14f) as an argument to the empty kernel.
Thoughts: 
- Define an empty __global__ kernel that takes a float parameter.
- From host, launch the kernel with a literal like 3.14f.
- Use a single thread block (<<<1,1>>>).
- Include minimal error checking and device synchronization.
- Keep the code simple and self-contained so it can be compiled with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void emptyKernel(float val)
{
    // Kernel does nothing with the passed float
    (void)val;
}

int main(void)
{
    // Launch the kernel with a float literal argument
    emptyKernel<<<1, 1>>>(3.14f);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Kernel executed successfully with float literal 3.14f.\n");
    return 0;
}
