/*
Pass an integer literal (e.g., 5) as an argument to the empty kernel.

Thought process:
1. The kernel should accept an integer parameter and do nothing (empty body).
2. Host code will launch the kernel with the literal value 5.
3. Include necessary CUDA headers and handle basic error checking.
4. Use one block with one thread for simplicity.
5. After kernel launch, synchronize and check for errors.
6. Print a confirmation message on the host to indicate completion.
7. Ensure the program compiles as a .cu file and runs without errors.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Empty kernel that accepts an integer argument
__global__ void emptyKernel(int val) {
    // Kernel intentionally left empty; val is received but unused.
}

int main(void) {
    // Launch the kernel with integer literal 5
    emptyKernel<<<1, 1>>>(5);

    // Check for any launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for device to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Device synchronization failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Kernel executed with integer literal 5.\n");
    return 0;
}
