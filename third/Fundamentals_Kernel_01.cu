/*
Write a 'Hello World' CUDA program. The kernel should be launched with a single block containing a single thread. The host code (the `main` function) should print 'Kernel launched successfully!' after the kernel launch. The kernel itself will be empty.

Thinking:
- The program requires a minimal CUDA .cu file.
- We'll define an empty kernel named helloKernel.
- Launch it with <<<1,1>>> from host.
- After launch, print the success message.
- Include minimal headers and call cudaDeviceReset() to clean up.
- No error handling is specified, but we keep it simple.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloKernel() {
    // Empty kernel
}

int main(void) {
    // Launch the kernel with one block and one thread
    helloKernel<<<1, 1>>>();

    // Print success message after kernel launch
    printf("Kernel launched successfully!\n");

    // Clean up the device
    cudaDeviceReset();

    return 0;
}
