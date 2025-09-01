/*
Aim of the program: Write a 'Hello World' CUDA program. The kernel should be launched with a single block containing a single thread. The host code (the main function) should print 'Kernel launched successfully!' after the kernel launch. The kernel itself will be empty.

Thinking process:
1. The program needs to be a valid CUDA source file with a .cu extension.
2. The kernel should be declared with __global__ and contain no statements (empty body).
3. In main, we launch the kernel with <<<1,1>>> which means one block with one thread.
4. After the kernel launch, we print the message using printf.
5. For good practice we include cuda_runtime.h and <stdio.h>.
6. Synchronize the device to ensure the kernel completes before printing, though the message is printed right after the launch. Using cudaDeviceSynchronize() guarantees the kernel finished.
7. Add minimal error checking for the kernel launch or synchronization; but not required by the prompt, but good practice. 
8. The code compiles with nvcc and runs on any CUDA-enabled device.

The final code below follows these steps and compiles to a CUDA executable.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Empty kernel
__global__ void helloKernel() {
    // Intentionally left blank
}

int main(void) {
    // Launch kernel with one block and one thread
    helloKernel<<<1, 1>>>();

    // Optional: synchronize to ensure kernel completion
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Print message after kernel launch
    printf("Kernel launched successfully!\n");

    return 0;
}
