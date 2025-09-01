/*
Aim: Pass a float literal (e.g., 3.14f) as an argument to the empty kernel.
Thinking: 
- Create a minimal CUDA program with a kernel that takes a single float parameter.
- The kernel body will be empty because the goal is just to pass the literal, not to perform any computation.
- In the host code (main), launch the kernel with <<<1,1>>> and pass the literal 3.14f directly as the argument.
- Use cudaDeviceSynchronize to wait for kernel completion.
- No additional output is required; the program should compile and run without errors.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Empty kernel that accepts a float argument
__global__ void myKernel(float f) {
    // No operation performed
}

int main(void) {
    // Launch the kernel with a float literal argument
    myKernel<<<1, 1>>>(3.14f);

    // Wait for the kernel to finish execution
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
