/*
Aim of the program: Create a second empty kernel with a different name and launch both kernels sequentially from main.

Thinking:
- The requirement is to have two empty CUDA kernels, each with a distinct name.
- We will define them as __global__ void kernelA() and __global__ void kernelB().
- Both kernels will contain no executable statements; they just exist for the purpose of demonstrating kernel launches.
- In main(), we will launch kernelA first, then kernelB, using simple <<<1,1>>> launch configuration.
- After launching each kernel we will call cudaDeviceSynchronize() to ensure completion before moving on.
- We will also check for launch errors using cudaGetLastError() after each kernel invocation and print any errors to the console.
- The program includes <stdio.h> for printing, and uses the standard CUDA runtime API.
- Since no device memory allocation or kernel logic is needed, the code remains minimal yet fulfills the task.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Empty kernel 1 */
__global__ void kernelA()
{
    // No operation
}

/* Empty kernel 2 */
__global__ void kernelB()
{
    // No operation
}

int main()
{
    // Launch kernelA
    kernelA<<<1, 1>>>();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error launching kernelA: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error after kernelA execution: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Launch kernelB
    kernelB<<<1, 1>>>();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error launching kernelB: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error after kernelB execution: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Both kernels launched and executed sequentially.\n");
    return 0;
}
