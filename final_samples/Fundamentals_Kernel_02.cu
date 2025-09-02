```cpp
/*
Aim of the program:
Modify the previous 'Hello World' kernel to print a message from the GPU itself. The single thread in the kernel should use `printf` to print 'Hello from the GPU!'.

Thinking:
The task requires us to write a CUDA kernel that uses device-side printf to emit a message from the GPU. Since we only need one thread, the kernel will be launched with a single block and single thread configuration. Device printf requires that the GPU supports the feature and that we call cudaDeviceSynchronize() (or any CUDA runtime function that waits for kernel completion) before the program exits to ensure the output is flushed. We also want to show host-side printing for contrast, so we will print a simple message from the host before launching the kernel and one after synchronization. The code will include the necessary headers, define the kernel, and contain the main function that handles launching and synchronization. The final file is a complete, compilable .cu file that can be built with nvcc. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that prints a message from the GPU
__global__ void helloGPU()
{
    // Only the single thread will execute this
    printf("Hello from the GPU!\n");
}

int main()
{
    // Inform the host about the upcoming GPU message
    printf("Hello from the host!\n");

    // Launch the kernel with one block and one thread
    helloGPU<<<1, 1>>>();

    // Wait for GPU to finish and flush printf output
    cudaDeviceSynchronize();

    // Indicate completion
    printf("GPU work completed.\n");

    return 0;
}
```