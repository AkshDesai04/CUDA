```cpp
/*
Aim of the program:
Modify the program to print 'Hello from host!' before the kernel launch and 'Kernel finished!' after the kernel launch.

Thinking:
- The task is to produce a CUDA C++ (.cu) file that performs the following:
- 1. Prints a message from the host ("Hello from host!") before launching a kernel.
- 2. Launches a simple kernel that optionally prints a message from the device.
- 3. After the kernel finishes executing, prints another message from the host ("Kernel finished!").
- The kernel can be a minimal example (e.g., a single-thread block doing nothing or printing a message).
- Error handling should be included to catch CUDA runtime errors.
- The program must compile with nvcc and run on a system with a CUDA-capable GPU.
- The comment block at the top must contain the exact aim string and the explanation of reasoning.

Implementation details:
- Use <stdio.h> for printf.
- Use <cuda_runtime.h> for CUDA API.
- Define a __global__ kernel that prints "Hello from kernel!" (optional).
- In main():
  - Print host message before kernel launch.
  - Launch kernel<<<1,1>>>(); (one block, one thread).
  - Use cudaGetLastError() to check kernel launch.
  - Use cudaDeviceSynchronize() to wait for kernel completion.
  - Print host message after kernel finished.
  - Check for errors after synchronization.
  - Return 0.

This yields a complete, self‑contained CUDA program that satisfies the requirement.

*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void simpleKernel()
{
    // Optional: device side output
    printf("Hello from kernel!\\n");
}

int main()
{
    // Print message from host before kernel launch
    printf("Hello from host!\\n");

    // Launch kernel with 1 block and 1 thread
    simpleKernel<<<1, 1>>>();

    // Check for errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Wait for kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Print message from host after kernel finished
    printf("Kernel finished!\\n");

    return EXIT_SUCCESS;
}
```