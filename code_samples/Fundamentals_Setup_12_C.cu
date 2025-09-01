/*
Aim of the program: Create an error by forgetting the `<<<...>>>` syntax for the kernel launch.

Thought Process:
- The task is to intentionally generate a compilation error in a CUDA program by omitting the required launch configuration syntax `<<<...>>>` when calling a kernel.
- To keep the program simple, define a single kernel that performs a trivial operation, such as printing a message.
- In the `main` function, invoke the kernel without the launch configuration. This omission will trigger a compiler error indicating that the `<<<...>>>` syntax is missing.
- The code includes the necessary headers for CUDA and C standard I/O, but the error is purely due to the missing launch syntax.
- No additional logic or error handling is required because the purpose is to illustrate the compile-time failure.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Simple kernel that prints a message (note: printf requires device-side printf support)
__global__ void helloKernel()
{
    printf("Hello from GPU kernel!\n");
}

int main()
{
    // Intentionally missing the <<<grid, block>>> launch configuration
    helloKernel();  // This line will cause a compilation error: missing '<<<...>>>'
    cudaDeviceSynchronize();
    return 0;
}
