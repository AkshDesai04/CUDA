/*
Aim: Create a CUDA program that deliberately fails to compile due to a syntax error within the __global__ function (e.g., missing a semicolon). Compile it with nvcc and carefully analyze the error message produced. This helps in learning how to debug compilation issues.

Thinking:
- The goal is to produce a source file that cannot compile because of a clear syntax mistake in a __global__ kernel.
- A missing semicolon is a classic easy-to-detect error that will produce a compilation error from nvcc.
- Include a minimal kernel that performs a simple operation; the kernel should have a line missing a semicolon to trigger the error.
- Provide a main function that launches the kernel so that the compiler will process the kernel code.
- The rest of the file should be syntactically correct to isolate the error to the intentional missing semicolon.
- The code will be a .cu file, using only standard CUDA constructs and minimal includes.
- No additional explanations or commentary should appear in the output beyond the initial comment and the source code itself.
*/

#include <stdio.h>

__global__ void faultyKernel()
{
    int a = 10;
    int b = 20      // <--- missing semicolon here
    int c = a + b;
    printf("Sum: %d\n", c);
}

int main()
{
    faultyKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
