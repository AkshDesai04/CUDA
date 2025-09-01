/*
Aim: Try to declare a non-const static variable inside a kernel and observe the compilation error.

Thinking: 
- In CUDA, __global__ functions (kernels) cannot have non-const static local variables. 
- The compiler should emit an error when encountering "static int counter;" inside the kernel. 
- This example includes a simple kernel that declares a static int counter, increments it, and prints it. 
- When compiled with nvcc, we expect a compilation error indicating that static local variables are not allowed in device code. 
- The rest of the code simply launches the kernel and synchronizes.
*/

#include <stdio.h>

__global__ void testKernel()
{
    static int counter;            // <--- This line should cause a compilation error
    counter++;
    printf("counter: %d\n", counter);
}

int main()
{
    testKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
