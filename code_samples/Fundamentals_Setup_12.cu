/*
Aim: Create a CUDA program that deliberately fails to compile due to a syntax error within the __global__ function (e.g., missing a semicolon). 
This program is intentionally erroneous to help learners understand how compiler error messages are generated and how to debug them.

Thinking process:
1. The kernel function is defined with __global__ to trigger the CUDA compiler. 
2. Inside the kernel we introduce a classic syntax mistake: forgetting the semicolon after a variable declaration.
3. The missing semicolon will cause nvcc to emit an error similar to:
   error: expected ‘;’ before ‘int’ token
   (and possibly additional diagnostic lines pointing to the exact location of the mistake).
4. The main() function simply launches the kernel, but because the kernel fails to compile, the whole compilation fails.
5. The user can compile this file with `nvcc -arch=sm_30 test.cu` (or any architecture) and observe the error output, then fix the semicolon to see a successful compile.
*/

#include <stdio.h>

__global__ void faultyKernel()
{
    int a = 10   // <--- Missing semicolon here will cause compilation error
    int b = a * 2;
}

int main()
{
    faultyKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    printf("Kernel launched.\n");
    return 0;
}
