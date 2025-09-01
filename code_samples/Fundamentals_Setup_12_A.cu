/*
Create a syntax error by misspelling `__global__` as `__globol__` and observe the compiler error.

Thinking:
To deliberately trigger a compiler error in CUDA, I will replace the correct CUDA kernel qualifier `__global__` with a misspelled version `__globol__`. The compiler will not recognize this qualifier and will produce a syntax error indicating that the keyword is unknown or unexpected. I will write a minimal CUDA program that defines such a kernel and attempts to launch it from `main`. The error should occur at compile time before the program can run, and it will demonstrate how the compiler reacts to an invalid qualifier.
*/

#include <stdio.h>

__globol__ void misspelledKernel()
{
    printf("This should never print.\n");
}

int main()
{
    misspelledKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
