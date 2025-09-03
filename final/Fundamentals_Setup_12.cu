/*
Create a CUDA program that deliberately fails to compile due to a syntax error within the `__global__` function (e.g., missing a semicolon). Compile it with `nvcc` and carefully analyze the error message produced. This helps in learning how to debug compilation issues.

My thinking:
- The goal is to produce a file that compiles with nvcc, but intentionally contains a syntax error that prevents successful compilation.
- A common syntax error is a missing semicolon. Inside a __global__ function, forgetting a semicolon after an assignment statement is straightforward and produces a clear compiler error.
- The rest of the code (memory allocation, kernel launch, cleanup) is standard and will not be executed because the compilation will fail early.
- The user can copy the file, compile with `nvcc` (e.g., `nvcc -o test test.cu`), and observe the error message. The message typically points to the line with the missing semicolon and indicates a syntax error near the unexpected token or end of file.
- No external files or support code is needed; the file is self-contained.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void faultyKernel(int *d_a)
{
    *d_a = 42   // <--- Missing semicolon here will cause a syntax error
}

int main()
{
    int *d_a;
    cudaMalloc(&d_a, sizeof(int));
    faultyKernel<<<1, 1>>>(d_a);
    cudaDeviceSynchronize();
    cudaFree(d_a);
    return 0;
}
