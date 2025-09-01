/*
Try to declare a non-const static variable inside a kernel and observe the compilation error.
The goal of this program is to deliberately trigger a compilation failure by placing a
non‑const static variable within a __global__ function.  In CUDA, device code has
strong restrictions on storage classes; a static variable inside a kernel must be
constant or otherwise explicitly marked as a device global.  Declaring a plain
static int inside a kernel violates these rules and should cause the compiler
to emit an error such as “static variable not allowed in device code” or
“device functions cannot contain static local variables”.

The thinking behind the code is straightforward:
1. Define a __global__ kernel that contains the statement `static int counter;`.
2. Increment the counter and print its value so that the kernel does something
   observable if the compiler (incorrectly) accepts it.
3. In the host `main`, launch the kernel with a single block and thread.
4. The compilation process (nvcc) should fail, demonstrating the intended error.
*/

#include <stdio.h>

__global__ void testKernel()
{
    // Non-const static variable inside a kernel – expected to fail compilation
    static int counter;          // <-- This line should trigger a compilation error
    counter++;
    printf("counter = %d\n", counter);
}

int main()
{
    // Launch the kernel
    testKernel<<<1, 1>>>();
    // Wait for the kernel to finish
    cudaDeviceSynchronize();
    return 0;
}
