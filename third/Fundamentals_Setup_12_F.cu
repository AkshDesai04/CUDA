/*
Aim: In `main`, try to call the kernel like a regular function `my_kernel()` instead of launching it with `<<<...>>>`.

Thinking:
The goal of this program is to explicitly demonstrate the mistake of invoking a CUDA kernel as if it were a normal C function. In CUDA, kernels are launched with a special syntax using triple angle brackets (`<<<...>>>`) that specifies the execution configuration (grid and block dimensions). Trying to call the kernel directly from host code (e.g., `my_kernel();`) will result in a compilation error, because a `__global__` function cannot be invoked from the host; it must be launched from the host with the correct execution configuration. 

The program below defines a simple kernel that prints a message. In the `main` function, it attempts to call `my_kernel()` directly. This will produce a compiler error similar to:
```
error: calling a __global__ function from a __host__ function
```
This demonstrates the compiler enforcement of CUDA's launch syntax rules. The rest of the code is minimal and self‑contained, intended to compile with `nvcc`. If you wish to see the kernel executed correctly, replace the direct call with a proper launch, e.g., `my_kernel<<<1,1>>>();`. However, the assignment explicitly requests the incorrect call to illustrate the limitation.
*/

#include <stdio.h>

__global__ void my_kernel() {
    printf("Hello from kernel\n");
}

int main() {
    // Attempt to call the kernel like a regular function (this will fail to compile)
    my_kernel(); // ❌ This is illegal in CUDA and will cause a compilation error

    // To properly launch the kernel, you would use:
    // my_kernel<<<1, 1>>>();

    return 0;
}
