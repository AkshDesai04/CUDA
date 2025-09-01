```cpp
/*
Aim of the program:
Try to declare a non-const static variable inside a kernel and observe the compilation error.

Thought process:
In CUDA, __global__ (kernel) functions are executed on the device and cannot contain static variables
with non-const storage duration because such variables would require a global address space that
cannot be assigned to a kernel's local execution context. The compiler will reject a declaration like
`static int counter;` inside a kernel with an error similar to:
  "error: static local variable 'counter' not allowed in __global__ function"
To demonstrate this, we write a simple kernel that declares a non-const static variable
and then attempts to increment it. The host code simply launches this kernel. The resulting
compilation failure is what the user wants to observe.

The code below intentionally triggers this error. To see the error, compile with nvcc:
  nvcc -arch=sm_70 -o test test.cu
*/

#include <stdio.h>

__global__ void testKernel()
{
    // This static variable is NOT const and should cause a compilation error
    static int counter = 0;
    counter++;
}

int main()
{
    // Launch the kernel
    testKernel<<<1, 1>>>();

    // Synchronize and check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    return 0;
}
```