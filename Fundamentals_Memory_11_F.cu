```cpp
/*
Aim: Can you write to a __constant__ variable from within a kernel? Try it and observe the compiler error.

Thinking:
- In CUDA, __constant__ memory is read-only from the device side. Attempting to write to it inside a kernel
  should trigger a compiler error because the language enforces that __constant__ variables cannot be
  modified by device code.
- We'll declare a __constant__ integer, then define a simple kernel that tries to assign a new value
  to that variable. When compiled with nvcc, we expect a compilation error.
- The rest of the code is just boilerplate to launch the kernel so that the compiler processes it.
- No host side modification of the constant memory is attempted; only a kernel write.
- Running this should produce an error similar to:
    "error: device code cannot write to __constant__ memory"
*/
#include <cstdio>
#include <cuda_runtime.h>

// Declare a constant memory variable
__constant__ int constVar = 0;

// Kernel that attempts to write to the constant memory
__global__ void tryWriteConst()
{
    // This assignment should trigger a compilation error
    constVar = 42;
}

int main()
{
    // Launch the kernel (single block, single thread)
    tryWriteConst<<<1, 1>>>();

    // Wait for any GPU errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    // Synchronize and exit
    cudaDeviceSynchronize();
    return 0;
}
```