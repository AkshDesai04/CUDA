/*
Create a macro `CHECK_KERNEL(kernel_launch)` that expands to `{ kernel_launch; checkKernelLaunch(...); }`.

[Thinking]
The goal is to define a macro that wraps a CUDA kernel launch with an error check.  
We need the macro to accept the entire kernel launch expression, execute it, and then call a helper function that queries `cudaGetLastError()` and reports any problems.  
The ellipsis in the specification suggests that `checkKernelLaunch` might accept additional arguments; a common pattern is to pass the name of the kernel or the launch expression as a string for clearer diagnostics.  
Using the preprocessor stringification operator `#`, we can turn the macro argument into a string and pass it to `checkKernelLaunch`.  

Implementation plan:
1. Include `<cuda_runtime.h>` and `<iostream>` for CUDA runtime API and I/O.
2. Define `checkKernelLaunch(const char *kernel_name)` that retrieves the last CUDA error, prints a message if the kernel failed, and exits.
3. Define the macro `CHECK_KERNEL(kernel_launch)` as `{ kernel_launch; checkKernelLaunch(#kernel_launch); }`.
4. Provide a small dummy kernel and a `main` that demonstrates the macro usage.

This .cu file compiles with `nvcc` and showcases the requested macro functionality.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

void checkKernelLaunch(const char *kernel_name = nullptr)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error";
        if (kernel_name)
        {
            std::cerr << " in kernel launch (" << kernel_name << ")";
        }
        std::cerr << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_KERNEL(kernel_launch) { kernel_launch; checkKernelLaunch(#kernel_launch); }

__global__ void dummyKernel(int *a)
{
    *a = 42;
}

int main()
{
    int *devPtr = nullptr;
    cudaMalloc(&devPtr, sizeof(int));

    CHECK_KERNEL(dummyKernel<<<1, 1>>>(devPtr));

    int hostVal = 0;
    cudaMemcpy(&hostVal, devPtr, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "hostVal = " << hostVal << std::endl;

    cudaFree(devPtr);
    return 0;
}
