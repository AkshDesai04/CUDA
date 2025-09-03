/*
Try to launch a kernel using a function pointer that is null. What error occurs?

My thinking process:

1. In CUDA C, kernels are launched either using the <<<...>>> syntax or via the low-level API function `cudaLaunchKernel`.  
2. The `<<<>>>` syntax expects a kernel name, not a function pointer, so attempting to pass a null pointer would not even compile.  
3. To test a null function pointer at runtime, we need to use `cudaLaunchKernel`, which accepts a `const void*` that should point to a device kernel function.  
4. If we pass a null pointer to `cudaLaunchKernel`, the CUDA runtime should detect that the function pointer is invalid and return an error.  
5. According to the CUDA Runtime API documentation, `cudaLaunchKernel` will return `cudaErrorInvalidDeviceFunction` when the function pointer does not refer to a valid device function.  
6. Therefore, the expected error message is something like “invalid device function”.  
7. The program below sets a `void*` kernel pointer to `nullptr`, calls `cudaLaunchKernel` with it, and then checks `cudaGetLastError()` to print the error string.  
8. This demonstrates that launching a kernel with a null function pointer produces the “invalid device function” error.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Dummy kernel just to satisfy the compiler
__global__ void dummyKernel() {}

int main(void)
{
    // Function pointer that is intentionally null
    void *kernel_ptr = nullptr;

    // Attempt to launch the kernel with a null pointer
    cudaError_t err = cudaLaunchKernel(
        kernel_ptr,                 // null function pointer
        dim3(1),                    // grid dimensions
        dim3(1),                    // block dimensions
        nullptr,                    // kernel arguments
        0,                          // shared memory size
        nullptr                     // stream
    );

    // Check and print the error returned by the launch
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaLaunchKernel error: %s\n", cudaGetErrorString(err));
    }
    else
    {
        printf("Kernel launched successfully (unexpected).\n");
    }

    return 0;
}
