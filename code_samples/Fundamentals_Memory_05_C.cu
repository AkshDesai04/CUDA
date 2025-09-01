/*
Aim: Try to call `cudaFree` on a `nullptr`. Check the return code. Is it an error?

Thinking:
The task is to create a minimal CUDA program that exercises the CUDA runtime API function `cudaFree` by passing it a `nullptr` (i.e. a zero pointer). According to the CUDA Runtime API documentation, `cudaFree(NULL)` is a no-op and should return `cudaSuccess`. However, historically some drivers might return an error such as `cudaErrorInvalidDevicePointer`. Therefore, the program will:
1. Include the necessary headers (`stdio.h` and `cuda_runtime.h`).
2. Call `cudaFree((void*)nullptr)` and capture the return value.
3. Use `cudaGetErrorString()` to convert the return value into a human-readable message.
4. Print the numeric return code, the string message, and a simple verdict indicating whether the call was considered an error (`ret != cudaSuccess`).
No device allocation or kernel launches are required; the focus is solely on the behaviour of `cudaFree` with a null pointer.

The program is written in standard C with CUDA extensions, compiled with `nvcc` into a `.cu` file. The output will differ depending on the CUDA driver/runtime version but should correctly report whether `cudaFree(NULL)` is treated as an error or a success.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    // Call cudaFree with a nullptr (NULL pointer)
    cudaError_t err = cudaFree((void*)nullptr);

    // Get the string representation of the error code
    const char *err_str = cudaGetErrorString(err);

    // Print the results
    printf("cudaFree(nullptr) returned code: %d\n", err);
    printf("Error string: %s\n", err_str);

    if (err == cudaSuccess)
    {
        printf("Result: No error (cudaSuccess)\n");
    }
    else
    {
        printf("Result: Error occurred\n");
    }

    return 0;
}
