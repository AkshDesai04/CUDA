/*
Pass a `nullptr` to `cudaMemset` and use error checking to catch the invalid value error.

The goal of this CUDA program is to demonstrate how CUDA error checking works when an invalid
pointer is supplied to a runtime API function. In this case, `cudaMemset` expects a valid
device pointer, but we intentionally pass `nullptr`. According to the CUDA Runtime
Specification, this should result in a `cudaErrorInvalidValue` error. The program will:

1. Call `cudaMemset` with a `nullptr` target, a value of `0`, and a size of 10 bytes.
2. Capture the return code (`cudaError_t`) from `cudaMemset`.
3. Use `cudaGetErrorString` to convert the error code into a human‑readable message.
4. Print the error code and message to the standard output.
5. Reset the device with `cudaDeviceReset` before exiting to clean up.

By examining the printed output, a user can verify that CUDA correctly identifies the
invalid pointer and reports the appropriate error. This pattern of checking the return
code after each CUDA call is essential for robust GPU programming.

The program uses the CUDA Runtime API and requires a CUDA‑enabled compiler (nvcc).
Compile with: `nvcc -o cuda_memset_null cuda_memset_null.cu`
Run with: `./cuda_memset_null`
*/

#include <cuda_runtime.h>
#include <iostream>

int main()
{
    // Attempt to memset a null pointer. This should trigger an error.
    size_t bytes = 10;
    cudaError_t err = cudaMemset(nullptr, 0, bytes);

    // Check the error returned by cudaMemset
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMemset returned error code: " << err << std::endl;
        std::cerr << "Error message: " << cudaGetErrorString(err) << std::endl;
    }
    else
    {
        std::cout << "cudaMemset succeeded (unexpected)." << std::endl;
    }

    // Reset the device to clean up.
    cudaDeviceReset();

    return 0;
}
