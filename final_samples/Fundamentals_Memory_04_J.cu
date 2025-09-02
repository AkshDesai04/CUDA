```cpp
/*
Pass a `nullptr` to `cudaMemset` and use error checking to catch the invalid value error.

The goal of this small CUDA program is to intentionally invoke `cudaMemset` with a null pointer to demonstrate how CUDA’s runtime error handling works.  
By passing `nullptr` as the destination pointer, `cudaMemset` should return an error status (specifically `cudaErrorInvalidValue`).  
The program uses a helper macro `CUDA_CHECK` to immediately check the return value of each CUDA API call and print a descriptive error message if something goes wrong.  
After calling `cudaMemset(nullptr, 0, 10)`, the error is captured by `CUDA_CHECK`, printed to `stderr`, and the program exits with a non‑zero status.  
This example can serve as a quick test for verifying that the CUDA runtime correctly rejects invalid arguments and that the error‑checking infrastructure is functioning as expected.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Helper macro to check CUDA API calls
#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = (call);                                  \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

int main(void) {
    // Intentionally pass nullptr to cudaMemset to trigger an error
    printf("Calling cudaMemset with nullptr...\n");
    CUDA_CHECK(cudaMemset(nullptr, 0, 10));

    // If we reach this point, something went wrong (unlikely)
    printf("Unexpected success: cudaMemset should have failed.\n");
    return 0;
}
```