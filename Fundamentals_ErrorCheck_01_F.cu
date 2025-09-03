/*
Aim of the program: Use your macro on a call that is guaranteed to fail, like `cudaMalloc` for 100TB, to test its output.

Thinking:
1. The goal is to create a simple CUDA C program that demonstrates a macro which checks the return status of CUDA API calls.
2. We will define a macro `CUDA_CHECK` that takes a CUDA API call, executes it, checks if the result is `cudaSuccess`, and if not, prints an error message with the file and line number.
3. In `main`, we will deliberately invoke `cudaMalloc` with a very large size (100 terabytes) that is far beyond the capacity of any typical device memory. This allocation is guaranteed to fail.
4. By wrapping the `cudaMalloc` call with `CUDA_CHECK`, the macro should capture the failure and output a descriptive error to `stderr`.
5. The program will compile as a single `.cu` file and can be built with `nvcc`. Running it will produce the expected error message, confirming that the macro behaves correctly.
*/

#include <cstdio>
#include <cuda_runtime.h>

/* Macro to check CUDA API calls for errors */
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in file '%s' in line %d: %s.\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            /* Optionally, exit or handle the error here */                   \
            /* exit(EXIT_FAILURE); */                                          \
        }                                                                      \
    } while (0)

int main(void) {
    // Pointer for device memory
    void* d_ptr = nullptr;

    // Attempt to allocate 100 TB (guaranteed to fail)
    size_t largeSize = static_cast<size_t>(100ULL * 1024ULL * 1024ULL * 1024ULL * 1024ULL);
    CUDA_CHECK(cudaMalloc(&d_ptr, largeSize));

    // Clean up if allocation somehow succeeded (unlikely)
    if (d_ptr != nullptr) {
        CUDA_CHECK(cudaFree(d_ptr));
    }

    return 0;
}
