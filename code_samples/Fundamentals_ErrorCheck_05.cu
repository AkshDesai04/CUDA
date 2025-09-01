```cuda
/*
Aim of the program:
Write a program that deliberately triggers a `cudaErrorInvalidDevice` by trying to `cudaSetDevice()` to a device index that doesn't exist (e.g., 99). Use your `CHECK()` macro to catch it.

Thinking process:
- The goal is to intentionally cause a CUDA runtime error by requesting a device index that is guaranteed not to exist on the system. The most common error returned by `cudaSetDevice()` for an out-of-range index is `cudaErrorInvalidDevice`.
- We need a helper macro to check the return status of CUDA API calls. The macro will call a function that compares the returned `cudaError_t` against `cudaSuccess`. If an error occurs, the function prints the error string, the file name, and line number, then optionally aborts execution.
- In `main()`, we simply call `CHECK(cudaSetDevice(99));`. Since device 99 almost certainly does not exist, this call will trigger the error and the CHECK macro will report it.
- To keep the program minimal, we don't launch any kernels; we just exercise the error handling logic.

The code below implements this logic and compiles as a standard CUDA C file (.cu).
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error [%d] at %s:%d : %s\n", code, file, line, cudaGetErrorString(code));
        if (abort) exit(code);
    }
}

int main(void)
{
    // Intentionally use an invalid device index to trigger cudaErrorInvalidDevice
    CHECK(cudaSetDevice(99));

    // If we reach this point, something went wrong with error handling
    printf("This line should not be printed.\n");
    return 0;
}
```