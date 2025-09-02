/*
Aim of the program:
Use your macro on a call that is guaranteed to succeed, like `cudaGetDeviceCount`, 
to ensure it doesn't trigger incorrectly.

Thinking:
1. Define a reusable error-checking macro that wraps CUDA API calls. 
   The macro will invoke an inline helper `gpuAssert` that checks the 
   returned `cudaError_t`. If the error code is not `cudaSuccess`, 
   it prints a detailed error message including the file name, line 
   number, and the CUDA error string, and aborts the program.

2. Use this macro on a CUDA API call that is expected to succeed under 
   normal conditions: `cudaGetDeviceCount`. Since `cudaGetDeviceCount` 
   simply reports the number of available GPU devices, it should never 
   return an error on a system with at least one CUDA-capable device. 
   This test demonstrates that the macro does not falsely report errors.

3. The program will print the number of devices found and then exit. 
   No additional kernels or GPU memory allocation is required. 
   The focus is purely on the correctness of the error-checking macro.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s in %s at line %d\n", 
                cudaGetErrorString(code), file, line);
        if (abort) exit(static_cast<int>(code));
    }
}

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

int main() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    printf("Number of CUDA-capable devices: %d\n", deviceCount);
    return 0;
}
