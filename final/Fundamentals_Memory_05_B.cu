```cpp
/*
Aim of the program:
Try to call cudaFree on the same pointer twice. What happens? Use error checking to see the result of the second call.

Thinking:
The objective is to demonstrate the behavior of CUDA's memory deallocation routine when invoked twice on the same device pointer. 
We will:
1. Allocate a small block of device memory using cudaMalloc.
2. Free that memory with cudaFree and check the return status.
3. Attempt to free the same pointer a second time and capture the return status.
4. Print the numerical error code and the human‑readable error string for both calls.
5. Observe that the first call succeeds (cudaSuccess), while the second call typically returns an error
   such as cudaErrorInvalidDevicePointer or cudaErrorInvalidValue, indicating that the pointer is no longer
   valid for deallocation.

We will use the standard CUDA runtime API, and basic error checking via the returned cudaError_t
and cudaGetErrorString. No kernel launches or complex device code are required.
*/

#include <cstdio>
#include <cuda_runtime.h>

int main() {
    const size_t numElements = 10;
    const size_t size = numElements * sizeof(int);
    int *d_ptr = nullptr;

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    printf("cudaMalloc succeeded, pointer = %p\n", (void*)d_ptr);

    // First cudaFree
    err = cudaFree(d_ptr);
    if (err == cudaSuccess) {
        printf("First cudaFree succeeded.\n");
    } else {
        printf("First cudaFree failed: %s\n", cudaGetErrorString(err));
    }

    // Second cudaFree on the same pointer
    err = cudaFree(d_ptr);
    if (err == cudaSuccess) {
        printf("Second cudaFree succeeded (unexpected).\n");
    } else {
        printf("Second cudaFree failed as expected: %s\n", cudaGetErrorString(err));
    }

    return 0;
}
```