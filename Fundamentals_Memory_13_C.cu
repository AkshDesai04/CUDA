```cpp
/*
Does a failed `cudaMalloc` return a `nullptr` and a success code, or does it return an error code directly? Check the documentation and your test.

[Thinking]
The CUDA Runtime API specifies that functions like `cudaMalloc` return a value of type `cudaError_t`.  
- If the call succeeds, the function returns `cudaSuccess`.  
- If the call fails (for example, due to insufficient device memory), the function returns an error code such as `cudaErrorMemoryAllocation`.  

The pointer argument is an output parameter; the documentation does not guarantee that it will be set to `nullptr` on failure, but it is common for implementations to set it to `NULL` to avoid using an uninitialized pointer.  
Therefore, the return code itself indicates failure; a `nullptr` is just a possible state of the pointer argument.  

To verify this, the program below:
1. Attempts a normal allocation that should succeed and prints the returned error code and the pointer value.
2. Attempts a huge allocation that will fail (e.g., >1 TB) and prints the returned error code and the pointer value.
The results demonstrate that the failure case returns an error code directly, not a success code with a null pointer.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper to print the result of a cudaMalloc attempt
void testCudaMalloc(size_t bytes, const char *label) {
    void *d_ptr = NULL;
    cudaError_t err = cudaMalloc(&d_ptr, bytes);

    printf("Test: %s\n", label);
    printf("  Requested size: %zu bytes\n", bytes);
    printf("  Return code: %d (%s)\n", err, cudaGetErrorString(err));
    printf("  Pointer returned: %p\n", d_ptr);
    printf("  d_ptr == NULL? %s\n", (d_ptr == NULL) ? "YES" : "NO");
    printf("\n");
}

int main() {
    // Ensure CUDA runtime is initialized
    cudaError_t initErr = cudaGetDeviceCount(NULL);
    if (initErr != cudaSuccess) {
        printf("CUDA initialization failed: %s\n", cudaGetErrorString(initErr));
        return 1;
    }

    // 1. Normal allocation (should succeed)
    size_t normalSize = 10 * 1024 * 1024; // 10 MB
    testCudaMalloc(normalSize, "Normal allocation (10 MB)");

    // 2. Very large allocation (likely to fail)
    size_t hugeSize = (size_t)1 << 40; // 1 TB
    testCudaMalloc(hugeSize, "Huge allocation (1 TB)");

    return 0;
}
```