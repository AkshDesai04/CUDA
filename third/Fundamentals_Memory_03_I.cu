```cpp
/*
Aim of the program:
Perform a three-way copy: HtoD to d_a, DtoD from d_a to d_b, DtoD from d_b to d_c. Verify d_c.

Thinking:
The goal is to demonstrate that data can be moved from host to device (HtoD),
then device-to-device (DtoD) twice, and that the final device array contains the same data
as the original host array. We'll use cudaMemcpy with appropriate flags for each
copy direction. After all copies, we copy the final device array back to the host
and compare element by element. We'll use a small integer array for simplicity.
We will also include basic error checking for all CUDA API calls.
*/

#include <iostream>
#include <cuda_runtime.h>

// Macro for CUDA error checking
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

int main() {
    const int N = 10;                 // Number of elements
    const size_t bytes = N * sizeof(int);

    // Host array initialization
    int h_a[N];
    for (int i = 0; i < N; ++i) h_a[i] = i;

    // Device pointers
    int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, bytes));

    // 1. Host to Device: copy h_a to d_a
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    // 2. Device to Device: copy d_a to d_b
    CUDA_CHECK(cudaMemcpy(d_b, d_a, bytes, cudaMemcpyDeviceToDevice));

    // 3. Device to Device: copy d_b to d_c
    CUDA_CHECK(cudaMemcpy(d_c, d_b, bytes, cudaMemcpyDeviceToDevice));

    // Verify: copy d_c back to host
    int h_c[N];
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Check that h_c matches original h_a
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != h_a[i]) {
            std::cerr << "Mismatch at index " << i << ": "
                      << "expected " << h_a[i] << ", got " << h_c[i] << std::endl;
            success = false;
        }
    }

    if (success) {
        std::cout << "Verification succeeded: d_c contains the correct data." << std::endl;
    } else {
        std::cout << "Verification failed: d_c does not match original data." << std::endl;
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}
```