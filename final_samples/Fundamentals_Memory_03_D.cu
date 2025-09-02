```cpp
/*
Use `cudaMemset` to initialize `d_a` to a value (e.g., 5) instead of copying from the host. Then perform the DtoD copy.

Thinking:
- Allocate two device arrays `d_a` and `d_b`.
- Use `cudaMemset` to set every byte of `d_a` to the value 5.
  Note that `cudaMemset` works on bytes, so each 4‑byte integer will become 0x05050505 (84215045 in decimal), not the integer 5.
  If we truly wanted each element to be 5, we would need a kernel, but the requirement is to use `cudaMemset`.
- Perform a device‑to‑device copy from `d_a` to `d_b` using `cudaMemcpy` with the flag `cudaMemcpyDeviceToDevice`.
- Copy the data back to the host to verify the copy and print the values.
- Include error checking for all CUDA API calls.
*/

#include <cuda_runtime.h>
#include <iostream>

#define N 10

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__        \
                      << " : " << cudaGetErrorString(err) << std::endl;          \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

int main() {
    int *d_a = nullptr;
    int *d_b = nullptr;
    size_t bytes = N * sizeof(int);

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));

    // Initialize d_a using cudaMemset to byte value 5
    // This sets each byte to 5, so each 4-byte int becomes 0x05050505
    CHECK_CUDA(cudaMemset(d_a, 5, bytes));

    // Perform device-to-device copy from d_a to d_b
    CHECK_CUDA(cudaMemcpy(d_b, d_a, bytes, cudaMemcpyDeviceToDevice));

    // Copy data back to host for verification
    int h_a[N], h_b[N];
    CHECK_CUDA(cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost));

    std::cout << "Values in host array h_a (from d_a):" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << h_a[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Values in host array h_b (from d_b after DtoD copy):" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << h_b[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));

    return 0;
}
```